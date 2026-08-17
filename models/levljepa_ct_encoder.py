"""LeVLJEPA-CT vision encoder for the fair benchmark evaluation.

Self-contained port of the multimodal-ct ``VisionTower`` (Primus-M 3D ViT +
COLIPRI ``Conv3d(864->768)`` projector + ``AttentionPool1D(768, 12)`` pooler),
vendored here so the DALE-CT eval container can load our trained checkpoint
WITHOUT importing ``multimodal_ct`` (not installed in the eval env). The
architecture is identical to COLIPRI's ``ImageEncoder`` -- the same Primus
backbone, projector, and pooler -- so feature extraction reuses COLIPRI's
preprocessing verbatim (see ``_extract_levljepa_ct`` in
``utils/benchmark_backbone_loader.py``). ``build_primus_backbone`` +
``AttentionPool1D`` are copied verbatim from
``multimodal-ct/src/multimodal_ct/model/{vision,pooling}.py``; ``Primus`` comes
from ``dynamic_network_architectures`` (a transitive dep of ``colipri``, present
in the eval env).

The trained Lightning checkpoint stores the encoder under
``model.vision_encoder.{backbone,projector,pooler}.*``; ``load_levljepa_ct_weights``
strips that prefix and loads with ``strict=False`` (the ``probe_head.*`` monitor
head and any text-encoder keys are correctly absent).

Two extract variants share one encoder:
  - raw    : ``forward(x, project=False, pool=False)`` -> (1, 864, 24, 24, 24)
             -> flatten -> (13824, 864)   [matches COLIPRI's input_dim 864]
  - pooled : ``forward(x, project=True,  pool=True)  -> (1, 768)
             [the trained image_raw the online probe monitored]
"""
from __future__ import annotations

import torch
from dynamic_network_architectures.architectures.primus import Primus
from einops import rearrange
from torch import nn

# COLIPRI default config (multimodal-ct model/vision.py).
INPUT_SIZE = 192
IMAGE_EMBED_DIM = 864  # Primus-M embed dim
PROJECTION_DIM = 768  # COLIPRI aligned dim
PATCH_SIZE = (8, 8, 8)
POOL_NUM_HEADS = 12


def build_primus_backbone(
    *,
    input_size: int = INPUT_SIZE,
    embed_dim: int = IMAGE_EMBED_DIM,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> Primus:
    """Primus-M with the COLIPRI default config.

    depth 16, 12 heads, 8^3 patches, 3D RoPE (``use_rot_pos_emb``,
    ``use_abs_pos_embed=False``), LayerScale 0.1, DropPath 0.2, EVA02-MLP
    (``scale_attn_inner``), 0 register tokens.
    """
    return Primus(
        input_channels=1,
        num_classes=1,  # Primus builds a classification head; unused for encoding.
        eva_depth=16,
        eva_numheads=12,
        embed_dim=embed_dim,
        patch_embed_size=list(patch_size),
        input_shape=[input_size, input_size, input_size],
        use_rot_pos_emb=True,
        use_abs_pos_embed=False,
        drop_path_rate=0.2,
        init_values=0.1,
        scale_attn_inner=True,
        num_register_tokens=0,
    )


class AttentionPool1D(nn.Module):
    """AttentionPool1D, ported verbatim from multimodal-ct model/pooling.py (MIT,
    originally vendor/colipri/src/colipri/pooling.py)."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = x.mean(dim=1, keepdim=True)
        key = value = x
        pooled, _ = self.attn(query, key, value)
        return rearrange(pooled, "batch 1 embed_dim -> batch embed_dim")

    def to_dense(self) -> nn.Sequential:
        v_proj_in_weight_qkv = self.get_parameter("attn.in_proj_weight")
        v_proj_in_bias_qkv = self.get_parameter("attn.in_proj_bias")
        v_proj_out_weight = self.get_parameter("attn.out_proj.weight")
        v_proj_out_bias = self.get_parameter("attn.out_proj.bias")
        dim = v_proj_in_weight_qkv.shape[0] // 3
        v_proj_in_weight_v = v_proj_in_weight_qkv[2 * dim:]
        v_proj_in_bias_v = v_proj_in_bias_qkv[2 * dim:]

        value_projection = nn.Conv3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
        )
        value_projection.weight.data = rearrange(
            v_proj_in_weight_v,
            "c_out c_in -> c_out c_in 1 1 1",
        )
        assert value_projection.bias is not None
        value_projection.bias.data = v_proj_in_bias_v

        out_projection = nn.Conv3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
        )
        out_projection.weight.data = rearrange(
            v_proj_out_weight,
            "c_out c_in -> c_out c_in 1 1 1",
        )
        assert out_projection.bias is not None
        out_projection.bias.data = v_proj_out_bias

        return nn.Sequential(
            value_projection,
            out_projection,
        )


class LeVLJEPACTEncoder(nn.Module):
    """Primus-M 3D encoder + COLIPRI ``Conv3d(864->768)`` projector +
    ``AttentionPool1D(768, 12)`` pooler. Trimmed port of multimodal-ct
    ``VisionTower`` (model/vision.py); ``forward`` mirrors
    ``VisionTower.forward`` (project -> optional pool)."""

    def __init__(
        self,
        *,
        input_size: int = INPUT_SIZE,
        embed_dim: int = IMAGE_EMBED_DIM,
        proj_dim: int = PROJECTION_DIM,
        num_heads: int = POOL_NUM_HEADS,
    ) -> None:
        super().__init__()
        backbone = build_primus_backbone(input_size=input_size, embed_dim=embed_dim)
        # COLIPRI nullifies Primus's decoder (up_projection) for encoding-only use.
        if hasattr(backbone, "up_projection"):
            backbone.up_projection = nn.Identity()  # type: ignore[assignment]
        self.backbone = backbone
        self.projector = nn.Conv3d(embed_dim, proj_dim, kernel_size=1)
        self.pooler = AttentionPool1D(proj_dim, num_heads)
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Dense token grid ``[B, embed_dim, X, Y, Z]`` (192^3 / 8 -> 24^3 tokens)."""
        return self.backbone(images.to(self.device))

    def forward(
        self,
        images: torch.Tensor,
        *,
        project: bool = True,
        pool: bool = True,
    ) -> torch.Tensor:
        if pool and not project:
            msg = "Pooling requires project=True."
            raise ValueError(msg)
        emb = self.encode(images)
        if project:
            emb = self.projector(emb)
            if pool:
                sequence = rearrange(emb, "b c x y z -> b (x y z) c")
                emb = self.pooler(sequence)
            else:
                emb = self.pooler.to_dense()(emb)
        return emb


def load_levljepa_ct_weights(encoder: LeVLJEPACTEncoder, ckpt_path: str) -> tuple[list[str], list[str]]:
    """Load the ``model.vision_encoder.*`` subset of a LeVLJEPA-CT Lightning
    checkpoint into ``encoder`` (prefix stripped, ``strict=False``).

    Returns ``(missing, unexpected)``. ``unexpected`` must be empty (every
    trained vision_encoder key maps onto the encoder); ``missing`` holds encoder
    params with no checkpoint source (e.g. Primus's unused classification head).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    prefix = "model.vision_encoder."
    enc_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    print(f"[LeVLJEPA-CT] loading {len(enc_sd)} vision_encoder keys from {ckpt_path}")
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"[LeVLJEPA-CT] unexpected keys loading encoder: {unexpected}")
    print(f"[LeVLJEPA-CT] load_state_dict: {len(missing)} missing keys (unused heads if any)")
    return missing, unexpected

"""
Unified HuggingFace Backbone Loader for Ablation Study.

Loads all 6 frozen ViT backbones:
  - Kentucky-Open-Science/DALE-CT-0   (timm, patch_size=16, img_size=512,  no registers)
  - Kentucky-Open-Science/DALE-CT-1S  (timm, patch_size=14, img_size=518,  no registers)
  - Kentucky-Open-Science/DALE-CT-2S  (timm, patch_size=16, img_size=512,  no registers)
  - DALE-CT-1S-v2                       (timm, patch_size=16, img_size=512,  no registers, LOCAL ckpt)
  - Kentucky-Open-Science/Finetuned-DINOv2-Chest-CT  (HF transformers, patch_size=14, img_size=518, 4 registers)
  - DALE-CT-0-L ("chest")               (timm, patch_size=16, img_size=512,  no registers, LOCAL ckpt; full 296k pool)

Each model is loaded once and cached for reuse across embedding extraction runs.

Usage:
    from utils.hf_backbone_loader import load_backbone, BACKBONE_SPECS
    model, spec = load_backbone("lejepa_0")
"""

import os
import sys

import torch
import torch.nn as nn

# --- Model specifications ---
BACKBONE_SPECS = {
    "lejepa_0": {
        "repo_id": "Kentucky-Open-Science/DALE-CT-0",
        "backbone_type": "timm",
        "patch_size": 16,
        "native_img_size": 512,
        "in_chans": 1,
        "embed_dim": 1024,
        "num_register_tokens": 0,
        "display_name": "LeJEPA-0",
    },
    "lejepa_1s": {
        "repo_id": "Kentucky-Open-Science/DALE-CT-1S",
        "backbone_type": "timm",
        "patch_size": 14,
        "native_img_size": 518,
        "in_chans": 1,
        "embed_dim": 1024,
        "num_register_tokens": 0,
        "display_name": "LeJEPA-1S",
    },
    "lejepa_2s": {
        "repo_id": "Kentucky-Open-Science/DALE-CT-2S",
        "backbone_type": "timm",
        "patch_size": 16,
        "native_img_size": 512,
        "in_chans": 1,
        "embed_dim": 1024,
        "num_register_tokens": 0,
        "display_name": "LeJEPA-2S",
    },
    "lejepa_1s_v2": {
        # DALE-CT-1S-v2: 2S architecture (patch16, dense+global TS soft labels, ReX-inert).
        "repo_id": "Kentucky-Open-Science/DALE-CT-1S-v2",
        "backbone_type": "timm",
        "patch_size": 16,
        "native_img_size": 512,
        "in_chans": 1,
        "embed_dim": 1024,
        "num_register_tokens": 0,
        "display_name": "LeJEPA-1S-v2",
    },
    "dinov2_ct": {
        "repo_id": "Kentucky-Open-Science/Finetuned-DINOv2-Chest-CT",
        "backbone_type": "hf",
        "patch_size": 14,
        "native_img_size": 518,
        "in_chans": 1,
        "embed_dim": 1024,
        "num_register_tokens": 4,
        "display_name": "DINOv2-CT",
    },
    # --- Depth-aware (2.5D) vs pure-2D ViT-Base LeJEPA ablation backbones
    #     (manuscript Table V). dinov2_with_registers ViT-Base (depth12, dim768,
    #     heads12, patch14, 518, 4 regs, 1ch) -- SAME model_type as dinov2_ct
    #     but Base scale. DINOv2-init, 35k iter, CT-RATE, norm_type=
    #     div1000. ONLY difference
    #     between the two arms is slab_size=12.0 (2.5D) vs single_slice=true (2D).
    #     NOT on the Hub -- loaded from LOCAL iter_35000 dirs via
    #     AutoModel.from_pretrained (dir holds config.json + model.safetensors). ---
    "lejepa_base_25d": {
        "repo_id": "/app/project/ibi-staff/CT-JEPA/outputs/Guided_Chest_CT_LeJEPA_V2_base_iso_pretrained/model_checkpoints/iter_35000",
        "backbone_type": "hf",
        "patch_size": 14,
        "native_img_size": 518,
        "in_chans": 1,
        "embed_dim": 768,
        "num_register_tokens": 4,
        "display_name": "LeJEPA-Base-2.5D",
    },
    "lejepa_base_2d": {
        "repo_id": "/app/project/ibi-staff/CT-JEPA/outputs/Guided_Chest_CT_LeJEPA_V2_base_iso_pretrained_2d/model_checkpoints/iter_35000",
        "backbone_type": "hf",
        "patch_size": 14,
        "native_img_size": 518,
        "in_chans": 1,
        "embed_dim": 768,
        "num_register_tokens": 4,
        "display_name": "LeJEPA-Base-2D",
    },
    "lejepa_0_chest": {
        # DALE-CT-0-L: variant-0 pure SSL on the full ~296k multi-source pool
        # (iter_191357, no aux head, ViT-L/patch16). The clip/mean/std fields are
        # its TRAINING z-score stats (full-pool foreground fit); the embed
        # preprocessor reads these per-model so 0-L features are normalized with
        # its own stats, not the CT-RATE defaults the other models use.
        "repo_id": "Kentucky-Open-Science/DALE-CT-0-L",
        "backbone_type": "timm",
        "patch_size": 16,
        "native_img_size": 512,
        "in_chans": 1,
        "embed_dim": 1024,
        "num_register_tokens": 0,
        "display_name": "DALE-CT-0-L",
        "clip_min": -940.8,
        "clip_max": 923.1,
        "mean_hu": -25.03,
        "std_hu": 246.87,
    },
}

# All model keys in canonical order
ALL_MODEL_KEYS = ["lejepa_0", "lejepa_1s", "lejepa_2s", "lejepa_1s_v2", "dinov2_ct", "lejepa_0_chest"]


def load_backbone_timm(repo_id: str, in_chans: int, patch_size: int, img_size: int,
                       weights_path: str | None = None) -> nn.Module:
    """
    Load a timm-based LeJEPA backbone from HuggingFace Hub, or from a local
    safetensors checkpoint when weights_path is given (used for DALE-CT-1S-v2,
    which is not published to the Hub).

    Uses the pattern from the model card:
        timm.create_model("vit_large_patch14_dinov2", ...)
        hf_hub_download + safetensors load_file
    """
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    src = weights_path if weights_path is not None else f"{repo_id}/model.safetensors"
    print(f"[Backbone] Loading timm model from {src}")
    print(f"           in_chans={in_chans}, patch_size={patch_size}, img_size={img_size}")

    model = timm.create_model(
        "vit_large_patch14_dinov2",
        pretrained=False,
        num_classes=0,
        in_chans=in_chans,
        patch_size=patch_size,
        img_size=img_size,
        dynamic_img_size=True,
    )

    if weights_path is not None:
        state_dict = load_file(weights_path)
    else:
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = load_file(model_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Backbone] WARNING missing keys ({len(missing)}): {missing[:5]}"
              f"{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[Backbone] INFO unexpected keys ({len(unexpected)}): {unexpected[:5]}"
              f"{'...' if len(unexpected) > 5 else ''}")
    model.eval()

    print(f"[Backbone] Loaded successfully.")
    return model


def load_backbone_hf(repo_id: str, attn_implementation: str | None = None) -> nn.Module:
    """
    Load a HuggingFace transformers-based DINOv2 backbone.

    Uses the pattern from the model card:
        AutoModel.from_pretrained(repo_id, trust_remote_code=True)

    attn_implementation: if set (e.g. "eager"), requested at load time. Needed to
    materialize attention matrices — DINOv2 defaults to "sdpa" which silently
    ignores output_attentions=True.
    """
    from transformers import AutoModel

    print(f"[Backbone] Loading HF transformers model from {repo_id}")

    kwargs = {"trust_remote_code": True}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModel.from_pretrained(repo_id, **kwargs)
    model.eval()

    print(f"[Backbone] Loaded successfully.")
    return model


def load_backbone(model_key: str, attn_implementation: str | None = None) -> tuple:
    """
    Load a frozen backbone by model key.

    Args:
        model_key: One of 'lejepa_0', 'lejepa_1s', 'lejepa_2s', 'lejepa_1s_v2', 'dinov2_ct'.
        attn_implementation: forwarded to HF loader only (e.g. "eager" to
            materialize attentions). timm path is unaffected.

    Returns:
        (model, spec_dict) where model is a frozen nn.Module and spec_dict
        contains the model specification.
    """
    if model_key not in BACKBONE_SPECS:
        raise ValueError(
            f"Unknown model_key '{model_key}'. "
            f"Must be one of: {list(BACKBONE_SPECS.keys())}"
        )

    spec = BACKBONE_SPECS[model_key]
    backbone_type = spec["backbone_type"]

    if backbone_type == "timm":
        model = load_backbone_timm(
            repo_id=spec["repo_id"],
            in_chans=spec["in_chans"],
            patch_size=spec["patch_size"],
            img_size=spec["native_img_size"],
            weights_path=spec.get("weights_path"),
        )
    elif backbone_type == "hf":
        model = load_backbone_hf(repo_id=spec["repo_id"],
                                 attn_implementation=attn_implementation)
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")

    # Freeze
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model, spec


def extract_cls_token(model, x: torch.Tensor, backbone_type: str,
                      num_register_tokens: int = 0) -> torch.Tensor:
    """
    Run backbone and extract [CLS] token.

    Args:
        model: Frozen ViT backbone.
        x: Input tensor [B, C, H, W].
        backbone_type: "timm" or "hf".
        num_register_tokens: Number of register tokens to skip (DINOv2 has 4).

    Returns:
        cls_token: [B, embed_dim]
    """
    if backbone_type == "timm":
        out = model.forward_features(x)
        if out.dim() == 3:
            return out[:, 0, :]
        else:
            return out
    else:
        out = model(pixel_values=x)
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state[:, 0, :]
        elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        elif out.dim() == 3:
            return out[:, 0, :]
        else:
            return out


def extract_patch_tokens(model, x: torch.Tensor, backbone_type: str,
                         num_register_tokens: int = 0) -> torch.Tensor:
    """
    Run backbone and extract patch tokens (excluding CLS and registers).

    Args:
        model: Frozen ViT backbone.
        x: Input tensor [B, C, H, W].
        backbone_type: "timm" or "hf".
        num_register_tokens: Number of register tokens to skip.

    Returns:
        patch_tokens: [B, N_patches, embed_dim]
    """
    if backbone_type == "timm":
        out = model.forward_features(x)
        if out.dim() == 3:
            hidden = out
        else:
            raise RuntimeError(
                f"Expected backbone output of shape [B, N, D], got {out.shape}"
            )
    else:
        out = model(pixel_values=x)
        if hasattr(out, 'last_hidden_state'):
            hidden = out.last_hidden_state
        elif out.dim() == 3:
            hidden = out
        else:
            raise RuntimeError(
                f"Expected backbone output of shape [B, N, D], got {out.shape}"
            )

    # Remove [CLS] token (index 0) and register tokens (indices 1..num_register_tokens)
    patch_tokens = hidden[:, 1 + num_register_tokens:, :]
    return patch_tokens

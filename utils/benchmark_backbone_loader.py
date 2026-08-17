"""
Benchmark CT Foundation Model Backbone Loader.

Loads 6 frozen benchmark backbones and extracts per-volume token-sequence
embeddings that drop straight into the error_bars MIL probe pipeline
(dataloaders/dataloader_embeddings.py reads .npz key "embeddings" of shape
(seq, D)):

  - rad_dino : 2D HF Dinov2Model          per-slice CLS  -> (n_slices, 768)
  - dinov3   : 2D HF DINOv3ViTModel       per-slice CLS  -> (n_slices, 1024)
  - colipri  : 3D Primus-M (colipri)      raw 3D patches -> (13824, 864)
  - ct_clip  : 3D CTViT (transformer_maskgit) tokens   -> (13824, 512)
  - merlin   : 3D i3_resnet (merlin-vlm)  pooled embed  -> (1, 2048)
  - ct_fm    : 3D SegResEncoder (MONAI)   1 token/patch -> (N_patches, 512)
  - levljepa_ct_raw    : 3D Primus-M (our LeVLJEPA-CT)  raw 3D patches -> (13824, 864)
  - levljepa_ct_pooled : 3D Primus-M (our LeVLJEPA-CT)  pooled embed  -> (1, 768)

Each model is loaded once (frozen, .eval()) and cached for reuse across
volumes. Native per-model preprocessing is applied INSIDE
extract_volume_features -- NO DALE-CT body-crop. Only CT-FM natively
foreground-crops; the others are full-FOV (see BENCHMARK_EMBEDDINGS_README.md).

3D volumes arrive as raw-HU (D, H, W) arrays with no affine/spacing sidecar
(the .npy / WebDataset sources carry no metadata). Per the plan's documented
approximation, spacing resampling is replaced by a direct resize to each
model's native input shape; this is smoke-test-gated. Axis order is fed as
(1, 1, D, H, W); flagged in the README.

Usage:
    from utils.benchmark_backbone_loader import load_benchmark_model, extract_volume_features
    model, spec = load_benchmark_model("rad_dino", device)
    feats = extract_volume_features(model, spec, volume_hu, device)  # (seq, D) float32
"""

import numpy as np
import torch
import torch.nn.functional as F

# --- Model specifications ---
# `weights` are defaults pointing at the local benchmark_models/ layout; override
# via the config on the DGX (configs/benchmark_embeddings.yaml).
BENCHMARK_SPECS = {
    "rad_dino": {
        "family": "2d_hf",
        "weights": "benchmark_models/rad-dino",          # HF dir (model.safetensors)
        "embed_dim": 768,
        "hu_window": [-1500.0, 600.0],                    # lung window -> [0,255] uint8
        "processor_use_fast": False,                      # BitImageProcessor (matches rad_dino src)
        "display_name": "RAD-DINO",
    },
    "dinov3": {
        "family": "2d_hf",
        "weights": "benchmark_models/dinov3-vitl16-pretrain-lvd1689m",
        "embed_dim": 1024,
        "hu_window": [-1500.0, 600.0],
        "processor_use_fast": True,                       # DINOv3ViTImageProcessorFast
        "display_name": "DINOv3",
    },
    "colipri": {
        "family": "3d_colipri",
        "weights": "benchmark_models/colipri/model.safetensors",
        "embed_dim": 864,                                 # raw backbone (projector->768 unused)
        "input_size": 192,                                # 192^3, patch 8^3 -> 24^3 = 13824 tokens
        "hu_clip": [-1000.0, 1000.0],
        "display_name": "COLIPRI",
    },
    "levljepa_ct_raw": {
        "family": "3d_levljepa_ct",
        # Our trained LeVLJEPA-CT (Primus-M arch == COLIPRI); weights overridden by
        # the config to a Lightning .ckpt on the DGX. Raw backbone tokens, matching
        # COLIPRI's input_dim 864 (projector->768 + pooler unused on this path).
        "weights": "benchmark_models/levljepa_ct/24987_last.ckpt",
        "embed_dim": 864,
        "input_size": 192,                                # 192^3, patch 8^3 -> 24^3 = 13824 tokens
        "hu_clip": [-1000.0, 1000.0],
        "pool": False,                                    # project=False, pool=False -> raw backbone
        "display_name": "LeVLJEPA-CT (raw)",
    },
    "levljepa_ct_pooled": {
        "family": "3d_levljepa_ct",
        "weights": "benchmark_models/levljepa_ct/24987_last.ckpt",
        "embed_dim": 768,                                 # trained pooled image_raw (project+pool)
        "input_size": 192,
        "hu_clip": [-1000.0, 1000.0],
        "pool": True,                                     # project=True, pool=True -> (1, 768)
        "display_name": "LeVLJEPA-CT (pooled)",
    },
    "levljepa_ct_raw_lr1e5": {
        "family": "3d_levljepa_ct",
        # 25117 = LeVLJEPA-CT WSD lr_backbone=1e-5 (the 'cosine' run that was actually
        # WSD on the :lazy6 image — lr_schedule=cosine was a no-op). Same arch/extract as
        # 24987; weights overridden by the config to 25117_last.ckpt on the DGX. Own output
        # dirs (no 24987 overwrite).
        "weights": "benchmark_models/levljepa_ct/25117_last.ckpt",
        "embed_dim": 864,
        "input_size": 192,
        "hu_clip": [-1000.0, 1000.0],
        "pool": False,
        "display_name": "LeVLJEPA-CT 1e-5 (raw)",
    },
    "levljepa_ct_pooled_lr1e5": {
        "family": "3d_levljepa_ct",
        "weights": "benchmark_models/levljepa_ct/25117_last.ckpt",
        "embed_dim": 768,
        "input_size": 192,
        "hu_clip": [-1000.0, 1000.0],
        "pool": True,
        "display_name": "LeVLJEPA-CT 1e-5 (pooled)",
    },
    # --- multimodal-ct master-log arms (region-vs-global experiment) -------------------
    # Same Primus-M architecture and extraction path as the 24987/25117 entries above;
    # only the checkpoint differs. `raw` (pool=False) is the readout that won the earlier
    # benchmark, so it is the one registered for each arm. Checkpoints live in the
    # multimodal-ct repo, so `weights` is an absolute DGX path rather than the local
    # benchmark_models/ layout -- no copy to keep in sync with a still-training run.
    #
    # The pair that carries the experiment: g3_fix_lr1e4 is the GLOBAL baseline and
    # r1_region_k1 is the REGION arm at MATCHED SIGReg N, so the difference between those
    # two is pairing granularity alone. r1_region_full is the method uncapped.
    "levljepa_ct_g3_global": {
        "family": "3d_levljepa_ct",
        "weights": "/project/ibi-staff/CT-JEPA/public/multimodal-ct/runs/g3/"
                   "g3_fix_lr1e4_global_ep1/ckpt/last.ckpt",
        "embed_dim": 864,
        "input_size": 192,
        "hu_clip": [-1000.0, 1000.0],
        "pool": False,
        "display_name": "LeVLJEPA-CT global (g3 lr1e-4)",
    },
    "levljepa_ct_r1_region_k1": {
        "family": "3d_levljepa_ct",
        "weights": "/project/ibi-staff/CT-JEPA/public/multimodal-ct/runs/r1/"
                   "r1_region_k1_region_ep1/ckpt/last.ckpt",
        "embed_dim": 864,
        "input_size": 192,
        "hu_clip": [-1000.0, 1000.0],
        "pool": False,
        "display_name": "LeVLJEPA-CT region k=1 (matched N)",
    },
    "levljepa_ct_r1_region_full": {
        "family": "3d_levljepa_ct",
        "weights": "/project/ibi-staff/CT-JEPA/public/multimodal-ct/runs/r1/"
                   "r1_region_full_region_ep1/ckpt/last.ckpt",
        "embed_dim": 864,
        "input_size": 192,
        "hu_clip": [-1000.0, 1000.0],
        "pool": False,
        "display_name": "LeVLJEPA-CT region uncapped",
    },
    "ct_clip": {
        "family": "3d_ctclip",
        "weights": "benchmark_models/CT-CHAT/CT-RATE-huggingface-downloads/CT-CLIP_v2.pt",
        "embed_dim": 512,
        "input_shape": (240, 480, 480),                   # (D, H, W); patch 10x20x20 -> 24^3
        "hu_clip": [-1000.0, 1000.0],
        "pad_value": -1.0,
        "display_name": "CT-CLIP",
    },
    "merlin": {
        "family": "3d_merlin",
        # Loaded manually from this .pt by _load_merlin (no auto-download; offline).
        "weights": "benchmark_models/Merlin/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt",
        "embed_dim": 2048,                                # pooled i3_resnet layer4 (matches CT-FM paper Merlin LP)
        "input_shape": (224, 224, 160),                   # (D,H,W) post-permute; 160 = slice/z = i3_resnet temporal dim
        "hu_clip": [-1000.0, 1000.0],
        "display_name": "Merlin",
    },
    "ct_fm": {
        "family": "3d_ctfm",
        "weights": "benchmark_models/CT-FM/huggingface/pretrained_segresnet.torch",
        "embed_dim": 512,
        "patch_size": (24, 128, 128),
        "overlap": 0.625,
        "batch_size": 32,
        "hu_clip": [-1024.0, 2048.0],
        "max_tokens": 4096,                               # optional safety cap (variable patch count)
        "display_name": "CT-FM",
    },
}

ALL_BENCHMARK_KEYS = ["rad_dino", "dinov3", "colipri", "ct_clip", "merlin", "ct_fm",
                      "levljepa_ct_raw", "levljepa_ct_pooled",
                      "levljepa_ct_raw_lr1e5", "levljepa_ct_pooled_lr1e5"]


# ---------------------------------------------------------------------------
# Preprocessing helpers (operate on raw-HU torch volumes of shape (D, H, W))
# ---------------------------------------------------------------------------
def _as_tensor(volume_hu):
    if isinstance(volume_hu, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(volume_hu)).float()
    else:
        t = volume_hu.float()
    if t.ndim != 3:
        raise ValueError(f"Expected (D,H,W) volume, got shape {tuple(t.shape)}")
    return t


def _resize_3d(vol, target):
    """Trilinear resize (D,H,W) -> target. Replaces affine-based resampling."""
    x = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    x = F.interpolate(x, size=tuple(target), mode="trilinear", align_corners=False)
    return x[0, 0]


def _crop_or_pad_3d(vol, target, pad_value=0.0):
    """Center crop/pad (D,H,W) -> target, filling with pad_value."""
    out = torch.full(tuple(target), float(pad_value), dtype=vol.dtype)
    d, h, w = vol.shape
    td, th, tw = target
    d0 = max((d - td) // 2, 0); h0 = max((h - th) // 2, 0); w0 = max((w - tw) // 2, 0)
    de, he, we = min(d0 + td, d), min(h0 + th, h), min(w0 + tw, w)
    src = vol[d0:de, h0:he, w0:we]
    sd, sh, sw = src.shape
    pd0 = (td - sd) // 2; ph0 = (th - sh) // 2; pw0 = (tw - sw) // 2
    out[pd0:pd0 + sd, ph0:ph0 + sh, pw0:pw0 + sw] = src
    return out


def _foreground_crop(vol, thr=0.0):
    """Tight bounding box of vol > thr (replicates MONAI CropForeground default
    select_fn x>0 on raw HU, as CT-FM's native pipeline applies)."""
    mask = vol > thr
    if not mask.any():
        return vol
    ds, hs, ws = torch.where(mask)
    return vol[ds.min():ds.max() + 1, hs.min():hs.max() + 1, ws.min():ws.max() + 1]


def _flatten_tokens(emb, embed_dim):
    """Normalize a backbone output to (B, N, C) and return (N, C).

    Handles both channel-first (B,C,D,H,W)/(B,C,N) and channel-last layouts.
    """
    if emb.ndim == 5:
        b = emb.shape[0]
        if emb.shape[1] == embed_dim:            # (B,C,D,H,W)
            emb = emb.permute(0, 2, 3, 4, 1)
        elif emb.shape[-1] != embed_dim:
            raise RuntimeError(f"Cannot locate {embed_dim}-d channel in shape {tuple(emb.shape)}")
        emb = emb.reshape(b, -1, embed_dim)
    elif emb.ndim == 4:
        b = emb.shape[0]
        if emb.shape[1] == embed_dim:            # (B,C,N)
            emb = emb.permute(0, 2, 1)
        elif emb.shape[-1] != embed_dim:
            raise RuntimeError(f"Cannot locate {embed_dim}-d channel in shape {tuple(emb.shape)}")
        emb = emb.reshape(b, -1, embed_dim)
    elif emb.ndim == 3:                          # (B,N,C) or (B,C,N)
        if emb.shape[-1] != embed_dim and emb.shape[1] == embed_dim:
            emb = emb.permute(0, 2, 1)
    else:
        raise RuntimeError(f"Unexpected embedding ndim {emb.ndim}, shape {tuple(emb.shape)}")
    if emb.shape[-1] != embed_dim:
        raise RuntimeError(
            f"Embedding dim {emb.shape[-1]} != expected {embed_dim} (shape {tuple(emb.shape)})"
        )
    return emb[0]  # (N, C)


# ---------------------------------------------------------------------------
# Per-family loaders (lazy imports so the module loads without all deps)
# ---------------------------------------------------------------------------
def _load_2d_hf(spec, attn_implementation=None):
    from transformers import AutoModel, AutoImageProcessor

    path = spec["weights"]
    kw = {}
    if attn_implementation is not None:
        kw["attn_implementation"] = attn_implementation
    print(f"[Backbone] Loading 2D HF model from {path}"
          + (f" (attn_implementation={attn_implementation})" if attn_implementation else ""))
    model = AutoModel.from_pretrained(path, **kw).eval()
    processor = AutoImageProcessor.from_pretrained(path, use_fast=spec.get("processor_use_fast", False))
    spec["_processor"] = processor
    return model


def _load_colipri(spec):
    import colipri

    print(f"[Backbone] Loading COLIPRI from {spec['weights']}")
    # image_only=True drops the text encoder; load_checkpoint_and_dispatch ignores
    # the unused text_encoder.* keys. If this ever rejects, the documented fallback
    # is to instantiate dynamic_network_architectures.Primus directly (see README).
    model = colipri.get_model(checkpoint_path=spec["weights"], image_only=True)
    return model


def _load_levljepa_ct(spec):
    """Load our trained LeVLJEPA-CT encoder (Primus-M arch == COLIPRI) from a
    Lightning .ckpt. Builds the encoder via the vendored
    ``models.levljepa_ct_encoder`` (no ``multimodal_ct`` import) and loads the
    ``model.vision_encoder.*`` subset; the same encoder serves both the raw and
    pooled extract variants (dispatched at extract time by ``spec["pool"]``)."""
    from models.levljepa_ct_encoder import LeVLJEPACTEncoder, load_levljepa_ct_weights

    print(f"[Backbone] Loading LeVLJEPA-CT from {spec['weights']}")
    enc = LeVLJEPACTEncoder()
    load_levljepa_ct_weights(enc, spec["weights"])
    return enc


def _load_ctclip(spec):
    from transformer_maskgit import CTViT

    print(f"[Backbone] Loading CT-CLIP from {spec['weights']}")
    enc = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )
    ckpt = torch.load(spec["weights"], map_location="cpu")
    sd = _extract_ctclip_visual(ckpt)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print(f"[CT-CLIP] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return enc


def _extract_ctclip_visual(ckpt):
    """Pull the visual_transformer state_dict out of a CT-CLIP_v2.pt checkpoint.

    The full checkpoint may be a flat state_dict with 'visual_transformer.*'
    keys, a nested {'visual_transformer': {...}, ...}, or already visual-only.
    """
    if isinstance(ckpt, dict) and "visual_transformer" in ckpt and isinstance(ckpt["visual_transformer"], dict):
        return ckpt["visual_transformer"]
    if isinstance(ckpt, dict) and any(k.startswith("visual_transformer.") for k in ckpt):
        return {k[len("visual_transformer."):]: v for k, v in ckpt.items() if k.startswith("visual_transformer.")}
    return ckpt  # assume already visual-only


def _load_merlin(spec):
    """Load Merlin's i3_resnet image encoder from a mounted .pt (offline).

    Builds the bare I3ResNet (3D ResNet-152) the way merlin-vlm's ImageEncoder
    does, then loads only the ``encode_image.i3_resnet.*`` weights from the
    checkpoint -- stripping that prefix so the keys line up with the bare encoder.
    This avoids ``Merlin(ImageEmbedding=True)``, whose constructor always pulls
    Clinical-Longformer from the HF Hub (unreachable on offline compute nodes) and
    auto-downloads the .pt to a site-packages cache. Verified from the merlin-vlm
    source (merlin/models/{i3res,build,load}.py); smoke-test-gated.
    """
    import copy
    import torchvision
    from merlin.models import i3res

    weights_path = spec["weights"]
    print(f"[Backbone] Loading Merlin i3_resnet from {weights_path} (manual, offline)")
    # resnet152(weights=None) builds the 2D architecture only -- no ImageNet
    # download; the inflated 3D weights come from the checkpoint.
    resnet2d = torchvision.models.resnet152(weights=None)
    encoder = i3res.I3ResNet(
        copy.deepcopy(resnet2d),
        class_nb=1692,          # matches merlin ImageEncoder; unused by ImageEmbedding forward
        conv_class=True,
        ImageEmbedding=True,
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    prefix = "encode_image.i3_resnet."
    enc_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"[Merlin] unexpected keys loading i3_resnet: {unexpected}")
    print(f"[Merlin] load_state_dict: {len(missing)} missing keys (unused heads if any)")
    return encoder


def _load_ctfm(spec):
    from monai.networks.nets.segresnet_ds import SegResEncoder

    print(f"[Backbone] Loading CT-FM from {spec['weights']}")
    weights = torch.load(spec["weights"], map_location="cpu")
    weights = {k.replace("encoder.", ""): v for k, v in weights.items()}  # encoder.* -> SegResEncoder
    model = SegResEncoder(
        blocks_down=(1, 2, 2, 4, 4),
        head_module=lambda x: F.adaptive_avg_pool3d(x[-1], 1).flatten(start_dim=1),  # 1 token/patch
    )
    model.load_state_dict(weights, strict=False)
    return model


def load_benchmark_model(model_key, device=None, spec_override=None, attn_implementation=None):
    """Load a frozen benchmark backbone by key.

    Args:
        model_key: one of ALL_BENCHMARK_KEYS.
        device: torch device to place the model on.
        spec_override: optional dict of spec keys to override (e.g. DGX weights
            paths from the config); merged on top of BENCHMARK_SPECS[model_key].
        attn_implementation: forwarded to the 2D HF loader only (e.g. "eager" to
            materialize attentions for output_attentions=True; 2D HF defaults to
            "sdpa" which silently ignores it). Ignored for other families.

    Returns:
        (model, spec) where model is a frozen .eval() nn.Module (on `device` if
        given) and spec is a copy of BENCHMARK_SPECS[model_key] with overrides
        applied (2D HF specs also carry an attached "_processor").
    """
    if model_key not in BENCHMARK_SPECS:
        raise ValueError(
            f"Unknown model_key '{model_key}'. Must be one of: {list(BENCHMARK_SPECS.keys())}"
        )
    spec = dict(BENCHMARK_SPECS[model_key])
    if spec_override:
        for k, v in dict(spec_override).items():
            spec[k] = v
    family = spec["family"]

    if family == "2d_hf":
        model = _load_2d_hf(spec, attn_implementation=attn_implementation)
    elif family == "3d_colipri":
        model = _load_colipri(spec)
    elif family == "3d_levljepa_ct":
        model = _load_levljepa_ct(spec)
    elif family == "3d_ctclip":
        model = _load_ctclip(spec)
    elif family == "3d_merlin":
        model = _load_merlin(spec)
    elif family == "3d_ctfm":
        model = _load_ctfm(spec)
    else:
        raise ValueError(f"Unknown family: {family}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    if device is not None:
        model = model.to(device)
    print(f"[Backbone] {spec['display_name']} loaded (embed_dim={spec['embed_dim']}, family={family})")
    return model, spec


# ---------------------------------------------------------------------------
# Per-family feature extraction (each applies its model's native preprocessing)
# ---------------------------------------------------------------------------
def _extract_2d_hf(model, spec, volume_hu, device):
    from PIL import Image

    processor = spec["_processor"]
    win_min, win_max = spec["hu_window"]
    # .cpu(): accelerate's prepared DataLoader auto-moves batches to the device
    # (device_placement=True), so volume_hu may arrive on cuda:0. The per-slice
    # windowing + PIL work below is CPU-side (model input is re-sent to `device`
    # at the processor call), so force CPU here -- vol[s].numpy() below would
    # otherwise raise "can't convert cuda:0 tensor to numpy".
    vol = _as_tensor(volume_hu).cpu()        # (D,H,W)
    n_slices = vol.shape[0]

    feats = []
    chunk = 16
    for i in range(0, n_slices, chunk):
        images = []
        for s in range(i, min(i + chunk, n_slices)):
            arr = vol[s].numpy()
            arr = np.clip(arr, win_min, win_max)
            arr = (arr - win_min) / (win_max - win_min)
            arr = (arr * 255.0).astype(np.uint8)
            images.append(Image.fromarray(arr, mode="L").convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        # CLS is index 0 for both Dinov2 (no registers) and DINOv3 (registers follow CLS).
        cls = out.last_hidden_state[:, 0, :]
        feats.append(cls.float().cpu())
    feats = torch.cat(feats, dim=0).numpy().astype(np.float32)   # (n_slices, embed_dim)
    return feats


def _extract_colipri(model, spec, volume_hu, device):
    vol = _as_tensor(volume_hu)
    lo, hi = spec["hu_clip"]
    size = spec["input_size"]
    vol = _resize_3d(vol, (size, size, size))         # approximates resample-to-2mm
    vol = torch.clamp(vol, lo, hi)
    vol = (vol - lo) / (hi - lo) * 2.0 - 1.0          # -> [-1, 1]
    vol = _crop_or_pad_3d(vol, (size, size, size), pad_value=-1.0)
    x = vol.unsqueeze(0).unsqueeze(0).to(device)      # (1,1,D,H,W)
    with torch.no_grad():
        emb = model.encode_image(x, project=False, pool=False, normalize=False)  # raw backbone
    tokens = _flatten_tokens(emb, spec["embed_dim"])  # (13824, 864)
    return tokens.float().cpu().numpy().astype(np.float32)


def _extract_levljepa_ct(model, spec, volume_hu, device):
    # Same architecture + preprocessing as COLIPRI (Primus-M, 192^3, HU clip
    # [-1000,1000] -> [-1,1]); differs only in the forward call, dispatched by
    # spec["pool"]: raw backbone tokens (project=False,pool=False -> (13824,864))
    # or trained pooled vector (project=True,pool=True -> (1,768)).
    vol = _as_tensor(volume_hu)
    lo, hi = spec["hu_clip"]
    size = spec["input_size"]
    vol = _resize_3d(vol, (size, size, size))         # approximates resample-to-2mm
    vol = torch.clamp(vol, lo, hi)
    vol = (vol - lo) / (hi - lo) * 2.0 - 1.0          # -> [-1, 1]
    vol = _crop_or_pad_3d(vol, (size, size, size), pad_value=-1.0)
    x = vol.unsqueeze(0).unsqueeze(0).to(device)      # (1,1,D,H,W)
    pool = spec["pool"]
    with torch.no_grad():
        emb = model(x, project=pool, pool=pool)
    if pool:
        feats = emb.reshape(-1, spec["embed_dim"])    # (1, 768)
    else:
        feats = _flatten_tokens(emb, spec["embed_dim"])  # (13824, 864)
    return feats.float().cpu().numpy().astype(np.float32)


def _extract_ctclip(model, spec, volume_hu, device):
    vol = _as_tensor(volume_hu)
    lo, hi = spec["hu_clip"]
    shape = spec["input_shape"]                       # (D,H,W)
    vol = torch.clamp(vol, lo, hi)
    vol = _resize_3d(vol, shape)
    vol = vol / 1000.0                                # clipped [-1000,1000] -> [-1,1]
    vol = _crop_or_pad_3d(vol, shape, pad_value=spec["pad_value"])
    x = vol.unsqueeze(0).unsqueeze(0).to(device)      # (1,1,D,H,W)
    with torch.no_grad():
        tokens = model(x, return_encoded_tokens=True)
    tokens = _flatten_tokens(tokens, spec["embed_dim"])  # (13824, 512)
    return tokens.float().cpu().numpy().astype(np.float32)


def _extract_merlin(model, spec, volume_hu, device):
    vol = _as_tensor(volume_hu)                       # (D, H, W); D = slices (z)
    lo, hi = spec["hu_clip"]
    shape = spec["input_shape"]                       # (D,H,W) post-permute; 160 = slice axis
    # i3_resnet treats the LAST input axis as the temporal/frame dim (forward
    # permutes 0,1,4,2,3; conv1 kernel is (3,7,7)). Native input is
    # (B,1,224,224,160) with 160 = z (slices), so move the volume's slice axis
    # (D) to last before resizing so it becomes the 160-wide temporal dim.
    vol = vol.permute(1, 2, 0)                        # (H, W, D_slices)
    vol = _resize_3d(vol, shape)                      # (224, 224, 160)
    vol = torch.clamp(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)                      # -> [0, 1]
    vol = _crop_or_pad_3d(vol, shape, pad_value=0.0)
    x = vol.unsqueeze(0).unsqueeze(0).to(device)      # (1,1,224,224,160)
    with torch.no_grad():
        out = model(x)
    # encoder returns (1, B, 2048) -- one pooled token per volume.
    emb = out[0] if isinstance(out, (tuple, list)) else out
    emb = emb.reshape(-1, spec["embed_dim"])          # (1, 2048)
    return emb.float().cpu().numpy().astype(np.float32)


def _extract_ctfm(model, spec, volume_hu, device):
    from monai.inferers import SlidingWindowSplitter

    vol = _as_tensor(volume_hu)
    lo, hi = spec["hu_clip"]
    vol = _foreground_crop(vol, 0.0)                  # native CropForeground (x>0 on raw HU)
    vol = torch.clamp(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)                      # -> [0, 1]
    x = vol.unsqueeze(0).unsqueeze(0)                 # (1,1,D,H,W)

    splitter = SlidingWindowSplitter(spec["patch_size"], spec["overlap"])

    def _to_5d(p):
        while p.ndim < 5:
            p = p.unsqueeze(0)
        while p.ndim > 5:
            p = p.squeeze(0)
        return p  # (1, 1, pd, ph, pw) = (B=1, C=1, D, H, W)

    patches = []
    for item in splitter(x):
        p = item[0] if isinstance(item, (tuple, list)) else item
        patches.append(_to_5d(p))

    feats = []
    bs = spec["batch_size"]
    with torch.no_grad():
        for i in range(0, len(patches), bs):
            batch = torch.cat(patches[i:i + bs], dim=0).to(device)   # (b, 1, pd, ph, pw)
            out = model(batch)                                        # (b, 512) -- 1 token/patch
            feats.append(out.float().cpu())
    feats = torch.cat(feats, dim=0)                                   # (N_patches, 512)

    cap = spec.get("max_tokens")
    if cap and feats.shape[0] > cap:
        feats = feats[:cap]
    return feats.numpy().astype(np.float32)


def extract_volume_features(model, spec, volume_hu, device):
    """Run a frozen benchmark backbone on one raw-HU volume -> (seq, D) float32.

    Args:
        model: frozen backbone from load_benchmark_model.
        spec: the spec dict returned by load_benchmark_model.
        volume_hu: raw-HU volume, shape (D, H, W) (numpy or torch).
        device: torch device for the forward pass.

    Returns:
        np.ndarray of shape (seq, embed_dim), float32 -- the per-volume token
        bag consumed by the error_bars MIL probe.
    """
    family = spec["family"]
    if family == "2d_hf":
        feats = _extract_2d_hf(model, spec, volume_hu, device)
    elif family == "3d_colipri":
        feats = _extract_colipri(model, spec, volume_hu, device)
    elif family == "3d_levljepa_ct":
        feats = _extract_levljepa_ct(model, spec, volume_hu, device)
    elif family == "3d_ctclip":
        feats = _extract_ctclip(model, spec, volume_hu, device)
    elif family == "3d_merlin":
        feats = _extract_merlin(model, spec, volume_hu, device)
    elif family == "3d_ctfm":
        feats = _extract_ctfm(model, spec, volume_hu, device)
    else:
        raise ValueError(f"Unknown family: {family}")

    if feats.ndim != 2 or feats.shape[1] != spec["embed_dim"]:
        raise RuntimeError(
            f"[{spec['display_name']}] expected (seq, {spec['embed_dim']}), got {feats.shape}"
        )
    return feats

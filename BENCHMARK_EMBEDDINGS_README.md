# Benchmark Foundation-Model Embeddings → Error-Bars Pipeline

Feature extraction for the 6 CT foundation-model baselines compared against the
DALE-CT family in `manuscript/main.tex`: **RAD-DINO, DINOv3, COLIPRI, CT-CLIP,
Merlin, CT-FM**. Each model's frozen backbone produces a per-volume
token-sequence embedding that drops straight into the existing
`ERROR_BARS_README.md` / `run_error_bars.py` pipeline — the *same* frozen-probe
MIL + 5-seed + bootstrap protocol as DALE-CT — so the resulting CI-equipped
numbers are directly comparable.

3D models contribute their **native 3D-patch token grids** (no pooling); 2D
models contribute **per-slice CLS**. Everything is saved as
`<base_name>.npz`, key `embeddings`, shape `(seq, D)` float32 — exactly what
`dataloaders/dataloader_embeddings.py` `CTScanMILDataset` consumes.

## The 6 models at a glance

| Key | Family | Output / volume | Dim | Native token count | Weights (mounted, under `benchmark_models/`) | Image |
|---|---|---|---|---|---|---|
| `rad_dino` | 2D HF `Dinov2Model` | `(n_slices, 768)` | 768 | n_slices (~100–300) | `rad-dino` | `evandamron/benchmark-rad_dino` |
| `dinov3` | 2D HF `DINOv3ViTModel` | `(n_slices, 1024)` | 1024 | n_slices | `dinov3-vitl16-pretrain-lvd1689m` | `evandamron/benchmark-dinov3` |
| `colipri` | 3D Primus-M (`colipri`) | `(13824, 864)` | 864 | 24³ = 13824 | `colipri/model.safetensors` | `evandamron/benchmark-colipri` |
| `ct_clip` | 3D CTViT (`transformer_maskgit`) | `(13824, 512)` | 512 | 24³ = 13824 | `CT-CHAT/CT-RATE-huggingface-downloads/CT-CLIP_v2.pt` | `evandamron/benchmark-ct_clip` |
| `merlin` | 3D i3_resnet (`merlin-vlm`) | `(1, 2048)` | 2048 | 1 (pooled) | `Merlin/i3_resnet_..._epoch_99.pt` | `evandamron/benchmark-merlin` |
| `ct_fm` | 3D `SegResEncoder` (MONAI) | `(N_patches, 512)` | 512 | N_patches (variable; ≤4096) | `CT-FM/huggingface/pretrained_segresnet.torch` | `evandamron/benchmark-ct_fm` |

The `input_dim` column is wired into `configs/error_bars.yaml` per model so the
probe (and the reloaded frozen-arm probe) use each model's real embed dim —
DALE-CT's 4 models fall back to the global `1024`. Source of truth for the
specs: `utils/benchmark_backbone_loader.py` `BENCHMARK_SPECS`; the values are
mirrored in `configs/benchmark_embeddings.yaml` and `configs/error_bars.yaml`.

## Per-model details

### RAD-DINO (`rad_dino`)
- **Image:** `FROM evandamron/lejepa:latest`; DEPS ONLY (no extra pip install — base `transformers` provides `Dinov2Model`). Weights + code are mounted at runtime; nothing is baked in. (Build via `build_benchmark_images.sh`.)
- **Load:** `AutoModel.from_pretrained` (HF `Dinov2Model`, 768-d, 518px) + `AutoImageProcessor` (`use_fast=False`, `BitImageProcessor`). Uses `model.safetensors` (not `backbone_compatible.safetensors`).
- **Native preprocess:** per slice, lung HU window `[-1500, 600] → [0, 255]` uint8 → PIL `L`→`RGB` → processor (518px, mean/std 0.5307/0.2583, `/255`). Full slice, no crop.
- **Output:** `last_hidden_state[:, 0, :]` per slice → `(n_slices, 768)`. CLS is index 0 (Dinov2 has no registers).

### DINOv3 (`dinov3`)
- **Image:** `dinov3/Dockerfile` adds `pip install -U "transformers>=4.56"` (required for `DINOv3ViTModel`); DEPS ONLY (weights + code mounted). Upgrading transformers is safe — this image only runs benchmark extraction; error_bars probes are torch-only and DALE-CT embeddings are already extracted.
- **Load:** `AutoModel.from_pretrained` (HF `DINOv3ViTModel`, 1024-d, 4 registers) + `AutoImageProcessor` (`use_fast=True`, 224px, ImageNet mean/std).
- **Native preprocess:** same 2D HU→pixel mapping as RAD-DINO (`[-1500, 600] → [0,255]`), full slice, no crop.
- **Output:** `last_hidden_state[:, 0, :]` → `(n_slices, 1024)`. CLS is still index 0 — the 4 register tokens follow CLS, so they are skipped automatically.

### COLIPRI (`colipri`)
- **Image:** `colipri/Dockerfile` `pip install colipri==0.1.2` from PyPI (== the local HF checkout's version, so the `get_model`/`encode_image` API matches); DEPS ONLY (weights + code mounted). Avoids the package's `uv_build` backend (fragile under cross-arch emulation) and shrinks the image.
- **Load:** `colipri.get_model(checkpoint_path=…, image_only=True)` → `model.image_encoder`. `image_only=True` drops the text encoder; the loader ignores the unused `text_encoder.*` keys.
- **Native preprocess:** resize to 192³ (approximates resample-to-2mm — see "No spacing sidecar" below), clamp `[-1000, 1000] → [-1, 1]`, center crop/pad to 192³ (pad `-1`). Matches `src/colipri/configs/processor/image_transform/default.yaml`.
- **Output:** `encode_image(project=False, pool=False, normalize=False)` → raw 864-d backbone grid `(B, 864, 24, 24, 24)` → flatten → `(13824, 864)`. The 768-d `projector` (CLIP space) is a one-line alternative; the raw 864-d backbone is used ("3D patches").
- **COLIPRI-C assumption:** the local checkpoint is taken as the **COLIPRI-C** variant cited in the manuscript. Confirm the checkpoint identity on the DGX if a specific variant is required.

### CT-CLIP (`ct_clip`)
- **Image:** `ctclip/Dockerfile` `pip install "git+https://github.com/ibrahimethemhamamci/CT-CLIP.git#subdirectory=transformer_maskgit"` (`transformer_maskgit` is a subdir of the CT-CLIP repo, not a standalone repo); DEPS ONLY (weights + code mounted).
- **Load:** `transformer_maskgit.CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10, spatial_depth=4, temporal_depth=4, dim_head=32, heads=8)`; `visual_transformer.*` keys pulled from `CT-CLIP_v2.pt` (`strict=False`).
- **Native preprocess:** clamp `[-1000, 1000]`, resize to `(240, 480, 480)`, `/1000` → `[-1, 1]`, center crop/pad to `(240, 480, 480)` (pad `-1`). Matches `CT-CHAT/encode_script.py`.
- **Output:** `model(x, return_encoded_tokens=True)` → `(24, 24, 24)` grid → flatten → `(13824, 512)`.
- **Smoke-test-gated.** The `transformer_maskgit` API and the `visual_transformer.*` key extraction are verified from source/keys but **not executed locally** — confirm at `--limit 2` before the full run.

### Merlin (`merlin`)
- **Image:** `merlin/Dockerfile` `pip install merlin-vlm`; DEPS ONLY. The loader loads weights manually from the mounted `.pt` (no auto-download; runtime is offline).
- **Load:** weights loaded MANUALLY from the mounted `.pt` by `_load_merlin` (3D i3_resnet / ResNet-152) -- no `Merlin(ImageEmbedding=True)` auto-download (compute nodes are offline). The encoder is constructed bare and `load_state_dict`'d from `i3_resnet_..._epoch_99.pt`.
- **Native preprocess:** resize to `(224, 224, 160)` with the slice (z) axis as the 160-wide temporal dim (i3_resnet forward permutes the last axis to temporal; conv1 kernel `(3,7,7)`); 1-channel (replicated to 3 internally), clamp `[-1000, 1000] → [0, 1]`, center crop/pad (pad `0`). Matches the `merlin-vlm` `ImageTransforms` (RAS, spacing (1.5,1.5,3), SpatialPad+CenterSpatialCrop to 224x224x160); no foreground crop is native — **confirm at smoke test**.
- **Output:** `model(x)` → pooled `(1, 2048)`. This is a **single pooled embedding per volume**, matching the CT-FM paper's own Merlin linear-probe protocol (`extract_feat_LP.py`). Not a token grid.
- **Smoke-test-gated.** The `i3res.I3ResNet` constructor, the `encode_image.i3_resnet.` key prefix, and the native `224x224x160` input are verified from the `merlin-vlm` source but **not executed locally** (the package isn't installed on this Mac). Confirm at `--limit 2` that the manual load + forward produce a `(1, 2048)` embedding.

### CT-FM (`ct_fm`)
- **Image:** `ctfm/Dockerfile` `pip install monai`; DEPS ONLY (weights + code mounted).
- **Load:** `monai SegResEncoder(blocks_down=(1,2,2,4,4), head_module=lambda x: F.adaptive_avg_pool3d(x[-1],1).flatten(1))`; `encoder.` prefix stripped, `load_state_dict(strict=False)`.
- **Native preprocess:** `CropForeground` (tight bbox of `vol > 0` on raw HU — CT-FM's own native step), clamp `[-1024, 2048] → [0, 1]`. Matches `feature_extractor.py`. No body-crop beyond the native foreground crop.
- **Output:** `SlidingWindowSplitter((24, 128, 128), overlap=0.625)` → per patch, the native **1-token-per-patch** `head_module` → `(N_patches, 512)`. `max_tokens=4096` is an optional safety cap for the variable patch count. This is CT-FM's own native feature (the 128-token/patch alternative exceeds 2 TB and is not native).

## Native-preprocessing notes

- **2D HU→image window.** RAD-DINO and DINOv3 were not trained on CT, so a HU→pixel mapping must be chosen. Default = lung window `[-1500, 600] → [0, 255]`, per-model configurable via `hu_window` in the config. Verify visually in the smoke test (slices should look like lung CT, not washed-out or clipped).
- **No spacing sidecar.** The `.npy` / WebDataset sources carry no affine/spacing metadata, so spacing-based resampling is replaced by a direct **resize to each model's native input shape** (192³ / 240×480×480 / 160×224×224). CT-RATE's fairly uniform spacing makes this a reasonable approximation; the smoke test confirms volumes are not distorted. CT-FM resamples to its native `[3, 1, 1]` spacing only conceptually — in practice the foreground-crop + sliding window handles variable shapes, and the resize is skipped (the splitter pads internally).
- **Axis order.** Volumes are fed as `(1, 1, D, H, W)`; the loaders assume the dataset yields `(D, H, W)`.
- **No DALE-CT body-crop anywhere.** Each model gets its authors' intended input. Only CT-FM natively foreground-crops; the other 5 are full-FOV. See Disclosure (a).

## How to launch (DGX)

All commands run on the DGX (data is on `/project/ibi-staff/CT-JEPA`). The
images are DEPENDENCIES ONLY -- weights and code are mounted at runtime, so the
build context is small (`.dockerignore` excludes the 102 GB `benchmark_models/`).

### 0. Sync `benchmark_models/` to the DGX (one-time)
The weights are MOUNTED, not baked into images, so they must exist on the DGX
before any extraction job runs. `benchmark_models/` is untracked in git (102 GB),
so a `git push` will NOT move it -- rsync it to the repo path on the DGX:
```bash
# from your Mac:
rsync -aP --info=progress2 benchmark_models/ \
    <dgx-login>:/project/ibi-staff/CT-JEPA/public/benchmark_models/
```
The `/project:/app/project` Pyxis mount then exposes it in-container at
`/app/project/ibi-staff/CT-JEPA/public/benchmark_models/`, which is exactly where
`configs/benchmark_embeddings.yaml` points each model's `weights:`.

### 1. Build + push the 6 per-model images (on your Mac)
The DGX is x86_64, so builds MUST target `linux/amd64` — a native arm64 build on
Apple Silicon will not run on the DGX (and GPU work can't run under QEMU).
`build_benchmark_images.sh` handles the `<dir>:<tag>` mapping (`ctclip`→`ct_clip`,
`ctfm`→`ct_fm`), the platform flag, and the push:
```bash
docker login                          # one-time, to your Docker Hub account
docker buildx create --use             # one-time, if you have no buildx builder
bash build_benchmark_images.sh                  # build + push all 6
# bash build_benchmark_images.sh merlin ct_clip   # ...or a subset (by tag)
```
Cross-arch (arm64 host → amd64 image) builds run under QEMU, so they're slower
than native — especially **colipri/ctclip** (pip/git installs under QEMU). Build the risky ones one at a time first.
The colipri image installs `colipri==0.1.2` from PyPI (not the local editable
checkout) to sidestep the package's `uv_build` backend under emulation.

The sbatch array script pulls each image on the DGX via Pyxis
(`--container-image=evandamron/benchmark-<tag>:latest`) on first use per node —
there is no separate `docker pull` (compute nodes have no Docker daemon).

### 2. Smoke test per model (gates the full run)
One volume on `ctrate_valid`, asserting the saved `.npz` shape matches the spec:
```bash
# Inside the container (or via an interactive srun):
cd /app && PYTHONPATH=/app python -u scripts/generate_benchmark_embeddings.py \
    --config configs/benchmark_embeddings.yaml \
    --model_key <rad_dino|dinov3|colipri|ct_clip|merlin|ct_fm> \
    --split ctrate_valid --limit 2
```
Assert the saved `<base_name>.npz` has key `embeddings`, shape `(seq, D)` with
`D` = 768 / 1024 / 864 / 512 / 2048 / 512, and `seq` matches the native grid
(COLIPRI/CT-CLIP = 13824; Merlin = 1; CT-FM = N_patches, variable; 2D =
n_slices). **Merlin and CT-CLIP are mandatory** (their load paths could not be
executed locally). For COLIPRI/CT-CLIP, also check `free -g` headroom on the
first train volumes before the 24.7k array.

### 3. Full extraction array (6 models × 3 splits = 18 jobs)
```bash
bash run_benchmark_embeddings_master.sh
```
Submits 3 Slurm arrays, **capped at 8 concurrent GPUs** on the DGX:
`ctrate_train` (`--array=0-5%2`, 4 GPUs/task → 8 GPUs, 2 models at a time, DDP
over the 24.7k tars); once train finishes, `ctrate_valid` + `rad`
(`--array=0-5%4` each, 1 GPU/task → 4 + 4 = 8 GPUs) run concurrently (`afterok`
train). Images are pulled by Pyxis on first use per node. Outputs land under
`/project/ibi-staff/CT-JEPA/features/benchmark/<model>/<split>/*.npz`. The
script prints the 3 job IDs.

### 4. Probe + error bars (select → variance → frozen → aggregate)
Once the 3 extraction arrays finish, feed their job IDs to the probe master:
```bash
bash run_benchmark_error_bars_master.sh <JOB_TRAIN> <JOB_VALID> <JOB_RAD>
```
This submits the same 4-stage error-bars pipeline as `run_error_bars_master.sh`
but for the 6 benchmark models (it shares `run_error_bars_array.sh`, selecting
the benchmark keys via `MODELS` env and widening the `--array` ranges to 0-5 /
0-29). It is **also capped at 8 concurrent GPUs**: every 1-GPU probe array is
throttled to `%4`, and the `afterok` DAG keeps at most two arrays eligible at
once (4 + 4 = 8). `afterok` also gates the probe stages on extraction, so probe
and extraction never overlap — pass `none` for an ID only if that split's
embeddings are truly done.

Alignment check before trusting the numbers:
```bash
python scripts/run_error_bars.py --config configs/error_bars.yaml \
    --mode select --task ctrate --model <benchmark_key>
# confirm "Matched N / N volumes" is non-zero and n_test matches DALE-CT (3002 CT-RATE / RAD-equiv)
```

## Wiring into error_bars (what changed)

Surgical edits only (per CLAUDE.md §3):
- **`scripts/run_error_bars.py`** — `build_config_for_task(config, task, model_key=None)`: when `model_key` is set, `exp["input_dim"] = config["models"][model_key].get("input_dim", exp["input_dim"])`. The three call sites (`run_select`, `run_variance`, `run_frozen`) now pass `model_key`. This flows each model's real embed dim into the `ColipriProber` construction (variance + frozen) and the saved selected-config — critical for the frozen arm, which reloads a probe whose architecture depends on `input_dim`.
- **`configs/error_bars.yaml`** — 6 benchmark models added under `models:` with `ctrate_train`/`ctrate_valid`/`rad` paths under `features/benchmark/<model>/` and per-model `input_dim`. The 4 DALE-CT models are untouched (fall back to global 1024).
- **`run_error_bars_array.sh`** — reads `MODELS` from env when set (split back into an array via `read -ra`); default unchanged, so both masters share it.
- **`run_benchmark_error_bars_master.sh`** (new) — the 6-model probe master with `afterok` gating on the embedding jobs.

The probe, metrics, bootstrap, and JSON saving are **already fully
parameterized** by `input_dim` / `num_classes` — no other change. The aggregate
stage globs whatever `per_run/*.json` exist (graceful if some models are
absent), so a benchmark-only run aggregates just the benchmark models; once
DALE-CT + benchmark both land, one aggregate covers all 10.

## Resource notes

- **8-GPU cap.** Both masters throttle their arrays (`%2` for the 4-GPU train array, `%4` for every 1-GPU array) and chain with `afterok` so peak concurrent GPUs is 8: train alone (2×4), then valid+rad (4+4); error_bars keeps any two probe arrays at 4+4. Don't run the (unthrottled) DALE-CT `run_error_bars_master.sh` or other GPU jobs concurrently, or you'll exceed 8.
- **COLIPRI** is the binding constraint: 13824 × 864 × float32 ≈ 46 MiB/volume → ~1.2 TiB for CT-RATE train+valid under `preload_ram=True`. CT-RATE select/variance stages request **1600G** (fits a 2 TB node). CT-CLIP is ~0.7 TiB; the other 4 are <25 GB but share the array, so they over-request — acceptable, since colipri/ct_clip are the constraint.
- RAD stages are smaller (RAD has far fewer volumes); frozen loads RAD test only (64G).
- 3D forwards are expensive: full train (24.7k) × 4 3D models is an overnight/weekend run. DDP sharding + resume (skip-if-valid `.npz`) bound wall-clock and make failures recoverable.
- If COLIPRI RAM is tight, documented fallbacks: the 768-d projected output (12% smaller) or a light 12³=1728 pool (still 8× richer than a 6³=216 cap). Default = full 864-d raw.

## Disclosures (for the manuscript)

**(a) Preprocessing asymmetry — identical probe protocol, not identical crops.**
DALE-CT embeddings were extracted after a body-crop. The 6 baselines each use
their **authors' intended native preprocessing** (only CT-FM natively
foreground-crops; the other 5 are full-FOV). Applying DALE-CT's body-crop to
the baselines would be off-distribution for 5/6 and would conflict with each
3D model's own native crop/resize. Apples-to-apples here = the **identical
frozen-probe MIL + error-bars protocol**, not identical crops. The asymmetry is
disclosed.

**(b) Representation-richness asymmetry — per-slice CLS vs. 3D patch tokens.**
DALE-CT's "3D" models (1S/2S) are stored as **per-slice CLS** (~100–300 tokens
per volume), because the DALE-CT embeddings were already extracted slice-wise;
re-extracting DALE-CT as 3D patches is out of scope. The 3D baselines
contribute **true 3D patch grids** — COLIPRI/CT-CLIP = 13824 tokens, CT-FM =
N_patches — which is representationally richer. Merlin is the opposite extreme:
a single **pooled 2048-d** embedding (1 token), matching the CT-FM paper's own
Merlin linear-probe protocol. If a 3D baseline beats DALE-CT-1S/2S, part of the
margin could be token richness, not backbone quality. State this in the
writeup. The `ColipriProber` is permutation-equivariant with a parameter count
independent of seq length, so a larger token bag cannot lower the probe's
ceiling — more positions only helps.

**(c) 3D-baseline CIs vs. point estimates.**
Previously the 3D baselines (CT-CLIP, CT-FM, Merlin, COLIPRI-C) were cited as
**point estimates from the COLIPRI paper**. This pipeline replaces them with
**our own CI-equipped numbers** under the identical protocol. Where the
manuscript still references COLIPRI-paper point estimates, the comparison is
"our CI vs. their point estimate," not paired — note this in the results.

## Files

| File | Role |
|---|---|
| `utils/benchmark_backbone_loader.py` | `BENCHMARK_SPECS` + `load_benchmark_model` + `extract_volume_features` (per-family loaders/extractors, native preprocessing) |
| `scripts/generate_benchmark_embeddings.py` | DDP driver (reuses `get_wds_dataset` / `CTMultiScaleDataset`); `--model_key` / `--split` / `--limit` |
| `configs/benchmark_embeddings.yaml` | per-model specs (mounted weights paths) + 3 split data dirs + output root |
| `docker/benchmark/<model>/Dockerfile` | 6 per-model images (DEPS ONLY), `FROM evandamron/lejepa:latest` |
| `build_benchmark_images.sh` | builds + pushes the 6 linux/amd64 images to Docker Hub (run on Mac) |
| `run_benchmark_embeddings_array.sh` | single array task: `TASK_ID`→model, `SPLIT` from env, per-model container |
| `run_benchmark_embeddings_master.sh` | submits the 3 extraction arrays (train/valid/rad) |
| `run_benchmark_error_bars_master.sh` | submits select→variance→frozen→aggregate for the 6 models, gated on extraction |
| `configs/error_bars.yaml` (+`run_error_bars.py`, `run_error_bars_array.sh`) | surgical wiring (per-model `input_dim`, `MODELS` env) |

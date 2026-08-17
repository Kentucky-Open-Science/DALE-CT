# DALE-CT: Depth-Aware 2D Slice Encoders Learn an Anatomical World Model of Chest CT

Training and evaluation code for **DALE-CT** (Depth-Aware Latent-Euclidean
Computed Tomography) — a family of 2D slice-based Vision Transformers trained
entirely from scratch on chest CT with the heuristics-free
[LeJEPA](https://arxiv.org/abs/2511.08544) objective and **depth-aware slab
sampling**: self-supervised views are drawn from across a physical $z$-axis
slab rather than a single slice, so the frozen representations form an
**anatomical world model** — they linearly decode volumetric slice position
($R^2 \approx 0.97$), recover slice ordering without labels, and localize
organs and findings, despite no 3D or positional supervision.

**Paper:** [DALE-CT: Depth-Aware 2D Slice Encoders Learn an Anatomical World Model of Chest CT](https://arxiv.org/abs/2606.07775)

## Released models

All weights are on the Hugging Face Hub (CC-BY-NC-SA-4.0) and load in one line
via `timm`:

```python
import timm
model = timm.create_model("hf-hub:Kentucky-Open-Science/DALE-CT-0-L", pretrained=True)
```

| Model | CT-RATE Macro AUROC | RAD-ChestCT AUROC (frozen / retrained probe) | Role |
|---|---|---|---|
| [DALE-CT-0-L](https://huggingface.co/Kentucky-Open-Science/DALE-CT-0-L) ⭐ | 0.8156 | 0.6281 / **0.7572** | **Recommended general-purpose backbone** — best 2D external transfer; supervision-free at ~287k-scan scale |
| [DALE-CT-2S](https://huggingface.co/Kentucky-Open-Science/DALE-CT-2S) | **0.8247** | 0.6252 / 0.7389 | Best in-domain (CT-RATE) |
| [DALE-CT-1S-v2](https://huggingface.co/Kentucky-Open-Science/DALE-CT-1S-v2) | 0.8098 | 0.6284 / 0.7334 | Anatomical (TotalSegmentator) dense supervision only |
| [DALE-CT-0](https://huggingface.co/Kentucky-Open-Science/DALE-CT-0) | 0.8057 | 0.5946 / 0.7477 | Pure self-supervised, CT-RATE |
| [Finetuned DINOv2](https://huggingface.co/Kentucky-Open-Science/Finetuned-DINOv2-Chest-CT) | 0.7953 | 0.6252 / 0.7550 | Continual-pretraining baseline |

All numbers are our own head-to-head measurements: every model in the paper
(including public 3D baselines COLIPRI-CRM, Merlin, CT-FM, CT-CLIP) is probed
under one linear-probing MIL protocol on shared splits with bootstrap
confidence intervals. Each model card documents the exact Hounsfield-Unit
preprocessing its backbone expects — **DALE-CT-0-L uses different
normalization statistics than the CT-RATE-trained variants.**

## Repository map

| Path | Contents |
|---|---|
| `train_lejepa.py` | Pre-training entry point (all DALE-CT variants) |
| `lejepa_core/` | LeJEPA architecture: SSL meta-arch, trainer, SIGReg |
| `models/` | ViT backbone, projector, MIL pooling heads |
| `supervised_heads/` | Dense auxiliary supervision (TotalSegmentator / ReXGroundingCT soft labels) |
| `dataloaders/` | CT-RATE WebDataset, multi-scale multi-crop, multi-source zarr, embedding loaders |
| `configs/` | OmegaConf YAML for every pretraining / probing / evaluation run |
| `scripts/run_error_bars.py` | Head-to-head benchmark: probe selection, seed variance, bootstrap CIs |
| `scripts/generate_benchmark_embeddings.py` | Embedding extraction for the public 3D baselines |
| `scripts/run_model_comparison_probes.py` | Dense 2D probing (auxiliary-task tables) |
| `scripts/exp_c_zposition.py`, `scripts/exp_c_worldmodel_probes.py` | Anatomical world-model probes ($z$-regression, slice ordering, organ identity) |
| `train_gridsearch.py`, `eval_rad.py` | Linear-probe grid search; RAD-ChestCT transfer |
| `train_e2e_lora.py` | LoRA fine-tuning |
| `ERROR_BARS_README.md`, `BENCHMARK_EMBEDDINGS_README.md` | Protocol documentation for the benchmark pipelines |

Dataset preparation (CT-RATE → WebDataset shards, TotalSegmentator masks) lives
in the companion repo
[Process-CT-Data](https://github.com/Kentucky-Open-Science/Process-CT-Data).

Paths in configs refer to our cluster layout; point them at your own data
roots. Raw datasets and checkpoints are never stored in this repository.

## Citation

```bibtex
@article{damron2026dalect,
  title   = {DALE-CT: Depth-Aware 2D Slice Encoders Learn an Anatomical World Model of Chest CT},
  author  = {Damron, Evan W. and Gokmen, Mahmut S. and Klusty, Mitchell A. and
             Leach, Caroline N. and Collier, Emily B. and Bumgardner, V. K. Cody},
  journal = {arXiv preprint arXiv:2606.07775},
  year    = {2026}
}
```

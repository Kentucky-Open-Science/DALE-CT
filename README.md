# Guided Chest CT

This repository contains the training and evaluation pipeline for LeJEPA (Latent-Euclidean Joint-Embedding Predictive Architecture) vision transformers pre-trained on the CT-RATE dataset. Three model variants are provided: a pure self-supervised baseline (LeJEPA-0), a single-source auxiliary variant with TotalSegmentator supervision (LeJEPA-1S), and a dual-source variant with both TotalSegmentator and ReXGroundingCT supervision (LeJEPA-2S).

---

## Repository Structure

```
.
├── train_lejepa.py              # LeJEPA pre-training entry point
├── train_gridsearch.py          # Linear probing grid search
├── train_e2e_lora.py            # LoRA fine-tuning
├── eval_rad.py                  # RAD-ChestCT transfer evaluation
├── test_ct_dataloader.py        # Dataloader debugging
├── requirements.txt
├── Dockerfile
│
├── lejepa_core/                 # Core LeJEPA architecture
│   ├── lejepa_ssl_arch.py       # SSLMetaArch training module
│   ├── main_lejepa_trainer.py   # Distributed training orchestration
│   └── SIGReg.py                # Sketched Isotropic Gaussian Regularization
│
├── models/                      # Model definitions
│   ├── vision_transformer.py    # ViT backbone builders
│   ├── lejepa_projector.py      # LeJEPA projection head
│   ├── colipri_pooling.py       # 5 pooling schemes for linear probing
│   └── e2e_colipri.py           # End-to-end model wrapper
│
├── data/                        # Data processing
│   ├── guided_data_augmentation_CT_RATE.py  # Multi-crop augmentation
│   └── collate.py               # Batch collation
│
├── dataloaders/                 # Dataset loaders
│   ├── datasetloader_web_ctrate.py          # WebDataset loader
│   ├── datasetloader_ctrate_multiscale.py   # Multi-scale .npy loader
│   ├── dataloader_embeddings.py             # Pre-computed embedding loader
│   ├── dataloader_linear.py                 # Linear probing loader
│   └── dataloader_rad_embeddings.py         # RAD-ChestCT loader
│
├── supervised_heads/            # Auxiliary supervision heads
│   ├── organ_supervision.py     # TotalSegmentator classification
│   ├── soft_label_supervision.py
│   └── example_supervised_head.py
│
├── scripts/                     # Evaluation scripts
│   ├── ctrate_generate_embeddings.py
│   ├── knn.py
│   ├── supervised_gap.py
│   ├── supervised_3D.py
│   ├── e2e_inference.py
│   ├── evaluate_gap.py
│   └── precompute_aggregations.py
│
├── utils/                       # Shared utilities
│   ├── config.py                # OmegaConf configuration
│   ├── dino_utils.py            # Model init, LoRA, checkpointing
│   ├── lejepa_scheduler.py      # LR/WD scheduling
│   ├── dist_utils.py            # FSDP/DDP distributed training
│   ├── distributed_checkpointer.py
│   ├── model_checkpointer_ddp.py
│   ├── model_checkpointer_fsdp.py
│   ├── global_state.py
│   ├── logger_utils.py
│   ├── wandb_utils.py
│   ├── metrics.py
│   ├── config_utils.py
│   ├── model_utils.py
│   └── standardization.py
│
└── configs/                     # YAML configs (one per use case)
    ├── pretrain_lejepa_0.yaml   # Unguided baseline
    ├── pretrain_lejepa_1s.yaml  # Single-source TS supervision
    ├── pretrain_lejepa_2s.yaml  # Dual-source TS + ReX supervision
    ├── linear_probe.yaml        # Grid search over pooling schemes
    ├── finetune_lora.yaml       # LoRA fine-tuning
    ├── rad_transfer.yaml        # RAD-ChestCT transfer
    ├── generate_embeddings.yaml # Embedding extraction
    ├── knn_eval.yaml            # KNN evaluation
    └── models/                  # Model architecture definitions
```

---

## Model Variants

| Model | Supervision | Crop Sizes |
|-------|------------|------------|
| LeJEPA-0 | None (pure SSL) | 256 global, 144 local |
| LeJEPA-1S | TotalSegmentator (118 classes) | 224 global, 140 local |
| LeJEPA-2S | TS (118) + ReXGroundingCT (14) | 256 global, 144 local |

All models use a `vit_large_patch14` backbone trained from scratch with the LeJEPA objective. The loss combines a predictive term (L2 distance between local and global view representations) with SIGReg regularization (lambda = 0.02) to prevent collapse without teacher-student heuristics. Guided variants add a BCE auxiliary loss (lambda_aux = 0.1) with positive class weighting to handle severe label imbalance.

---

## Evaluation Methods

- **Linear Probing**: Grid search over 4 learning rates and 5 pooling schemes (average, max, learned attention, average attention, multi-learned attention). Best probe selected by validation AUPRC with per-class F1-optimal thresholds.
- **KNN**: 5-fold cross-validation using mean-pooled slice embeddings, reporting accuracy, balanced accuracy, and macro F1 under 1-NN and 5-NN.
- **LoRA Fine-Tuning**: Low-rank adaptation (r=8, alpha=16) on QKV projection matrices with learned attention pooling.
- **RAD-ChestCT Transfer**: Cross-dataset evaluation mapping 18 CT-RATE classes to 14 RAD-ChestCT classes.

---

## Data

CT-RATE volumes are pre-processed with HU clipping to [-997, 888] and Z-score normalization (mu = -142, sigma = 361, derived from 0.5%/99.5% foreground percentiles). Training uses a multi-crop strategy: 2 global crops from the center slice and 8 local crops sampled across a 12 mm volumetric window, with 80% probability of centering on RAD-ChestCT labels when present.

---

# DALE-CT Pretraining

This repository contains the training and evaluation pipeline for **DALE-CT** (**Depth-Aware Latent-Euclidean Computed Tomography**), a family of vision transformers pre-trained on the **CT-RATE** dataset using the **LeJEPA** framework.

Three model variants are provided:

* **DALE-CT-0**: pure self-supervised baseline
* **DALE-CT-1S**: single-source auxiliary supervision using TotalSegmentator
* **DALE-CT-2S**: dual-source auxiliary supervision using TotalSegmentator and ReXGroundingCT

---

## Repository Structure

```text
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
└── configs/                     # YAML configs
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

| Model          | Supervision                                                | Crop Sizes            |
| -------------- | ---------------------------------------------------------- | --------------------- |
| **DALE-CT-0**  | None; pure self-supervised learning                        | 256 global, 144 local |
| **DALE-CT-1S** | TotalSegmentator, 118 classes                              | 224 global, 140 local |
| **DALE-CT-2S** | TotalSegmentator, 118 classes + ReXGroundingCT, 14 classes | 256 global, 144 local |

All models use a `vit_large_patch14` backbone trained from scratch with the LeJEPA objective.

The training loss combines:

* A **spatial invariance loss**, implemented as the L2 distance between local and global view representations
* **SIGReg regularization** with `lambda = 0.02`, which prevents representation collapse without teacher-student heuristics
* For guided variants, an auxiliary **binary cross-entropy loss** with `lambda_aux = 0.1`

The guided auxiliary losses use positive class weighting to handle severe label imbalance.

---

## Evaluation Methods

### Linear Probing

Linear probing performs a grid search over:

* 4 learning rates
* 5 pooling schemes:

  * average pooling
  * max pooling
  * learned attention pooling
  * average attention pooling
  * multi-learned attention pooling

The best probe is selected by validation AUPRC, with per-class thresholds chosen to maximize F1 score.

### KNN Evaluation

KNN evaluation uses 5-fold cross-validation on mean-pooled slice embeddings.

Reported metrics include:

* Accuracy
* Balanced accuracy
* Macro F1

Both 1-NN and 5-NN settings are evaluated.

### RAD-ChestCT Transfer

RAD-ChestCT transfer evaluates cross-dataset generalization by mapping 18 CT-RATE classes to 14 RAD-ChestCT classes.

---

## Data

CT-RATE volumes are pre-processed using:

* HU clipping to `[-997, 888]`
* Z-score normalization with:

  * `mu = -142`
  * `sigma = 361`

The normalization statistics are derived from the 0.5% and 99.5% foreground intensity percentiles.

Training uses a multi-crop strategy:

* 2 global crops sampled from the center slice
* 8 local crops sampled across a 12 mm volumetric window
* 80% probability of centering crops on RAD-ChestCT labels when present, TotalSegmentator labels otherwise

This strategy encourages the model to learn spatially consistent CT representations across local and global anatomical context.


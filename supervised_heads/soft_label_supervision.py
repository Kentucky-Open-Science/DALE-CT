import torch
import torch.nn as nn


class SoftLabelSupervisionHead(nn.Module):
    """
    V2 Auxiliary head: Predicts fractional coverage of tissues and conditions
    for both the global volume (CLS token) and individual patches (Patch tokens).
    """

    def __init__(self, config, input_dim=768):
        super().__init__()
        self.num_ts_classes = 118
        self.num_rex_classes = 14

        # TS-only mode: skip the ReX head/loss entirely (no Rex supervision).
        self.use_rex = getattr(config.auxiliary, 'use_rex', True)

        # Independent heads for TotalSegmentator and ReX
        self.ts_head = nn.Linear(input_dim, self.num_ts_classes)
        if self.use_rex:
            self.rex_head = nn.Linear(input_dim, self.num_rex_classes)

        self.aux_weight = getattr(config.auxiliary, 'aux_weight', 1.0)

        # 1. Define the customized TS array (Index 0 is background, OOD anatomy demoted to 1.0)
        ts_weights_list = [
            1.0, 184.42, 635.22, 541.35, 1000.00, 30.53, 154.25, 745.32, 1000.00, 1000.00,
            47.51, 49.00, 55.92, 127.37, 45.82, 1000.00, 1000.00, 1000.00, 412.64, 1000.00,
            241.72, 1.0, 1.0, 1000.00, 1000.00, 1.0, 1.0, 1.0, 1.0, 1.0,
            1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00,
            1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00,
            1000.00, 87.24, 232.62, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00,
            1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1.0, 1.0, 1.0, 1.0, 939.88,
            890.98, 455.42, 471.30, 1000.00, 1000.00, 1.0, 1.0, 1.0, 1.0, 727.56,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 159.89, 164.19, 1.0, 1.0,
            1.0, 1.0, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00,
            1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00,
            1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 746.09, 338.73
        ]

        # 2. Define the uniform ReX array for highly sparse abnormalities
        rex_weights_list = [
            800.0, 800.0, 800.0, 800.0, 800.0, 800.0,
            800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0
        ]

        # 3. Register as buffers so PyTorch manages their device placement
        self.register_buffer('ts_pos_weight', torch.tensor(ts_weights_list, dtype=torch.float32))
        if self.use_rex:
            self.register_buffer('rex_pos_weight', torch.tensor(rex_weights_list, dtype=torch.float32))

        # 4. Initialize the loss functions with reduction='none' and our custom weights
        self.criterion_ts = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.ts_pos_weight)
        if self.use_rex:
            self.criterion_rex = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.rex_pos_weight)

    def forward(self, features, labels, is_rex):
        cls_tokens = features[:, 0, :]
        num_patches = labels['ts_patch'].shape[1]
        patch_tokens = features[:, -num_patches:, :]

        # --- TotalSegmentator Loss (Uses criterion_ts) ---
        ts_cls_logits = self.ts_head(cls_tokens)
        ts_patch_logits = self.ts_head(patch_tokens)

        loss_ts_cls = self.criterion_ts(ts_cls_logits, labels['ts_crop']).mean()
        loss_ts_patch = self.criterion_ts(
            ts_patch_logits.reshape(-1, self.num_ts_classes),
            labels['ts_patch'].reshape(-1, self.num_ts_classes)
        ).mean()

        # --- ReX Loss (Uses criterion_rex and is masked) ---
        if self.use_rex:
            rex_cls_logits = self.rex_head(cls_tokens)
            rex_patch_logits = self.rex_head(patch_tokens)

            raw_loss_rex_cls = self.criterion_rex(rex_cls_logits, labels['rex_crop'])
            raw_loss_rex_patch = self.criterion_rex(
                rex_patch_logits.reshape(-1, self.num_rex_classes),
                labels['rex_patch'].reshape(-1, self.num_rex_classes)
            )

            # Apply the mask (Same as before)
            rex_mask_cls = is_rex.unsqueeze(1).float()
            rex_mask_patch = is_rex.repeat_interleave(num_patches).unsqueeze(1).float()

            valid_cls_count = rex_mask_cls.sum() + 1e-8
            valid_patch_count = rex_mask_patch.sum() + 1e-8

            loss_rex_cls = (raw_loss_rex_cls * rex_mask_cls).sum() / (valid_cls_count * self.num_rex_classes)
            loss_rex_patch = (raw_loss_rex_patch * rex_mask_patch).sum() / (valid_patch_count * self.num_rex_classes)

            # --- Combine (TS + ReX, 0.5/0.5) ---
            loss_cls = 0.5 * (loss_ts_cls + loss_rex_cls)
            loss_patch = 0.5 * (loss_ts_patch + loss_rex_patch)
        else:
            loss_rex_cls = torch.zeros((), device=features.device)
            loss_rex_patch = torch.zeros((), device=features.device)
            # --- Combine (TS only) ---
            loss_cls = loss_ts_cls
            loss_patch = loss_ts_patch

        total_aux_loss = 0.5 * loss_cls + 0.5 * loss_patch

        stats = {
            "aux_loss": total_aux_loss.item() * self.aux_weight,
            "aux_cls_ts": loss_ts_cls.item(),
            "aux_cls_rex": loss_rex_cls.item(),
            "aux_patch_ts": loss_ts_patch.item(),
            "aux_patch_rex": loss_rex_patch.item(),
        }

        return total_aux_loss * self.aux_weight, stats
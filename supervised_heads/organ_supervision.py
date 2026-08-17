import torch
import torch.nn as nn


class OrganSupervisionHead(nn.Module):
    """
    Auxiliary head for supervised organ segmentation/classification.
    Attached to the backbone during training to provide gradients.
    """

    def __init__(self, config, input_dim=768):
        super().__init__()
        # TotalSegmentator uses classes 1-117.
        # We use 118 output neurons to map indices 0-117 directly.
        self.num_classes = 118

        self.head = nn.Linear(input_dim, self.num_classes)
        pos_weight_val = getattr(config.auxiliary, 'pos_weight', 15.0)
        self.register_buffer('pos_weight', torch.tensor([pos_weight_val] * self.num_classes))

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.aux_weight = getattr(config.auxiliary, 'aux_weight', 1.0)

    def forward(self, features, labels):
        """
        Args:
            features: (Batch * Views, Embed_Dim) - Attached backbone features
            labels: (Batch * Views, Num_Classes) - Multi-hot labels
        Returns:
            weighted_loss: Scalar tensor
            stats: Dict containing raw loss and F1 score
        """
        logits = self.head(features)

        # Compute Loss
        loss = self.criterion(logits, labels)

        # Compute Metrics (No Grad)
        stats = {"aux_loss": loss.item()}
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()

            # Micro-F1 Calculation
            tp = (preds * labels).sum()
            fp = (preds * (1 - labels)).sum()
            fn = ((1 - preds) * labels).sum()

            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            stats["aux_f1"] = f1.item()

        return loss * self.aux_weight, stats
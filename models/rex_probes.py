"""
Lightweight classification and segmentation probe heads for ReX-GroundingCT.

These modules sit on top of a frozen ViT backbone and produce:
  - Slice-level multi-label classification logits (from [CLS] token)
  - Patch-level 2D semantic segmentation logits (from patch tokens)

Supports both timm-based backbones (LeJEPA family) and HuggingFace
transformers backbones (DINOv2-CT).  Handles register tokens correctly
for DINOv2-with-registers architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RexSliceClassifier(nn.Module):
    """
    Global slice-level multi-label classification head.

    Extracts the [CLS] token from the frozen ViT backbone and passes it
    through a single linear layer to produce 14-class logits.

    Parameters
    ----------
    backbone : nn.Module
        Frozen ViT backbone.
        - timm models: ``forward_features(x)`` returns [B, N, D]
        - HF models: ``(pixel_values=x)`` returns BaseModelOutput with
          ``last_hidden_state`` of shape [B, N, D]
    embed_dim : int
        Dimensionality of the backbone's output embeddings.
    num_classes : int
        Number of target classes (default: 14 for ReX subcategories).
    backbone_type : str
        "timm" or "hf".  Determines how to call the backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int = 14,
        backbone_type: str = "timm",
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_type = backbone_type
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _extract_cls(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run backbone and extract [CLS] token.

        Parameters
        ----------
        x : torch.Tensor  shape [B, C, H, W]

        Returns
        -------
        cls_token : torch.Tensor  shape [B, embed_dim]
        """
        if self.backbone_type == "timm":
            out = self.backbone.forward_features(x)
            # timm returns [B, N, D] — CLS is index 0
            if out.dim() == 3:
                return out[:, 0, :]
            else:
                return out
        else:
            # HuggingFace transformers
            out = self.backbone(pixel_values=x)
            if hasattr(out, 'last_hidden_state'):
                return out.last_hidden_state[:, 0, :]
            elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
                return out.pooler_output
            elif out.dim() == 3:
                return out[:, 0, :]
            else:
                return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, C, H, W]
            Batch of 2D CT slices (already preprocessed).

        Returns
        -------
        logits : torch.Tensor  shape [B, num_classes]
        """
        cls_token = self._extract_cls(x)
        logits = self.classifier(cls_token)
        return logits


class RexDenseProbe(nn.Module):
    """
    Patch-level 2D semantic segmentation head.

    Extracts the sequence of patch tokens (ignoring [CLS] and register
    tokens), reshapes them into a 2D spatial grid, applies a 1x1
    convolution, and upsamples back to the original input resolution.

    Parameters
    ----------
    backbone : nn.Module
        Frozen ViT backbone.
    embed_dim : int
        Dimensionality of the backbone's output embeddings.
    num_classes : int
        Number of target segmentation classes (default: 14).
    patch_size : int
        Patch size of the ViT (14 for LeJEPA-1S/DINOv2, 16 for LeJEPA-0/2S).
    image_size : int
        Input image size after resizing (default: 256).
    num_register_tokens : int
        Number of register tokens to skip after [CLS].
        0 for LeJEPA, 4 for DINOv2-with-registers.
    backbone_type : str
        "timm" or "hf".
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int = 14,
        patch_size: int = 14,
        image_size: int = 256,
        num_register_tokens: int = 0,
        backbone_type: str = "timm",
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_register_tokens = num_register_tokens
        self.backbone_type = backbone_type
        self.grid_size = image_size // patch_size

        # 1x1 convolution to project patch embeddings to class logits
        self.conv_seg = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=num_classes,
            kernel_size=1,
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run backbone and extract patch tokens (excluding CLS and registers).

        Parameters
        ----------
        x : torch.Tensor  shape [B, C, H, W]

        Returns
        -------
        patch_tokens : torch.Tensor  shape [B, N_patches, D]
        """
        if self.backbone_type == "timm":
            out = self.backbone.forward_features(x)
            # timm returns [B, N, D]
            if out.dim() == 3:
                hidden = out
            else:
                raise RuntimeError(
                    f"Expected backbone output of shape [B, N, D], got {out.shape}"
                )
        else:
            # HuggingFace transformers
            out = self.backbone(pixel_values=x)
            if hasattr(out, 'last_hidden_state'):
                hidden = out.last_hidden_state
            elif out.dim() == 3:
                hidden = out
            else:
                raise RuntimeError(
                    f"Expected backbone output of shape [B, N, D], got {out.shape}"
                )

        # Remove [CLS] token (index 0) and register tokens (indices 1..num_register_tokens)
        # DINOv2: tokens are [CLS, R1, R2, R3, R4, P1, P2, ...]
        # LeJEPA: tokens are [CLS, P1, P2, ...]
        patch_tokens = hidden[:, 1 + self.num_register_tokens:, :]
        return patch_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, C, H, W]
            Batch of 2D CT slices.

        Returns
        -------
        mask_logits : torch.Tensor  shape [B, num_classes, H, W]
        """
        B = x.shape[0]

        patch_tokens = self._extract_patch_tokens(x)  # [B, N_patches, D]

        # Reshape to 2D spatial grid: [B, D, grid_size, grid_size]
        patch_tokens = patch_tokens.transpose(1, 2)  # [B, D, N_patches]
        patch_tokens = patch_tokens.reshape(
            B, -1, self.grid_size, self.grid_size
        )

        # 1x1 convolution → [B, num_classes, grid_size, grid_size]
        mask_logits = self.conv_seg(patch_tokens)

        # Upsample to original resolution [B, num_classes, H, W]
        mask_logits = F.interpolate(
            mask_logits,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )

        return mask_logits

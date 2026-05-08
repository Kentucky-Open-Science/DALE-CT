import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.colipri_pooling import ColipriProber


class EndToEndColipri(nn.Module):
    def __init__(self, vit_backbone, colipri_state_dict_path=None, input_dim=1024,
                 pooling_scheme="learned_attention", multi_adapter_config=None):
        super().__init__()
        self.backbone = vit_backbone
        self.multi_adapter_config = multi_adapter_config
        self.is_multi_adapter = multi_adapter_config is not None and multi_adapter_config.get('enabled', False)

        unwrapped = getattr(self.backbone, '_orig_mod', getattr(self.backbone, 'module', self.backbone))
        if hasattr(unwrapped, 'base_model'):
            unwrapped = unwrapped.base_model
        self._is_timm = hasattr(unwrapped, 'forward_features')

        # Initialize multiple heads if using multi-adapter
        self.colipri_heads = nn.ModuleDict()

        if self.is_multi_adapter:
            for adapter_name, cfg in self.multi_adapter_config['adapters'].items():
                num_classes = len(cfg['classes'])
                self.colipri_heads[adapter_name] = ColipriProber(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    pooling_scheme=pooling_scheme,
                    pooling_mode="embedding"
                )
        else:
            # Fallback to standard single head
            self.colipri_heads['default'] = ColipriProber(
                input_dim=input_dim,
                num_classes=18,
                pooling_scheme=pooling_scheme,
                pooling_mode="embedding"
            )

        # Note: If loading pre-trained Colipri weights, you will need to slice the
        # classifier weights to match the subset of classes for each adapter head.

    def forward(self, x, mask=None, chunk_size=32, active_adapters=None):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        out_logits = {}

        # Default to all adapters if none are explicitly provided
        if active_adapters is None:
            active_adapters = self.multi_adapter_config['adapters'].keys() if self.is_multi_adapter else ['default']

        # Loop ONLY over active adapters
        for adapter_name in active_adapters:
            if self.is_multi_adapter:
                self.backbone.set_adapter(adapter_name)

            embs = []
            for i in range(0, B * S, chunk_size):
                chunk = x[i: i + chunk_size]
                if chunk.requires_grad is False:
                    chunk.requires_grad_()

                chunk_emb = checkpoint(self._forward_backbone, chunk, use_reentrant=False)
                embs.append(chunk_emb)

            features = torch.cat(embs, dim=0).view(B, S, -1)
            logits, _ = self.colipri_heads[adapter_name](features, mask=mask)
            out_logits[adapter_name] = logits

        return out_logits if self.is_multi_adapter else out_logits['default']

    @torch.compiler.disable()
    def _forward_backbone(self, x):
        """Helper to extract the CLS token or pooled feature from the backbone.
        
        Decorated with torch.compiler.disable() to prevent torch.compile/dynamo
        from tracing this function. This avoids a known torch._inductor bug where
        _upsample_bicubic2d_aa is incorrectly selected for downsample operations
        when the backbone is torch.compile'd and called via gradient checkpointing.
        """
        # Unwrap DDP/FSDP wrappers to access the underlying model
        model = self.backbone
        unwrapped = getattr(model, '_orig_mod', getattr(model, 'module', model))
        # For PEFT models, unwrap further to the base model
        if hasattr(unwrapped, 'base_model'):
            unwrapped = unwrapped.base_model

        if hasattr(unwrapped, 'forward_features'):
            # TIMM model: forward_features returns (B, N, D) token sequence
            features = model(x)
            if isinstance(features, torch.Tensor):
                if features.ndim == 2:
                    return features
                elif features.ndim == 3:
                    return features[:, 0, :]
            return features
        else:
            # HuggingFace model
            features = model(x)
            if hasattr(features, 'last_hidden_state'):
                return features.last_hidden_state[:, 0, :]
            elif isinstance(features, tuple):
                return features[0]
            elif isinstance(features, torch.Tensor) and features.ndim == 3:
                return features[:, 0, :]
            return features
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.colipri_pooling import ColipriProber


class PatchAttentionPooler(nn.Module):
    """Learned single-query attention over the N patch tokens of one slice.

    Mirrors ColipriProber's single-query cross-attention (scaled dot-product,
    no positional encoding, permutation-invariant over patches) to distill the
    N per-slice patch tokens into one (B, D) summary, which is then fused with
    the CLS token.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim) / (input_dim ** 0.5))
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, patches):
        # patches: (B, N, D) -> (B, D)
        scale = patches.shape[-1] ** -0.5
        scores = torch.matmul(patches, self.query) * scale   # (B, N)
        attn = scores.softmax(dim=-1)                         # (B, N)
        summary = (attn.unsqueeze(-1) * patches).sum(dim=1)   # (B, D)
        return self.norm(summary)


class SliceOrderTransformer(nn.Module):
    """Small pre-norm transformer over the S per-slice vectors with learned
    absolute slice-order positional embeddings, so slices can attend to each
    other along the z-axis. Runs after per-slice CLS+patch pooling and before
    the order-agnostic ColipriProber head.

    Output projections are zero-initialized so the stack starts as identity:
    pretrained backbone features flow through unmodified at step 0, and the
    transformer learns perturbations from there.
    """

    def __init__(self, input_dim, num_layers=2, num_heads=8, dim_feedforward=None,
                 max_slices=512, dropout=0.0):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * input_dim
        self.max_slices = max_slices
        self.slice_pos_embed = nn.Parameter(torch.zeros(1, max_slices, input_dim))
        nn.init.trunc_normal_(self.slice_pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(input_dim)

        # Zero-init each layer's output projections -> stack starts as identity.
        for layer in self.transformer.layers:
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)

    def forward(self, x, mask=None):
        # x: (B, S, D); mask: (B, S) bool, True = valid slice
        B, S, D = x.shape
        if S > self.max_slices:
            # Defensive: uniformly subsample long volumes preserving z-order so
            # the learned positional table is never indexed out of range.
            idx = torch.linspace(0, S - 1, self.max_slices, device=x.device).long()
            x = x[:, idx, :]
            if mask is not None:
                mask = mask[:, idx]
            S = self.max_slices
        x = x + self.slice_pos_embed[:, :S, :]
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # nn.MultiheadAttention convention: True = ignore
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)


class EndToEndColipri(nn.Module):
    def __init__(self, vit_backbone, colipri_state_dict_path=None, input_dim=1024,
                 pooling_scheme="learned_attention", multi_adapter_config=None,
                 use_patch_pooling=False, use_slice_transformer=False,
                 slice_transformer_config=None, use_gradient_checkpointing=True):
        super().__init__()
        self.backbone = vit_backbone
        self.multi_adapter_config = multi_adapter_config
        self.is_multi_adapter = multi_adapter_config is not None and multi_adapter_config.get('enabled', False)
        # When True (default), the per-slice backbone forward runs under
        # torch.utils.checkpoint -- activations are dropped and recomputed in
        # backward (fixed ~1x forward tax, low VRAM). When False, one unchunked
        # forward holds all B*S slice activations (no recompute, ~33% faster) --
        # only safe with GPU headroom; the trainer's step-1 peak-VRAM log confirms.
        # PERF KNOB, NOT AN ABLATION AXIS: use one consistent setting per run.
        # Off vs on does NOT give identical gradients -- use_reentrant=False only
        # guarantees forward/recompute RNG consistency WITHIN the checkpointed
        # region, not across the off/on paths. A chunked checkpointed forward
        # consumes RNG per-block-per-chunk while an unchunked forward consumes it
        # per-block-for-all-slices, so LoRA-dropout / drop-path masks diverge
        # (empirically grad maxdiff ~0.009 at dropout=0.1). So pick a setting and
        # keep it for the whole run; do not flip mid-comparison.
        self.use_gradient_checkpointing = use_gradient_checkpointing

        unwrapped = getattr(self.backbone, '_orig_mod', getattr(self.backbone, 'module', self.backbone))
        if hasattr(unwrapped, 'base_model'):
            unwrapped = unwrapped.base_model
        self._is_timm = hasattr(unwrapped, 'forward_features')

        # Per-slice CLS+patch fusion. When disabled, _forward_backbone returns
        # the pooled CLS token only (legacy behavior).
        self.use_patch_pooling = use_patch_pooling
        if self.use_patch_pooling:
            self.patch_pooler = PatchAttentionPooler(input_dim)
            self.patch_combine = nn.Sequential(
                nn.Linear(2 * input_dim, input_dim),
                nn.GELU(),
                nn.LayerNorm(input_dim),
            )

        # Inter-slice order-aware transformer over the S per-slice vectors.
        self.use_slice_transformer = use_slice_transformer
        if self.use_slice_transformer:
            st_cfg = slice_transformer_config or {}
            self.slice_transformer = SliceOrderTransformer(
                input_dim=input_dim,
                num_layers=st_cfg.get('num_layers', 2),
                num_heads=st_cfg.get('num_heads', 8),
                dim_feedforward=st_cfg.get('dim_feedforward', None),
                max_slices=st_cfg.get('max_slices', 512),
                dropout=st_cfg.get('dropout', 0.0),
            )

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

            if self.use_gradient_checkpointing:
                embs = []
                for i in range(0, B * S, chunk_size):
                    chunk = x[i: i + chunk_size]
                    embs.append(checkpoint(self._forward_backbone, chunk, use_reentrant=False))
                features = torch.cat(embs, dim=0).view(B, S, -1)
            else:
                # Checkpointing off: one unchunked forward (no recompute). chunk_size
                # is ignored -- all B*S slices go through in a single call. See the
                # __init__ flag docstring for the VRAM/speed tradeoff.
                features = self._forward_backbone(x).view(B, S, -1)

            # Inter-slice order-aware enrichment (gated).
            if self.use_slice_transformer:
                features = self.slice_transformer(features, mask=mask)

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
            if self.use_patch_pooling:
                # CLS + attention-pooled patches: needs the raw token sequence.
                # forward_features returns (B, 1+N, D) = [CLS, *patches].
                features = unwrapped.forward_features(x)
                if isinstance(features, torch.Tensor) and features.ndim == 3:
                    cls = features[:, 0, :]                       # (B, D)
                    patches = features[:, 1:, :]                  # (B, N, D)
                    patch_summary = self.patch_pooler(patches)    # (B, D)
                    return self.patch_combine(torch.cat([cls, patch_summary], dim=-1))  # (B, D)
                # Unexpected shape: fall through to legacy CLS extraction.
                features = model(x)
                if isinstance(features, torch.Tensor):
                    if features.ndim == 2:
                        return features
                    elif features.ndim == 3:
                        return features[:, 0, :]
                return features
            else:
                # Legacy CLS-only path: pooled feature via the full forward.
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

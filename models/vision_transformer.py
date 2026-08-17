
# models/vision_transformer.py

import copy
import math 
import os 
import warnings

from transformers import ViTConfig
from safetensors.torch import load_file
import timm 
from torch import Tensor, nn
from transformers import AutoModel, AutoConfig  
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel, ViTConfig, Dinov2Config, Dinov2WithRegistersConfig, DINOv3ViTConfig
from torch.nn.utils.parametrizations import weight_norm 
from utils.config_utils import find_closest_model_config 
from utils.logger_utils import write_to_main_log
from utils.model_utils import transfer_weights_from_pretrained
from utils.standardization import standardize_model_config_object 


class MaskedPatchEmbedding(nn.Module):
    def __init__(self, original_patch_embed, mask_token, num_register_tokens=0):
        super().__init__()
        self.patch_embed = original_patch_embed
        self.mask_token = mask_token
        self.num_register_tokens = num_register_tokens
        self._mask_indices = None
        self._batch_size = None
        self._num_patches = None

    @property
    def projection(self):
        """HF embeddings.forward() → self.patch_embeddings.projection.weight.dtype erişiyor."""
        if hasattr(self.patch_embed, 'projection'):
            return self.patch_embed.projection
        raise AttributeError(f"{type(self.patch_embed).__name__} has no 'projection'")

    
    def set_mask(self, mask_indices, batch_size, num_patches):
        """
        Set mask for next forward pass.
        
        Args:
            mask_indices: (total_masked_patches,) flattened indices
            batch_size: B
            num_patches: N (patches per image)
        """
        self._mask_indices = mask_indices
        self._batch_size = batch_size
        self._num_patches = num_patches
    
    def clear_mask(self):
        """Clear mask after forward pass"""
        self._mask_indices = None
        self._batch_size = None
        self._num_patches = None
    
    def forward(self, x):
        # Normal patch embedding
        # Output shape depends on backbone: could be (B, N, D) or (B, C, H, W) before flatten
        x = self.patch_embed(x)
        
        # Handle different output formats
        if x.ndim == 4:  # (B, C, H, W) -> (B, N, D)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Now x should be (B, N, D)
        if self._mask_indices is not None and self._batch_size is not None:
            B, N, D = x.shape
            
            # Create mask tokens expanded to batch
            mask_tokens = self.mask_token.expand(B, N, -1).to(x.dtype)
            
            # Create 2D mask from flattened indices
            # mask_indices are flattened across batch: index = b * N + n
            mask_2d = torch.zeros(B * N, dtype=torch.bool, device=x.device)
            mask_2d[self._mask_indices] = True
            mask_2d = mask_2d.view(B, N)
            
            # Apply mask: where mask is True, use mask_token
            x = torch.where(mask_2d.unsqueeze(-1), mask_tokens, x)
        
        return x


class DINOiBOTWrapper(nn.Module): 
    def __init__(self, config, backbone, dino_head, ibot_head=None ):
        super().__init__()
        self.config = config 
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head  
        self.n_global_crops = 2
        self.num_register_tokens = self.config.num_register_tokens
        
        # Mask token parameter - learnable
        embed_dim = config.hidden_size if hasattr(config, 'hidden_size') else 768
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Wrap patch embedding with masked version
        self._setup_masked_patch_embedding()
    
    def _setup_masked_patch_embedding(self):
        """Replace patch embedding layer with masked version"""
        self._original_patch_embed = None
        self._masked_patch_embed = None
        
        # Check different backbone structures
        if hasattr(self.backbone, 'embeddings'):
            embed = self.backbone.embeddings
            
            # HuggingFace ViT/DINOv2 structure
            if hasattr(embed, 'patch_embeddings'):
                self._original_patch_embed = embed.patch_embeddings
                self._masked_patch_embed = MaskedPatchEmbedding(
                    self._original_patch_embed, 
                    self.mask_token,
                    self.num_register_tokens
                )
                embed.patch_embeddings = self._masked_patch_embed
                
            elif hasattr(embed, 'patch_embedding'):
                self._original_patch_embed = embed.patch_embedding
                self._masked_patch_embed = MaskedPatchEmbedding(
                    self._original_patch_embed,
                    self.mask_token,
                    self.num_register_tokens
                )
                embed.patch_embedding = self._masked_patch_embed
                
            elif hasattr(embed, 'proj'):  # Conv projection
                self._original_patch_embed = embed.proj
                self._masked_patch_embed = MaskedPatchEmbedding(
                    embed.proj,
                    self.mask_token,
                    self.num_register_tokens
                )
                embed.proj = self._masked_patch_embed
        
        # timm model structure
        elif hasattr(self.backbone, 'patch_embed'):
            self._original_patch_embed = self.backbone.patch_embed
            self._masked_patch_embed = MaskedPatchEmbedding(
                self.backbone.patch_embed,
                self.mask_token,
                self.num_register_tokens
            )
            self.backbone.patch_embed = self._masked_patch_embed
        
        if self._masked_patch_embed is None:
            warnings.warn(
                "Could not find patch embedding layer to wrap. "
                "Mask token will not be applied. "
                "Falling back to output-only masking."
            )
    
    def _set_mask(self, mask_indices_list, pixel_values):
        """Set mask - calculate batch size and num_patches"""
        if self._masked_patch_embed is None or mask_indices_list is None:
            return
        
        B = pixel_values.shape[0]
        
        # Calculate number of patches
        if hasattr(self.config, 'patch_size'):
            patch_size = self.config.patch_size
        else:
            patch_size = 14  # default
        
        H = W = pixel_values.shape[-1] // patch_size
        N = H * W  # patches per image
        
        self._masked_patch_embed.set_mask(mask_indices_list, B, N)
    
    def _clear_mask(self):
        """Clear mask"""
        if self._masked_patch_embed is not None:
            self._masked_patch_embed.clear_mask()
    
    def forward(self, pixel_values, interpolate_pos_encoding=False, 
                mask_indices_list=None, shuffle_global=None, no_input_masking=False):
        
        if not no_input_masking:  # ← Teacher forward'da True geçiliyor, mask uygulanmaz
            self._set_mask(mask_indices_list, pixel_values)
        
        
        try:
            # Backbone forward (mask will be applied automatically)
            if interpolate_pos_encoding:
                backbone_output = self.backbone(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
            else: 
                backbone_output = self.backbone(pixel_values)
            
            # Process outputs
            if hasattr(backbone_output, "last_hidden_state"):
                model_cls = backbone_output.last_hidden_state[:, 0]
                model_patches = backbone_output.last_hidden_state[:, 1 + self.num_register_tokens:]
            else:
                model_cls = backbone_output[:, 0]
                model_patches = backbone_output[:, 1 + self.num_register_tokens:]
            
            # Shuffle global crops (for DINO)
            if shuffle_global:
                model_cls_chunks = model_cls.chunk(self.n_global_crops) 
                model_cls = torch.cat((model_cls_chunks[1], model_cls_chunks[0]))
            
            # DINO head (CLS token)
            dino_output = self.dino_head(model_cls)
            
            # iBOT head (masked patches)
            ibot_output = None
            if mask_indices_list is not None:
                # Select masked patches (now replaced with mask tokens)
                model_masked_patches = torch.index_select(
                    model_patches.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list
                )
                
                if self.ibot_head: 
                    ibot_output = self.ibot_head(model_masked_patches)
                else: 
                    ibot_output = self.dino_head(model_masked_patches)
            
            return dino_output, ibot_output, model_cls
            
        finally:
            # Always clear
            self._clear_mask()

class DinoWrapper(nn.Module): 
    def __init__(self, config, backbone, dino_head ):
        super().__init__()
        self.config = config 
        self.backbone = backbone
        self.dino_head = dino_head 
 
    def forward(self, pixel_values, interpolate_pos_encoding=False ) : 
        if interpolate_pos_encoding:
            backbone_output = self.backbone(  pixel_values,interpolate_pos_encoding=interpolate_pos_encoding  )
        else: 

            backbone_output = self.backbone(  pixel_values ) 

        if hasattr(backbone_output, "last_hidden_state"):
            embeddings = backbone_output.last_hidden_state[:, 0]
        else:
            embeddings = backbone_output 
        output = self.dino_head(embeddings) 

        return output


class LeJEPA_Wrapper(nn.Module):
    def __init__(self, config, backbone, projector):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.projector = projector

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        # --- 1. Extract Raw Tokens (Bypass Pooling) ---
        if hasattr(self.backbone, "forward_features"):
            # This is standard for timm/custom ViTs to get [Batch, Tokens, Dim]
            if interpolate_pos_encoding:
                embeddings = self.backbone.forward_features(pixel_values,
                                                            interpolate_pos_encoding=interpolate_pos_encoding)
            else:
                embeddings = self.backbone.forward_features(pixel_values)
        else:
            # Fallback for HuggingFace style models
            if interpolate_pos_encoding:
                backbone_output = self.backbone(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
            else:
                backbone_output = self.backbone(pixel_values)

            embeddings = backbone_output.last_hidden_state if hasattr(backbone_output,
                                                                      "last_hidden_state") else backbone_output

        # --- 2. Safety Check ---
        if embeddings.dim() == 2:
            raise RuntimeError(f"Backbone returned a 2D tensor {embeddings.shape}. V2 requires [Batch, Tokens, Dim]. "
                               "Ensure your ViT returns the unpooled sequence of tokens.")

        # --- 3. Isolate the [CLS] token (Index 0) ---
        # Shape goes from [Batch, Tokens, Dim] -> [Batch, Dim]
        cls_token = embeddings[:, 0, :] if embeddings.dim() == 3 else embeddings[:, 0]

        # --- 4. Project for LeJEPA SSL Loss ---
        output = self.projector(cls_token)

        # Return projected CLS for main loss, and full sequence for V2 auxiliary loss
        return output, embeddings

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
 


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

 
def load_pretrained_weights_onto_model(model, model_type, accelerator):
    """
    Updates the current model's weights with pretrained weights and returns the model.
    Model reference doesn't change, only weights are updated.
    """
    try:
        # First check for local checkpoint
        local_path = os.path.join('checkpoints', model_type)
        
        if os.path.exists(local_path):
            write_to_main_log(accelerator, f'Loading weights from local: {local_path}')
            pretrained_model = AutoModel.from_pretrained(local_path)
        else:
            write_to_main_log(accelerator, f'Loading weights from HuggingFace: {model_type}')
            pretrained_model = AutoModel.from_pretrained(model_type)
        
        # Copy weights to current model
        missing_keys, unexpected_keys = model.load_state_dict(
            pretrained_model.state_dict(), 
            strict=False
        )
        
        # Report missing or unexpected keys
        if missing_keys:
            write_to_main_log(accelerator, f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...", 'warning')
        if unexpected_keys:
            write_to_main_log(accelerator, f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...", 'warning')
            
        write_to_main_log(accelerator, 'Pretrained weights loaded onto existing model')
         
        return model, True
        
    except Exception as e:
        write_to_main_log(accelerator, f"Failed to load pretrained weights: {e}", 'error')
        return model, False

def set_dropout_from_config(config,model_config, accelerator): 
    if hasattr(config, 'dropout_config') and config.dropout_config is not None:
        write_to_main_log(accelerator, "✅ Found custom dropout/regularization configuration.")
        dropout_settings=  dict(config.dropout_config) 
    else:
        return model_config
    
    for key, value in dropout_settings.items():
        if key == 'attention_dropout':

            if 'dinov3' in model_config.model_type:
                if hasattr(config.dropout_config, 'attention_dropout'):
                        setattr(model_config, 'attention_dropout', value)
                        write_to_main_log(accelerator, f"  - Set DINOv3 style 'attention_dropout' = {value}")
            else: 
                if hasattr(config.dropout_config, 'attention_dropout'):
                        setattr(model_config, 'attention_probs_dropout_prob', value)
                        write_to_main_log(accelerator, f"  - Set DINOv1 and DINOv2 style 'attention_dropout' = {value}") 
        else:
            if hasattr(config.dropout_config, key):
                    setattr(model_config, key, value)
                    write_to_main_log(accelerator, f"  - Set {key} = {value}")
            else:
                    write_to_main_log(accelerator, f"  - Warning: Attribute '{key}' not found in model_config.", 'warning')
                    
    return model_config
def check_timm_model(model_type):  
    try:
        model_config = AutoModel.from_pretrained(model_type)
        print(model_config)
        if 'timm' in model_config.model_type: 
            return True
        else: 
            return False
    except Exception as  e: 
        return False
def build_transformer_model_from_config(config, model_type, accelerator, custom_config, load_pretrained, checkpoint_path=None): 

    write_to_main_log(accelerator, "=== STEP 1: Loading Configuration ===") 
    try: 
        model_config = None # facebook/dinov2-with-registers-base
        try: 
            write_to_main_log(accelerator, f"Trying to load model configuration from '{model_type}'...", 'info')
            model_config = AutoConfig.from_pretrained(model_type)
            write_to_main_log(accelerator, f"Successfully loaded model configuration from '{model_type}'.", 'info')

        except Exception as e: 
            write_to_main_log(
                accelerator,  f"Model configuration couldn't be found for '{model_type}'. Reason: {e}. Trying to load it from checkpoint path.", 'warning'    )
            if checkpoint_path:
                try: 
                    write_to_main_log(accelerator, f"Loading configuration from checkpoint: '{checkpoint_path}'", 'info')
                    model_config = AutoConfig.from_pretrained(checkpoint_path)
                    write_to_main_log(accelerator, "Successfully loaded configuration from checkpoint.", 'info')
                except Exception as e2: 
                    write_to_main_log( accelerator,  f"FATAL: Failed to load configuration from checkpoint path '{checkpoint_path}' as well. Error: {e2}",'error'
                    ) 
            else: 
                write_to_main_log(
                    accelerator,  "FATAL: Model couldn't be loaded from model_type and no checkpoint_path was provided. Please check your configuration file!", 'error'
                ) 
        if model_config is None:
            write_to_main_log(accelerator, "Exiting due to missing model configuration.", 'error')
            # exit() veya raise Exception()

######
        if custom_config:
            for key, value in vars(custom_config).items():
                if key != "architectures":
                    setattr(model_config, key, value)
            if hasattr(config, 'dropout_config'): 
                model_config = set_dropout_from_config(config,model_config, accelerator)
            
            if hasattr(config.train, 'drop_path_rate'): 
                setattr(model_config, 'drop_path_rate', config.train.drop_path_rate)
                write_to_main_log(accelerator, f"Drop path rate is adjusted as: {config.train.drop_path_rate} in config for {model_type}")
            write_to_main_log(accelerator, f"Pretrained config loaded and customized for {model_type}")
            
    except Exception as e:
        write_to_main_log(accelerator, f"Pretrained config failed: {e}. Creating from scratch.", 'warning')
         
        config_dict = {**vars(custom_config), "_name_or_path": custom_config.model_type}
        
        config_classes = {
            'vit': ViTConfig,
            'dinov2_with_registers': Dinov2WithRegistersConfig,
            'dinov2': Dinov2Config,
            'dinov3_vit': DINOv3ViTConfig
        }
        
        config_class = config_classes.get(custom_config.model_type)
        
        if not config_class:
            raise ValueError(f"Unsupported model type: {custom_config.model_type}")
        
        model_config = config_class(**config_dict)
        write_to_main_log(accelerator, f"Config created from scratch for {custom_config.model_type}")

    # 2. Create model
    write_to_main_log(accelerator, "=== STEP 2: Initializing Model ===")
    
    model = AutoModel.from_config(model_config) # single channel 
    write_to_main_log(accelerator, "Empty model initialized from config")
    
    # 3. Load weights
    write_to_main_log(accelerator, "=== STEP 3: Loading Weights ===")
    
    # Check for ambiguous configuration
    if checkpoint_path and load_pretrained:
        write_to_main_log(accelerator, 
            f"WARNING: Both load_pretrained=True and checkpoint_path provided. "
            f"Prioritizing checkpoint over pretrained weights as per user preference.", 
            'warning')
    
    if checkpoint_path:
        # Option A: Load complete new model from checkpoint (prioritized over pretrained)
        write_to_main_log(accelerator, f"Loading complete model from checkpoint: {checkpoint_path}")
        try:
            model = AutoModel.from_pretrained(checkpoint_path)
            write_to_main_log(accelerator, f'✅ New model loaded from checkpoint: {checkpoint_path}')
        except Exception as e:
            write_to_main_log(accelerator, f'Failed to load from checkpoint: {checkpoint_path} - {e}', 'error')
            write_to_main_log(accelerator, "Falling back to other weight loading methods...", 'warning')
            # Fallback to pretrained if checkpoint loading failed
            if load_pretrained:
                write_to_main_log(accelerator, f"Attempting to load pretrained weights after checkpoint failure")
                model, success = load_pretrained_weights_onto_model(model, model_type, accelerator)
                if not success:
                    write_to_main_log(accelerator, "Pretrained loading failed, trying transfer learning...", 'warning')
                    _transfer_from_closest_model(model, custom_config, accelerator)
            else:
                write_to_main_log(accelerator, "Keeping empty model", 'warning')
                
    elif load_pretrained:
        # Option B: Load pretrained weights onto current model (no checkpoint)
        write_to_main_log(accelerator, f"Loading pretrained weights onto current model")
        model, success = load_pretrained_weights_onto_model(model, model_type, accelerator)
        # where 3 channel doesn't fit with multi channel 
        if not success:
            write_to_main_log(accelerator, "Pretrained loading failed, trying transfer learning...", 'warning')
            # Fallback: transfer weights from closest compatible model
            _transfer_from_closest_model(model, custom_config, accelerator)
    else:
        # Option C: No weight loading, continue with random weights
        write_to_main_log(accelerator, "No weight loading requested, using random initialization")

    write_to_main_log(accelerator, "=== MODEL BUILD COMPLETE ===")
    return model, model_config

def _transfer_from_closest_model(model, custom_config, accelerator):
    """Helper function to transfer weights from closest compatible model."""
    write_to_main_log(accelerator=accelerator, result="Attempting weight transfer from closest model", type='info')
    
    closest_model_type = find_closest_model_config(custom_config, accelerator)
    if not closest_model_type:
        write_to_main_log(accelerator=accelerator, result="No suitable model found for weight transfer", type='warning')
        return
    
    try:
        # Try local checkpoint first, then remote
        checkpoint_path = os.path.join('checkpoints', closest_model_type)
        source_path = checkpoint_path if os.path.exists(checkpoint_path) else closest_model_type
        
        source_model = AutoModel.from_pretrained(source_path)
        write_to_main_log(accelerator=accelerator, result=f"Source model loaded from {source_path}")
        
        # Transfer weights
        model = transfer_weights_from_pretrained(model, source_model, method='mean', accelerator=accelerator)
        write_to_main_log(accelerator=accelerator, result=f"Successfully transferred weights from {closest_model_type}")
        
        del source_model
    except Exception as e:
        write_to_main_log(accelerator=accelerator, result=f"Error during weight transfer: {e}", type='error')

def build_dino_head_from_config(config,model_params):  
    in_dim =model_params.hidden_size
    out_dim=config.dino_head.out_dim
    use_bn_in_head=False
    norm_last_layer=config.dino_head.norm_last_layer 

    hidden_dim   =getattr(config.dino_head, "hidden_dim", 2048) 
    bottleneck_dim   =getattr(config.dino_head, "bottleneck_dim", 256) 
    nlayers   =getattr(config.dino_head, "nlayers", 3) 

    return DINOHead(  in_dim,  out_dim,  use_bn=use_bn_in_head,  norm_last_layer=norm_last_layer,
                     hidden_dim=hidden_dim,bottleneck_dim=bottleneck_dim, nlayers=nlayers) 

def build_ibot_head_from_config(config,model_params): 
    in_dim =model_params.hidden_size
    out_dim=config.ibot.out_dim
    use_bn_in_head=False
    norm_last_layer=config.dino_head.norm_last_layer 
    
    hidden_dim   =getattr(config.ibot, "hidden_dim", 2048) 
    bottleneck_dim   =getattr(config.ibot, "bottleneck_dim", 256) 
    nlayers   =getattr(config.ibot, "nlayers", 3) 
    return DINOHead(  in_dim,  out_dim,  use_bn=use_bn_in_head,  norm_last_layer=norm_last_layer, 
                    hidden_dim=hidden_dim,bottleneck_dim=bottleneck_dim, nlayers=nlayers) 

def build_transformer_model_from_timm(model_config, accelerator, use_pretrained=False, weights_path=None):
    """
    Paranoid builder for TIMM models.
    Enforces strict loading and verifies weight statistics.
    """
    model_type = model_config.model_type
    base_arch_name = getattr(model_config, 'timm_arch', 'vit_large_patch16_224')

    write_to_main_log(accelerator, f"Building TIMM model: {model_type} (Arch: {base_arch_name})")

    # Validate configuration compatibility
    if use_pretrained and weights_path:
        write_to_main_log(accelerator, 
            f"WARNING: Both use_pretrained=True and checkpoint_path provided. "
            f"Prioritizing checkpoint over pretrained weights as per user preference.", 
            'warning')
        # When both are provided, we'll prioritize checkpoint over pretrained
        # This means we'll treat it as use_pretrained=False for weight loading
        # but we still need to create the model structure
        use_pretrained_for_structure = use_pretrained  # Keep for model creation
        load_from_checkpoint = True
    else:
        use_pretrained_for_structure = use_pretrained
        load_from_checkpoint = bool(weights_path)

    native_img_size = 224
    if hasattr(model_config, 'extra_kwargs'):
        extra = vars(model_config.extra_kwargs) if not isinstance(model_config.extra_kwargs,
                                                                  dict) else model_config.extra_kwargs
        native_img_size = extra.get('img_size', 224)

    # 1. Map config keys to TIMM arguments
    timm_kwargs = {
        'img_size': native_img_size,
        'patch_size': model_config.patch_size,
        'embed_dim': model_config.hidden_size,
        'depth': model_config.num_hidden_layers,
        'num_heads': model_config.num_attention_heads,
        'mlp_ratio': model_config.mlp_ratio,
        'init_values': 1e-5,
        'num_classes': 0,
        'dynamic_img_size': True,
    }

    if hasattr(model_config, 'num_register_tokens') and model_config.num_register_tokens > 0:
        timm_kwargs['reg_tokens'] = model_config.num_register_tokens

    if hasattr(model_config, 'qkv_bias'):
        timm_kwargs['qkv_bias'] = model_config.qkv_bias

    # Handle extra arguments, like input channels for CT data
    if hasattr(model_config, 'extra_kwargs'):
        extra = vars(model_config.extra_kwargs) if not isinstance(model_config.extra_kwargs,
                                                                  dict) else model_config.extra_kwargs
        
        # Convert string layer names to actual class references
        # timm expects mlp_layer and act_layer to be callable classes, not strings
        if 'mlp_layer' in extra and isinstance(extra['mlp_layer'], str):
            mlp_layer_name = extra['mlp_layer']
            if mlp_layer_name == 'SwiGLUPacked':
                extra['mlp_layer'] = timm.layers.SwiGLUPacked
            elif mlp_layer_name == 'Mlp':
                extra['mlp_layer'] = timm.layers.Mlp
            else:
                # Try to get the class from timm.layers module
                try:
                    extra['mlp_layer'] = getattr(timm.layers, mlp_layer_name)
                except AttributeError:
                    write_to_main_log(accelerator, 
                        f"WARNING: Could not find mlp_layer '{mlp_layer_name}' in timm.layers. Using default Mlp.", 
                        'warning')
                    extra['mlp_layer'] = timm.layers.Mlp
        
        if 'act_layer' in extra and isinstance(extra['act_layer'], str):
            act_layer_name = extra['act_layer']
            if act_layer_name == 'SiLU':
                extra['act_layer'] = torch.nn.SiLU
            elif act_layer_name == 'GELU':
                extra['act_layer'] = torch.nn.GELU
            elif act_layer_name == 'ReLU':
                extra['act_layer'] = torch.nn.ReLU
            elif act_layer_name == 'gelu':
                extra['act_layer'] = torch.nn.GELU
            elif act_layer_name == 'relu':
                extra['act_layer'] = torch.nn.ReLU
            else:
                # Try to get the class from torch.nn module
                try:
                    extra['act_layer'] = getattr(torch.nn, act_layer_name)
                except AttributeError:
                    write_to_main_log(accelerator, 
                        f"WARNING: Could not find act_layer '{act_layer_name}' in torch.nn. Using default GELU.", 
                        'warning')
                    extra['act_layer'] = torch.nn.GELU
        
        timm_kwargs.update(extra)

    # 2. Instantiate Skeleton with appropriate initialization
    # Determine if we should create model with pretrained weights or random init
    # Note: Even if we have checkpoint, we create model without pretrained weights
    # because we'll load checkpoint weights separately
    create_with_pretrained = use_pretrained_for_structure and not load_from_checkpoint
    
    if create_with_pretrained:
        write_to_main_log(accelerator, f"Loading pretrained weights from TIMM repository: {base_arch_name}")
        model = timm.create_model(base_arch_name, pretrained=True, **timm_kwargs)
        write_to_main_log(accelerator, "✅ Model loaded with pretrained TIMM weights")
    else:
        write_to_main_log(accelerator, f"Creating model with random initialization: {base_arch_name}")
        model = timm.create_model(base_arch_name, pretrained=False, **timm_kwargs)
        write_to_main_log(accelerator, "✅ Model created with random initialization")

    # 3. Load Weights from checkpoint (if available)
    if load_from_checkpoint and weights_path:
        write_to_main_log(accelerator, f"Attempting to load weights from checkpoint: {weights_path}")
        state_dict = None
        safetensors_path = os.path.join(weights_path, "model.safetensors")
        bin_path = os.path.join(weights_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
            write_to_main_log(accelerator, f"Found weights: {safetensors_path}")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            write_to_main_log(accelerator, f"Found weights: {bin_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {weights_path}")

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
            write_to_main_log(accelerator, f"✅ Loaded weights from checkpoint: {weights_path}")
    elif weights_path and use_pretrained and not load_from_checkpoint:
        write_to_main_log(accelerator, 
            f"NOTE: Checkpoint path provided but use_pretrained=True. "
            f"Using pretrained TIMM weights instead of checkpoint.", 
            'info')

        # --- 3D ADAPTER INJECTION ---
        # Check if the config requests 3D inflation
        is_3d = False
        frames_per_clip = 32
        if hasattr(model_config, 'extra_kwargs'):
            extra = vars(model_config.extra_kwargs) if not isinstance(model_config.extra_kwargs,
                                                                      dict) else model_config.extra_kwargs
            is_3d = extra.get('is_3d', False)
            frames_per_clip = extra.get('frames_per_clip', 32)

        if is_3d:
            write_to_main_log(accelerator,
                              f"🚀 Converting 2D TIMM model to 3D with factorized PE (Depth: {frames_per_clip})")
            # Wrap the initialized/loaded 2D model in our 3D Adapter
            model = Timm3DViTAdapter(model, frames_per_clip=frames_per_clip)
        # ---------------------------------

    # Log final initialization method
    if load_from_checkpoint and weights_path:
        write_to_main_log(accelerator, "📊 INITIALIZATION: Checkpoint weights (prioritized over pretrained)")
    elif use_pretrained and not load_from_checkpoint:
        write_to_main_log(accelerator, "📊 INITIALIZATION: Pretrained TIMM weights")
    else:
        write_to_main_log(accelerator, "📊 INITIALIZATION: Random weights")
        
    return model, model_config


class Timm3DViTAdapter(nn.Module):
    """
    Wraps a 2D TIMM ViT to support 3D volumetric data using inflated weights
    and factorized positional embeddings.
    """

    def __init__(self, model_2d, frames_per_clip=32):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.model_2d = model_2d

        # 1. Inflate Patch Embeddings
        # Extract the native 2D Conv projection layer from TIMM
        proj_2d = model_2d.patch_embed.proj
        out_ch, in_ch, k_h, k_w = proj_2d.weight.shape
        stride_h, stride_w = proj_2d.stride

        # Initialize the new 3D Conv layer
        self.patch_embed_3d = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(frames_per_clip, k_h, k_w),
            stride=(frames_per_clip, stride_h, stride_w),
            bias=proj_2d.bias is not None
        )

        # Inflate the 2D weights to 3D and copy them over
        with torch.no_grad():
            # weight_2d: [out_ch, in_ch, k_h, k_w] -> [out_ch, in_ch, 1, k_h, k_w] -> repeat Z times
            w_3d = proj_2d.weight.unsqueeze(2).repeat(1, 1, frames_per_clip, 1, 1)
            self.patch_embed_3d.weight.copy_(w_3d / frames_per_clip)
            if proj_2d.bias is not None:
                self.patch_embed_3d.bias.copy_(proj_2d.bias)

        # 2. Factorized Depth Positional Embeddings
        # We initialize a small 1D learnable parameter for the Z-axis
        self.depth_pos_embed = nn.Parameter(torch.zeros(1, frames_per_clip, 1, out_ch))
        nn.init.trunc_normal_(self.depth_pos_embed, std=.02)

    def forward_features(self, x, interpolate_pos_encoding=False):
        # Expected input: [Batch, Channels, Depth, Height, Width]
        B, C, D_vol, H_vol, W_vol = x.shape

        # 1. Extract 3D Patches
        x = self.patch_embed_3d(x)  # Output: [B, embed_dim, D_prime, H_prime, W_prime]
        _, embed_dim, D_prime, H_prime, W_prime = x.shape

        # Flatten spatial and depth dims -> [B, D'*H'*W', embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # 2. Add Factorized Positional Embeddings
        num_prefix = self.model_2d.num_prefix_tokens  # Usually 1 (CLS) or more (Registers)

        # Extract native 2D spatial embeddings [1, H'*W', embed_dim]
        pos_embed_spatial = self.model_2d.pos_embed[:, num_prefix:]

        # Broadcast spatial and depth PEs together
        # [1, 1, H'*W', D] + [1, D_prime, 1, D] -> [1, D_prime, H'*W', D]
        pos_embed_3d = pos_embed_spatial.unsqueeze(1) + self.depth_pos_embed

        # Flatten the PE grid to match token sequence: [1, D_prime * H' * W', D]
        pos_embed_3d = pos_embed_3d.view(1, D_prime * H_prime * W_prime, embed_dim)
        x = x + pos_embed_3d

        # 3. Handle Prefix Tokens (CLS + Registers)
        prefix_tokens = []
        if self.model_2d.cls_token is not None:
            prefix_tokens.append(self.model_2d.cls_token.expand(B, -1, -1))
        if hasattr(self.model_2d, 'reg_token') and self.model_2d.reg_token is not None:
            prefix_tokens.append(self.model_2d.reg_token.expand(B, -1, -1))

        prefix_tokens = torch.cat(prefix_tokens, dim=1)
        # Add the native PE for the prefix tokens
        prefix_tokens = prefix_tokens + self.model_2d.pos_embed[:, :num_prefix]

        # Concatenate prefix to the front of our 3D patch tokens
        x = torch.cat((prefix_tokens, x), dim=1)

        # 4. Pass through native TIMM transformer blocks
        x = self.model_2d.blocks(x)
        x = self.model_2d.norm(x)

        return x

    def gradient_checkpointing_enable(self):
        # Expose this method so LeJEPA setup doesn't crash if GC is enabled
        self.model_2d.set_grad_checkpointing(True)
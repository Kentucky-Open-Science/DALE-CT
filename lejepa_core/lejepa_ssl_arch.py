import math
import sys
import torch 
import torch.distributed as dist
from torch import nn    
from lejepa_core.SIGReg import SIGReg
from lejepa_core.VISReg import VISReg
from utils.config import load_model_configs
from utils.logger_utils import write_to_main_log
from utils.wandb_utils import log_metrics_wandb

# --- LeJEPA Scheduler Integration ---
from utils.lejepa_scheduler import LeJEPAScheduler 

from utils.dino_utils import (  
    clip_gradients,
    freeze_layers, 
    get_params_groups,              
    init_lejepa_training_models_wrapper, 
)


class SSLMetaArch(nn.Module):
    """
    LeJEPA Self-Supervised Learning Architecture
    
    Key differences from DINO: 
    - No stop-gradient
    - Uses SIGReg for regularization instead of centering
    - Single hyperparameter λ for loss weighting
    """
    def __init__(
        self,
        config,
        accelerator 
    ):
        super().__init__()
        self.config = config
        self.accelerator = accelerator 
        self.model_type = config.train.model_type
        self.model_params = load_model_configs(self.model_type) 
        self.device = accelerator.device 
        
        # --- Regularizer config (SIGReg default; VISReg opt-in via config.regularizer.type) ---
        reg_conf = getattr(config, 'regularizer', None)
        sigreg_conf = getattr(config, 'sigreg', None)
        self.regularizer_type = getattr(reg_conf, 'type', 'sigreg') if reg_conf is not None else 'sigreg'
        # λ: prefer config.regularizer.lejepa_lambda, fall back to config.sigreg for backward compat.
        if reg_conf is not None and hasattr(reg_conf, 'lejepa_lambda'):
            self.lejepa_lambda = reg_conf.lejepa_lambda
        else:
            self.lejepa_lambda = getattr(sigreg_conf, 'lejepa_lambda', 0.02)
        # SIGReg params (used only by the SIGReg path; kept for the log message).
        self.num_slices = getattr(sigreg_conf, 'num_slices', 256)
        self.sigreg_knots = getattr(sigreg_conf, 'knots', 17)
        # VISReg params (galaxy-best "Shape 4:1" defaults).
        self.visreg_K = getattr(reg_conf, 'K', 4096) if reg_conf is not None else 4096
        self.visreg_lambda_scale = getattr(reg_conf, 'lambda_scale', 0.5) if reg_conf is not None else 0.5
        self.visreg_lambda_shape = getattr(reg_conf, 'lambda_shape', 2.0) if reg_conf is not None else 2.0
        self.visreg_lambda_center = getattr(reg_conf, 'lambda_center', 0.5) if reg_conf is not None else 0.5
        # Collapse fixups (defaults preserve 2S/SIGReg behavior):
        #  - apply_to: where the regularizer reads the representation. 'projector' (default)
        #    regularizes the projector output (same space as the invariance loss). 'backbone_cls'
        #    regularizes the backbone [CLS] — the saved/evaluated representation — directly,
        #    bypassing the projector's LayerNorms that mask collapse from VISReg's bounded L_scale
        #    (L_shape is scale-invariant and cannot detect uniform collapse).
        #  - detach_centers: stop-gradient on the invariance centroid, removing the collapse driver
        #    (without it, collapse drives inv_loss -> 0, i.e. invariance rewards collapse).
        self.visreg_apply_to = getattr(reg_conf, 'apply_to', 'projector') if reg_conf is not None else 'projector'
        self.invariance_detach_centers = getattr(reg_conf, 'detach_centers', False) if reg_conf is not None else False
        
        # --- LoRA Configuration ---
        self.use_lora = getattr(config.train, 'use_lora', False)
        self.lora_r = getattr(config.lora_config, 'lora_r', None) if self.use_lora else None
        self.lora_alpha = getattr(config.lora_config, 'lora_alpha', None) if self.use_lora else None
        self.lora_dropout = getattr(config.lora_config, 'lora_dropout', None) if self.use_lora else None
        self.clip = getattr(config.train, 'clip_grad', 0.0)
        self.early_stop_triggered = False 
        self.is_best_checkpoint = False
        
        # --- Dimensions --- 
        self.patch_size = self.model_params.patch_size 
        self.frozen_layers_count = 0
        
        # --- Initialize Model (ModuleDict with backbone + projector) ---
        self.lejepa_model, self.model_config = init_lejepa_training_models_wrapper(
            config=self.config,  
            accelerator=self.accelerator, 
            lora_alpha=self.lora_alpha, 
            lora_dropout=self.lora_dropout, 
            lora_r=self.lora_r
        )

        self.aux_head = None
        aux_conf = getattr(config, 'auxiliary', None)
        if aux_conf and getattr(aux_conf, 'enable', False):
            input_dim = getattr(self.model_params, 'hidden_size', 768)
            
            # Dynamically load the specified supervised head
            head_type = getattr(aux_conf, 'head_type', 'organ_supervision')
            
            if head_type == 'organ_supervision':
                from supervised_heads.organ_supervision import OrganSupervisionHead
                self.aux_head = OrganSupervisionHead(config=config, input_dim=input_dim).to(self.device)
                
            elif head_type == 'example_supervised_head':
                from supervised_heads.example_supervised_head import ExampleSupervisedHead
                head_params = getattr(aux_conf, 'head_params', {})
                task_type = head_params.get('task_type', 'classification')
                num_classes = head_params.get('num_classes', 10)
                hidden_dim = head_params.get('hidden_dim', 512)
                
                self.aux_head = ExampleSupervisedHead(
                    config=config,
                    input_dim=input_dim,
                    task_type=task_type,
                    num_classes=num_classes,
                    hidden_dim=hidden_dim
                ).to(self.device)
            elif head_type == 'soft_label_supervision':
                from supervised_heads.soft_label_supervision import SoftLabelSupervisionHead
                self.aux_head = SoftLabelSupervisionHead(config=config, input_dim=input_dim).to(self.device)
                
            else:
                write_to_main_log(
                    accelerator=self.accelerator,
                    result=f"Unrecognized supervised head type '{head_type}'. Skipping.",
                    type='warning'
                )

        # --- Initialize Regularizer (SIGReg or VISReg) ---
        if self.regularizer_type == 'visreg':
            self.regularizer = VISReg(
                K=self.visreg_K,
                lambda_scale=self.visreg_lambda_scale,
                lambda_shape=self.visreg_lambda_shape,
                lambda_center=self.visreg_lambda_center,
            ).to(self.device)
        else:
            self.regularizer = SIGReg(
                knots=self.sigreg_knots,
                num_slices=self.num_slices
            ).to(self.device)
        write_to_main_log(
            accelerator=self.accelerator,
            result=(f"Regularizer: type={self.regularizer_type}, lambda={self.lejepa_lambda}, "
                    f"apply_to={self.visreg_apply_to}, detach_centers={self.invariance_detach_centers}"),
            type='info'
        )
         
        self.apply_interpolate = self.model_params.model_type == 'vit'
         
        # --- Freeze Layers (if specified) ---
        self.freeze_backbone_layers = 0 if self.use_lora else getattr(config.train, 'freeze_backbone_layers', 0) 
        freeze_layers(
            num_layers_to_freeze=self.freeze_backbone_layers,   
            accelerator=self.accelerator, 
            use_lora=self.use_lora, 
            model=self.lejepa_model.backbone  
        )
   
        # --- Optimizer Setup ---
        wd_val = getattr(config.train, 'weight_decay', 0.04)
        params_groups = get_params_groups(self.lejepa_model)
        self.optimizer = torch.optim.AdamW(params_groups, weight_decay=wd_val, eps=1e-6)
         
        # --- Accelerator Preparation ---
        self.lejepa_model, self.optimizer = accelerator.prepare(
            self.lejepa_model, self.optimizer
        )
 
        # --- Scheduler ---
        self.scheduler = LeJEPAScheduler(
            config=config, 
            accelerator=accelerator, 
            optimizer=self.optimizer, 
            model=self.lejepa_model
        )
        
        # --- Logging ---
        if self.accelerator.is_main_process:
            write_to_main_log(
                accelerator=self.accelerator, 
                result=(f"✅ LeJEPA initialized: regularizer={self.regularizer_type}, λ={self.lejepa_lambda}"
                        + (f", M={self.num_slices}, knots={self.sigreg_knots}" if self.regularizer_type == 'sigreg'
                           else f", K={self.visreg_K}, λ_scale={self.visreg_lambda_scale}, "
                                f"λ_shape={self.visreg_lambda_shape}, λ_center={self.visreg_lambda_center}"))
            )
         
            write_to_main_log(
                accelerator=self.accelerator, 
                result=f"✅ Clip Grad:  {self.clip}"
            ) 
         
    def forward(self, iteration, data, initialize=False): 
        """
        Forward pass for LeJEPA training.
        
        Args:
            iteration: Current training iteration
            data: Dict with 'aug_imgs' key, shape (B, V, C, H, W)
            initialize: Whether this is initialization step
        """
        self.optimizer.zero_grad()
         
        # --- Scheduler Step ---
        self.early_stop_triggered, self.is_best_checkpoint = self.scheduler.step(iteration) 

        with self.accelerator.autocast():
            loss_dict = self._compute_loss(data=data, iteration=iteration)
            total_loss = loss_dict['total_loss']
            
            # --- Safety Check ---
            if not math.isfinite(total_loss.item()):
                write_to_main_log(
                    accelerator=self.accelerator, 
                    result=f"Loss is {total_loss.item()}, stopping training", 
                    type='error'
                )   
                sys.exit(1)
 
            self.accelerator.backward(total_loss) 
                 
            if self.clip>0.0:
                clip_gradients(self.lejepa_model, self.clip)

            self.optimizer.step() 

        # --- Sync losses across GPUs ---
        if dist.is_initialized():
            with torch.no_grad():
                for k, v in loss_dict.items():
                    if k != 'probe_features' and isinstance(v, torch.Tensor):
                        dist.all_reduce(v, op=dist.ReduceOp.AVG)

        self.accelerator.wait_for_everyone()

        # --- Logging & Return ---
        if self.accelerator.is_main_process:
            metrics_to_log = {
                "train/total_loss": loss_dict['total_loss'].item(),
                "train/inv_loss": loss_dict['inv_loss'].item(),
                "train/reg_loss": loss_dict['sigreg_loss'].item(),
                "train/learning_rate": self.optimizer.param_groups[0]["lr"]
            }

            # Standard keys we don't need to dynamically loop over
            standard_keys = {'total_loss', 'inv_loss', 'sigreg_loss', 'lejepa_loss', 'probe_features'}

            # Dynamically add all auxiliary and multi-task metrics to WandB
            for key, value in loss_dict.items():
                if key not in standard_keys:
                    # Safely extract the value whether it's a tensor or a float
                    val = value.item() if isinstance(value, torch.Tensor) else value
                    # Prefix with 'train/' to keep your WandB dashboard organized
                    metrics_to_log[f"train/{key}"] = val

            log_metrics_wandb(metrics_to_log, step=iteration)

            # Optional per-iter CSV trace (verification only; off unless
            # config.train.log_csv_path is set). Used for the resume-spike A/B/C
            # comparison so SIGReg/inv_loss trajectories can be diffed exactly.
            csv_path = getattr(self.config.train, 'log_csv_path', None)
            if csv_path:
                with open(csv_path, 'a') as f:
                    f.write(f"{iteration},{loss_dict['total_loss'].item():.6f},"
                            f"{loss_dict['inv_loss'].item():.6f},"
                            f"{loss_dict['sigreg_loss'].item():.6f},"
                            f"{self.optimizer.param_groups[0]['lr']:.8f}\n")

        return self._format_log_string(loss_dict, iteration), loss_dict['probe_features'].detach()

    '''
    def _compute_loss_loop(self, data, iteration):
        loss_dict = {}
        
        # =========================================================================
        # SCENARIO:
        # Batch Size (B) = 256
        # Global Views = 2, Local Views = 6 -> Total 8 Views
        # Projection Dimension (Dim) = 128
        # =========================================================================

        # 1. DATA LOADING
        # global_crops Shape: (B, 2, 3, 224, 224) -> (256, 2, 3, 224, 224)
        global_crops = data['global_crops'].to(self.device, non_blocking=True) 
        
        # local_crops Shape:  (B, 6, 3, 96, 96)   -> (256, 6, 3, 96, 96)
        local_crops = data['local_crops'].to(self.device, non_blocking=True)   
        
        B = global_crops.shape[0] # 256
        
        # --- STEP 1: Global Crops Forward ---
        
        # Flatten: Merge Batch and View dimensions to feed into the model.
        # (256 * 2, 3, 224, 224) -> (512, 3, 224, 224)
        flat_global = global_crops.flatten(0, 1) 
        
        # Forward pass through the model
        # Output: (512, 128)
        global_out = self.lejepa_model(
            flat_global,
            interpolate_pos_encoding=self.apply_interpolate  
        )
        
        # Reshape (Restore dimensions):
        # (512, 128) -> (256, 2, 128)
        global_out = global_out.view(B, -1, global_out.shape[-1])

        # --- STEP 2: Local Crops Forward ---
        
        # Flatten:
        # (256 * 6, 3, 96, 96) -> (1536, 3, 96, 96)
        flat_local = local_crops.flatten(0, 1)
        
        # Forward pass through the model:
        # Output: (1536, 128)
        local_out = self.lejepa_model(
            flat_local,
            interpolate_pos_encoding=self.apply_interpolate  
        )
        
        # Reshape:
        # (1536, 128) -> (256, 6, 128)
        local_out = local_out.view(B, -1, local_out.shape[-1])
         
        # --- STEP 3: Concatenation ---
        
        # torch.cat (Concatenate along Dim=1, i.e., the View axis):
        # (256, 2, 128) + (256, 6, 128) -> (256, 8, 128)
        all_views = torch.cat([global_out, local_out], dim=1)
         
        # Transpose (Swap Batch and View dimensions):
        # (256, 8, 128) -> (8, 256, 128)
        # Now Dim 0 is: "Which View Group?" (1st Global, 2nd Global, 1st Local, etc.)
        proj = all_views.transpose(0, 1) 
        
        # --- STEP 4: Loss Calculation ---
        
        # A) Invariance Loss (Converging to the Target)
        
        # Determine Target (Virtual Teacher):
        # Only global_out is used: (256, 2, 128)
        # Take the mean along Dim=1 (View).
        # Result: (256, 128) -> One "Centroid Embedding" per image
        centers = global_out.mean(dim=1) 
        
        # Calculate Difference (Broadcasting):
        # centers.unsqueeze(0) -> (1, 256, 128)
        # proj                 -> (8, 256, 128)
        # Operation: (1, 256, 128) - (8, 256, 128) 
        # Math: One center is subtracted from 8 different views one by one.
        inv_loss = (centers.unsqueeze(0) - proj).square().mean()
        
        # B) SIGReg Loss (Gaussian Distribution - Per View)
        
        sigreg_losses = []
        # proj.shape[0] = 8 (Total Number of Views)
        # Loop runs 8 times.
        for v in range(proj.shape[0]): 
            # proj[v] -> The entire batch for the current View group
            # Shape: (256, 128) -> "v-th view" of all images in the batch
            view_embedding = proj[v]   
            
            # Check if this matrix (256, 128) follows a Gaussian distribution.
            # Statistically safe since Batch is 256.
            loss = self.sigreg(x=view_embedding, global_step=iteration) 
            sigreg_losses.append(loss)
            
        # Take the mean of 8 loss values.
        sigreg_loss = torch.stack(sigreg_losses).mean()
        
        # C) Total LeJEPA Loss
        lejepa_loss = self.lejepa_lambda * sigreg_loss + (1 - self.lejepa_lambda) * inv_loss
        
        # --- Logging ---
        loss_dict['inv_loss'] = inv_loss
        loss_dict['sigreg_loss'] = sigreg_loss
        loss_dict['lejepa_loss'] = lejepa_loss
        loss_dict['total_loss'] = lejepa_loss
        
        return loss_dict

    '''

    def _compute_loss(self, data,iteration):
        """
        Computes the LeJEPA loss using a fully vectorized approach.
        
        Example Dimensions used below:
        B (Batch Size) = 256
        V_global = 2 | V_local = 6 | V_total = 8
        D (Projection Dim) = 128
        """
        loss_dict = {}
        
        # 1. Load Data
        # global_crops: (256, 2, 3, 224, 224) -> [B, V_g, C, H, W]
        # local_crops:  (256, 6, 3, 98, 98)   -> [B, V_l, C, H, W]
        global_crops = data['global_crops'].to(self.device, non_blocking=True) 
        local_crops = data['local_crops'].to(self.device, non_blocking=True)    
        B = global_crops.shape[0]
        V_g = global_crops.shape[1]
        V_l = local_crops.shape[1]
        # --- STEP 1: Forward Pass (Processing all views through backbone + projector) ---
        
        # Global Views: Flatten (B, V_g) into a single batch dimension for the model
        # (256 * 2, 3, 224, 224) -> (512, 3, 224, 224)
        flat_global = global_crops.flatten(0, 1) 
        global_out, global_feat = self.lejepa_model(
            flat_global,
            interpolate_pos_encoding=self.apply_interpolate  
        )
        # Reshape back to separate Batch and View dimensions
        # global_out: [B*V_g, D] -> [B, V_g, D]
        global_out = global_out.view(B, V_g, -1)
        # global_feat: [B*V_g, Tokens, D] -> [B, V_g, Tokens, D]
        global_feat = global_feat.view(B, V_g, global_feat.shape[1], global_feat.shape[2])

        flat_local = local_crops.flatten(0, 1)
        local_out, local_feat = self.lejepa_model(
            flat_local, interpolate_pos_encoding=self.apply_interpolate
        )

        local_out = local_out.view(B, V_l, -1)
        local_feat = local_feat.view(B, V_l, local_feat.shape[1], local_feat.shape[2])
         
        # --- STEP 2: Preparation for Loss Calculation ---
        
        # Concatenate Global and Local embeddings along the View dimension (dim=1)
        # (256, 2, 128) + (256, 6, 128) -> (256, 8, 128) | Shape: [B, V_total, D]
        all_views = torch.cat([global_out, local_out], dim=1)
        
        # Transpose to View-First format. This is CRITICAL for SIGReg's internal mean(-3) operation.
        # (256, 8, 128) -> (8, 256, 128) | Shape: [V_total, B, D]
        proj = all_views.transpose(0, 1) 
        
        # --- STEP 3: Loss Calculation (Fully Vectorized) ---
        
        # A) Invariance Loss (Feature Clustering)
        # Target: The 'Consensus' (mean) of the 2 Global views for each image in the batch
        # global_out.mean(dim=1) -> (256, 128) | Shape: [B, D]
        centers = global_out.mean(dim=1)
        if self.invariance_detach_centers:
            # Stop-gradient on the centroid: without this, collapse drives inv_loss -> 0, so
            # invariance actively rewards collapse. Detaching makes the centroid a fixed target
            # (SimSiam-style) and removes the collapse driver.
            centers = centers.detach()

        # Calculate MSE between the Target (centers) and all 8 views.
        # centers.unsqueeze(0) -> (1, 256, 128) broadcasts against proj (8, 256, 128)
        inv_loss = (centers.unsqueeze(0) - proj).square().mean()

        # B) Regularizer Loss (Collapse Prevention)
        # Default: regularize the projector output `proj` (V_total, B, D_proj) — same space as
        # the invariance loss. 'backbone_cls' option: regularize the backbone [CLS] (the saved/
        # evaluated representation) directly, bypassing the projector's LayerNorms which mask
        # collapse from VISReg's bounded L_scale term. SIGReg is unbounded on collapse so it did
        # not need this; VISReg's L_scale is capped at 1.0 and L_shape is scale-invariant, so
        # regularizing the backbone [CLS] directly is what guarantees the saved rep stays alive.
        if self.visreg_apply_to == 'backbone_cls':
            global_cls = global_feat[:, :, 0, :]   # (B, V_g, D_bb)
            local_cls = local_feat[:, :, 0, :]     # (B, V_l, D_bb)
            reg_input = torch.cat([global_cls, local_cls], dim=1).transpose(0, 1)  # (V_total, B, D_bb)
        else:
            reg_input = proj
        reg_loss = self.regularizer(x=reg_input, global_step=iteration)
        
        # C) Total LeJEPA Loss: Weighted sum of Invariance and Regularization
        lejepa_loss = self.lejepa_lambda * reg_loss + (1 - self.lejepa_lambda) * inv_loss

        total_loss = lejepa_loss
        # --- Soft Label Auxiliary Loss ---
        if self.aux_head is not None and 'global_ts_crop_ratios' in data:
            flat_global_bb = global_feat.flatten(0, 1)
            flat_local_bb = local_feat.flatten(0, 1)

            # Replicate the is_rex flag for all global and local views
            is_rex = data['is_rex_shard']  # Shape: [B]
            is_rex_global = is_rex.repeat_interleave(V_g, dim=0)
            is_rex_local = is_rex.repeat_interleave(V_l, dim=0)

            # Package Global Labels (sending to device)
            global_labels = {
                'ts_crop': data['global_ts_crop_ratios'].flatten(0, 1).to(self.device, non_blocking=True),
                'ts_patch': data['global_ts_patch_ratios'].flatten(0, 1).to(self.device, non_blocking=True),
                'rex_crop': data['global_rex_crop_ratios'].flatten(0, 1).to(self.device, non_blocking=True),
                'rex_patch': data['global_rex_patch_ratios'].flatten(0, 1).to(self.device, non_blocking=True)
            }

            # Package Local Labels (sending to device)
            local_labels = {
                'ts_crop': data['local_ts_crop_ratios'].flatten(0, 1).to(self.device, non_blocking=True),
                'ts_patch': data['local_ts_patch_ratios'].flatten(0, 1).to(self.device, non_blocking=True),
                'rex_crop': data['local_rex_crop_ratios'].flatten(0, 1).to(self.device, non_blocking=True),
                'rex_patch': data['local_rex_patch_ratios'].flatten(0, 1).to(self.device, non_blocking=True)
            }

            # Compute loss for Global and Local views separately
            aux_loss_global, aux_stats_global = self.aux_head(flat_global_bb, global_labels, is_rex_global)
            aux_loss_local, aux_stats_local = self.aux_head(flat_local_bb, local_labels, is_rex_local)

            # Average the resulting losses
            aux_loss = (aux_loss_global + aux_loss_local) / 2.0
            total_loss = lejepa_loss + aux_loss

            # Average the stats for clean WandB logging
            loss_dict['aux_loss'] = torch.tensor(aux_loss.item(), device=self.device)
            for k in aux_stats_global.keys():
                if k != 'aux_loss':
                    avg_stat = (aux_stats_global[k] + aux_stats_local[k]) / 2.0
                    loss_dict[k] = torch.tensor(avg_stat, device=self.device)

        # --- Logging and Output ---
        loss_dict['inv_loss'] = inv_loss
        loss_dict['sigreg_loss'] = reg_loss
        loss_dict['lejepa_loss'] = lejepa_loss
        loss_dict['total_loss'] = total_loss

        # Extract CLS token from the first global view for the probe: [B, D]
        loss_dict['probe_features'] = global_feat[:, 0, 0, :]

        return loss_dict

    def _format_log_string(self, loss_dict, iteration):
        """Format logging string for console output."""

        if torch.cuda.is_available():
            device = self.accelerator.device
            total_mem = torch.cuda.get_device_properties(device.index).total_memory / (1024 ** 3)
            if device.type == 'cuda':
                mem = torch.cuda.memory_reserved(device.index) / (1024 ** 3)
            else:
                mem = 0
        else:
            mem = 0
            total_mem = 0

        lr = self.optimizer.param_groups[0]["lr"]

        # 1. Start with the standard LeJEPA baseline metrics
        log_parts = [
            f"Total: {loss_dict['total_loss'].item():.4f}",
            f"Inv: {loss_dict['inv_loss'].item():.4f}",
            f"Reg: {loss_dict['sigreg_loss'].item():.4f}"
        ]

        # 2. Define standard keys to filter out from the dynamic loop
        standard_keys = {'total_loss', 'inv_loss', 'sigreg_loss', 'lejepa_loss', 'probe_features'}

        # 3. Dynamically append any auxiliary or multi-task metrics
        for key, value in loss_dict.items():
            if key not in standard_keys:
                # Check if it's a tensor before calling .item()
                val = value.item() if isinstance(value, torch.Tensor) else value
                log_parts.append(f"{key}: {val:.4f}")

        # 4. Append the training state metrics
        log_parts.extend([
            f"LR: {lr:.6f}",
            f"λ: {self.lejepa_lambda:.3f}",
            f"Iter: {int(iteration):5d}/{int(self.config['train']['max_iterations']):5d}",
            f"Mem: {mem:.2f}/{total_mem:.2f}GB"
        ])

        # Join all parts with a tab separator for a clean console output
        return "\t".join(log_parts)
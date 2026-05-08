import gc
import torch   
from data.collate import lejepa_collate_fn
from data.guided_data_augmentation_CT_RATE import BatchedGuidedDataAugmentationDINO_CT
from lejepa_core.lejepa_ssl_arch import SSLMetaArch
from utils.config import load_model_configs   
from utils.dino_utils import (
    generate_samples, 
    get_periodic_backbone_checkpointer_ddp, 
    get_periodic_backbone_checkpointer_fsdp, 
    get_periodic_train_checkpointer, 
    print_model_info, 
    write_to_main_log
)
  
from accelerate.state import DistributedType
from utils.dist_utils import get_transformer_layers_to_wrap
from utils.logger_utils import write_to_node_logs 


class Trainer:
    def __init__(
        self,
        config,
        accelerator, 
        dataset 
    ) -> None:
        self.config = config 
        self.accelerator = accelerator  
        self.model_name = self.config.train.model_name 
        self.dataset = dataset   
        self.train_loader = dataset.train_loader

        self.model_params = load_model_configs(self.config.train.model_type)
        self.dist_type = accelerator.state.distributed_type
        self.gc_freq = 50
        self.num_register_tokens = getattr(self.model_params, 'num_register_tokens', 0)
         
        # Initialize model based on distribution type
        if self.dist_type == DistributedType.FSDP: 
            if self.accelerator.is_main_process:
                write_to_main_log(accelerator=self.accelerator, result='FSDP LeJEPA Training activated')
            self.ssl_model = SSLMetaArch(
                config=config,
                accelerator=accelerator 
            ).cuda()  
            self.verify_fsdp_wrapping()
        else:    
            if self.accelerator.is_main_process:
                write_to_main_log(accelerator=self.accelerator, result='DDP LeJEPA Training activated')
            self.ssl_model = SSLMetaArch(
                config=config,
                accelerator=accelerator 
            ).cuda() 
 
        print_model_info(accelerator=self.accelerator, model=self.ssl_model.lejepa_model)
        self.gpu_aug = getattr(self.config.dataset, 'gpu_augmentations', False)
        if self.gpu_aug:
            if self.accelerator.is_main_process:
                write_to_main_log(accelerator=self.accelerator, result='GPU Data Augmentations Enabled.')
            self.gpu_augmentor = BatchedGuidedDataAugmentationDINO_CT(self.config).to(self.accelerator.device)

    def verify_fsdp_wrapping(self, verbose=False):
        """Verify FSDP wrapping."""
        if self.dist_type != DistributedType.FSDP:
            return

        write_to_main_log(accelerator=self.accelerator, result="--- FSDP Wrapping Verification ---")
        
        try: 
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except ImportError: 
            write_to_main_log(accelerator=self.accelerator, result="FSDP not available")
            return

        target_classes = get_transformer_layers_to_wrap(
            model_type=self.model_params.model_type, 
            use_lora=self.config.train.use_lora
        )
        target_class_names = {cls.__name__ for cls in target_classes}

        model = self.ssl_model.module if hasattr(self.ssl_model, 'module') else self.ssl_model
        
        wrapped_count = 0
        for name, mod in model.named_modules():
            if isinstance(mod, FSDP):
                orig_mod = getattr(mod, 'module', getattr(mod, '_fsdp_wrapped_module', None))
                if orig_mod and type(orig_mod) in target_classes:
                    wrapped_count += 1

        status = "SUCCESS" if wrapped_count > 0 else "WARNING"
        write_to_main_log(
            accelerator=self.accelerator, 
            result=f"Targets ({','.join(target_class_names)}): {status}. {wrapped_count} wrapped."
        )
        write_to_main_log(accelerator=self.accelerator, result="--- End FSDP Verification ---")

    def save_checkpoint(self, iteration, sample_imgs=None):  
        """Save training checkpoint."""
        self.checkpoint_manager.save(iteration=iteration) 
        self.backbone_checkpoint_manager.save(iteration=iteration) 
        
        # Generate sample visualizations
        if self.dist_type == DistributedType.MULTI_GPU and self.accelerator.is_main_process:  
            if sample_imgs is not None and not hasattr(self.model_params, 'timm_arch'):
                unwrapped_lejepa = self.accelerator.unwrap_model(self.ssl_model.lejepa_model)
                generate_samples(
                    dataset=self.dataset,
                    accelerator=self.accelerator, 
                    config=self.config, 
                    model_params=self.model_params,  
                    num_register_tokens=self.num_register_tokens, 
                    imgs=sample_imgs, 
                    masks=None,
                    iteration=iteration, 
                    backbone_model=unwrapped_lejepa.backbone
                )
                     
    def train(self): 
        max_iter = self.config.train.max_iterations 

        # Setup checkpointers
        self.start_iteration, self.checkpoint_manager = get_periodic_train_checkpointer(
            config=self.config, 
            model=self.ssl_model, 
            optimizer=self.ssl_model.optimizer,
            accelerator=self.accelerator
        )
        
        if self.dist_type == DistributedType.MULTI_GPU:
            self.backbone_checkpoint_manager = get_periodic_backbone_checkpointer_ddp(
                config=self.config, 
                model=self.ssl_model,  
                accelerator=self.accelerator
            )
        elif self.dist_type == DistributedType.FSDP:
            self.backbone_checkpoint_manager = get_periodic_backbone_checkpointer_fsdp(
                config=self.config, 
                model=self.ssl_model,  
                accelerator=self.accelerator
            )
            
        # Log start
        if self.start_iteration > 0:
            write_to_main_log(
                accelerator=self.accelerator, 
                result=f"Resuming {self.model_name} from iteration {self.start_iteration}"
            ) 
        else:
            write_to_main_log(
                accelerator=self.accelerator, 
                result=f"Starting training: {self.model_name}"
            )
        
        # Prepare data
        if self.gpu_aug:
            train_data = self.train_loader
        else:
            train_data = self.accelerator.prepare(self.train_loader)
        iteration = self.start_iteration  
        do_initialize = iteration > 0
        
        # Training loop
        for data in train_data:
            masks = None
            labels = None

            if self.gpu_aug and isinstance(data, dict) and 'volumes_list' in data:
                # --- GPU Augmentation Path ---
                # Device placement is now natively handled inside the augmentor
                # to support varying nested lists of reconstructions cleanly.
                crops = self.gpu_augmentor(data['volumes_list'], data['ts_masks_list'], data['rex_masks_list'])

                # Stack the image crops
                crops['global_crops'] = torch.stack(crops['global_crops'], dim=1)
                crops['local_crops'] = torch.stack(crops['local_crops'], dim=1)

                # Dynamically stack all ratio labels generated by the augmentor
                for k in crops.keys():
                    if 'ratios' in k and isinstance(crops[k], list):
                        crops[k] = torch.stack(crops[k], dim=1)

                if 'dataset_labels' in data: crops['labels'] = data['dataset_labels']
                if 'is_rex_shard' in data: crops['is_rex_shard'] = data['is_rex_shard'].to(self.accelerator.device,
                                                                                           non_blocking=True)
            else:
                # --- Existing Safe Data Unpacking (CPU Path) ---
                if isinstance(data, dict):
                    crops_raw = data
                elif isinstance(data, (list, tuple)):
                    crops_raw = data[0]
                    if len(data) == 2:
                        labels = data[1]
                    elif len(data) > 2:
                        masks = data[1]; labels = data[2]
                else:
                    crops_raw = data

                crops = lejepa_collate_fn(crops_raw)
                if labels is not None: crops['labels'] = labels
                if masks is not None: crops['masks'] = masks

            if iteration >= max_iter or self.ssl_model.early_stop_triggered:
                break
            log_str, probe_features = self.ssl_model(iteration, crops, initialize=do_initialize)
            
            if do_initialize: 
                write_to_node_logs(accelerator=self.accelerator, result='INITIALIZATION COMPLETE') 
                do_initialize = False 
                
            write_to_node_logs(accelerator=self.accelerator, result=log_str, check_main_process=True)
             
            iteration += 1 
            
            # Garbage collection
            if iteration % self.gc_freq == 0: 
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Checkpointing
            should_save = False
            if self.ssl_model.scheduler.save_best and self.ssl_model.is_best_checkpoint:
                should_save = True
            elif self.config.train.saveckp_freq > 0 and iteration % self.config.train.saveckp_freq == 0:
                should_save = True
                
            if should_save and iteration > 0:
                # Get first view of first few images for visualization 
                self.save_checkpoint(iteration=iteration, sample_imgs=crops['global_crops'].flatten(0, 1)  )
                    
            self.accelerator.wait_for_everyone()
        
        # Final checkpoint 
        self.save_checkpoint(iteration=iteration, sample_imgs=crops['global_crops'].flatten(0, 1) )

        # Plot training health if using WeightWatcher
        if self.accelerator.is_main_process and getattr(self.ssl_model.scheduler, 'use_ww', False):
            self.ssl_model.scheduler.plot_training_health()
            
        self.accelerator.wait_for_everyone()
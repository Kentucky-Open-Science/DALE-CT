import os
import re
import shutil 
import json
from typing import Optional,  Any
import torch
import torch.distributed as dist

import gc  
from utils.logger_utils import write_to_main_log
from accelerate import Accelerator
from accelerate.utils import DistributedType # Needed for checking init type

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from safetensors.torch import save_file  

class FSDPModelBackboneCheckpointManager:
    """
    Manages saving only the model's backbone state dictionary
    for DDP or non-distributed training using safetensors.
    Handles LoRA adapters if configured and cleans old checkpoints.
    Loading is NOT supported by this manager.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 config: Any, 
                 accelerator: Accelerator,
                 save_dir: Optional[str] = None,  
                 max_to_keep: Optional[int] = 3,
                 ):
        
        if accelerator.state.distributed_type != DistributedType.NO and (not dist.is_available() or not dist.is_initialized()):
             raise RuntimeError("Distributed environment is not initialized properly by Accelerator.")

        self.model = model
        self.config = config
        self.accelerator = accelerator
        self.is_main_process = accelerator.is_main_process
 
        self.save_dir = save_dir

        self.max_to_keep = max_to_keep if max_to_keep is not None and max_to_keep > 0 else None

        # Ensure save directory exists on main process
        if self.is_main_process: # No need to check os.path.exists first before makedirs with exist_ok=True
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Ensuring backbone checkpoint directory exists: {self.save_dir}")
            os.makedirs(self.save_dir, exist_ok=True)

        # Synchronize after directory creation
        self.accelerator.wait_for_everyone()
        write_to_main_log(accelerator=self.accelerator, result=f"[{self.accelerator.process_index}] ModelBackboneCheckpointManager initialized. Save directory: {self.save_dir}")

    def save(self, iteration): 
        try:         
            current_ckpt_dir = os.path.join(self.save_dir, f"iter_{iteration}")
            os.makedirs(current_ckpt_dir, exist_ok=True)
            self.accelerator.wait_for_everyone()
            
            # --- Determine target model ---
            if hasattr(self.model, "student_shadow"): 
                target_model = self.model.student_shadow
            elif hasattr(self.model, "lejepa_model"):
                if hasattr(self.model, "teacher_model") and self.model.teacher_model is not None:
                    target_model = self.model.teacher_model
                else:
                    target_model = self.model.lejepa_model
            elif hasattr(self.model, "teacher_model"):
                target_model = self.model.teacher_model
            else: 
                target_model = self.model
                write_to_main_log(self.accelerator, result="Backbone model couldn't be found!", type='warning')
            
            # --- Get state dict ---
            is_fsdp_wrapped = isinstance(target_model, FSDP)
            if is_fsdp_wrapped: 
                full_state_dict = self.accelerator.get_state_dict(target_model)
            else: 
                full_state_dict = target_model.state_dict()

            # --- Save on main process ---
            if self.accelerator.is_main_process and full_state_dict:
                if self.config.train.use_lora: 
                    # LoRA save logic (existing code)
                    lora_state_dict = {}
                    for key, value in full_state_dict.items():
                        if any(lora_key in key for lora_key in ["lora_A", "lora_B", "lora_E", "lora_dropout"]):
                            clean_key = key.replace("._fsdp_wrapped_module", "").replace("backbone.", "").replace(".default", "")
                            lora_state_dict[clean_key] = value.cpu()
                    
                    if lora_state_dict:
                        serializable_config = self.serialize_config_parameters(self.model.model_config).get('default', {})
                        with open(os.path.join(current_ckpt_dir, "adapter_config.json"), "w") as f:
                            json.dump(serializable_config, f, indent=2)
                        save_file(lora_state_dict, os.path.join(current_ckpt_dir, "adapter_model.safetensors"))
                else: 
                    gathered_backbone_state_dict = self.remove_prefix(full_state_dict)
                    if gathered_backbone_state_dict:
                        save_file(gathered_backbone_state_dict, os.path.join(current_ckpt_dir, "model.safetensors"))
                        serializable_config = self.serialize_config_parameters(self.model.model_config)
                        with open(os.path.join(current_ckpt_dir, "config.json"), "w") as f:
                            json.dump(serializable_config, f, indent=2)

                if self.max_to_keep is not None:
                    self._clean_checkpoints()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.accelerator.wait_for_everyone()
            
        except Exception as e:
            write_to_main_log(self.accelerator, result=f"Error in save: {str(e)}", type='error')
            self.accelerator.wait_for_everyone()
    def _extract_iteration_from_dirname(self, dirname):
        """
        Extract iteration number from directory name like 'iter_1250'
        Returns -1 if the format doesn't match expected pattern
        """
        match = re.match(r'iter_(\d+)', dirname)
        if match:
            return int(match.group(1))
        return -1


    def _clean_checkpoints(self): 
        if not self.is_main_process or self.max_to_keep is None:
            return

        try:
            base_dir = self.save_dir
            ckpt_dirs = []
            
            # Nothing to clean if dir doesn't exist
            if not os.path.isdir(base_dir):
                return 

            # List directories and extract iteration numbers from directory names
            for dname in os.listdir(base_dir):
                full_path = os.path.join(base_dir, dname)
                
                # Check if it's a directory and follows our naming pattern
                if os.path.isdir(full_path) and dname.startswith("iter_"):
                    # Extract iteration number from directory name
                    iter_num = self._extract_iteration_from_dirname(dname)
                    
                    if iter_num != -1:
                        # Check if directory contains checkpoint files to ensure it's valid
                        if any(fname.endswith(".safetensors") for fname in os.listdir(full_path)):
                            ckpt_dirs.append((iter_num, full_path))
                        else:
                            # Directory appears to be empty or incomplete
                            write_to_main_log(
                                accelerator=self.accelerator, 
                                result=f"[Rank 0] Found possibly incomplete checkpoint dir: {full_path}", 
                                type='warning'
                            )
                    
            if len(ckpt_dirs) <= self.max_to_keep:
                return

            # Sort by iteration number
            ckpt_dirs.sort(key=lambda x: x[0])

            # Identify checkpoints to delete (all except the last max_to_keep)
            checkpoints_to_delete = ckpt_dirs[:-self.max_to_keep]

            for iter_num, dir_path in checkpoints_to_delete:
                try:
                    shutil.rmtree(dir_path)
                    write_to_main_log(
                        accelerator=self.accelerator, 
                        result=f"[Rank 0] Deleted old checkpoint: {dir_path} (iteration {iter_num})"
                    )
                except OSError as e:
                    write_to_main_log(
                        accelerator=self.accelerator, 
                        result=f"[Rank 0] Error deleting {dir_path}: {e}", 
                        type='error'
                    )

        except Exception as e:
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Error during backbone checkpoint cleaning process: {e}", type='error')
            import traceback
            write_to_main_log(accelerator=self.accelerator, result=traceback.format_exc(), type='error')


    def serialize_config_parameters(self,peft_config): 
        def _convert_to_serializable(value): 
            if value is None:
                return None
            elif isinstance(value, (int, float, bool, str)):
                return value
            elif isinstance(value, (list, tuple)):
                return [_convert_to_serializable(item) for item in value]
            elif isinstance(value, set):
                return [_convert_to_serializable(item) for item in value]
            elif isinstance(value, dict):
                return {k: _convert_to_serializable(v) for k, v in value.items()}
            elif hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
                return _convert_to_serializable(value.to_dict())
            elif hasattr(value, "__dict__"): 
                return _convert_to_serializable(vars(value))
            else:
                # Convert to string as a fallback
                return str(value)
        
        # Create a deep copy of the original config to avoid modifying it
        result = {}
        
        if peft_config is None:
            return result
        
        # Handle dictionary or object
        if isinstance(peft_config, dict):
            for key, value in peft_config.items():
                result[key] = _convert_to_serializable(value)
        elif hasattr(peft_config, "to_dict") and callable(getattr(peft_config, "to_dict")):
            result = _convert_to_serializable(peft_config.to_dict())
        elif hasattr(peft_config, "__dict__"):
            result = _convert_to_serializable(vars(peft_config))
        else: 
            result = {"config": str(peft_config)}
        
        return result

    def remove_prefix(self, model_state_dict):
        if not model_state_dict: 
            return None

        prefix = 'backbone.'
        new_state_dict = {}
        
        for k, v in model_state_dict.items():
            if not k.startswith(prefix):
                continue
            
            clean_key = k[len(prefix):]
             
            if 'patch_embeddings.mask_token' in clean_key:
                continue
             
            clean_key = clean_key.replace(
                "patch_embeddings.patch_embed.", 
                "patch_embeddings."
            )
            
            new_state_dict[clean_key] = v

        return new_state_dict or None
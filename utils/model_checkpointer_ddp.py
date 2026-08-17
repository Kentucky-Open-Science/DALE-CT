import os
import shutil 
from typing import Optional,  Any
import torch
import torch.distributed as dist
import re

import json
from utils.logger_utils import write_to_main_log
from accelerate import Accelerator
from accelerate.utils import DistributedType  

from safetensors.torch import save_file 

class DDPModelBackboneCheckpointManager:
    """
    Manages saving only the model's backbone state dictionary
    for DDP or non-distributed training using safetensors.
    Handles LoRA adapters if configured and cleans old checkpoints.
    Loading is NOT supported by this manager.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 config: Any, # Configuration object
                 accelerator: Accelerator,
                 save_dir: Optional[str] = None, # Can override default save_dir from config
                 max_to_keep: Optional[int] = 3,
                 ):
        # Check only if Accelerator reports distributed is initialized.
        if accelerator.state.distributed_type != DistributedType.NO and (not dist.is_available() or not dist.is_initialized()):
             raise RuntimeError("Distributed environment is not initialized properly by Accelerator.")

        self.model = model
        self.config = config
        self.accelerator = accelerator
        self.is_main_process = accelerator.is_main_process


        self.save_dir=save_dir
        self.max_to_keep = max_to_keep if max_to_keep is not None and max_to_keep > 0 else None

        # Ensure save directory exists on main process
        if self.is_main_process: # No need to check os.path.exists first before makedirs with exist_ok=True
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Ensuring backbone checkpoint directory exists: {self.save_dir}")
            os.makedirs(self.save_dir, exist_ok=True)

        # Synchronize after directory creation
        self.accelerator.wait_for_everyone()
        write_to_main_log(accelerator=self.accelerator, result=f"[{self.accelerator.process_index}] ModelBackboneCheckpointManager initialized. Save directory: {self.save_dir}")


    def save(self, iteration: int):
        """
        Saves the model's backbone state dictionary for the given iteration.
        Only performed by the main process. Handles LoRA or standard saving.
        Automatically triggers cleaning of old checkpoints after a successful save.
        """
        if not self.is_main_process: 
            self.accelerator.wait_for_everyone()
            return

        current_ckpt_dir = os.path.join(self.save_dir, f"iter_{iteration}")
        
        try: 

            os.makedirs(current_ckpt_dir, exist_ok=True)
            if hasattr(self.model, "student_shadow"): 
                backbone = self.accelerator.unwrap_model(self.model.student_shadow.backbone)
            elif hasattr(self.model, "lejepa_model"):  
                unwrapped_lejepa = self.accelerator.unwrap_model(self.model.lejepa_model)
                backbone = unwrapped_lejepa.backbone  
            elif hasattr(self.model, "teacher_model"):
                backbone = self.accelerator.unwrap_model(self.model.teacher_model.backbone)
            else: 
                backbone = self.accelerator.unwrap_model(self.model.backbone) 
                write_to_main_log(accelerator=self.accelerator, result=f"Backbone model couldn't be find! Please check model DDP model checkpointer!", type='warning')


            if self.config.train.use_lora: 
                # LoRA keylerini filtrele ve temizle
                lora_state_dict = {}
                for k, v in backbone.state_dict().items():
                    if any(x in k for x in ["lora_A", "lora_B", "lora_E", "lora_dropout"]):
                        clean_key = k.replace("backbone.", "").replace(".default", "")
                        # MaskedPatchEmbedding wrapper temizliği
                        clean_key = clean_key.replace("patch_embeddings.patch_embed.", "patch_embeddings.")
                        lora_state_dict[clean_key] = v.cpu()
                
                if lora_state_dict:
                    # adapter_config.json kaydet
                    if hasattr(self.model, 'model_config'):
                        serializable_config = self.serialize_config_parameters(
                            self.model.model_config
                        ).get('default', {})
                        with open(os.path.join(current_ckpt_dir, "adapter_config.json"), "w") as f:
                            json.dump(serializable_config, f, indent=2)
                    
                    save_file(lora_state_dict, os.path.join(current_ckpt_dir, "adapter_model.safetensors"))
                    write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] LoRA adapter saved to {current_ckpt_dir}")
                else:
                    write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Warning: No LoRA keys found in state dict.", type='warning')
            else: 
                state_dict = backbone.state_dict()
                clean_state_dict = self._clean_backbone_state_dict(state_dict)
                safetensors_path = os.path.join(current_ckpt_dir, "model.safetensors")
                save_file(clean_state_dict, safetensors_path) 
                if hasattr(backbone, 'config'):
                    backbone.config.save_pretrained(current_ckpt_dir)
                write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Backbone state dict saved to {safetensors_path}")
 

            write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Backbone checkpoint for iteration {iteration} saved successfully.")

            # Clean old checkpoints *only if max_to_keep is configured*
            if self.max_to_keep is not None:
                self._clean_checkpoints()

        except Exception as e:
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Failed to save backbone checkpoint for iteration {iteration}: {e}", type='error')
            import traceback
            write_to_main_log(accelerator=self.accelerator, result=traceback.format_exc(), type='error')

            # Clean up potentially incomplete directory if save failed
            if os.path.exists(current_ckpt_dir):
                write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Attempting to remove possibly incomplete checkpoint directory: {current_ckpt_dir}", type='warning')
                try:
                    shutil.rmtree(current_ckpt_dir)
                except Exception as clean_e:
                    write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Error removing incomplete checkpoint directory: {clean_e}", type='error')

        finally:
            # Synchronize all ranks after the save attempt (successful or failed)
            self.accelerator.wait_for_everyone()


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
        """
        Cleans old checkpoint directories, keeping only the latest `max_to_keep`.
        Only runs on the main process. Requires max_to_keep is configured.
        Uses directory names instead of metadata files.
        """
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

    
    def remove_prefix(self, model_state_dict):
        if not model_state_dict: 
            return None

        prefix = 'backbone.'
        gathered_backbone_state_dict = {
            k[len(prefix):]: v for k, v in model_state_dict.items() if k.startswith(prefix)
                }

        if not gathered_backbone_state_dict:
            del model_state_dict
            return None
        return gathered_backbone_state_dict

    def _clean_backbone_state_dict(self, state_dict): 
        if not state_dict:
            return None

        clean = {}
        for k, v in state_dict.items():
            if 'patch_embeddings.mask_token' in k:
                continue
            k = k.replace("patch_embeddings.patch_embed.", "patch_embeddings.")
            clean[k] = v

        return clean or None
import os
import shutil
import logging
import re
from typing import Optional, Tuple
import torch
import torch.distributed as dist
from accelerate.state import DistributedType

from utils.logger_utils import write_to_main_log


class DistributedCheckpointManager: 
    def __init__(self,
                 model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer,  
                 save_dir: str,
                 accelerator, 
                 max_to_keep: int = 3,
                 scheduler = None  
                 ): 
        if not dist.is_available() or not dist.is_initialized(): 
            raise RuntimeError("Distributed environment is not initialized properly.")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.accelerator = accelerator 
        self.max_to_keep = max_to_keep if max_to_keep is not None and max_to_keep > 0 else None
        self.is_main_process = accelerator.is_main_process
        self.rank = accelerator.process_index
        self.world_size = accelerator.num_processes
        
        # Determine if we're using FSDP
        self.is_fsdp = accelerator.state.distributed_type == DistributedType.FSDP
        
        if self.is_fsdp:
            # For FSDP: Each GPU has its own checkpoint subfolder
            self.rank_save_dir = os.path.join(self.save_dir, f"rank_{self.rank}")
            write_to_main_log(accelerator=accelerator, result=f"[Rank {self.rank}] FSDP mode: Using rank-specific checkpoint directory: {self.rank_save_dir}")
        else:
            # For DDP: All GPUs use the same directory (only rank 0 writes)
            self.rank_save_dir = self.save_dir
            write_to_main_log(accelerator=accelerator, result=f"[Rank {self.rank}] DDP mode: Using shared checkpoint directory: {self.rank_save_dir}")
        
        # Create directory (only rank 0 creates in DDP mode)
        if self.is_fsdp or self.is_main_process:
            os.makedirs(self.rank_save_dir, exist_ok=True)
        
        # Wait for all processes to reach this point
        self.accelerator.wait_for_everyone()
        write_to_main_log(accelerator=accelerator, result=f"[Rank {self.rank}] DistributedCheckpointManager initialized.")

    def save(self, iteration: int):
        """
        Save checkpoint based on training mode:
        - FSDP: Each GPU saves its own checkpoint
        - DDP: Only rank 0 saves the checkpoint
        """
        # For DDP, only the main process saves
        if not self.is_fsdp and not self.is_main_process:
            # Non-main processes in DDP just wait
            self.accelerator.wait_for_everyone()
            return
        
        current_ckpt_dir = os.path.join(self.rank_save_dir, f"iter_{iteration}")
        
        write_to_main_log(accelerator=self.accelerator, result=f"[Rank {self.rank}] Saving checkpoint for iteration {iteration} to {current_ckpt_dir}...")

        try: 
            # Create directory for this checkpoint
            os.makedirs(current_ckpt_dir, exist_ok=True)
            
            # Unwrap model from accelerator
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            # Save model state dict
            model_path = os.path.join(current_ckpt_dir, "model.pt")
            torch.save(unwrapped_model.state_dict(), model_path)
             
            optimizer_path = os.path.join(current_ckpt_dir, "optimizer.pt")
            torch.save(self.optimizer.state_dict(), optimizer_path)
            
            # Save scheduler if available
            if self.scheduler:
                scheduler_path = os.path.join(current_ckpt_dir, "scheduler.pt")
                torch.save(self.scheduler.state_dict(), scheduler_path)
            
            # Save iteration info
            app_state = {"iteration": iteration}
            app_state_path = os.path.join(current_ckpt_dir, "app_state.pt")
            torch.save(app_state, app_state_path)
            
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank {self.rank}] Checkpoint for iteration {iteration} saved successfully.")
            
            # Clean old checkpoints
            self._clean_checkpoints()
            
        except Exception as e:
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank {self.rank}] Failed to save checkpoint for iteration {iteration}: {e}", type='error') 
            if os.path.exists(current_ckpt_dir):
                write_to_main_log(accelerator=self.accelerator, result=f"[Rank {self.rank}] Removing incomplete checkpoint directory: {current_ckpt_dir}", type='warning')
                try:
                    shutil.rmtree(current_ckpt_dir)
                except Exception as clean_e:
                    write_to_main_log(accelerator=self.accelerator, result=f"[Rank {self.rank}] Error removing incomplete checkpoint directory: {clean_e}", type='error')

        finally: 
            # Wait for all processes to complete their save operations
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

    def _is_valid_checkpoint_dir(self, dir_path):
        """
        Check if a directory contains a valid checkpoint
        Returns True if the directory has necessary files
        """
        # Check for required files
        model_file = os.path.join(dir_path, "model.pt")
        optimizer_file = os.path.join(dir_path, "optimizer.pt")
        app_state_file = os.path.join(dir_path, "app_state.pt")
        
        # Basic files must exist
        basic_files_exist = (
            os.path.exists(model_file) and 
            os.path.exists(optimizer_file) and
            os.path.exists(app_state_file)
        )
        
        return basic_files_exist

    def _find_latest_checkpoint_dir(self) -> Tuple[Optional[str], int]:
        """
        Find the latest checkpoint directory
        For FSDP: Find in this rank's directory
        For DDP: Find in the main directory (all processes look, but typically only rank 0 will find)
        """
        latest_iter = -1
        latest_ckpt_dir = None

        if not os.path.isdir(self.rank_save_dir):
            return None, -1

        try:
            for dname in os.listdir(self.rank_save_dir):
                full_path = os.path.join(self.rank_save_dir, dname)
                
                # Check if it's a directory and follows our naming pattern
                if os.path.isdir(full_path) and dname.startswith("iter_"):
                    # Extract iteration number directly from directory name
                    iter_num = self._extract_iteration_from_dirname(dname)
                    
                    # Check if directory contains valid checkpoint files
                    if iter_num != -1 and self._is_valid_checkpoint_dir(full_path):
                        if iter_num > latest_iter:
                            latest_iter = iter_num
                            latest_ckpt_dir = full_path
                    elif iter_num != -1:
                        # Directory might be incomplete
                        write_to_main_log(
                            accelerator=self.accelerator, 
                            result=f"[Rank {self.rank}] Found potential checkpoint dir {dname} but missing required files. Skipping.", 
                            type='warning'
                        )
        except Exception as e:
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank {self.rank}] Error finding latest checkpoint directory: {e}", type='error')
            return None, -1

        return latest_ckpt_dir, latest_iter

    def load_latest(self) -> int: 
        """
        Load the latest checkpoint:
        - FSDP: Each process loads from its own checkpoint
        - DDP: All processes load from the main checkpoint directory
        Returns the loaded iteration number, or -1 if no checkpoint was found
        """
        # Each process finds the latest checkpoint in its directory
        load_dir, loaded_iteration = self._find_latest_checkpoint_dir()
        
        # In both cases, we need all processes to agree on the iteration
        iter_tensor = torch.tensor([loaded_iteration], dtype=torch.long, device=self.accelerator.device)
        dist.all_reduce(iter_tensor, op=dist.ReduceOp.MAX)  # Use max to find highest iteration across all ranks
        loaded_iteration = iter_tensor.item()
        
        # If we found a checkpoint
        if loaded_iteration >= 0:
            if not self.is_fsdp:
                # For DDP, all ranks use the main checkpoint directory
                load_dir = os.path.join(self.save_dir, f"iter_{loaded_iteration}")
            else:
                # For FSDP, each rank has its own directory
                load_dir = os.path.join(self.rank_save_dir, f"iter_{loaded_iteration}")
            
            # Check if the checkpoint exists and is valid
            if not os.path.exists(load_dir) or not self._is_valid_checkpoint_dir(load_dir):
                write_to_main_log(
                    accelerator=self.accelerator, 
                    result=f"[Rank {self.rank}] Warning: No valid checkpoint found for iteration {loaded_iteration}. This may cause state inconsistency.", 
                    type='warning'
                )
                return loaded_iteration
            
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank {self.rank}] Loading checkpoint from {load_dir}...")

            try: 
                # Load model state dict
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                model_path = os.path.join(load_dir, "model.pt")
                model_state = torch.load(model_path, weights_only=False, map_location='cpu')
                unwrapped_model.load_state_dict(model_state)
                
                # Load optimizer state dict
                optimizer_path = os.path.join(load_dir, "optimizer.pt")
                optimizer_state = torch.load(optimizer_path, weights_only=False, map_location='cpu')
                self.optimizer.load_state_dict(optimizer_state)
                
                # Load scheduler if available
                if self.scheduler:
                    scheduler_path = os.path.join(load_dir, "scheduler.pt")
                    if os.path.exists(scheduler_path):
                        scheduler_state = torch.load(scheduler_path, weights_only=False, map_location='cpu')
                        self.scheduler.load_state_dict(scheduler_state)
                
                # Verify iteration number
                app_state_path = os.path.join(load_dir, "app_state.pt")
                app_state = torch.load(app_state_path, weights_only=False, map_location='cpu')
                loaded_iteration_check = app_state.get("iteration", -1)
                
                if loaded_iteration_check != loaded_iteration:
                    write_to_main_log(
                        accelerator=self.accelerator, 
                        result=f"[Rank {self.rank}] Loaded iteration mismatch! Expected {loaded_iteration}, got {loaded_iteration_check} from app_state.", 
                        type='warning'
                    )
                else:
                    write_to_main_log(
                        accelerator=self.accelerator, 
                        result=f"[Rank {self.rank}] Checkpoint loaded successfully for iteration {loaded_iteration}."
                    )
            except Exception as e:
                write_to_main_log(
                    accelerator=self.accelerator, 
                    result=f"[Rank {self.rank}] Failed to load checkpoint from {load_dir}: {e}", 
                    type='error'
                )
                # Don't change loaded_iteration here, as we need to keep all ranks in sync
 

        self.accelerator.wait_for_everyone()
        return loaded_iteration

    def _clean_checkpoints(self): 
        """
        Clean old checkpoints:
        - FSDP: Each rank cleans its own checkpoints
        - DDP: Only rank 0 cleans checkpoints
        """
        # For DDP, only the main process cleans
        if not self.is_fsdp and not self.is_main_process:
            return
            
        if self.max_to_keep is None:
            return

        try:
            ckpt_dirs = []
            
            # List directories and extract iteration numbers
            for dname in os.listdir(self.rank_save_dir):
                full_path = os.path.join(self.rank_save_dir, dname) 
                
                # Check if it's a directory and follows our naming pattern
                if os.path.isdir(full_path) and dname.startswith("iter_"):
                    # Extract iteration number from directory name
                    iter_num = self._extract_iteration_from_dirname(dname)
                    
                    if iter_num != -1:
                        ckpt_dirs.append((iter_num, full_path))
 
            if len(ckpt_dirs) <= self.max_to_keep:
                return 
                
            # Sort by iteration number
            ckpt_dirs.sort(key=lambda x: x[0])
 
            # Identify checkpoints to delete (all except the last max_to_keep)
            checkpoints_to_delete = ckpt_dirs[:-self.max_to_keep]

            write_to_main_log(
                accelerator=self.accelerator, 
                result=f"[Rank {self.rank}] Cleaning checkpoints, keeping last {self.max_to_keep}. Deleting {len(checkpoints_to_delete)} old checkpoints."
            )
            
            for iter_num, dir_path in checkpoints_to_delete:
                try:
                    shutil.rmtree(dir_path) 
                except OSError as e:
                    write_to_main_log(
                        accelerator=self.accelerator, 
                        result=f"[Rank {self.rank}] Error deleting {dir_path}: {e}", 
                        type='error'
                    )
        except Exception as e:
            write_to_main_log(
                accelerator=self.accelerator, 
                result=f"[Rank {self.rank}] Error during checkpoint cleaning: {e}", 
                type='error'
            )
 
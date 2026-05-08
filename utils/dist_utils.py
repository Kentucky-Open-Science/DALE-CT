import os
import time

from accelerate import Accelerator

from typing import Set, Type

import sys
import socket
import functools 
import torch
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer
from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import  Dinov2WithRegistersLayer
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer 
from peft.tuners.lora import LoraLayer
from accelerate.utils import FullyShardedDataParallelPlugin
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from utils.config import load_model_configs
from utils.logger_utils import write_to_main_log
from torch import distributed as dist

def get_transformer_layers_to_wrap(model_type: str, use_lora: bool) -> Set[Type]:
    """Get the transformer layer classes to wrap for FSDP"""
    transformer_layer_classes_to_wrap = set() 
    if 'dinov3' in model_type.lower():
        transformer_layer_classes_to_wrap.add(DINOv3ViTLayer)
    elif 'dinov2' in model_type.lower() and 'register' in model_type.lower():
        transformer_layer_classes_to_wrap.add(Dinov2WithRegistersLayer)
    elif 'dinov2' in model_type.lower() and 'register' not in model_type.lower():
        transformer_layer_classes_to_wrap.add(Dinov2Layer)
    elif 'vit' in model_type.lower():  # En genel koşul en sonda
        transformer_layer_classes_to_wrap.add(ViTLayer)
    if use_lora:
        transformer_layer_classes_to_wrap.add(LoraLayer)
        
    return transformer_layer_classes_to_wrap


def initialize_ddp_accelerator_from_config(config) -> Accelerator:
    # ===== CRITICAL FIX: Manual distributed initialization =====
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    # Set device FIRST
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize distributed BEFORE Accelerator with explicit device_id
    if not dist.is_initialized():
        is_main_early = rank == 0
        if is_main_early:
            print(f"Manually initializing distributed on device cuda:{local_rank}")
            print(f"Rank {rank}/{world_size}, Local rank: {local_rank}")
        
        # Key fix: Initialize with device_id parameter
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            device_id=device  # This eliminates the warning!
        )
        
        if is_main_early:
            print("✅ Distributed initialized successfully")
    
    # Set config-based parameters
    freeze_backbone_layers = 0
    mixed_precision = 'bf16'

    if hasattr(config, 'distribution'):
        mixed_precision = config.distribution.mixed_precision
        # Set ACCELERATE_DOWNCAST_BF16 from config
        desired_downcast_bf16 = getattr(config.distribution, 'downcast_bf16', 'no').lower()
        os.environ['ACCELERATE_DOWNCAST_BF16'] = desired_downcast_bf16
    
    if hasattr(config.train, 'freeze_backbone_layers'):
        freeze_backbone_layers = config.train.freeze_backbone_layers
        find_unused_parameters = (freeze_backbone_layers <= 0)
    else: 
        find_unused_parameters = False
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters,static_graph=True)
    
    accelerator_kwargs = {
        "mixed_precision": mixed_precision,
        "kwargs_handlers": [ddp_kwargs], 
    }
    
    # Now Accelerator won't reinitialize distributed (already done)
    accelerator = Accelerator(**accelerator_kwargs)

    # Print statement happens only on main process *after* Accelerator init
    if accelerator.is_main_process:
        print("✅ Accelerator initialized for DDP.")
        print(f"Device: {accelerator.device}, Process: {accelerator.process_index}/{accelerator.num_processes}")

    return accelerator

'''
def initialize_fsdp_accelerator_from_config(config) -> Accelerator:
    # ===== SAME FIX FOR FSDP =====
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    # Set device FIRST
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize distributed BEFORE Accelerator
    if not dist.is_initialized():
        is_main_early = rank == 0
        if is_main_early:
            print(f"Manually initializing distributed for FSDP on device cuda:{local_rank}")
        
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            device_id=device  # This eliminates the warning!
        )
    
    mixed_precision = config.distribution.mixed_precision

    # Get wrapping targets based on model config
    model_params = load_model_configs(config.train.model_type)

    transformer_layer_classes = get_transformer_layers_to_wrap(
        model_params.model_type,
        config.train.use_lora
    ) 

    # Create custom wrapping policy
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_classes
    )
 
    fsdp_config_source_dict = {
        'fsdp_backward_prefetch_policy': 'BACKWARD_PRE',
        'fsdp_forward_prefetch': False,
        'fsdp_offload_params': False,
        'fsdp_sharding_strategy': 'FULL_SHARD',
        'fsdp_state_dict_type': 'FULL_STATE_DICT',
        'fsdp_sync_module_states': True,
        'fsdp_use_orig_params': True,
        'fsdp_state_dict_config': {
            'offload_to_cpu': True,
            'rank0_only': True,
        },
        'fsdp_optim_state_dict_config': {
            'offload_to_cpu': True,
            'rank0_only': True,
        }, 
    } 
    def activation_checkpointing_policy(module):
        return isinstance(module, tuple(transformer_layer_classes))
    
    fsdp_plugin = FullyShardedDataParallelPlugin(
        #activation_checkpointing=activation_checkpointing_policy, 
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=fsdp_config_source_dict.get('fsdp_sharding_strategy', 'FULL_SHARD'),
        backward_prefetch=fsdp_config_source_dict.get('fsdp_backward_prefetch_policy', 'NO_PREFETCH'),
        cpu_offload=fsdp_config_source_dict.get('fsdp_offload_params', False),
        state_dict_type=fsdp_config_source_dict.get('fsdp_state_dict_type', 'FULL_STATE_DICT'),
        sync_module_states=fsdp_config_source_dict.get('fsdp_sync_module_states', False),
        use_orig_params=fsdp_config_source_dict.get('fsdp_use_orig_params', False),
        forward_prefetch=fsdp_config_source_dict.get('fsdp_forward_prefetch', False),
        state_dict_config=fsdp_config_source_dict.get('fsdp_state_dict_config', None),
        optim_state_dict_config=fsdp_config_source_dict.get('fsdp_optim_state_dict_config', None),
    ) 
    
    accelerator_kwargs = {
        "mixed_precision": mixed_precision,
        "fsdp_plugin": fsdp_plugin,
    }
    accelerator = Accelerator(**accelerator_kwargs)

    if accelerator.is_main_process:
        print("✅ Accelerator initialized for FSDP.")
        print("FSDP Plugin prepared with fixed config.")
        print(f"FSDP Wrapping Targets: {[cls.__name__ for cls in transformer_layer_classes]}")
        
    return accelerator

'''
def initialize_fsdp_accelerator_from_config(config) -> Accelerator:
    # ===== SAME FIX FOR FSDP =====
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    # Set device FIRST
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize distributed BEFORE Accelerator
    if not dist.is_initialized():
        is_main_early = rank == 0
        if is_main_early:
            print(f"Manually initializing distributed for FSDP on device cuda:{local_rank}")
        
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            device_id=device  # This eliminates the warning!
        )
    
    mixed_precision = config.distribution.mixed_precision

    # Get wrapping targets based on model config
    model_params = load_model_configs(config.train.model_type)

    transformer_layer_classes = get_transformer_layers_to_wrap(
        model_params.model_type,
        config.train.use_lora
    ) 

    # Create ModuleWrapPolicy instead of transformer_auto_wrap_policy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    
    my_auto_wrap_policy = ModuleWrapPolicy(transformer_layer_classes)
 
    fsdp_config_source_dict = {
        'fsdp_backward_prefetch_policy': 'BACKWARD_PRE',
        'fsdp_forward_prefetch': False,
        'fsdp_offload_params': False,
        'fsdp_sharding_strategy': 'FULL_SHARD',
        'fsdp_state_dict_type': 'FULL_STATE_DICT',
        'fsdp_sync_module_states': True,
        'fsdp_use_orig_params': True,
        'fsdp_state_dict_config': {
            'offload_to_cpu': True,
            'rank0_only': True,
        },
        'fsdp_optim_state_dict_config': {
            'offload_to_cpu': True,
            'rank0_only': True,
        }, 
    } 
    def activation_checkpointing_policy(module):
        return isinstance(module, tuple(transformer_layer_classes))
    
    fsdp_plugin = FullyShardedDataParallelPlugin(
        activation_checkpointing=activation_checkpointing_policy if config.train.gradient_checkpointing_enable else None , 
        
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=fsdp_config_source_dict.get('fsdp_sharding_strategy', 'FULL_SHARD'),
        backward_prefetch=fsdp_config_source_dict.get('fsdp_backward_prefetch_policy', 'NO_PREFETCH'),
        cpu_offload=fsdp_config_source_dict.get('fsdp_offload_params', False),
        state_dict_type=fsdp_config_source_dict.get('fsdp_state_dict_type', 'FULL_STATE_DICT'),
        sync_module_states=fsdp_config_source_dict.get('fsdp_sync_module_states', False),
        use_orig_params=fsdp_config_source_dict.get('fsdp_use_orig_params', False),
        forward_prefetch=fsdp_config_source_dict.get('fsdp_forward_prefetch', False),
        state_dict_config=fsdp_config_source_dict.get('fsdp_state_dict_config', None),
        optim_state_dict_config=fsdp_config_source_dict.get('fsdp_optim_state_dict_config', None),
    ) 
    
    accelerator_kwargs = {
        "mixed_precision": mixed_precision,
        "fsdp_plugin": fsdp_plugin,
    }
    accelerator = Accelerator(**accelerator_kwargs)

    if accelerator.is_main_process:
        print("✅ Accelerator initialized for FSDP with ModuleWrapPolicy.")
        print("FSDP Plugin prepared with fixed config.")
        print(f"FSDP Wrapping Targets: {[cls.__name__ for cls in transformer_layer_classes]}")
        
    return accelerator

def print_cluster_info(accelerator): 
    hostname = socket.gethostname() 
    world_size = accelerator.num_processes
 
    node_info = {
        "hostname": hostname,
        "rank": accelerator.process_index,
        "device": str(accelerator.device),
        "local_rank": int(os.environ.get('LOCAL_RANK', '0'))
    }

    if dist.is_initialized():
        node_info_list = [None] * world_size
        if accelerator.is_main_process:  
            write_to_main_log(accelerator=accelerator, result=f"Collecting basic node information from all {world_size} processes...")
        
        try:
            # Use barrier for synchronization before all_gather
            dist.barrier()
            dist.all_gather_object(node_info_list, node_info)
            
            if accelerator.is_main_process:
                write_to_main_log(accelerator=accelerator, result="✅ Node information collection completed successfully!")
                
        except Exception as e:
            if accelerator.is_main_process:
                write_to_main_log(accelerator=accelerator, result=f"⚠️ Error in all_gather: {e}")
            node_info_list = [node_info]  # Fallback
    else:
        node_info_list = [node_info]  
 
    if accelerator.is_main_process:
        nodes = {}
        for info in node_info_list:
            if info and info["hostname"] not in nodes:
                nodes[info["hostname"]] = []
            if info:
                nodes[info["hostname"]].append(info)

        output = "=" * 20 + " CLUSTER INFO SUMMARY " + "=" * 20
        write_to_main_log(accelerator=accelerator, result=output)   
        
        for hostname, gpu_list in sorted(nodes.items()):
            devices = [f"cuda:{info.get('local_rank', '?')}" for info in gpu_list]
            output = f"NODE: {hostname} has {len(gpu_list)} GPUs: {devices}"
            write_to_main_log(accelerator=accelerator, result=output)   

        output = f"TOTAL GPUs IN CLUSTER: {world_size}" 
        write_to_main_log(accelerator=accelerator, result=output)


def setup_accelerate_seed(accelerator, config):
    """
    Sets up the random seed for distributed training.
    - Defaults to 42.
    - Uses config.train.seed if it's an integer.
    - Generates a synchronized random seed if config.train.seed == 'random'.
    """
    configured_seed = getattr(config.train, 'seed', None)

    # Determine Seed Mode
    if configured_seed is None:
        seed = 42
        mode = "default"
    elif str(configured_seed).strip().lower() == "random":
        seed = None
        mode = "random"
    else:
        try:
            seed = int(configured_seed)
            mode = "fixed"
        except ValueError:
            raise ValueError(
                f"Invalid seed in config: '{configured_seed}'. "
                f"Expected an integer or the string 'random'."
            )

    # Generate and Synchronize (if Random)
    if mode == "random":
        if accelerator.is_main_process:
            # Multiply by 1000 for better sub-second variance, fit in positive 32-bit int
            generated_seed = int(time.time() * 1000) % (2 ** 31 - 1)
            seed_tensor = torch.tensor([generated_seed], dtype=torch.long, device=accelerator.device)
        else:
            seed_tensor = torch.tensor([0], dtype=torch.long, device=accelerator.device)

        # Broadcast from Rank 0 to all other processes
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(seed_tensor, src=0)

        seed = seed_tensor.item()

        if accelerator.is_main_process:
            write_to_main_log(accelerator=accelerator, result=f"Using Synchronized Random Seed: {seed}")

    # Log Deterministic Seeds
    else:
        if accelerator.is_main_process:
            log_prefix = "Default Fixed" if mode == "default" else "Configured Fixed"
            write_to_main_log(accelerator=accelerator, result=f"Using {log_prefix} Seed: {seed}")

    # Apply the Seed
    set_seed(seed)
    return seed

# freeze_last_layer 50 
# generate_samples True 
# use_pretrained True 
# save_checkpoint_frequency 1000
# warmup_iterations %30 of max_iterations or  max_iterations * 0.3
# warmup_teacher_temp_iterations == max_iterations 
#  
#python clearml_training_wrapper.py  --dataset_path /pscratch/scheu2_dgxbdsa24/neuropath_dataset --do_distillation False --distribution_type ddp --model_name clearml_test --global_batch_size 128 --max_iterations 5000 --gradient_checkpointing_enable False --use_lora False --freeze_backbone_layers 0 --warmup_iterations 1500 --momentum_teacher 0.999 --ibot_separate_head False --save_checkpoint_frequency 1000 --model_type facebook/dinov2-base 
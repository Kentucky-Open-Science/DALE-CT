# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Model initialization, LoRA, checkpointing, and training utilities for LeJEPA.
"""
import copy
import os
import types

import numpy as np
import torch
from torch import nn

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate.state import DistributedType

from utils.distributed_checkpointer import DistributedCheckpointManager
from utils.config import load_model_configs
from utils.global_state import GlobalState
from utils.logger_utils import write_to_main_log
from utils.model_checkpointer_ddp import DDPModelBackboneCheckpointManager
from utils.model_checkpointer_fsdp import FSDPModelBackboneCheckpointManager
from models.lejepa_projector import LeJEPAProjector
from models.vision_transformer import (
    LeJEPA_Wrapper,
    build_transformer_model_from_config,
    build_transformer_model_from_timm,
)


# ---------------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------------

def build_model_with_config(config, model_params, accelerator, checkpoint_path=None,
                            load_pretrained=False, model_type=None):
    """Build a model based on config, handling both TIMM and HuggingFace models."""
    if model_type is None:
        model_type = config.train.model_type

    if hasattr(model_params, 'timm_arch'):
        write_to_main_log(accelerator, "Initializing model via TIMM builder")
        model_backbone, model_config = build_transformer_model_from_timm(
            model_config=model_params,
            accelerator=accelerator,
            use_pretrained=load_pretrained,
            weights_path=checkpoint_path
        )
    else:
        write_to_main_log(accelerator, "Initializing model via HuggingFace builder")
        model_backbone, model_config = build_transformer_model_from_config(
            config=config,
            load_pretrained=load_pretrained,
            custom_config=model_params,
            model_type=model_type,
            accelerator=accelerator,
            checkpoint_path=checkpoint_path
        )
    return model_backbone, model_config


# ---------------------------------------------------------------------------
# Gradient & Parameter Utilities
# ---------------------------------------------------------------------------

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def freeze_layers(model, accelerator, use_lora=False, num_layers_to_freeze=0):
    if num_layers_to_freeze <= 0:
        return

    if use_lora:
        write_to_main_log(accelerator=accelerator, result="LoRA is active, skipping layer freezing")
        return

    write_to_main_log(accelerator=accelerator,
                      result=f"Attempting to freeze first {num_layers_to_freeze} layers/blocks in model")

    backbone = model.backbone
    transformer = backbone.module if hasattr(backbone, "module") else backbone

    if hasattr(transformer, "encoder"):
        encoder = transformer.encoder
        layers_attr = "layer" if hasattr(encoder, "layer") else "layers"
    elif hasattr(transformer, "blocks"):
        encoder = transformer
        layers_attr = "blocks"
    else:
        write_to_main_log(accelerator=accelerator, result="Encoder structure not found!")
        return

    layers = getattr(encoder, layers_attr)
    freeze_count = min(num_layers_to_freeze, len(layers))
    for i in range(freeze_count):
        for param in layers[i].parameters():
            param.requires_grad = False

    if accelerator.is_main_process:
        frozen_params = total_params = 0
        for param in model.parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()
        frozen_percentage = (frozen_params / total_params) * 100
        write_to_main_log(accelerator=accelerator,
                          result=f"Total: {total_params:,}, Frozen: {frozen_params:,} ({frozen_percentage:.2f}%)")


# ---------------------------------------------------------------------------
# LeJEPA Model Initialization
# ---------------------------------------------------------------------------

def init_lejepa_training_models_wrapper(config, accelerator,
                                        lora_r=None, lora_alpha=None, lora_dropout=None):
    """Initialize the LeJEPA model (backbone + projector)."""
    model_type = config.train.model_type
    model_params = load_model_configs(model_type)

    proj_dim = getattr(config.projector, 'proj_dim', 128)
    hidden_dim = getattr(config.projector, 'proj_hidden_dim', 2048)

    checkpoint_path = getattr(config.train, 'checkpoint_path', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_path = config.train.checkpoint_path
        if config.train.use_pretrained:
            write_to_main_log(accelerator=accelerator,
                              result="WARNING: Both use_pretrained=True and checkpoint_path provided. "
                                     "Checkpoint will be prioritized over pretrained weights.",
                              type='warning')

    model_backbone, model_config = build_model_with_config(
        config=config,
        model_params=model_params,
        accelerator=accelerator,
        checkpoint_path=checkpoint_path,
        load_pretrained=config.train.use_pretrained
    )

    if config.train.use_lora:
        model_backbone = apply_lora_to_model(
            backbone=model_backbone,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            accelerator=accelerator,
            adapter_path=None
        )
        model_config = model_backbone.peft_config

    backbone_dim = model_config.hidden_size if hasattr(model_config, 'hidden_size') else 768
    projector = LeJEPAProjector(
        input_dim=backbone_dim,
        hidden_dim=hidden_dim,
        output_dim=proj_dim
    )

    if accelerator.is_main_process:
        write_to_main_log(
            accelerator=accelerator,
            result=f"LeJEPAProjector initialized: input_dim={backbone_dim}, hidden_dim={hidden_dim}, output_dim={proj_dim}"
        )

    model = LeJEPA_Wrapper(
        config=config,
        backbone=model_backbone,
        projector=projector
    )

    write_to_main_log(accelerator=accelerator, result="LeJEPA_Wrapper instance created.")
    return model, model_config


# ---------------------------------------------------------------------------
# Evaluation Model Initialization
# ---------------------------------------------------------------------------

def init_dino_evaluiaton_model(config, accelerator):
    """Initialize evaluation model with support for .safetensors and LoRA."""
    model_type = config.train.model_type
    model_path = config.train.vit_ckpt_path if hasattr(config.train, 'vit_ckpt_path') else None
    use_pretrained = config.train.use_pretrained
    adapter_path = os.path.join(model_path, 'adapter_config.json') if model_path else None
    is_lora = os.path.exists(adapter_path) if adapter_path else False

    model_params = load_model_configs(model_type)

    checkpoint_path = None
    if model_path:
        checkpoint_path = model_path
        if use_pretrained:
            write_to_main_log(accelerator=accelerator,
                              result="WARNING: Both use_pretrained=True and checkpoint_path provided. "
                                     "Checkpoint will be prioritized over pretrained weights.",
                              type='warning')

    base_model, _ = build_model_with_config(
        config=config,
        model_params=model_params,
        accelerator=accelerator,
        checkpoint_path=checkpoint_path,
        load_pretrained=use_pretrained,
        model_type=model_type
    )

    if hasattr(model_params, 'timm_arch'):
        write_to_main_log(accelerator=accelerator, result=f"Loaded TIMM model: {model_type}")
        return base_model

    if is_lora:
        write_to_main_log(accelerator=accelerator, result=f"Applying LoRA to HuggingFace model: {model_type}")
        base_model = apply_lora_to_model(
            backbone=base_model,
            accelerator=accelerator,
            adapter_path=model_path
        )
    else:
        write_to_main_log(accelerator=accelerator, result=f"Loaded HuggingFace model: {model_type}")

    return base_model


# ---------------------------------------------------------------------------
# LoRA Utilities
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    backbone,
    accelerator,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    adapter_path=None,
    target_modules=None,
):
    write_to_main_log(accelerator=accelerator, result="Applying LoRA to model")

    if adapter_path and os.path.exists(adapter_path):
        backbone = PeftModel.from_pretrained(backbone, adapter_path)
    else:
        if adapter_path and not os.path.exists(adapter_path):
            write_to_main_log(accelerator=accelerator,
                              result=f"Warning: Adapter path {adapter_path} does not exist. "
                                     "Creating a new LoRA adapter instead.",
                              type='warning')

        write_to_main_log(accelerator=accelerator,
                          result=f"Creating new LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

        if target_modules is None:
            target_modules = ["query", "key", "value"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="lora_only",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        backbone = get_peft_model(backbone, lora_config)

    # Count LoRA modules
    lora_module_count = 0
    lora_layer_set = set()
    for name, _ in backbone.named_modules():
        if "lora_A" in name or "lora_B" in name:
            if "encoder.layer." in name:
                layer_num = name.split("encoder.layer.")[1].split(".")[0]
                lora_layer_set.add(f"layer_{layer_num}")
            if "lora_A" in name:
                lora_module_count += 1

    write_to_main_log(accelerator=accelerator,
                      result=f"LoRA applied to {lora_module_count} modules across {len(lora_layer_set)} layers")
    write_to_main_log(accelerator=accelerator,
                      result=f"Affected layers: {sorted(list(lora_layer_set))}")

    # Monkey-patch forward for flexible input handling
    def monkey_patch_forward(module):
        def custom_forward(self, x=None, *args, **kwargs):
            if kwargs.get("input_ids") is not None:
                x = kwargs.pop("input_ids")
            elif x is None and len(args) > 0:
                x = args[0]
                args = args[1:]
            if hasattr(self.base_model, "model"):
                return self.base_model.model(x, *args, **kwargs)
            else:
                return self.base_model(x, *args, **kwargs)
        module.forward = types.MethodType(custom_forward, module)
        return module

    backbone = monkey_patch_forward(backbone)
    return backbone


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------

def print_model_info(accelerator, model):
    import torch.distributed as dist

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    local_params = sum(p.numel() for p in model.parameters())
    local_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_tensor = torch.tensor([local_params, local_trainable], dtype=torch.long,
                                device='cuda' if torch.cuda.is_available() else 'cpu')
    gathered_tensors = [torch.zeros_like(param_tensor) for _ in range(world_size)]

    if world_size > 1:
        dist.all_gather(gathered_tensors, param_tensor)
    else:
        gathered_tensors[0] = param_tensor

    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    else:
        allocated_memory = total_memory = 0

    if rank == 0:
        per_gpu_params = [t[0].item() for t in gathered_tensors]
        per_gpu_trainable = [t[1].item() for t in gathered_tensors]

        is_sharded = False
        for module in model.modules():
            if 'FullyShardedDataParallel' in str(type(module)):
                is_sharded = True
                break

        if is_sharded:
            total_params = sum(per_gpu_params)
            total_trainable = sum(per_gpu_trainable)
        else:
            total_params = per_gpu_params[0]
            total_trainable = per_gpu_trainable[0]

        write_to_main_log(accelerator=accelerator, result="===== Model Parameter Distribution =====")
        write_to_main_log(accelerator=accelerator,
                          result=f"Total parameters: {total_params:,}")
        write_to_main_log(accelerator=accelerator,
                          result=f"Total trainable: {total_trainable:,} ({total_trainable / total_params * 100:.1f}%)")
        write_to_main_log(accelerator=accelerator, result="Per-GPU Distribution:")
        for i, (params, trainable) in enumerate(zip(per_gpu_params, per_gpu_trainable)):
            write_to_main_log(accelerator=accelerator,
                              result=f"GPU {i}: {params:,} params, {trainable:,} trainable ({trainable / params * 100:.1f}%)")

    if torch.cuda.is_available():
        write_to_main_log(accelerator=accelerator,
                          result=f"GPU {rank} memory: {allocated_memory:.2f}GB / {total_memory:.2f}GB ({allocated_memory / total_memory * 100:.1f}%)")
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def get_periodic_train_checkpointer(config, model, optimizer, accelerator, max_to_keep=3):
    checkpoint_manager = DistributedCheckpointManager(
        model=model,
        optimizer=optimizer,
        save_dir=GlobalState.get('train_checkpoint_dir'),
        accelerator=accelerator,
        max_to_keep=max_to_keep,
    )
    # iter_N is written AFTER step N-1 completes (save() is called post-increment,
    # after ssl_model(N-1) which runs optimizer.step internally): it means
    # "N steps done, step N pending". So resume does step N. The previous `+ 1`
    # skipped step N on every resume (lost update + lost batch). No checkpoint ->
    # load_latest() returns -1 -> start at 0.
    loaded = checkpoint_manager.load_latest()
    start_iteration = 0 if loaded < 0 else loaded
    return start_iteration, checkpoint_manager


def get_periodic_backbone_checkpointer_ddp(config, model, accelerator, max_to_keep=3):
    if GlobalState.has('teacher_checkpoint_dir'):
        save_dir = GlobalState.get('teacher_checkpoint_dir')
    else:
        save_dir = GlobalState.get('model_checkpoint_dir')
    checkpoint_manager = DDPModelBackboneCheckpointManager(
        model=model,
        config=config,
        save_dir=save_dir,
        accelerator=accelerator,
        max_to_keep=max_to_keep,
    )
    return checkpoint_manager


def get_periodic_backbone_checkpointer_fsdp(config, model, accelerator, max_to_keep=3):
    if GlobalState.has('teacher_checkpoint_dir'):
        save_dir = GlobalState.get('teacher_checkpoint_dir')
    else:
        save_dir = GlobalState.get('model_checkpoint_dir')
    checkpoint_manager = FSDPModelBackboneCheckpointManager(
        model=model,
        config=config,
        save_dir=save_dir,
        accelerator=accelerator,
        max_to_keep=max_to_keep,
    )
    return checkpoint_manager


# ---------------------------------------------------------------------------
# Sample Generation (stub -- visualization modules removed for public release)
# ---------------------------------------------------------------------------

def generate_samples(accelerator, config, model_params, imgs, masks, iteration,
                     backbone_model, num_register_tokens=0, dataset=None):
    """Stub: sample visualization is disabled in the public release."""
    pass


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def cosine_scheduler(base_value, final_value, max_iters, warmup_iters=0,
                     start_warmup_value=0, keep_constant_after_warmup=False):
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    if keep_constant_after_warmup:
        remaining_iters = max_iters - warmup_iters
        constant_schedule = np.ones(remaining_iters) * base_value
        schedule = np.concatenate((warmup_schedule, constant_schedule))
    else:
        iters = np.arange(max_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == max_iters
    return schedule


def init_lejepa_schedulers(config):
    lr_schedule = cosine_scheduler(
        config.train.lr,
        config.train.min_lr,
        config.train.max_iterations,
        warmup_iters=config.train.warmup_iterations,
    )
    wd_schedule = np.full(
        config.train.max_iterations,
        config.train.weight_decay
    )
    return lr_schedule, wd_schedule

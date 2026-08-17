import os
import sys
import argparse
import json
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import math
import signal
from contextlib import nullcontext

from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
from sklearn.metrics import roc_auc_score, average_precision_score
from peft import LoraConfig, get_peft_model

# Ensure project root is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils.config import load_config, create_output_dirs
from utils.global_state import GlobalState
from utils.dino_utils import init_dino_evaluiaton_model, apply_lora_to_model, freeze_layers
from utils.logger_utils import write_to_main_log
from utils.wandb_utils import setup_wandb, log_metrics_wandb
from dataloaders.datasetloader_ctrate_multiscale import get_wds_dataset, get_npy_validation_dataset, \
    MultiScaleSliceProcessor, CTMultiScaleDataset
from models.e2e_colipri import EndToEndColipri


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def compute_metrics(logits_list, labels_list):
    """Compute macro AUROC and AUPRC across all classes from accumulated predictions."""
    all_logits = torch.cat(logits_list, dim=0).cpu().numpy()
    all_labels = torch.cat(labels_list, dim=0).cpu().numpy()
    try:
        auroc_scores = []
        auprc_scores = []
        for c in range(all_labels.shape[1]):
            if all_labels[:, c].sum() == 0 or all_labels[:, c].sum() == all_labels.shape[0]:
                auroc_scores.append(0.5)  # Undefined for single-class columns
                auprc_scores.append(0.0)
            else:
                auroc_scores.append(roc_auc_score(all_labels[:, c], all_logits[:, c]))
                auprc_scores.append(average_precision_score(all_labels[:, c], all_logits[:, c]))
        return float(np.mean(auroc_scores)), float(np.mean(auprc_scores))
    except Exception:
        return 0.5, 0.0


CHECKPOINT_DIR_NAME = "training_checkpoints"

# Global flag set by signal handler to request a graceful save-and-exit
_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True


def _get_checkpoint_dir(save_dir):
    """Return the path to the checkpoint directory."""
    return os.path.join(save_dir, CHECKPOINT_DIR_NAME)


def _get_latest_checkpoint_path(save_dir):
    """Find the latest training checkpoint, if any. Returns (path, step) or (None, -1)."""
    ckpt_dir = _get_checkpoint_dir(save_dir)
    if not os.path.isdir(ckpt_dir):
        return None, -1
    # Look for training_state_step_*.pt files
    pattern = os.path.join(ckpt_dir, "training_state_step_*.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None, -1
    # Extract step numbers and find the max
    best_path = None
    best_step = -1
    for m in matches:
        try:
            basename = os.path.basename(m)
            # Format: training_state_step_<number>.pt
            step_str = basename.replace("training_state_step_", "").replace(".pt", "")
            step = int(step_str)
            if step > best_step:
                best_step = step
                best_path = m
        except ValueError:
            continue
    return best_path, best_step


def save_checkpoint(model, optimizer, scheduler, training_state, save_dir, accelerator, multi_cfg, finetune_method):
    """Save a full training checkpoint that can be used to resume.

    Saves:
      - LoRA adapter weights (PEFT save_pretrained per adapter)
      - Colipri head weights (state dict per adapter)
      - Optimizer state dict
      - Scheduler state dict
      - Training state dict (global_step, epoch, batch_idx, best metrics, etc.)
      - RNG states (Python, NumPy, PyTorch)
    """
    if not accelerator.is_main_process:
        return

    ckpt_dir = _get_checkpoint_dir(save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = training_state['global_step']
    unwrapped = accelerator.unwrap_model(model)

    if finetune_method == 'lora':
        # 1. Save LoRA adapter weights
        if multi_cfg and multi_cfg.get('enabled', False):
            for adapter_name in training_state['all_adapter_names']:
                adapter_ckpt_dir = os.path.join(ckpt_dir, f"lora_{adapter_name}_step_{global_step}")
                unwrapped.backbone.save_pretrained(adapter_ckpt_dir, adapter_name=adapter_name)
        else:
            adapter_ckpt_dir = os.path.join(ckpt_dir, f"lora_default_step_{global_step}")
            unwrapped.backbone.save_pretrained(adapter_ckpt_dir, adapter_name="default")

        # 2. Save Colipri head weights
        if multi_cfg and multi_cfg.get('enabled', False):
            for adapter_name in training_state['all_adapter_names']:
                head_path = os.path.join(ckpt_dir, f"colipri_{adapter_name}_step_{global_step}.pth")
                torch.save(unwrapped.colipri_heads[adapter_name].state_dict(), head_path)
        else:
            head_path = os.path.join(ckpt_dir, f"colipri_default_step_{global_step}.pth")
            torch.save(unwrapped.colipri_heads['default'].state_dict(), head_path)
    else:
        # 1+2. Save full trainable state_dict (unfrozen backbone + arch + head)
        # for partial/full unfreeze. Frozen backbone params are excluded.
        trainable_names = set(n for n, p in unwrapped.named_parameters() if p.requires_grad)
        trainable_state_dict = {k: v for k, v in unwrapped.state_dict().items() if k in trainable_names}
        model_path = os.path.join(ckpt_dir, f"model_step_{global_step}.pth")
        torch.save(trainable_state_dict, model_path)

    # 3. Save optimizer state
    optimizer_path = os.path.join(ckpt_dir, f"optimizer_step_{global_step}.pth")
    torch.save(optimizer.state_dict(), optimizer_path)

    # 4. Save scheduler state
    scheduler_path = os.path.join(ckpt_dir, f"scheduler_step_{global_step}.pth")
    torch.save(scheduler.state_dict(), scheduler_path)

    # 5. Save training state (JSON-serializable + tensors)
    serializable_state = {}
    for k, v in training_state.items():
        if isinstance(v, dict):
            serializable_state[k] = {kk: vv for kk, vv in v.items()}
        elif isinstance(v, list):
            serializable_state[k] = list(v)
        else:
            serializable_state[k] = v

    # 6. Save RNG states
    rng_state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state['torch_cuda'] = torch.cuda.get_rng_state_all()

    training_state_path = os.path.join(ckpt_dir, f"training_state_step_{global_step}.pt")
    torch.save({
        'training_state': serializable_state,
        'rng_state': rng_state,
    }, training_state_path)

    # 7. Clean up old checkpoints (keep only the 3 most recent by step)
    all_state_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "training_state_step_*.pt")),
        key=lambda x: int(os.path.basename(x).replace("training_state_step_", "").replace(".pt", ""))
    )
    for old_file in all_state_files[:-3]:
        old_step = int(os.path.basename(old_file).replace("training_state_step_", "").replace(".pt", ""))
        # Remove associated files for this step
        for pattern in [
            f"optimizer_step_{old_step}.pth",
            f"scheduler_step_{old_step}.pth",
        ]:
            p = os.path.join(ckpt_dir, pattern)
            if os.path.isfile(p):
                os.remove(p)
        # Remove LoRA dirs
        for adapter_name in training_state.get('all_adapter_names', []):
            adapter_dir = os.path.join(ckpt_dir, f"lora_{adapter_name}_step_{old_step}")
            if os.path.isdir(adapter_dir):
                import shutil
                shutil.rmtree(adapter_dir, ignore_errors=True)
        # Remove head files
        for adapter_name in training_state.get('all_adapter_names', []):
            head_p = os.path.join(ckpt_dir, f"colipri_{adapter_name}_step_{old_step}.pth")
            if os.path.isfile(head_p):
                os.remove(head_p)
        # Remove full-model state files (partial/full unfreeze methods)
        if finetune_method != 'lora':
            model_p = os.path.join(ckpt_dir, f"model_step_{old_step}.pth")
            if os.path.isfile(model_p):
                os.remove(model_p)
        # Remove the state file itself
        if os.path.isfile(old_file):
            os.remove(old_file)

    write_to_main_log(accelerator, f"Checkpoint saved at step {global_step}.")


def load_checkpoint(model, optimizer, scheduler, save_dir, accelerator, multi_cfg, finetune_method, device):
    """Attempt to load the latest checkpoint. Returns training_state dict or None."""
    ckpt_path, resumed_step = _get_latest_checkpoint_path(save_dir)
    if ckpt_path is None:
        return None

    write_to_main_log(accelerator, f"Found checkpoint at step {resumed_step}. Resuming...")

    ckpt_dir = _get_checkpoint_dir(save_dir)
    unwrapped = accelerator.unwrap_model(model)

    # 1. Load training state and RNG
    state_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    training_state = state_data['training_state']
    rng_state = state_data['rng_state']

    # Restore RNG states
    random.setstate(rng_state['python'])
    np.random.set_state(rng_state['numpy'])
    torch.set_rng_state(rng_state['torch'])
    if torch.cuda.is_available() and 'torch_cuda' in rng_state:
        torch.cuda.set_rng_state_all(rng_state['torch_cuda'])

    if finetune_method == 'lora':
        # 2. Load LoRA adapter weights
        if multi_cfg and multi_cfg.get('enabled', False):
            for adapter_name in training_state['all_adapter_names']:
                adapter_ckpt_dir = os.path.join(ckpt_dir, f"lora_{adapter_name}_step_{resumed_step}")
                if os.path.isdir(adapter_ckpt_dir):
                    unwrapped.backbone.load_adapter(adapter_ckpt_dir, adapter_name=adapter_name)
                    write_to_main_log(accelerator, f"  Loaded LoRA adapter '{adapter_name}'")
        else:
            adapter_ckpt_dir = os.path.join(ckpt_dir, f"lora_default_step_{resumed_step}")
            if os.path.isdir(adapter_ckpt_dir):
                unwrapped.backbone.load_adapter(adapter_ckpt_dir, adapter_name="default")
                write_to_main_log(accelerator, "  Loaded LoRA adapter 'default'")

        # 3. Load Colipri head weights
        if multi_cfg and multi_cfg.get('enabled', False):
            for adapter_name in training_state['all_adapter_names']:
                head_path = os.path.join(ckpt_dir, f"colipri_{adapter_name}_step_{resumed_step}.pth")
                if os.path.isfile(head_path):
                    unwrapped.colipri_heads[adapter_name].load_state_dict(
                        torch.load(head_path, map_location='cpu', weights_only=True)
                    )
                    write_to_main_log(accelerator, f"  Loaded Colipri head '{adapter_name}'")
        else:
            head_path = os.path.join(ckpt_dir, f"colipri_default_step_{resumed_step}.pth")
            if os.path.isfile(head_path):
                unwrapped.colipri_heads['default'].load_state_dict(
                    torch.load(head_path, map_location='cpu', weights_only=True)
                )
                write_to_main_log(accelerator, "  Loaded Colipri head 'default'")
    else:
        # 2+3. Load full trainable state_dict (unfrozen backbone + arch + head).
        # strict=False: frozen backbone params are absent from the checkpoint.
        model_path = os.path.join(ckpt_dir, f"model_step_{resumed_step}.pth")
        if os.path.isfile(model_path):
            unwrapped.load_state_dict(
                torch.load(model_path, map_location='cpu', weights_only=True),
                strict=False
            )
            write_to_main_log(accelerator, "  Loaded full model state_dict")
        else:
            write_to_main_log(accelerator, f"  WARNING: model checkpoint not found at {model_path}")

    # 4. Load optimizer state
    optimizer_path = os.path.join(ckpt_dir, f"optimizer_step_{resumed_step}.pth")
    if os.path.isfile(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu', weights_only=True))
        write_to_main_log(accelerator, "  Loaded optimizer state")

    # 5. Load scheduler state
    scheduler_path = os.path.join(ckpt_dir, f"scheduler_step_{resumed_step}.pth")
    if os.path.isfile(scheduler_path):
        scheduler.load_state_dict(torch.load(scheduler_path, map_location='cpu', weights_only=True))
        write_to_main_log(accelerator, "  Loaded scheduler state")

    write_to_main_log(accelerator, f"Resumed from step {resumed_step}, epoch {training_state.get('epoch', 0) + 1}")
    return training_state


@torch.no_grad()
def run_validation(model, val_loader, processor, criterion_dict, device, config, active_adapters, max_batches=None):
    """Run validation loop and return per-adapter metrics."""
    model.eval()

    total_loss = {adapter: 0.0 for adapter in active_adapters}
    num_batches = 0
    all_logits = {adapter: [] for adapter in active_adapters}
    all_labels = {adapter: [] for adapter in active_adapters}

    # Safely check if multi_adapter is enabled to prevent OmegaConf attribute errors
    multi_cfg = getattr(config, 'multi_adapter', None)
    is_multi = multi_cfg and multi_cfg.get('enabled', False)

    for batch_idx, (volumes, labels, filenames) in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break

        raw_volume = volumes.squeeze(0).to(device)
        labels = labels.to(device)
        processed_slices, _ = processor.process_batch(raw_volume)
        processed_slices = processed_slices.unsqueeze(0)

        # Pass active adapters to avoid wasted computation
        model_output = model(processed_slices, chunk_size=32, active_adapters=active_adapters)

        for adapter_name in active_adapters:
            if is_multi:
                adapter_cfg = config.multi_adapter.adapters[adapter_name]
                adapter_labels = labels[:, adapter_cfg.classes]
                adapter_logits = model_output[adapter_name]
            else:
                # Fallback for standard single-adapter training
                adapter_labels = labels
                adapter_logits = model_output  # Model returns a raw tensor, not a dict, for single adapter

            loss = criterion_dict[adapter_name](adapter_logits, adapter_labels)
            total_loss[adapter_name] += loss.item()

            all_logits[adapter_name].append(adapter_logits.detach())
            all_labels[adapter_name].append(adapter_labels.detach())

        num_batches += 1

    model.train()

    metrics = {}
    for adapter_name in active_adapters:
        avg_loss = total_loss[adapter_name] / max(num_batches, 1)
        if all_logits[adapter_name]:
            auroc, auprc = compute_metrics(all_logits[adapter_name], all_labels[adapter_name])
        else:
            auroc, auprc = 0.5, 0.0

        metrics[adapter_name] = {'loss': avg_loss, 'auroc': auroc, 'auprc': auprc}

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/finetune_lora.yaml")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Automatically resume from latest checkpoint if available (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Start training from scratch, ignoring any existing checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save a full training checkpoint every N accumulation steps "
                             "(default: same as val_every_n_steps)")
    args = parser.parse_args()

    # Use the framework's OmegaConf-based config loader
    config = load_config(args.config)

    # Initialize Accelerator for DDP and logging
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))],
        split_batches=True,
        mixed_precision='bf16',
    )

    set_seed(config.experiment.seed)
    device = accelerator.device

    if accelerator.is_main_process:
        create_output_dirs(config)
    GlobalState.synchronize()
    save_dir = GlobalState.get('e2e_save_dir')
    write_to_main_log(accelerator, f"Model checkpoints will be saved to: {save_dir}")

    # 1. Data Processing Strategy (normalization_mode comes from the config;
    #    must match the 2S pretrain: 'z_score' for the Guided-Chest-CT-LeJEPA-2S backbone).
    write_to_main_log(accelerator, f"Normalization mode: {config.dataset.normalization_mode}")

    # 2. Setup Data Pipelines
    write_to_main_log(accelerator, "Preparing dataloaders...")

    import pandas as pd
    train_label_csv = getattr(config.data, 'train_label_path', None)
    labels_map = None
    if train_label_csv and os.path.exists(train_label_csv):
        write_to_main_log(accelerator, f"Loading training labels from {train_label_csv}...")
        df = pd.read_csv(train_label_csv)
        id_col = df.columns[0]
        labels_map = df.set_index(id_col)
        write_to_main_log(accelerator, f"Loaded labels for {len(labels_map)} volumes")

    # Train data: opt-in whole-volume .npy (Stage 2 fair-train) via
    # config.validation.train_dataset_format='npy'; default 'webdataset' = the
    # existing tar-shard path (ablation configs unaffected). The npy path reads
    # CTMultiScaleDataset filtered to train_allowed_volumes_path (.nii.gz names),
    # so the 245 test-overlap patients can be excluded from ckpt-val while train
    # stays the full fair-train.
    train_dataset_format = getattr(config.validation, 'train_dataset_format', 'webdataset')
    if train_dataset_format == 'npy':
        train_data_dir = config.validation.train_data_dir
        train_allowed_path = getattr(config.validation, 'train_allowed_volumes_path', None)
        train_allowed = None
        if train_allowed_path:
            with open(train_allowed_path) as _f:
                train_allowed = set(_l.strip() for _l in _f if _l.strip())
            write_to_main_log(accelerator,
                              f"Train npy: {len(train_allowed)} allowed volumes from {train_allowed_path}")
        train_dataset = CTMultiScaleDataset(
            config, data_dir=train_data_dir,
            label_csv=train_label_csv,
            allowed_volume_names=train_allowed,
        )
        write_to_main_log(accelerator,
                          f"Train npy dataset: {len(train_dataset)} volumes from {train_data_dir}")
        # Pre-flight (npy): every allowed train volume must (a) have an 18-class
        # label in the CSV -- CTMultiScaleDataset raises KeyError inside a WORKER
        # on a miss, and the train loop's `continue` fallback below can't catch
        # it, so one gap among 4,642 would kill the run mid-epoch -- and (b) have
        # its .npy on disk (catches an incomplete CT-RATE_train_hu build before
        # wasting GPU time). Runs on all ranks so every rank fails fast (a
        # rank-local assert exit avoids DDP hangs at the next collective).
        if labels_map is not None and train_allowed:
            _missing = [v for v in train_allowed if v not in labels_map.index]
            assert not _missing, (f"Train pre-flight: {len(_missing)}/{len(train_allowed)} volumes "
                                  f"have no label in {train_label_csv}; first: {_missing[:5]}")
            assert len(train_dataset) == len(train_allowed), (
                f"Train pre-flight: {len(train_dataset)} .npy on disk vs {len(train_allowed)} "
                f"allowed -- CT-RATE_train_hu build incomplete?")
            write_to_main_log(accelerator,
                              f"Train pre-flight OK: {len(train_allowed)} vols all labeled + on disk")
    else:
        train_dataset = get_wds_dataset(config, tar_pattern=config.validation.train_tar_pattern)

    # DDP sharding for the npy path. CTMultiScaleDataset is a plain map-style
    # Dataset (no wds.split_by_node like the tar path), so without a sampler each
    # rank would iterate ALL 4,642 vols -> 8x redundant compute, eff. batch
    # collapses to 8 (not 64), and the cosine LR schedule (total_steps=219) hits
    # min_lr at step 219 then pins for ~1,500 steps. A bare DistributedSampler
    # shards indices directly (each rank a disjoint ~1/8 share -> eff. batch =
    # 8 ranks x 1 x acc(8) = 64). We do NOT use accelerator.prepare here: under
    # split_batches=True + batch_size=1 + world>1 it wraps with BatchSamplerShard
    # which RAISES "batch size 1 not a round multiple of num_processes" -- so the
    # 1-GPU smoke passes but 8-GPU crashes at startup. The tar path needs no
    # sampler (wds shards internally) and keeps the plain DataLoader. shuffle is
    # omitted: a non-None sampler is mutually exclusive with shuffle in PyTorch,
    # and None-sampler defaults to SequentialSampler (= shuffle=False) anyway.
    train_sampler = None
    if train_dataset_format == 'npy' and accelerator.num_processes > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=False)
    _nw = config.validation.num_workers
    train_loader = DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler,
        num_workers=_nw,
        pin_memory=True, timeout=300,
        prefetch_factor=3 if _nw > 0 else None,
        persistent_workers=_nw > 0,
    )

    val_loader = None
    val_data_dir = getattr(config.validation, 'val_data_dir', None)
    if val_data_dir and os.path.isdir(val_data_dir):
        write_to_main_log(accelerator, f"Loading validation .npy dataset from {val_data_dir}...")
        val_label_csv = getattr(config.validation, 'val_label_csv', None)
        val_max_patients = getattr(config.validation, 'val_max_patients', 200)
        val_allowed_path = getattr(config.validation, 'val_allowed_volumes_path', None)
        val_allowed = None
        if val_allowed_path:
            with open(val_allowed_path) as _f:
                val_allowed = set(_l.strip() for _l in _f if _l.strip())
            write_to_main_log(accelerator,
                              f"Val npy: {len(val_allowed)} allowed volumes from {val_allowed_path}")
        val_dataset = get_npy_validation_dataset(
            config=config,
            data_dir=val_data_dir,
            label_csv=val_label_csv,
            max_patients=val_max_patients,
            seed=config.experiment.seed,
            allowed_volume_names=val_allowed,
        )
        # val_loader is NOT prepared on the npy path (see N3 note above -- preparing
        # would raise BatchSamplerShard under split_batches=True + batch_size=1 + 8 GPU).
        # Match train_loader's robustness knobs: timeout=300 catches a stalled .npy read
        # on the shared FS (val has no NCCL collective to trip the 60-min watchdog, so
        # without a fetch timeout a stall hangs the run indefinitely). persistent_workers
        # / prefetch mirror train and avoid re-spawning across the ~4 val cycles.
        val_loader = DataLoader(
            val_dataset, batch_size=1, num_workers=_nw,
            shuffle=False, pin_memory=True, timeout=300,
            prefetch_factor=3 if _nw > 0 else None,
            persistent_workers=_nw > 0,
        )
        write_to_main_log(accelerator, f"Validation dataset: {len(val_dataset)} .npy files")

    processor = MultiScaleSliceProcessor(config)

    # 3. Load Base ViT Backbone
    write_to_main_log(accelerator, "Loading pre-trained ViT backbone...")
    base_vit = init_dino_evaluiaton_model(config, accelerator)

    # 4. Fine-tuning method: LoRA adapters, partial unfreeze (freeze first N
    #    blocks), or full unfreeze. LoRA attaches PEFT qkv adapters; the unfreeze
    #    methods train the backbone directly (no PEFT).
    finetune_method = getattr(config.experiment, 'finetune_method', 'lora')
    multi_cfg = getattr(config, 'multi_adapter', None)
    adapter_names = ['default']

    if finetune_method == 'lora':
        lora_config = LoraConfig(
            r=config.experiment.lora_r,
            lora_alpha=config.experiment.lora_alpha,
            lora_dropout=config.experiment.lora_dropout,
            target_modules=["qkv"]
        )

        # Check if multi_adapter config exists to initialize PEFT correctly
        if multi_cfg and multi_cfg.get('enabled', False):
            adapter_names = list(multi_cfg.adapters.keys())
            base_vit = get_peft_model(base_vit, lora_config, adapter_name=adapter_names[0])
            for name in adapter_names[1:]:
                base_vit.add_adapter(name, lora_config)
            write_to_main_log(accelerator, f"Initialized Multi-Adapter LoRA: {', '.join(adapter_names)}")
        else:
            # Fallback to standard
            base_vit = get_peft_model(base_vit, lora_config, adapter_name="default")
            write_to_main_log(accelerator, "Initialized Standard Single LoRA Adapter.")
    else:
        # partial_unfreeze / full_unfreeze: backbone trains directly, no adapters.
        write_to_main_log(accelerator, f"Fine-tuning method: {finetune_method} (no LoRA adapters)")

    # 5. Initialize End-to-End Model
    use_patch_pooling = getattr(config.experiment, 'use_patch_pooling', False)
    use_slice_transformer = getattr(config.experiment, 'use_slice_transformer', False)
    slice_transformer_config = getattr(config.experiment, 'slice_transformer', None)
    use_gradient_checkpointing = getattr(config.experiment, 'use_gradient_checkpointing', True)
    model = EndToEndColipri(
        vit_backbone=base_vit,
        colipri_state_dict_path=None,
        input_dim=config.experiment.input_dim,
        pooling_scheme=config.experiment.best_pooling_scheme,
        multi_adapter_config=multi_cfg,
        use_patch_pooling=use_patch_pooling,
        use_slice_transformer=use_slice_transformer,
        slice_transformer_config=slice_transformer_config,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    # Freeze the first N transformer blocks for partial unfreeze. Full unfreeze
    # leaves all backbone params trainable. (freeze_layers no-ops for LoRA, but
    # we only call it for partial_unfreeze here.)
    if finetune_method == 'partial_unfreeze':
        n_freeze = getattr(config.experiment, 'unfreeze_layers', 0)
        freeze_layers(model, accelerator, use_lora=False, num_layers_to_freeze=n_freeze)

    # 6. Optimization Setup
    # Param groups: backbone (LoRA adapters OR unfrozen ViT params) at the
    # backbone lr; the new patch-pooler / slice-transformer modules and the
    # Colipri head at head_lr.
    backbone_params = []
    head_params = []
    arch_params = []
    arch_keys = ('patch_pooler', 'patch_combine', 'slice_transformer')

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'colipri' in n:
            head_params.append(p)
        elif any(k in n for k in arch_keys):
            arch_params.append(p)
        else:
            backbone_params.append(p)

    backbone_lr = getattr(config.experiment, 'backbone_lr', None)
    if finetune_method == 'lora':
        bb_lr = config.experiment.lora_lr
    else:
        bb_lr = backbone_lr if backbone_lr is not None else config.experiment.head_lr

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': bb_lr})
    if arch_params:
        param_groups.append({'params': arch_params, 'lr': config.experiment.head_lr})
    if head_params:
        param_groups.append({'params': head_params, 'lr': config.experiment.head_lr})

    optimizer = optim.AdamW(param_groups, weight_decay=config.experiment.weight_decay)

    # 7. Multi-Adapter Loss and Tracking Setup
    pos_weights = torch.tensor(config.experiment.pos_weights, dtype=torch.float32).to(device)
    accumulation_steps = config.experiment.accumulation_steps

    criterion_dict = {}
    if multi_cfg and multi_cfg.get('enabled', False):
        active_adapters = list(multi_cfg.adapters.keys())
        for adapter_name, adapter_cfg in multi_cfg.adapters.items():
            adapter_pos_weights = pos_weights[adapter_cfg.classes]
            criterion_dict[adapter_name] = nn.BCEWithLogitsLoss(pos_weight=adapter_pos_weights)
    else:
        active_adapters = ['default']
        criterion_dict['default'] = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    best_val_auprc = {a: 0.0 for a in active_adapters}
    epochs_without_improvement = {a: 0 for a in active_adapters}
    early_stopping_patience = getattr(config.experiment, 'early_stopping_patience', 10)

    # 8. DDP
    model, optimizer = accelerator.prepare(model, optimizer)
    # NOTE: train_loader is NOT prepared here. The npy path shards via the
    # explicit DistributedSampler built above; the tar path shards via wds
    # internally. accelerator.prepare(train_loader) is deliberately avoided:
    # under split_batches=True + batch_size=1 + world>1 it wraps with
    # BatchSamplerShard which RAISES ("batch size 1 not a round multiple of
    # num_processes") -- the 1-GPU smoke passes but 8-GPU crashes at startup.
    #
    # npy/Stage-2 val path: do NOT prepare val_loader either. run_validation
    # accumulates logits/labels locally (no all_gather), so a sharded val would
    # make rank 0's best-checkpoint AUPRC cover only ~1/8 of the 300
    # patient-disjoint val set; leaving val_loader unprepared means every rank
    # evals all 300 and rank 0 logs the full-set metric (8x redundant val compute,
    # but val is small). Preparing it would ALSO crash at startup on 8 GPU (same
    # BatchSamplerShard raise). val_loader is ALWAYS the batch_size=1 npy loader
    # (built above), so it must NOT be prepared on either train path: preparing
    # crashes on 8 GPU (above) and run_validation needs no sharding (it
    # accumulates logits/labels locally). Every rank evals the full val set;
    # rank 0 logs the full-set metric (redundant compute, but val is small).

    # 9. LR Scheduler
    steps_per_epoch = getattr(config.experiment, 'steps_per_epoch', 1000)
    total_steps = steps_per_epoch * config.experiment.epochs
    warmup_steps = config.experiment.warmup_steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(config.experiment.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ============================================================
    # CHECKPOINT RESUME LOGIC
    # ============================================================
    global_step = 0
    start_epoch = 0
    skip_batches = 0  # Number of batches to skip within the first epoch

    if args.resume:
        training_state = load_checkpoint(model, optimizer, scheduler, save_dir, accelerator, multi_cfg, finetune_method, device)
        if training_state is not None:
            global_step = training_state.get('global_step', 0)
            start_epoch = training_state.get('epoch', 0)
            skip_batches = training_state.get('batch_idx', 0) + 1  # +1 to skip past the saved batch
            best_val_auprc = training_state.get('best_val_auprc', best_val_auprc)
            epochs_without_improvement = training_state.get('epochs_without_improvement', epochs_without_improvement)
            active_adapters = training_state.get('active_adapters', active_adapters)

            # If we finished the epoch, advance to next epoch and reset skip.
            # WebDataset (IterableDataset) has no __len__: len(train_loader)
            # raises TypeError on the WDS train path. Guard it so resume works
            # for both npy (map-style, has len) and WDS (treat as unbounded ->
            # skip stays in-epoch; the skip loop below consumes the batches).
            try:
                loader_len = len(train_loader)
            except (TypeError, NotImplementedError):
                loader_len = float('inf')
            if skip_batches >= loader_len:
                start_epoch += 1
                skip_batches = 0
            write_to_main_log(accelerator,
                              f"Resuming: epoch={start_epoch + 1}, global_step={global_step}, "
                              f"skip_batches={skip_batches}, active_adapters={active_adapters}")
        else:
            write_to_main_log(accelerator, "No checkpoint found. Starting from scratch.")
    else:
        write_to_main_log(accelerator, "Resume disabled via --no-resume. Starting from scratch.")

    # Determine checkpoint save interval
    checkpoint_every = args.checkpoint_every
    if checkpoint_every is None:
        checkpoint_every = getattr(config.experiment, 'val_every_n_steps', 50)

    # ============================================================
    # SIGNAL HANDLERS FOR GRACEFUL INTERRUPTION
    # ============================================================
    signal.signal(signal.SIGINT, _signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, _signal_handler)  # kill / SLURM preemption

    # 10. Training Loop
    model.train()
    optimizer.zero_grad()

    if accelerator.is_main_process:
        setup_wandb(config, accelerator)

    write_to_main_log(accelerator,
                      f"Starting End-to-End Multi-Adapter Training on {accelerator.num_processes} GPU(s)...")
    write_to_main_log(accelerator,
                      f"Checkpointing every {checkpoint_every} steps. Resume: {args.resume}")

    running_loss = 0.0

    for epoch in range(start_epoch, config.experiment.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # no-op for shuffle=False, but canonical
        if not active_adapters:
            write_to_main_log(accelerator, "All adapters have triggered early stopping. Exiting training loop.")
            break

        for batch_idx, (volumes, labels, filenames) in enumerate(train_loader):

            # Skip batches when resuming within the same epoch
            if epoch == start_epoch and batch_idx < skip_batches:
                continue

            # Check for interruption signal
            global _interrupted
            if _interrupted:
                write_to_main_log(accelerator, "Interruption signal received. Saving checkpoint...")
                training_state = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'best_val_auprc': best_val_auprc,
                    'epochs_without_improvement': epochs_without_improvement,
                    'active_adapters': active_adapters,
                    'all_adapter_names': adapter_names,
                }
                save_checkpoint(model, optimizer, scheduler, training_state, save_dir, accelerator, multi_cfg, finetune_method)
                write_to_main_log(accelerator, "Checkpoint saved. Exiting.")
                if accelerator.is_main_process:
                    wandb.finish()
                sys.exit(0)

            raw_volume = volumes.squeeze(0).to(device)
            # The wds shards carry no 18-class labels (meta['labels'] == []); look up
            # the abnormality labels from the train CSV by volume name. filenames[0]
            # is e.g. "train_679_a_1.npy" -> CSV key "train_679_a_1.nii.gz".
            csv_key = filenames[0].replace('.npy', '.nii.gz')
            if labels_map is not None and csv_key in labels_map.index:
                labels = torch.from_numpy(
                    labels_map.loc[csv_key].values.astype(np.float32)
                ).unsqueeze(0).to(device)
            else:
                write_to_main_log(accelerator, f"WARNING: no train label for {csv_key}; skipping")
                continue

            processed_slices, _ = processor.process_batch(raw_volume)
            processed_slices = processed_slices.unsqueeze(0)

            # Pass only the remaining active adapters
            logits_dict = model(processed_slices, chunk_size=config.experiment.chunk_size,
                                active_adapters=active_adapters)

            total_loss = 0.0

            if multi_cfg and multi_cfg.get('enabled', False):
                for adapter_name in active_adapters:
                    adapter_cfg = multi_cfg.adapters[adapter_name]
                    adapter_labels = labels[:, adapter_cfg.classes]
                    loss = criterion_dict[adapter_name](logits_dict[adapter_name], adapter_labels)
                    total_loss += loss

                # Sum losses without averaging. Each adapter's LoRA parameters only
                # participate in their own forward pass (set_adapter switches which LoRA
                # weights are active), so each adapter's params only receive gradient from
                # their own loss term. Averaging would scale each adapter's gradients by
                # 1/N, effectively reducing the LR. The accumulation_steps division below
                # handles gradient scaling correctly.
            else:
                total_loss = criterion_dict['default'](logits_dict, labels)

            current_loss = total_loss.item()
            running_loss += current_loss
            scaled_loss = total_loss / accumulation_steps
            # Sync gradients only on the final micro-batch of each accumulation
            # window. Without this, DDP allreduces on EVERY backward (~32/step),
            # which both wastes bandwidth and turns any transient I/O stall on one
            # rank into a full collective block (the 1h NCCL-watchdog hang).
            is_last_micro = (batch_idx + 1) % accumulation_steps == 0
            with accelerator.no_sync(model) if not is_last_micro else nullcontext():
                accelerator.backward(scaled_loss)

            if accelerator.is_main_process:
                vram_str = "VRAM: N/A"
                if torch.cuda.is_available():
                    vram_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    vram_rsvd = torch.cuda.memory_reserved(device) / (1024 ** 3)
                    vram_str = f"VRAM Alloc: {vram_alloc:.2f} GB | Rsvd: {vram_rsvd:.2f} GB"

                print(f"Epoch: {epoch + 1}/{config.experiment.epochs} | "
                      f"Batch: {batch_idx + 1} | "
                      f"Global Step: {global_step}/{total_steps} | "
                      f"Active Heads: {len(active_adapters)} | "
                      f"S={raw_volume.shape[0]}->{processed_slices.shape[1]} H={raw_volume.shape[1]} W={raw_volume.shape[2]} | "
                      f"file={filenames[0]} | "
                      f"Loss: {current_loss:.4f} | "
                      f"{vram_str}", flush=True)

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # One-time peak-VRAM log after the first backward (where memory
                # peaks). The model always sees (1, 256, 1, 256, 256) after
                # process_batch, so peak is volume-size-independent -- 2 smoke
                # steps measure it fully. Drives the use_gradient_checkpointing
                # call: a small chunk-32 checkpointed peak means the full run can
                # disable checkpointing (~33% faster, ~8x the peak VRAM).
                if global_step == 1 and torch.cuda.is_available():
                    _peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    write_to_main_log(accelerator,
                        f"Peak VRAM after step 1 (chunk={config.experiment.chunk_size}, "
                        f"grad_ckpt={use_gradient_checkpointing}): {_peak_gb:.2f} GB")

                if accelerator.is_main_process:
                    current_lr = scheduler.get_last_lr()[0]
                    log_metrics_wandb({
                        "train_loss": running_loss / accumulation_steps,
                        "learning_rate": current_lr,
                        "step": global_step,
                        "epoch": epoch + 1,
                    }, step=global_step)
                running_loss = 0.0

                # ---- PERIODIC CHECKPOINT SAVE ----
                if global_step % checkpoint_every == 0:
                    training_state = {
                        'global_step': global_step,
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'best_val_auprc': best_val_auprc,
                        'epochs_without_improvement': epochs_without_improvement,
                        'active_adapters': active_adapters,
                        'all_adapter_names': adapter_names,
                    }
                    save_checkpoint(model, optimizer, scheduler, training_state, save_dir, accelerator, multi_cfg, finetune_method)

                # Validation Phase
                if val_loader is not None and global_step % config.experiment.val_every_n_steps == 0:
                    metrics = run_validation(
                        model, val_loader, processor, criterion_dict, device, config, active_adapters,
                        max_batches=config.experiment.val_max_batches
                    )

                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)

                        # Iterate through a copy of the list so we can safely remove elements
                        for adapter_name in list(active_adapters):
                            val_auprc = metrics[adapter_name]['auprc']
                            val_loss = metrics[adapter_name]['loss']
                            val_auroc = metrics[adapter_name]['auroc']

                            log_metrics_wandb({
                                f"{adapter_name}_val_loss": val_loss,
                                f"{adapter_name}_val_auprc": val_auprc,
                                f"{adapter_name}_val_auroc": val_auroc,
                                "step": global_step,
                            }, step=global_step)

                            write_to_main_log(
                                accelerator,
                                f"[{adapter_name}] Step {global_step}: Val Loss={val_loss:.4f}, "
                                f"Val AUROC={val_auroc:.4f}, Val AUPRC={val_auprc:.4f}"
                            )

                            if val_auprc > best_val_auprc[adapter_name]:
                                best_val_auprc[adapter_name] = val_auprc
                                epochs_without_improvement[adapter_name] = 0

                                # Save localized model states (best model, separate from checkpoint)
                                if multi_cfg and multi_cfg.get('enabled', False):
                                    adapter_save_path = os.path.join(save_dir, f"lora_{adapter_name}")
                                    unwrapped.backbone.save_pretrained(adapter_save_path, adapter_name=adapter_name)

                                    head_save_path = os.path.join(save_dir, f"colipri_{adapter_name}.pth")
                                    torch.save(unwrapped.colipri_heads[adapter_name].state_dict(), head_save_path)
                                    write_to_main_log(accelerator,
                                                      f"New best '{adapter_name}' adapter saved! "
                                                      f"AUPRC={val_auprc:.4f}")
                                else:
                                    trainable_names = set(
                                        n for n, p in unwrapped.named_parameters() if p.requires_grad)
                                    trainable_state_dict = {k: v for k, v in unwrapped.state_dict().items() if
                                                            k in trainable_names}
                                    save_path = os.path.join(save_dir, "e2e_model_best.pth")
                                    torch.save(trainable_state_dict, save_path)
                            else:
                                epochs_without_improvement[adapter_name] += 1
                                write_to_main_log(
                                    accelerator,
                                    f"No AUPRC improvement for '{adapter_name}' "
                                    f"({epochs_without_improvement[adapter_name]}/{early_stopping_patience})."
                                )

                                if epochs_without_improvement[adapter_name] >= early_stopping_patience:
                                    write_to_main_log(accelerator,
                                                      f"Early stopping triggered for '{adapter_name}'. "
                                                      f"Removing from active training loop.")
                                    active_adapters.remove(adapter_name)

                                    # Turn off gradients for this specific head to be safe
                                    if multi_cfg and multi_cfg.get('enabled', False):
                                        for p in unwrapped.colipri_heads[adapter_name].parameters():
                                            p.requires_grad = False

                # Safety cap: stop at total_steps so the smoke exits cleanly
                # (steps_per_epoch=2, epochs=1 -> total_steps=2) instead of
                # iterating all 4,642 train batches into the SLURM timeout, and
                # the full run can't over-train past the LR schedule horizon.
                # (Full run reaches ~216 opt steps in 3 epochs < total_steps=219,
                # so this is a no-op there; fires only for the smoke.)
                if global_step >= total_steps:
                    write_to_main_log(accelerator, f"Reached total_steps={total_steps}; ending training.")
                    break

        # End of epoch fallback save (best models, not full checkpoints)
        if accelerator.is_main_process and active_adapters:
            unwrapped = accelerator.unwrap_model(model)
            if multi_cfg and multi_cfg.get('enabled', False):
                for adapter_name in active_adapters:
                    adapter_save_path = os.path.join(save_dir, f"lora_{adapter_name}_epoch_{epoch + 1}")
                    unwrapped.backbone.save_pretrained(adapter_save_path, adapter_name=adapter_name)
                    head_save_path = os.path.join(save_dir, f"colipri_{adapter_name}_epoch_{epoch + 1}.pth")
                    torch.save(unwrapped.colipri_heads[adapter_name].state_dict(), head_save_path)
            else:
                trainable_names = set(n for n, p in unwrapped.named_parameters() if p.requires_grad)
                trainable_state_dict = {k: v for k, v in unwrapped.state_dict().items() if k in trainable_names}
                save_path = os.path.join(save_dir, f"e2e_model_epoch_{epoch + 1}.pth")
                torch.save(trainable_state_dict, save_path)
            write_to_main_log(accelerator, f"Saved epoch {epoch + 1} checkpoint for active adapters.")

        # Outer half of the total_steps safety cap (inner break exits the batch
        # loop; this exits the epoch loop). See comment above.
        if global_step >= total_steps:
            break

        # Reset skip_batches after the first resumed epoch
        skip_batches = 0

    if accelerator.is_main_process:
        write_to_main_log(accelerator, "Training complete. Best validation AUPRCs:")
        for a_name, a_auprc in best_val_auprc.items():
            write_to_main_log(accelerator, f"  {a_name}: {a_auprc:.4f}")
        wandb.finish()


if __name__ == "__main__":
    main()

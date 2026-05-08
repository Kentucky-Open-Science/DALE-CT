import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
import wandb

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataloaders.datasetloader_ctrate_multiscale import CTMultiScaleDataset, MultiScaleSliceProcessor, get_wds_dataset
from utils.config import load_config, load_model_configs
from utils.dino_utils import init_dino_evaluiaton_model

import random


def set_seed(seed=25):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ==========================================
# MODEL DEFINITION
# ==========================================
class E2ECTClassifier(nn.Module):
    def __init__(self, config, accelerator, embed_dim, num_classes=18):
        super().__init__()
        self.backbone = init_dino_evaluiaton_model(config, accelerator)
        for param in self.backbone.parameters():
            param.requires_grad = True

        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=True)

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.extract_patches = getattr(config.inference, 'extract_patches', False)

    def forward(self, x, lengths=None):
        """
        x: Batched tensor of shape (B, S, C, H, W) where S is max_slices (padded)
        lengths: List of the true sequence lengths for each volume in the batch
        """
        B, S, C, H, W = x.shape

        # Flatten batch and depth dimensions for the backbone
        x_flat = x.view(B * S, C, H, W)

        unwrapped_model = getattr(self.backbone, '_orig_mod', getattr(self.backbone, 'module', self.backbone))
        is_timm = hasattr(unwrapped_model, 'forward_features')

        # Run the massive parallel forward pass
        if is_timm and self.extract_patches:
            out = unwrapped_model.forward_features(x_flat)
            if out.ndim == 4:
                out = out.flatten(2).transpose(1, 2)
            cls_tokens = out[:, 0, :] if not self.extract_patches else out.mean(dim=1)
        else:
            out = self.backbone(x_flat)
            if hasattr(out, 'last_hidden_state'):
                cls_tokens = out.last_hidden_state[:, 0, :]
            else:
                cls_tokens = out

        # Reshape tokens back to (B, S, D)
        cls_tokens = cls_tokens.view(B, S, -1)

        # Global Average Pooling per patient, IGNORING PADDED SLICES
        if lengths is not None:
            volume_embeddings = []
            for i in range(B):
                actual_s = lengths[i]
                # Mean over the true valid slices only
                vol_emb = cls_tokens[i, :actual_s, :].mean(dim=0, keepdim=True)
                volume_embeddings.append(vol_emb)
            volume_embeddings = torch.cat(volume_embeddings, dim=0)  # Shape: (B, D)
        else:
            # Fallback if no lengths are provided
            volume_embeddings = cls_tokens.mean(dim=1)

        logits = self.classifier(volume_embeddings)  # Shape: (B, num_classes)
        return logits


# ==========================================
# DATA LOADING UTILS
# ==========================================
def get_dataset(config, mode="train"):
    target_config = getattr(config, mode)
    dataset_format = getattr(target_config, 'dataset_format', 'npy')

    if dataset_format == 'webdataset':
        mock_config = argparse.Namespace(validation=target_config, dataset=config.dataset)
        return get_wds_dataset(mock_config)
    else:
        data_dir = target_config.data_dir
        label_csv = getattr(target_config, 'label_csv', None)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        return CTMultiScaleDataset(config=config, data_dir=data_dir, label_csv=label_csv)


def center_crop_to_multiple(tensor, patch_size):
    h, w = tensor.shape[-2], tensor.shape[-1]
    new_h, new_w = (h // patch_size) * patch_size, (w // patch_size) * patch_size
    if new_h == h and new_w == w: return tensor
    top, left = (h - new_h) // 2, (w - new_w) // 2
    return tensor[..., top:top + new_h, left:left + new_w]


def variable_depth_collate(batch):
    volumes = [item[0] for item in batch]
    labels = torch.stack([torch.tensor(item[1], dtype=torch.float32) if not isinstance(item[1], torch.Tensor) else item[
        1].clone().detach().float() for item in batch])
    filenames = [item[2] for item in batch]
    return volumes, labels, filenames


# ==========================================
# EVALUATION METRICS
# ==========================================
def get_optimal_f1_thresholds(y_true, y_prob):
    thresholds_out = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            thresholds_out.append(0.5)
            continue
        prec, rec, thresholds = precision_recall_curve(y_true[:, i], y_prob[:, i])
        f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-6)
        best_thresh = thresholds[np.argmax(f1_scores)]
        thresholds_out.append(best_thresh)
    return np.array(thresholds_out)


@torch.no_grad()
def evaluate(model, dataloader, criterion, accelerator, processor, config, class_names, fixed_thresholds=None):
    model.eval()
    total_loss = 0
    all_targets, all_probs = [], []
    patch_size = load_model_configs(config.train.model_type).patch_size
    max_slices = getattr(config.experiment, 'max_slices', 256)

    val_steps = getattr(config.validation, 'steps_per_epoch', None)
    dataset_format = getattr(config.validation, 'dataset_format', 'npy')

    if dataset_format == 'webdataset' and val_steps is None:
        raise ValueError("You MUST define 'steps_per_epoch' in config.validation when using WebDataset.")

    num_batches = len(dataloader) if val_steps is None else val_steps
    val_iter = iter(dataloader)

    # Use explicit stepping to prevent DDP deadlock during evaluation
    for step in tqdm(range(num_batches), desc="Evaluating", disable=not accelerator.is_local_main_process):
        try:
            batch = next(val_iter)
        except StopIteration:
            if dataset_format == 'webdataset':
                val_iter = iter(dataloader)
                batch = next(val_iter)
            else:
                break

        volumes_list, labels, filenames = batch
        labels = labels.to(accelerator.device)

        primary_views = []
        valid_labels = []
        lengths = []

        # PREPARE BATCH
        for i in range(len(volumes_list)):
            raw_volume = volumes_list[i].squeeze(0).to(accelerator.device)
            current_label = labels[i:i + 1]

            primary_view, _ = processor.process_batch(raw_volume, filename=filenames[i])
            primary_view = center_crop_to_multiple(primary_view, patch_size=patch_size)

            S = primary_view.shape[0]
            true_length = min(S, max_slices)
            lengths.append(true_length)

            if S > max_slices:
                indices = torch.linspace(0, S - 1, max_slices).long()
                primary_view = primary_view[indices]
            elif S < max_slices:
                pad_size = max_slices - S
                padding = torch.zeros((pad_size, *primary_view.shape[1:]), dtype=primary_view.dtype,
                                      device=primary_view.device)
                primary_view = torch.cat([primary_view, padding], dim=0)

            primary_views.append(primary_view)
            valid_labels.append(current_label)

        # Skip iteration safely if valid_labels is empty to avoid crashing torch.stack
        if len(valid_labels) == 0:
            continue

        # RUN INFERENCE ON FULL BATCH
        batched_volumes = torch.stack(primary_views, dim=0)
        batched_labels = torch.cat(valid_labels, dim=0)

        with accelerator.autocast():
            logits = model(batched_volumes, lengths=lengths)
            loss = criterion(logits, batched_labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits)

        # GATHER FOR METRICS - Prevents DDP duplicate padding issues
        gathered_probs = accelerator.gather_for_metrics(probs)
        gathered_targets = accelerator.gather_for_metrics(batched_labels)

        all_probs.append(gathered_probs.cpu().numpy())
        all_targets.append(gathered_targets.cpu().numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_targets)

    best_thresholds = fixed_thresholds if fixed_thresholds is not None else get_optimal_f1_thresholds(y_true, y_prob)
    y_pred = (y_prob >= best_thresholds).astype(int)

    metrics = {"val_loss": total_loss / max(1, num_batches)}

    try:
        metrics["val_macro_auc"] = roc_auc_score(y_true, y_prob, average="macro")
        metrics["val_macro_auprc"] = average_precision_score(y_true, y_prob, average="macro")
    except ValueError:
        metrics["val_macro_auc"], metrics["val_macro_auprc"] = 0.0, 0.0

    macro_f1, macro_ba = [], []
    for i, name in enumerate(class_names):
        y_t, y_p_prob, y_p_bin = y_true[:, i], y_prob[:, i], y_pred[:, i]
        if len(np.unique(y_t)) < 2: continue

        tn, fp, fn, tp = confusion_matrix(y_t, y_p_bin, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-6)
        spec = tn / (tn + fp + 1e-6)
        ppv = tp / (tp + fp + 1e-6)

        f1 = 2 * (ppv * sens) / (ppv + sens + 1e-6)
        macro_f1.append(f1)
        macro_ba.append((sens + spec) / 2.0)

    metrics["val_macro_f1"] = np.mean(macro_f1) if macro_f1 else 0.0
    metrics["val_macro_ba"] = np.mean(macro_ba) if macro_ba else 0.0
    metrics["thresholds"] = best_thresholds

    return metrics


# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def main():
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to unified config")
    args = parser.parse_args()

    config = load_config(args.config_file)

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[process_group_kwargs])

    # Extract Experiment Hyperparameters from Config
    learning_rate = getattr(config.experiment, 'learning_rate', 0.001)
    weight_decay = getattr(config.experiment, 'weight_decay', 0.05)
    max_epochs = getattr(config.experiment, 'max_epochs', 30)
    warmup_epochs = getattr(config.experiment, 'warmup_epochs', 5)
    patience_limit = getattr(config.experiment, 'patience', 10)
    min_lr_ratio = getattr(config.experiment, 'min_lr_ratio', 0.1)
    log_freq = getattr(config.experiment, 'log_freq', 1000)
    max_slices = getattr(config.experiment, 'max_slices', 256)
    embed_dim = getattr(config.experiment, 'embed_dim', 1024)
    num_classes = getattr(config.experiment, 'num_classes', 18)

    train_batch_size = getattr(config.train, 'batch_size', 1)
    val_batch_size = getattr(config.validation, 'batch_size', 1)

    if accelerator.is_local_main_process:
        os.makedirs(config.output_folders.main_output, exist_ok=True)
        run_id_path = os.path.join(config.output_folders.main_output, "wandb_run_id.txt")

        if os.path.exists(run_id_path):
            with open(run_id_path, "r") as f:
                run_id = f.read().strip()
        else:
            run_id = wandb.util.generate_id()
            with open(run_id_path, "w") as f:
                f.write(run_id)

        wandb_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "warmup_epochs": warmup_epochs,
            "patience": patience_limit,
            "min_lr_ratio": min_lr_ratio,
            "log_freq": log_freq,
            "max_slices": max_slices,
            "embed_dim": embed_dim,
            "num_classes": num_classes,
            "train_batch_size": train_batch_size,
            "model_type": getattr(config.train, 'model_type', "unknown"),
        }
        wandb.init(
            project=getattr(config.wandb, 'project', 'CT-RATE-E2E'),
            name=getattr(config.wandb, 'name', 'LeJEPA_E2E_GAP'),
            mode=getattr(config.wandb, 'mode', 'online'),
            config=wandb_config,
            id=run_id,
            resume="allow"
        )

    # Data Setup
    train_dataset = get_dataset(config, mode="train")
    val_dataset = get_dataset(config, mode="validation")
    class_names = train_dataset.class_names if hasattr(train_dataset, 'class_names') else [f"class_{i}" for i in
                                                                                           range(num_classes)]

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=getattr(config.train, 'num_workers', 8),
        collate_fn=variable_depth_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=getattr(config.validation, 'num_workers', 8),
        collate_fn=variable_depth_collate
    )

    dataset_format = getattr(config.train, 'dataset_format', 'npy')
    if dataset_format != 'webdataset':
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    processor = MultiScaleSliceProcessor(config, output_dir=config.output_folders.main_output)
    patch_size = load_model_configs(config.train.model_type).patch_size

    # Model Setup
    model = E2ECTClassifier(config, accelerator, embed_dim=embed_dim, num_classes=num_classes)

    # Optimizer Parameter Groups
    head_params, alpha_params, base_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name:
            head_params.append(param)
        elif 'alpha' in name:
            alpha_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': learning_rate},
        {'params': alpha_params, 'lr': learning_rate * 0.3},
        {'params': head_params, 'lr': learning_rate * 3.0}
    ], weight_decay=weight_decay)

    calculated_weights = [7.1040, 2.5246, 7.8826, 12.8186, 2.9209, 5.9840, 2.8580, 4.1687, 2.8448, 1.2051, 1.7066,
                          2.7453, 7.2645, 11.9601, 8.4810, 4.6676, 8.9639, 11.5899]
    pos_weights = torch.tensor(calculated_weights, dtype=torch.float32)
    pos_weights = torch.clamp(pos_weights, max=10.0).to(accelerator.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Scheduler Setup
    steps_per_epoch = getattr(config.experiment, 'steps_per_epoch', None)

    if steps_per_epoch is None:
        try:
            steps_per_epoch = len(train_loader)
        except TypeError:
            raise ValueError("You are using WebDataset. You MUST define 'steps_per_epoch' in config.experiment.")

    total_steps = max_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Auto-resume
    checkpoint_dir = os.path.join(config.output_folders.main_output, "latest_checkpoint")
    starting_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    step = 0

    if os.path.exists(checkpoint_dir):
        if accelerator.is_local_main_process:
            print(f"🔄 Resuming from checkpoint: {checkpoint_dir}")

        accelerator.load_state(checkpoint_dir)
        custom_state_path = os.path.join(checkpoint_dir, "custom_state.pt")
        if os.path.exists(custom_state_path):
            custom_state = torch.load(custom_state_path, map_location="cpu")
            starting_epoch = custom_state['epoch']
            best_val_loss = custom_state['best_val_loss']
            patience_counter = custom_state['patience_counter']
            step = custom_state['step']

            if accelerator.is_local_main_process:
                print(f"📈 Resumed at Epoch {starting_epoch + 1}, Global Step {step}.")

    # --- Training Loop ---
    train_iter = iter(train_loader)

    for epoch in range(starting_epoch, max_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{max_epochs}",
                            disable=not accelerator.is_local_main_process)

        # Force lockstep iterations across all DDP ranks
        for epoch_steps in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                # If a WebDataset shard exhausts early on one rank, wrap around safely!
                train_iter = iter(train_loader)
                batch = next(train_iter)

            volumes_list, labels, filenames = batch
            labels = labels.to(accelerator.device)

            optimizer.zero_grad()

            primary_views = []
            valid_labels = []
            lengths = []

            for i in range(len(volumes_list)):
                raw_volume = volumes_list[i].squeeze(0).to(accelerator.device)
                current_label = labels[i:i + 1]

                primary_view, _ = processor.process_batch(raw_volume, filename=filenames[i])
                primary_view = center_crop_to_multiple(primary_view, patch_size=patch_size)

                # Standardize shape to max_slices
                S = primary_view.shape[0]
                lengths.append(min(S, max_slices))

                if S > max_slices:
                    indices = torch.linspace(0, S - 1, max_slices).long()
                    primary_view = primary_view[indices]
                elif S < max_slices:
                    pad_size = max_slices - S
                    padding = torch.zeros((pad_size, *primary_view.shape[1:]), dtype=primary_view.dtype,
                                          device=primary_view.device)
                    primary_view = torch.cat([primary_view, padding], dim=0)

                primary_views.append(primary_view)
                valid_labels.append(current_label)

            # Build a standardized 5D tensor (B, max_slices, C, H, W)
            batched_volumes = torch.stack(primary_views, dim=0)
            batched_labels = torch.cat(valid_labels, dim=0)

            with accelerator.autocast():
                logits = model(batched_volumes, lengths=lengths)
                loss = criterion(logits, batched_labels)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step += 1
            progress_bar.update(1)

            if accelerator.is_local_main_process:
                current_lr = scheduler.get_last_lr()[0]
                vram_reserved = torch.cuda.memory_reserved(accelerator.device) / (
                            1024 ** 3) if torch.cuda.is_available() else 0.0

                tqdm.write(
                    f"[Epoch {epoch + 1} | Step {epoch_steps + 1}/{steps_per_epoch}] Loss: {loss.item():.4f} | LR: {current_lr:.6e} | Rank 0 VRAM: {vram_reserved:.2f} GB")

                if step % log_freq == 0:
                    wandb.log({"train/loss_step": loss.item(), "lr_base": current_lr, "vram_gb": vram_reserved},
                              step=step)

        # --- End of Epoch Evaluation ---
        metrics = evaluate(model, val_loader, criterion, accelerator, processor, config, class_names)

        # 1. Sync the validation loss across all ranks so they all make the same patience decision
        local_val_loss = torch.tensor(metrics["val_loss"], device=accelerator.device)
        avg_val_loss = accelerator.reduce(local_val_loss, reduction="mean").item()
        metrics["val_loss"] = avg_val_loss

        if accelerator.is_local_main_process:
            avg_train_loss = running_loss / max(1, steps_per_epoch)
            wandb.log({
                "train/epoch_loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/macro_auc": metrics["val_macro_auc"],
                "val/macro_auprc": metrics["val_macro_auprc"],
                "val/macro_f1": metrics["val_macro_f1"],
                "val/macro_ba": metrics["val_macro_ba"],
                "epoch": epoch + 1
            })

            print(f"\n--- Epoch {epoch + 1} Summary ---")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Val AUC: {metrics['val_macro_auc']:.4f} | Val AUPRC: {metrics['val_macro_auprc']:.4f}")
            print(f"Val F1: {metrics['val_macro_f1']:.4f} | Val BA: {metrics['val_macro_ba']:.4f}\n")

        # 2. Update patience counter on ALL ranks
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            patience_counter = 0

            if accelerator.is_local_main_process:
                unwrapped = accelerator.unwrap_model(model)
                os.makedirs(config.output_folders.main_output, exist_ok=True)
                torch.save(unwrapped.state_dict(),
                           os.path.join(config.output_folders.main_output, "best_loss_model.pth"))
                np.save(os.path.join(config.output_folders.main_output, "best_thresholds.npy"), metrics["thresholds"])
                print("💾 Saved new best model based on validation loss!")
        else:
            patience_counter += 1
            if accelerator.is_local_main_process:
                print(f"⚠️ No improvement. Patience: {patience_counter}/{patience_limit}")

        accelerator.wait_for_everyone()
        accelerator.save_state(checkpoint_dir)

        if accelerator.is_local_main_process:
            torch.save({
                'epoch': epoch + 1,
                'step': step,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }, os.path.join(checkpoint_dir, "custom_state.pt"))
            print(f"💾 Epoch {epoch + 1} complete. Saved 'latest_checkpoint' for auto-resume.")

        # 3. All ranks break together
        if patience_counter >= patience_limit:
            if accelerator.is_local_main_process:
                print("🛑 Early stopping triggered!")
            break

    if accelerator.is_local_main_process:
        wandb.finish()
        print(f"✅ Training Complete. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
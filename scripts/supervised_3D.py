import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from accelerate import Accelerator
import wandb

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataloaders.datasetloader_ctrate_multiscale import CTMultiScaleDataset, MultiScaleSliceProcessor, get_wds_dataset
from utils.config import load_config, load_model_configs
from utils.dino_utils import init_dino_evaluiaton_model


# ==========================================
# MODEL DEFINITION
# ==========================================
class E2ECTClassifier3D(nn.Module):
    def __init__(self, config, accelerator, embed_dim, num_classes=18):
        super().__init__()
        # Load backbone and UNFREEZE
        self.backbone = init_dino_evaluiaton_model(config, accelerator)
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Enable gradient checkpointing for memory safety
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=True)
        elif hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (S, C, H, W) - A batch of slices from a SINGLE volume.
        """
        # 1. Reshape to 3D format: (Batch, Channels, Depth, Height, Width)
        # Since batch_size=1 in the DataLoader, S represents our Depth.
        # (S, C, H, W) -> (1, C, S, H, W)
        x_3d = x.permute(1, 0, 2, 3).unsqueeze(0)

        # Robustly identify the unwrapped TIMM adapter
        unwrapped_model = getattr(self.backbone, '_orig_mod', getattr(self.backbone, 'module', self.backbone))

        # 2. Extract features using the 3D Adapter
        out = unwrapped_model.forward_features(x_3d)

        # 3. Isolate the 3D [CLS] Token
        # out shape is (Batch, Tokens, Dim)
        cls_token = out[:, 0, :]  # Shape: (1, D)

        # 4. Classify based solely on the volume-aware CLS token
        logits = self.classifier(cls_token)
        return logits

# ==========================================
# DATA LOADING UTILS
# ==========================================
def get_dataset(config, mode="train"):
    # Temporarily point validation config to the right section based on mode
    target_config = getattr(config, mode)
    dataset_format = getattr(target_config, 'dataset_format', 'npy')

    if dataset_format == 'webdataset':
        # Create a mock config to satisfy get_wds_dataset which expects 'validation'
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

    for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        volumes, labels, filenames = batch
        raw_volume = volumes.squeeze(0).to(accelerator.device)
        labels = labels.to(accelerator.device)

        # --- STRICT 3D SLICE SAMPLING ---
        max_slices = getattr(config.experiment, 'max_slices', 128)
        num_slices = raw_volume.shape[0]
        indices = torch.linspace(0, num_slices - 1, max_slices).long()
        raw_volume = raw_volume[indices]
        # -------------------------------

        # Dynamic morphology crop & resize
        primary_view, _ = processor.process_batch(raw_volume, filename=filenames[0])
        primary_view = center_crop_to_multiple(primary_view, patch_size=patch_size)

        # Forward
        with accelerator.autocast():
            logits = model(primary_view)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits)

        all_probs.append(accelerator.gather(probs).cpu().numpy())
        all_targets.append(accelerator.gather(labels).cpu().numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_targets)

    best_thresholds = fixed_thresholds if fixed_thresholds is not None else get_optimal_f1_thresholds(y_true, y_prob)
    y_pred = (y_prob >= best_thresholds).astype(int)
    metrics = {"val_loss": total_loss / len(dataloader)}

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to unified config")
    args = parser.parse_args()

    config = load_config(args.config_file)
    accelerator = Accelerator(mixed_precision="bf16")

    # Extract Experiment Hyperparameters from Config
    learning_rate = getattr(config.experiment, 'learning_rate', 1e-4)
    total_steps = getattr(config.experiment, 'total_steps', 15000)
    eval_freq = getattr(config.experiment, 'eval_freq', 2500)
    max_slices = getattr(config.experiment, 'max_slices', 256)
    embed_dim = getattr(config.experiment, 'embed_dim', 1024)
    num_classes = getattr(config.experiment, 'num_classes', 18)

    if accelerator.is_local_main_process:
        wandb_config = {
            "learning_rate": learning_rate,
            "total_steps": total_steps,
            "eval_freq": eval_freq,
            "max_slices": max_slices,
            "embed_dim": embed_dim,
            "num_classes": num_classes,
            "model_type": getattr(config.train, 'model_type', "unknown"),
        }
        wandb.init(
            project=getattr(config.wandb, 'project', 'CT-RATE-E2E'),
            name=getattr(config.wandb, 'name', 'LeJEPA_E2E_GAP'),
            mode=getattr(config.wandb, 'mode', 'online'),
            config=wandb_config
        )

    # Data Setup (Batch size MUST be 1 since volume depths vary)
    train_dataset = get_dataset(config, mode="train")
    val_dataset = get_dataset(config, mode="validation")
    class_names = train_dataset.class_names if hasattr(train_dataset, 'class_names') else [f"class_{i}" for i in
                                                                                           range(num_classes)]

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=getattr(config.train, 'num_workers', 8))
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=getattr(config.validation, 'num_workers', 8))

    # Do not let accelerate prepare WebDataset, only prepare standard iterables
    dataset_format = getattr(config.train, 'dataset_format', 'npy')
    if dataset_format != 'webdataset':
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    # Processor handles physical normalization (div1000) & batched RoI align cropping
    processor = MultiScaleSliceProcessor(config, output_dir=config.output_folders.main_output)
    patch_size = load_model_configs(config.train.model_type).patch_size

    # Model Setup
    model = E2ECTClassifier3D(config, accelerator, embed_dim=embed_dim, num_classes=num_classes)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Training Loop
    best_auprc = 0.0
    step = 0
    train_iter = iter(train_loader)

    model.train()
    running_loss = 0.0

    progress_bar = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        volumes, labels, filenames = batch
        raw_volume = volumes.squeeze(0).to(accelerator.device)
        labels = labels.to(accelerator.device)

        # --- STRICT 3D SLICE SAMPLING ---
        # Force the volume to exactly match max_slices (frames_per_clip)
        num_slices = raw_volume.shape[0]
        indices = torch.linspace(0, num_slices - 1, max_slices).long()
        raw_volume = raw_volume[indices]
        # --------------------------------

        # Dynamic pre-processing via MultiScaleSliceProcessor
        primary_view, _ = processor.process_batch(raw_volume, filename=filenames[0])
        primary_view = center_crop_to_multiple(primary_view, patch_size=patch_size)

        optimizer.zero_grad()

        with accelerator.autocast():
            logits = model(primary_view)
            loss = criterion(logits, labels)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1
        progress_bar.update(1)

        # Evaluation
        if step % eval_freq == 0 or step == total_steps:
            metrics = evaluate(model, val_loader, criterion, accelerator, processor, config, class_names)

            if accelerator.is_local_main_process:
                avg_train_loss = running_loss / eval_freq
                wandb.log({
                    "train/loss": avg_train_loss,
                    "val/loss": metrics["val_loss"],
                    "val/auprc": metrics["val_macro_auprc"],
                    "val/auroc": metrics["val_macro_auc"],
                    "val/f1": metrics["val_macro_f1"],
                    "lr": scheduler.get_last_lr()[0]
                }, step=step)

                print(f"\nStep {step} | Train Loss: {avg_train_loss:.4f} | Val AUPRC: {metrics['val_macro_auprc']:.4f}")

                if metrics["val_macro_auprc"] > best_auprc:
                    best_auprc = metrics["val_macro_auprc"]
                    unwrapped = accelerator.unwrap_model(model)
                    os.makedirs(config.output_folders.main_output, exist_ok=True)
                    torch.save(unwrapped.state_dict(),
                               os.path.join(config.output_folders.main_output, "best_e2e_model.pth"))
                    np.save(os.path.join(config.output_folders.main_output, "best_thresholds.npy"),
                            metrics["thresholds"])
                    print("💾 Saved new best model!")

            running_loss = 0.0
            model.train()

    if accelerator.is_local_main_process:
        wandb.finish()
        print(f"✅ Training Complete. Best AUPRC: {best_auprc:.4f}")


if __name__ == "__main__":
    main()
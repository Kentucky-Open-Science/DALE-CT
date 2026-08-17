"""
LeJEPA Self-Supervised Pre-Training Entry Point for CT-RATE-huggingface-downloads.

Usage:
    python train_lejepa.py --train_config_file configs/pretrain_lejepa_0.yaml
"""
import argparse

from dataloaders.datasetloader_web_ctrate import CTWebDatasetLoader
from dataloaders.datasetloader_multisource_zarr import CTMultisourceZarrLoader
from lejepa_core.main_lejepa_trainer import Trainer as MainTrainer
from utils.config import setup as setup_config
from utils.dist_utils import (
    setup_accelerate_seed,
    initialize_fsdp_accelerator_from_config,
    initialize_ddp_accelerator_from_config,
    print_cluster_info,
)
from utils.logger_utils import setup_accelerate_logger, write_to_main_log
from utils.wandb_utils import setup_wandb

import torch.distributed as dist


def main(config_file):
    config = setup_config(config_file=config_file)

    if config.distribution.type == 'fsdp':
        accelerator = initialize_fsdp_accelerator_from_config(config=config)
    else:
        accelerator = initialize_ddp_accelerator_from_config(config=config)

    accelerator.wait_for_everyone()
    setup_accelerate_seed(accelerator=accelerator, config=config)
    print_cluster_info(accelerator=accelerator)
    setup_accelerate_logger(accelerator, config)

    if accelerator.is_main_process:
        if "wandb" in config and config.wandb is not None:
            write_to_main_log(accelerator=accelerator, result="Starting WandB")
            setup_wandb(config, accelerator)
        write_to_main_log(accelerator=accelerator, result="Starting LeJEPA training on CT-RATE-huggingface-downloads")
        write_to_main_log(accelerator=accelerator, result=f"Running with {accelerator.num_processes} processes")
        write_to_main_log(accelerator=accelerator, result=f"Mixed precision: {accelerator.mixed_precision}")

    actual_dist_type = accelerator.state.distributed_type
    device = accelerator.device
    world_size = accelerator.num_processes

    write_to_main_log(accelerator=accelerator, result=f"Accelerator initialized. Device: {device}, World Size: {world_size}")
    write_to_main_log(accelerator=accelerator, result=f"Enabled Distributed Strategy: {actual_dist_type}")

    dataloader_type = getattr(config.train, 'dataloader_type', 'web_ctrate')
    if dataloader_type == 'multisource_zarr':
        train_data = CTMultisourceZarrLoader(cfg=config)
    elif dataloader_type == 'web_ctrate':
        train_data = CTWebDatasetLoader(cfg=config)
    else:
        raise ValueError(f"Unknown dataloader_type: {dataloader_type}")
    trainer = MainTrainer(config=config, accelerator=accelerator, dataset=train_data)
    trainer.train()

    if accelerator.is_main_process:
        if "wandb" in config and config.wandb is not None:
            import wandb
            if wandb.run is not None:
                wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LeJEPA Self-Supervised Pre-Training on CT-RATE-huggingface-downloads')
    parser.add_argument('--train_config_file', type=str, dest='train_config_file',
                        default='ssl_default_config', help='Configuration file for training parameters')
    args = parser.parse_args()
    config_file = args.train_config_file
    main(config_file=config_file)

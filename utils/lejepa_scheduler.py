"""
LeJEPA Learning Rate and Weight Decay Scheduler.

Handles cosine LR scheduling with warmup, optional WeightWatcher monitoring
(disabled by default), and early stopping logic.
"""
import os
import numpy as np
import torch
import pandas as pd
import torch.distributed as dist
import matplotlib.pyplot as plt

from utils.dino_utils import init_lejepa_schedulers
from utils.global_state import GlobalState
from utils.logger_utils import write_to_main_log


class LeJEPAScheduler:
    """
    Manages learning rate and weight decay schedules for LeJEPA training.

    Optionally integrates with WeightWatcher for training health monitoring,
    but this is disabled by default and requires the weightwatcher package.
    """

    def __init__(self, config, accelerator, optimizer, model):
        self.config = config
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.model = model

        self.lr_schedule, self.wd_schedule = init_lejepa_schedulers(config=self.config)
        self.max_iterations = getattr(self.config.train, 'max_iterations', len(self.lr_schedule))

        # WeightWatcher is optional and disabled by default
        self.use_ww = False
        self.enable_early_stop = False
        self.save_best = False
        self.ww_monitor = None

        self.stats_history = []
        self.early_stop_triggered = False

        self.best_checkpoint_info = {
            'iteration': None,
            'overall_score': float('-inf'),
            'global_weighted_alpha': None
        }

    def step(self, iteration):
        idx = min(iteration, len(self.lr_schedule) - 1)
        base_lr = self.lr_schedule[idx]
        base_wd = self.wd_schedule[idx]

        is_best = False

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = base_lr
            if i == 0:
                param_group["weight_decay"] = base_wd

        self.accelerator.wait_for_everyone()
        return self.early_stop_triggered, is_best

    def get_best_checkpoint_info(self):
        return self.best_checkpoint_info

    def plot_training_health(self):
        """Plot training health curves (only when WeightWatcher is active)."""
        if not self.accelerator.is_main_process or not self.stats_history:
            return

        output_dir = GlobalState.get('training_results_dir')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(self.stats_history)

        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        iters = df['iteration']

        # 1. LR vs Global Alpha
        ax1 = axes[0, 0]
        color = 'tab:purple'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Learning Rate', color=color)
        ax1.plot(iters, df['base_lr'], color=color, label='LR')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log')
        ax1_r = ax1.twinx()
        color = 'tab:green'
        ax1_r.set_ylabel('Global Weighted Alpha', color=color)
        ax1_r.plot(iters, df['global_weighted_alpha'], color=color, linewidth=2, label='Alpha')
        ax1_r.tick_params(axis='y', labelcolor=color)
        ax1_r.axhline(y=2.5, color='orange', linestyle='--', alpha=0.5, label='Target (2.5)')
        ax1.set_title('1. LR vs Global Weighted Alpha')

        # 2. LR vs Avg Layer Alpha
        ax2 = axes[0, 1]
        color = 'tab:purple'
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Learning Rate', color=color)
        ax2.plot(iters, df['base_lr'], color=color, label='LR')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yscale('log')
        ax2_r = ax2.twinx()
        color = 'tab:blue'
        ax2_r.set_ylabel('Avg Layer Alpha', color=color)
        ax2_r.plot(iters, df['avg_layer_alpha'], color=color, linewidth=2, label='Avg Alpha')
        ax2_r.tick_params(axis='y', labelcolor=color)
        ax2.set_title('2. LR vs Avg Layer Alpha')

        # 3. Overall Score
        ax3 = axes[1, 0]
        ax3.plot(iters, df['overall_score'], 'b-', linewidth=2)
        ax3.set_title('3. Overall Score (Combined)')
        ax3.set_ylabel('Score')
        ax3.set_ylim(-55, 105)
        ax3.grid(True, alpha=0.3)

        # 4. Avg Layer Score
        ax4 = axes[1, 1]
        ax4.plot(iters, df['avg_layer_score'], 'c-', linewidth=2)
        ax4.set_title('4. Average Layer Score')
        ax4.set_ylabel('Score')
        ax4.set_ylim(-55, 105)
        ax4.grid(True, alpha=0.3)

        # 5. Global Score
        ax5 = axes[2, 0]
        ax5.plot(iters, df['global_score'], 'g-', linewidth=2)
        ax5.set_title('5. Global Weighted Alpha Score')
        ax5.set_ylabel('Score')
        ax5.set_ylim(-55, 105)
        ax5.grid(True, alpha=0.3)

        axes[2, 1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_health_detailed.png'), dpi=150)
        plt.close()
        write_to_main_log(self.accelerator, f"Detailed plots saved to {output_dir}")

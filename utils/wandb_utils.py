import wandb
from utils.global_state import GlobalState
from utils.logger_utils import write_to_main_log


def setup_wandb(config, accelerator):
    # Only the main process should initialize WandB
    if not accelerator.is_main_process:
        return

    run_id = None

    # Initialize WandB with error handling
    try:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.run_name,
            group=config.wandb.get("group", None),
            job_type=config.wandb.get("job_type", None),
            config=dict(config),
            id=run_id,
            resume="allow" if run_id else None,
            mode=config.wandb.get("mode", "offline"),
            dir=GlobalState.get('log_dir')
        )
        write_to_main_log(accelerator=accelerator, result=f"WandB initialized successfully for project: {config.wandb.project}")
    except Exception as e:
        write_to_main_log(accelerator=accelerator, result=f"WandB initialization failed: {e}", type='warning')
        write_to_main_log(accelerator=accelerator, result="Training will continue without WandB logging", type='warning')
        return None


def log_metrics_wandb(metrics_dict, step):
    """Logs dynamic metrics (loss, LR) at every step."""
    if wandb.run is not None:
        try:
            wandb.log(metrics_dict, step=step)
        except Exception as e:
            # Silent failure for logging to avoid disrupting training
            pass
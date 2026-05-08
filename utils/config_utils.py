from utils.logger_utils import write_to_main_log


def find_closest_model_config(target_config, accelerator, config_path="configs/models/models.json"):
    import json
    import numpy as np
    
    # Load all model configurations
    with open(config_path, 'r') as f:
        all_configs = json.load(f)
    
    # Convert target_config to dict if it's a SimpleNamespace
    if not isinstance(target_config, dict):
        target_config = vars(target_config)
    
    # Parameters to compare (in order of importance)
    primary_params = [
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_register_tokens",
        "patch_size"
    ]
    
    secondary_params = [
        "mlp_ratio", 
        "hidden_act", 
        "layer_norm_eps",
        "qkv_bias",
        "use_swiglu_ffn"
    ]
    
    best_score = -float('inf')
    closest_model = None
    
    # Target model param values (ignoring num_channels)
    target_values = {}
    for param in primary_params + secondary_params:
        if param in target_config:
            target_values[param] = target_config[param]
    
    # Filter to only include Facebook models
    candidate_models = {k: v for k, v in all_configs.items() 
                       if k.startswith("facebook/")}
    
    results = []
    
    # Compare with each candidate model
    for model_name, config in candidate_models.items():
        # Calculate similarity score
        primary_score = 0
        secondary_score = 0
        
        # Check model type/architecture compatibility first (crucial)
        if "model_type" in config and "model_type" in target_config:
            if config["model_type"] == target_config["model_type"]:
                primary_score += 1000  # High weight for matching model type
            elif "registers" in config.get("model_type", "") and "registers" in target_config.get("model_type", ""):
                primary_score += 500  # Partial match for register-based models
        
        # Compare primary parameters (more important)
        for param in primary_params:
            if param in config and param in target_values:
                if param == "architectures":
                    # Compare architectures as sets
                    if set(config[param]) == set(target_values[param]):
                        primary_score += 100
                    elif len(set(config[param]) & set(target_values[param])) > 0:
                        primary_score += 50  # Partial match
                elif isinstance(target_values[param], (int, float)) and isinstance(config[param], (int, float)):
                    # For numerical values, score based on relative difference
                    diff = abs(config[param] - target_values[param])
                    if diff == 0:
                        primary_score += 100  # Exact match
                    else:
                        # Calculate relative similarity (0-100)
                        max_val = max(abs(config[param]), abs(target_values[param]))
                        similarity = max(0, 100 - (diff / max_val * 100))
                        primary_score += similarity
                elif config[param] == target_values[param]:
                    primary_score += 100  # Exact match for non-numeric
        
        # Compare secondary parameters (less important)
        for param in secondary_params:
            if param in config and param in target_values:
                if config[param] == target_values[param]:
                    secondary_score += 10
        
        # Total score (primary parameters are more important)
        total_score = primary_score + secondary_score
        
        # Store results for logging
        results.append({
            "model_name": model_name,
            "score": total_score,
            "primary_score": primary_score,
            "secondary_score": secondary_score
        })
        
        # Update best match
        if total_score > best_score:
            best_score = total_score
            closest_model = model_name
    
    # Sort results for logging
    results.sort(key=lambda x: x["score"], reverse=True)
    
    write_to_main_log( accelerator= accelerator, result=f"\nSelected {closest_model} as the closest model for weight transfer") 
 
    return closest_model
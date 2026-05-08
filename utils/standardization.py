
import sys
import types
import copy 
from typing import Any, Dict, Union 
def standardize_model_config(config ):
    
    model_type= config.model_type 
    if isinstance(config, dict): 
        std_config = types.SimpleNamespace()
        for k, v in config.items():
            setattr(std_config, k, v)
    else: 
        std_config = copy.deepcopy(config)
     
    if model_type is None: 
        if hasattr(config, 'model_type'):
            model_type = config.model_type 
        elif hasattr(config, 'download_path') and 'gigapath' in getattr(config, 'download_path', ''):
            model_type = 'timm'
     
    if model_type == 'timm':
        # hidden_size -> embed_dim

        if not hasattr(std_config, 'hidden_size') and hasattr(std_config, 'embed_dim'):
            std_config.hidden_size = std_config.embed_dim

        if not hasattr(std_config, 'hidden_size') and hasattr(std_config, 'num_features'):
            std_config.hidden_size = std_config.num_features
        
        # num_attention_heads -> num_heads
        if not hasattr(std_config, 'num_attention_heads') and hasattr(std_config, 'num_heads'):
            std_config.num_attention_heads = std_config.num_heads
        
        # num_hidden_layers -> depth
        if not hasattr(std_config, 'num_hidden_layers') and hasattr(std_config, 'depth'):
            std_config.num_hidden_layers = std_config.depth
            
        # Eğer intermediate_size yoksa, mlp_ratio'ya göre hesapla
        if not hasattr(std_config, 'intermediate_size') and hasattr(std_config, 'mlp_ratio') and hasattr(std_config, 'hidden_size'):
            std_config.intermediate_size = int(std_config.mlp_ratio * std_config.hidden_size)
    
    # DINOv2 modelleri için ek özellikler
    elif 'dino' in model_type.lower():
        # Ters mapping (HF -> timm)
        if not hasattr(std_config, 'embed_dim') and hasattr(std_config, 'hidden_size'):
            std_config.embed_dim = std_config.hidden_size
            
        if not hasattr(std_config, 'num_heads') and hasattr(std_config, 'num_attention_heads'):
            std_config.num_heads = std_config.num_attention_heads
            
        if not hasattr(std_config, 'depth') and hasattr(std_config, 'num_hidden_layers'):
            std_config.depth = std_config.num_hidden_layers
     
    if not hasattr(std_config, 'hidden_size'):
        std_config.hidden_size = 768
        
    if not hasattr(std_config, 'num_attention_heads'):
        std_config.num_attention_heads = 12
        
    if not hasattr(std_config, 'num_hidden_layers'):
        std_config.num_hidden_layers = 12
    
    return std_config



def standardize_model_config_object(config: Union[Dict[str, Any], Any]) -> types.SimpleNamespace:
    """
    Standardizes a model configuration dictionary or object and returns it
    as a SimpleNamespace object, allowing attribute access (dot notation).

    This applies mappings from common alternative names
    (like embed_dim, num_heads, depth, num_features) to the standard names
    (hidden_size, num_attention_heads, num_hidden_layers)
    REGARDLESS of the detected model_type.
    Adds default values for essential parameters if missing.
    Handles nested dictionaries by including them directly.

    Args:
        config (Union[Dict[str, Any], Any]): The input configuration.
            Can be a dictionary or an object with a __dict__ attribute.

    Returns:
        types.SimpleNamespace: The standardized configuration as a SimpleNamespace object.
    """
    # 1. Convert input to a dictionary if it's not already one
    if isinstance(config, dict):
        std_config_dict = copy.deepcopy(config) # Work on a copy to avoid modifying the original input dict
    elif hasattr(config, '__dict__'):
        # Convert object attributes to a dictionary
        std_config_dict = copy.deepcopy(vars(config))
    else:
        # If it's neither a dict nor has __dict__, raise an error or return as is
        raise TypeError(f"Input config must be a dictionary or an object with __dict__, but got {type(config)}")

    # 2. Determine model_type from the dictionary (optional, but kept for context)
    model_type = std_config_dict.get('model_type', None)

    # If model_type is not explicitly set, try to infer it
    if model_type is None:
        # Check for 'download_path' only if it exists and is a string
        if 'download_path' in std_config_dict and isinstance(std_config_dict['download_path'], str) and 'gigapath' in std_config_dict['download_path'].lower():
            model_type = 'timm' # Infer timm for models like GigaPath
        # Add other inference rules if needed

    # Store the determined model_type back in the dictionary
    std_config_dict['model_type'] = str(model_type) if model_type is not None else None


    # 3. Apply mappings from common alternative names to standard names
    # These mappings are applied regardless of the detected model_type

    # hidden_size -> embed_dim or num_features
    # Check for 'hidden_size' first, then try alternatives
    if 'hidden_size' not in std_config_dict:
        if 'embed_dim' in std_config_dict:
            std_config_dict['hidden_size'] = std_config_dict['embed_dim']
        elif 'num_features' in std_config_dict: # Maps num_features to hidden_size
             std_config_dict['hidden_size'] = std_config_dict['num_features']

    # num_attention_heads -> num_heads
    if 'num_attention_heads' not in std_config_dict and 'num_heads' in std_config_dict:
        std_config_dict['num_attention_heads'] = std_config_dict['num_heads']

    # num_hidden_layers -> depth
    if 'num_hidden_layers' not in std_config_dict and 'depth' in std_config_dict:
        std_config_dict['num_hidden_layers'] = std_config_dict['depth']

    # intermediate_size from mlp_ratio (if intermediate_size is missing)
    # Requires both mlp_ratio and hidden_size to be present for calculation
    if 'intermediate_size' not in std_config_dict and 'mlp_ratio' in std_config_dict and 'hidden_size' in std_config_dict:
        try:
            # Ensure types are numeric for calculation
            mlp_ratio = float(std_config_dict['mlp_ratio'])
            hidden_size = int(std_config_dict['hidden_size'])
            std_config_dict['intermediate_size'] = int(mlp_ratio * hidden_size)
        except (ValueError, TypeError):
            # Handle cases where mlp_ratio or hidden_size are not valid numbers
            print(f"Warning: Could not calculate intermediate_size due to invalid mlp_ratio ({std_config_dict.get('mlp_ratio')}) or hidden_size ({std_config_dict.get('hidden_size')})", file=sys.stderr)

    # Add reverse mappings if needed for compatibility with code expecting timm/dino names
    # These are less critical for standardization but can be included for convenience
    # Example: Ensure embed_dim exists if hidden_size is present
    if 'embed_dim' not in std_config_dict and 'hidden_size' in std_config_dict:
         std_config_dict['embed_dim'] = std_config_dict['hidden_size']

    # Example: Ensure num_heads exists if num_attention_heads is present
    if 'num_heads' not in std_config_dict and 'num_attention_heads' in std_config_dict:
         std_config_dict['num_heads'] = std_config_dict['num_attention_heads']

    # Example: Ensure depth exists if num_hidden_layers is present
    if 'depth' not in std_config_dict and 'num_hidden_layers' in std_config_dict:
         std_config_dict['depth'] = std_config_dict['num_hidden_layers']

 
    # Add other essential defaults if necessary
    # std_config_dict.setdefault('intermediate_size', None) # Or calculate a default if possible
    # std_config_dict.setdefault('architecture', None)
    # std_config_dict.setdefault('num_classes', None)
    # std_config_dict.setdefault('global_pool', None)
    # std_config_dict.setdefault('dynamic_img_size', False)
    # std_config_dict.setdefault('init_values', None)


    # 5. Convert the standardized dictionary to a SimpleNamespace object
    std_config_object = types.SimpleNamespace()
    for key, value in std_config_dict.items():
        # Directly assign values, including nested dictionaries
        setattr(std_config_object, key, value)

    # 6. Return the SimpleNamespace object
    return std_config_object
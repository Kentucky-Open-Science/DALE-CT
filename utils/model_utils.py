import torch
from utils.logger_utils import write_to_main_log

def transfer_weights_from_pretrained(to_model, from_model, accelerator, method='mean'):
    state_dict_1 = to_model.state_dict()
    state_dict_2 = from_model.state_dict()
    transferred = 0
    skipped = 0
    new_state_dict = {}
    
    for name, param in state_dict_2.items():
        if name in state_dict_1:
            if param.shape == state_dict_1[name].shape:
                new_state_dict[name] = param
                transferred += 1
            else:
                if name == 'embeddings.patch_embeddings.projection.weight':
                    source_channels = param.shape[1]  # from_model channels

                    target_channels = state_dict_1[name].shape[1]  # to_model channels 
                    if source_channels > target_channels:
                        # Multi-channel to fewer channels (existing logic)
                        if method == 'mean':
                            # Take average of all channels
                            converted_weights = param.mean(dim=1, keepdim=True)
                        elif method == 'first':
                            # Use only the first channel
                            converted_weights = param[:, 0:1, :, :]
                        else:
                            raise ValueError(f"Unknown method: {method}")
                        
                        # If target has more than 1 channel, repeat the converted weights
                        if target_channels > 1:
                            converted_weights = converted_weights.repeat(1, target_channels, 1, 1)
                            
                    elif source_channels < target_channels:
                        # Single/fewer channels to multi-channel (new logic)
                        if method == 'mean':
                            # Repeat the existing channels to match target channels
                            converted_weights = param.repeat(1, target_channels // source_channels, 1, 1)
                            # If target_channels is not divisible by source_channels, handle remainder
                            if target_channels % source_channels != 0:
                                remainder = target_channels % source_channels
                                extra_channels = param[:, :remainder, :, :]
                                converted_weights = torch.cat([converted_weights, extra_channels], dim=1)
                        elif method == 'first':
                            # Repeat the first channel to all target channels
                            first_channel = param[:, 0:1, :, :]
                            converted_weights = first_channel.repeat(1, target_channels, 1, 1)
                        else:
                            raise ValueError(f"Unknown method: {method}")
                    else:
                        # Same number of channels but different shape - shouldn't happen for this parameter
                        converted_weights = param
                    
                    new_state_dict[name] = converted_weights
                    transferred += 1
                else:
                    skipped += 1
        else:
            skipped += 1
    
    # Load the new state dict
    to_model.load_state_dict(new_state_dict, strict=False)
    
    # Print brief summary
    write_to_main_log(
        accelerator=accelerator, 
        result=f"Weight transfer complete: {transferred} parameters transferred, {skipped} parameters skipped using '{method}' method."
    )
    return to_model
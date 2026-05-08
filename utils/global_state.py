# utils/global_state.py
import torch.distributed as dist

class GlobalState:
    """
    A Singleton-like class to hold global path variables (Key-Value pairs).
    It allows Rank 0 to define paths and synchronize them to other GPUs.
    """
    _params = {}

    @classmethod
    def set(cls, key, value):
        """
        Sets a value in the local dictionary. 
        NOTE: This only affects the current process until synchronize() is called.
        """
        cls._params[key] = value

    @classmethod
    def get(cls, key):
        """
        Retrieves a value by key.
        Raises an error if the key doesn't exist (e.g., if synchronize wasn't called).
        """
        if key not in cls._params:
            raise KeyError(f"GlobalState: Key '{key}' not found. Did you call synchronize()?")
        return cls._params[key]
    @classmethod
    def has(cls, key): 
        return key in cls._params
    @classmethod
    def synchronize(cls):
        """
        MUST BE CALLED BY ALL PROCESSES.
        Broadcasts the _params dictionary from Rank 0 to all other processes.
        """
        if dist.is_initialized():
            # Prepare a list container. Rank 0 puts its data here.
            # Other ranks put an empty list (which will be overwritten).
            share_obj = [cls._params]
            
            # Broadcast from Rank 0 (src=0) to everyone
            dist.broadcast_object_list(share_obj, src=0)
            
            # Update local _params with the data received from Rank 0
            cls._params = share_obj[0]
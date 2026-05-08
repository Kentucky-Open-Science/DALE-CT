
'''
import torch
import torch.distributed as dist

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_projections=256 ):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.num_projections = num_projections 

    def forward(self, proj): 
        A = torch.randn(proj.size(-1), self.num_projections, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
'''

import torch
import torch.nn as nn
import torch.distributed as dist

class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg).
    Updated to handle dynamic input dimensions (Fix for FSDP/DDP sharding).
    """
    def __init__(self, knots=17, num_slices=256):
        super().__init__()
        self.num_slices = num_slices
        
        # Integration points on [-5, 5] as per Algorithm 1
        t = torch.linspace(-3, 3, knots, dtype=torch.float32)
        
        # Theoretical CF for N(0, 1)
        exp_f = torch.exp(-0.5 * t**2)
        
        self.register_buffer("t", t)
        self.register_buffer("exp_f", exp_f)

    def forward(self, x, global_step=0):
        """
        Args:
            x: Input tensor. Expected feature dim is the last dimension.
            global_step: Current iteration for seed synchronization.
        """
        device = x.device
        
        # --- FIX: Dynamic Dimension Handling ---
        # Instead of a hardcoded 128, we take the actual size from the input tensor.
        # This handles FSDP sharding (e.g., 64 instead of 128) automatically.
        current_feature_dim = x.size(-1) 
        
        # Slice sampling -- synced across devices using global_step --
        generator = torch.Generator(device=device)
        generator.manual_seed(global_step)
         
        proj_shape = (current_feature_dim, self.num_slices)
        A = torch.randn(proj_shape, generator=generator, device=device)
        A /= A.norm(p=2, dim=0) 
         
        x_t = (x @ A).unsqueeze(-1) * self.t
        
        # Average across the batch dimension (-3 or 0 depending on view format)
        ecf = (1j * x_t).exp().mean(-3)
        
        # Multi-GPU Synchronization (DDP/FSDP)
        if dist.is_initialized():
            dist.all_reduce(ecf, op=dist.ReduceOp.AVG)
            world_size = dist.get_world_size()
        else:
            world_size = 1
            
        # Weighted L2 distance
        err = (ecf - self.exp_f).abs().square().mul(self.exp_f)
        
        # Scale by global batch size
        global_batch_size = x.size(-2) * world_size
        
        # Final integration using Trapezoidal rule
        # Since t is 1D, err @ weights or trapz works; here we follow the paper's trapz logic.
        T = torch.trapz(err, self.t, dim=-1) * global_batch_size
        
        return T.mean()
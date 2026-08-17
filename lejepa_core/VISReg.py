import math

import torch
import torch.nn as nn
import torch.distributed as dist


class _AllGather(torch.autograd.Function):
    """Differentiable all-gather along dim 0.

    Forward concatenates each rank's tensor along dim 0 (same ordering on every
    rank). Backward scatters the gathered gradient back to each rank's local
    shard — each rank receives only the slice of grad_output that corresponds to
    its own contribution. No-op when distributed is not initialized or
    world_size == 1.
    """

    @staticmethod
    def forward(ctx, x):
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            ctx.world_size = 1
            return x
        ctx.world_size = dist.get_world_size()
        gathered = [torch.empty_like(x) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, x.contiguous())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.world_size == 1:
            return grad_output
        rank = dist.get_rank()
        chunks = torch.chunk(grad_output, ctx.world_size, dim=0)
        return chunks[rank].contiguous()


def all_gather(x):
    return _AllGather.apply(x)


class VISReg(nn.Module):
    """Variance-Invariance-Sketching Regularization (VISReg).

    Decoupled collapse-prevention regularizer from Balestriero (arXiv:2606.02572).
    Unlike SIGReg (which scales a batch-mean statistic by global batch size), the
    shape term sorts the *global* batch, so the embeddings must be all-gathered
    across ranks before the loss is computed. The random projection matrix W is
    seeded by ``global_step`` so every rank draws an identical W (matching the
    SIGReg sync convention).

    L = lambda_scale * L_scale + lambda_shape * L_shape + lambda_center * L_center
      L_center = mu.pow(2).mean()
      L_scale  = (1 - std).pow(2).mean()           # std over the global batch, biased
      L_shape  = SWD to N(0,1): project z_norm with random W, sort, compare to
                 Gaussian quantiles (Normal(0,1).icdf(u), u = arange(1,N+1)/(N+1))

    Args:
        K: number of random 1-D projections (slices).
        lambda_scale/lambda_shape/lambda_center: weights for the three terms
            (galaxy-best "Shape 4:1" = 0.5 / 2.0 / 0.5).
    """

    def __init__(self, K=4096, lambda_scale=0.5, lambda_shape=2.0, lambda_center=0.5):
        super().__init__()
        self.K = K
        self.lambda_scale = lambda_scale
        self.lambda_shape = lambda_shape
        self.lambda_center = lambda_center
        # Guard against a degenerate (zero-variance) channel producing inf/NaN.
        self.eps = 1e-6

    def forward(self, x, global_step=0):
        """
        Args:
            x: projector output, shape (V, B, D) — same convention as SIGReg.
            global_step: iteration index, used to seed the shared projection W.
        """
        device = x.device
        # Compute the regularizer in float32 for numerical stability under bf16.
        x = x.float()

        # Flatten views into the sample dim and gather the global batch.
        # (V, B, D) -> (N_local, D); all_gather -> (N_global, D).
        z = x.reshape(-1, x.size(-1))
        z = all_gather(z)
        N, D = z.shape

        # 1. Center loss.
        mu = z.mean(dim=0)
        l_center = mu.pow(2).mean()

        # 2. Scale loss (biased std over the global batch).
        z_cent = z - mu
        std = z_cent.std(dim=0, unbiased=False)
        l_scale = (1.0 - std).pow(2).mean()

        # 3. Shape loss: sliced-Wasserstein distance to N(0,1).
        z_norm = z_cent / (std.detach() + self.eps)

        generator = torch.Generator(device=device)
        generator.manual_seed(int(global_step))
        W = torch.randn(D, self.K, generator=generator, device=device, dtype=torch.float32)
        W = W / W.norm(p=2, dim=0, keepdim=True)

        p = z_norm @ W                      # (N, K)
        p_sorted = torch.sort(p, dim=0).values

        u = torch.arange(1, N + 1, device=device, dtype=torch.float32) / (N + 1)
        # Normal(0,1).icdf(u) == sqrt(2) * erfinv(2u - 1)
        target = torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)   # (N,)

        l_shape = (p_sorted - target.unsqueeze(1)).pow(2).mean()

        return (
            self.lambda_scale * l_scale
            + self.lambda_shape * l_shape
            + self.lambda_center * l_center
        )

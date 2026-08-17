"""Sequence-aware MIL probers built on torchmil (Apache-2.0) layers.

Drop-in alternative to ColipriProber for the error-bars / linear-probe protocol.
Same constructor contract and forward signature — ``forward(x, mask=None) ->
(logits, aux)`` — so it plugs into train_gridsearch.train_one_config and
run_error_bars.run_inference unchanged. ``aux`` is ``None`` for order-agnostic
methods and a scalar KL term for ProbSA (added to BCE with cyclical-annealing
weight by the training loop).

The cranio-caudal slice order — already preserved in the stored per-slice CLS
embeddings (see scripts/ctrate_generate_embeddings.py) — is exploited two ways:

  1. A 1D positional-chain adjacency A[i,j]=1 iff 0<|i-j|<=k (unnormalised 0/1;
     no self-loops). ProbSA's Dirichlet-energy prior mu^T L mu (L = D - A, the
     combinatorial graph Laplacian) is computed on this chain in the stable
     factored form 1/2 sum A[i,j](mu_i-mu_j)^2, so the attention map is
     encouraged to vary smoothly along z.
  2. 1D sinusoidal positional encoding added before the transformer encoders
     (TransMIL-1D, Transformer-ABMIL-1D).

Schemes:
  abmil                -- torchmil AttentionPool (MLP attention). Order-agnostic
                          sanity baseline; comparable to ColipriProber's
                          learned_attention.
  probsa_zero          -- ProbSmoothAttentionPool, covar_mode='zero' (deterministic
                          smooth attention). aux = masked KL.
  probsa_diag          -- ProbSmoothAttentionPool, covar_mode='diag' (per-instance
                          uncertainty, MC samples). aux = masked KL.
  transformer_abmil_1d -- TransformerEncoder (exact O(n^2) self-attention) + 1D PE
                          + AttentionPool.
  transmil_1d          -- NystromTransformerEncoder (O(n) Nyström self-attention,
                          the TransMIL signature) + 1D PE + AttentionPool. The 2D
                          PPEG of the original TransMIL is dropped (mismatched to a
                          1D slice sequence); 1D sinusoidal PE replaces it.

torchmil + einops are pip-installed (no deps) to /project/.../vendor_pkgs on the
DGX and put on PYTHONPATH by the sbatch wrapper; the lejepa container supplies
torch/numpy/scipy. See CLAUDE.md / memory for the vendoring rationale.
"""
import math

import torch
import torch.nn as nn

# torchmil lives in vendor_pkgs (on PYTHONPATH in the container). Import lazily
# so the module can still be imported (and unit-tested for the non-torchmil
# helpers) on machines without torchmil; only MILProber construction needs it.
_TORCHMIL = None


def _tm():
    global _TORCHMIL
    if _TORCHMIL is None:
        from torchmil.nn import (  # noqa: E402
            AttentionPool,
            NystromTransformerEncoder,
            ProbSmoothAttentionPool,
            TransformerEncoder,
        )
        _TORCHMIL = {
            "AttentionPool": AttentionPool,
            "NystromTransformerEncoder": NystromTransformerEncoder,
            "ProbSmoothAttentionPool": ProbSmoothAttentionPool,
            "TransformerEncoder": TransformerEncoder,
        }
    return _TORCHMIL


MIL_SCHEMES = {"abmil", "probsa_zero", "probsa_diag",
               "transformer_abmil_1d", "transmil_1d"}
PROBSA_SCHEMES = {"probsa_zero", "probsa_diag"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cyclical_lambda(step, total_steps, n_cycles=5, max_lambda=1.0):
    """Cyclical annealing schedule for the KL weight (ProbSA paper, M=5 cycles).

    Within each cycle lambda ramps 0 -> max_lambda over the first 80% of the
    cycle, then holds at max_lambda for the remaining 20%. Matches
    CT-MIL/utils/probsa_utils.get_cyclical_lambda.
    """
    if total_steps <= 0 or n_cycles <= 0:
        return max_lambda
    steps_per_cycle = max(1, total_steps // n_cycles)
    step_in_cycle = step % steps_per_cycle
    anneal_steps = max(1, int(steps_per_cycle * 0.8))
    if step_in_cycle < anneal_steps:
        return max_lambda * step_in_cycle / anneal_steps
    return max_lambda


def build_chain_adjacency(mask, k=1, normalize=True):
    """1D positional-chain adjacency over slice indices, masked to valid slices.

    A[i,j] = 1 iff 0 < |i-j| <= k  (no self-loops), then rows/cols of padded
    positions are zeroed. The KL (masked_kl_divergence) uses the UNNORMALISED
    0/1 form (normalize=False) so that L = D - A is the combinatorial graph
    Laplacian and the Dirichlet energy is the stable sum-of-squared-differences.
    The ``normalize`` flag (symmetric-normalise to A_norm = D^{-1/2} A D^{-1/2},
    giving I - A_norm = L_sym) is retained for the unit tests and as a helper
    capability; it is not used by the prober.

    mask: (B, N) bool/float. Returns (B, N, N) float on the same device.
    """
    device = mask.device
    B, N = mask.shape
    idx = torch.arange(N, device=device)
    A = ((idx[:, None] - idx[None, :]).abs() <= k).float().unsqueeze(0)  # (1,N,N)
    A = A * (idx[:, None] - idx[None, :]).abs().unsqueeze(0).gt(0).float()  # drop self-loops
    A = A.repeat(B, 1, 1)
    m = mask.float()
    A = A * m.unsqueeze(1) * m.unsqueeze(2)  # zero padded rows/cols
    if normalize:
        deg = A.sum(dim=-1)  # (B, N)
        deg_inv_sqrt = deg.clamp(min=1e-12).rsqrt()  # (B, N)
        A = A * deg_inv_sqrt.unsqueeze(2) * deg_inv_sqrt.unsqueeze(1)
    return A


def sinusoidal_pe_1d(N, D, device, dtype=torch.float32):
    """Standard 1D sinusoidal positional encoding. Returns (N, D)."""
    pe = torch.zeros(N, D, device=device, dtype=dtype)
    pos = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, device=device, dtype=dtype)
                    * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def masked_kl_divergence(mu_f, log_diag_Sigma_f, adj, mask):
    """Masked ProbSA KL with the combinatorial graph Laplacian L = D - A (the
    form the ProbSA paper specifies), in the numerically-stable factored form.

    torchmil's ProbSmoothAttentionPool._kl_div computes the explicit mu^T (I-A)
    mu (and divides by the FULL bag size, without masking mu/Sigma). Two problems
    with the explicit form: (1) it suffers catastrophic cancellation for smooth
    mu, where mu^T mu and mu^T A mu are near-equal and their difference loses
    precision -- and ProbSA's smoothness is imposed *entirely* through the KL
    gradient on mu_f (the pool's attention is a plain softmax; adj only enters
    via the KL), so a noise-dominated KL would make probsa behave like plain
    ABMIL; (2) torchmil's code uses I-A, not the D-A the paper documents. We use
    the paper's L = D - A in the factored sum-of-squared-differences form, which
    is non-negative by construction and exactly zero for a constant mu.

        Dirichlet energy:  mu^T L mu = 1/2 * sum_{i,j} A[i,j] (mu_i - mu_j)^2
        covariance trace:  tr(L Sigma) = sum_i d_i sigma_i^2   (A has no self-loops)
        log-det term:      -1/2 * sum_i log sigma_i^2
    All divided by N_valid; mu/Sigma are zeroed at padded positions (and padded
    rows/cols of A are already zero, so they contribute nothing).

    mu_f: (B, N, 1); log_diag_Sigma_f: (B, N, 1) or None (covar_mode='zero');
    adj: (B, N, N) UNNORMALIZED 0/1 chain adjacency (no self-loops, padded rows/
         cols zeroed -- i.e. build_chain_adjacency(..., normalize=False));
    mask: (B, N). Returns scalar.
    """
    m = mask.float().unsqueeze(-1)               # (B, N, 1)
    mu = mu_f * m                                # zero padded positions
    valid = m.sum(dim=(1, 2)).clamp(min=1.0)     # (B,) N_valid
    inv = 1.0 / valid

    # mu^T L mu = 1/2 * sum_{i,j} A[i,j] (mu_i - mu_j)^2  (sum of squares -> >= 0)
    diff = mu - mu.transpose(1, 2)               # (B, N, N): diff[b,i,j] = mu_i - mu_j
    muT_lap_mu = inv * 0.5 * (adj * (diff ** 2)).sum(dim=(1, 2))  # (B,)

    if log_diag_Sigma_f is None:
        kl_per = muT_lap_mu
    else:
        diag_Sigma = torch.exp(log_diag_Sigma_f) * m                 # (B, N, 1)
        deg = adj.sum(dim=-1)                                        # (B, N) degree d_i
        tr_lap_Sigma = inv * (deg.unsqueeze(-1) * diag_Sigma).sum(dim=(1, 2))  # sum d_i sigma_i^2 / N
        log_det_Sigma = inv * (log_diag_Sigma_f * m).sum(dim=(1, 2))
        kl_per = muT_lap_mu + tr_lap_Sigma - 0.5 * log_det_Sigma
    return kl_per.mean()


# --------------------------------------------------------------------------- #
# Prober
# --------------------------------------------------------------------------- #
class MILProber(nn.Module):
    """Sequence-aware MIL prober. See module docstring for the schemes.

    mil_cfg (optional dict, with defaults) keys:
        att_dim (int, 128 for abmil/probsa, =input_dim for transformers),
        n_heads (8), n_layers (2), n_landmarks (64), chain_k (1),
        n_samples_train (100), n_samples_test (500),
        n_cycles (5), max_lambda (1.0).
    """

    def __init__(self, input_dim, num_classes, pooling_scheme,
                 pooling_mode="embedding", mil_cfg=None):
        super().__init__()
        scheme = pooling_scheme.lower()
        if scheme not in MIL_SCHEMES:
            raise ValueError(f"Unknown MIL pooling scheme: {pooling_scheme}")
        if pooling_mode.lower() != "embedding":
            raise ValueError("MILProber only supports pooling_mode='embedding'")
        mil_cfg = mil_cfg or {}
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pooling_scheme = scheme
        self.classifier = nn.Linear(input_dim, num_classes)

        self.chain_k = mil_cfg.get("chain_k", 1)
        self.n_cycles = mil_cfg.get("n_cycles", 5)
        self.max_lambda = mil_cfg.get("max_lambda", 1.0)
        tm = _tm()

        if scheme == "abmil":
            self.pool = tm["AttentionPool"](
                in_dim=input_dim, att_dim=mil_cfg.get("att_dim", 128))
        elif scheme in PROBSA_SCHEMES:
            covar = "zero" if scheme == "probsa_zero" else "diag"
            self.pool = tm["ProbSmoothAttentionPool"](
                in_dim=input_dim, att_dim=mil_cfg.get("att_dim", 128),
                covar_mode=covar,
                n_samples_train=mil_cfg.get("n_samples_train", 100),
                n_samples_test=mil_cfg.get("n_samples_test", 500))
        elif scheme == "transformer_abmil_1d":
            self.encoder = tm["TransformerEncoder"](
                in_dim=input_dim, att_dim=mil_cfg.get("att_dim", input_dim),
                n_heads=mil_cfg.get("n_heads", 8),
                n_layers=mil_cfg.get("n_layers", 2), dropout=0.0)
            self.pool = tm["AttentionPool"](
                in_dim=input_dim, att_dim=mil_cfg.get("att_dim", 128))
        elif scheme == "transmil_1d":
            self.encoder = tm["NystromTransformerEncoder"](
                in_dim=input_dim, att_dim=mil_cfg.get("att_dim", input_dim),
                n_heads=mil_cfg.get("n_heads", 8),
                n_layers=mil_cfg.get("n_layers", 2),
                n_landmarks=mil_cfg.get("n_landmarks", 64), dropout=0.0)
            self.pool = tm["AttentionPool"](
                in_dim=input_dim, att_dim=mil_cfg.get("att_dim", 128))

    def forward(self, x, mask=None):
        B, N, D = x.shape
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)

        if self.pooling_scheme == "abmil":
            z = self.pool(x, mask=mask)
            aux = None
        elif self.pooling_scheme in PROBSA_SCHEMES:
            # Unnormalized 0/1 chain adjacency for the KL (combinatorial Laplacian
            # L = D - A, stable factored form). The pool only touches adj when
            # return_kl_div=True; we compute our own masked KL, so adj is passed
            # through but ignored by the pool.
            adj = build_chain_adjacency(mask, k=self.chain_k, normalize=False)
            z, mu_f, log_diag_Sigma_f = self.pool(
                x, adj=adj, mask=mask, return_att_dist=True)
            z = z.mean(dim=2)  # (B, D) average over MC samples
            aux = masked_kl_divergence(mu_f, log_diag_Sigma_f, adj, mask)
        else:  # transformer_abmil_1d | transmil_1d
            pe = sinusoidal_pe_1d(N, D, x.device, x.dtype)
            Y = self.encoder(x + pe.unsqueeze(0), mask=mask)
            z = self.pool(Y, mask=mask)
            aux = None

        logits = self.classifier(z)
        return logits, aux


def build_prober(input_dim, num_classes, pooling_scheme, pooling_mode, config=None):
    """Return MILProber for sequence-aware schemes, else ColipriProber.

    Reads optional ``config['mil']`` for hyperparameters (defaults apply if
    absent). This is the single switch point used by train_gridsearch and
    run_error_bars so every prober construction stays in sync.
    """
    if pooling_scheme.lower() in MIL_SCHEMES:
        mil_cfg = (config or {}).get("mil", {}) if isinstance(config, dict) else {}
        return MILProber(input_dim, num_classes, pooling_scheme, pooling_mode, mil_cfg)
    from models.colipri_pooling import ColipriProber
    return ColipriProber(input_dim, num_classes, pooling_scheme, pooling_mode)

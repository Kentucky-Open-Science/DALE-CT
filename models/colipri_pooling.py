import torch
import torch.nn as nn
import math


class ColipriProber(nn.Module):
    def __init__(self, input_dim, num_classes, pooling_scheme="average", pooling_mode="embedding"):
        super(ColipriProber, self).__init__()
        self.pooling_scheme = pooling_scheme.lower()
        self.pooling_mode = pooling_mode.lower()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # The classifier is used in BOTH modes
        self.classifier = nn.Linear(input_dim, num_classes)

        # ---------------------------------------------------------
        # MODE 1: Embedding Pooling (Late Fusion - COLIPRI Paper)
        # ---------------------------------------------------------
        if self.pooling_mode == "embedding":
            if self.pooling_scheme == "learned_attention":
                self.Q = nn.Parameter(torch.randn(1, input_dim) / math.sqrt(input_dim))
            elif self.pooling_scheme == "multilearned_attention":
                self.Q = nn.Parameter(torch.randn(4, input_dim) / math.sqrt(input_dim))
            elif self.pooling_scheme not in ["average", "max", "average_attention"]:
                raise ValueError(f"Unknown embedding pooling scheme: {self.pooling_scheme}")

        # ---------------------------------------------------------
        # MODE 2: Instance Pooling (Early Classification / MIL)
        # ---------------------------------------------------------
        elif self.pooling_mode == "instance":
            if self.pooling_scheme == "attention":
                self.attention_net = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.Tanh(),
                    nn.Linear(128, 1)
                )
            elif self.pooling_scheme not in ["average", "max"] and not self.pooling_scheme.startswith("top_k"):
                raise ValueError(f"Unknown instance pooling scheme: {self.pooling_scheme}")
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}. Use 'embedding' or 'instance'.")

    def forward(self, x, mask=None):
        # x shape: (Batch, Seq_Len, Input_Dim)
        B, N, D = x.shape

        # =========================================================
        # MODE 1: Embedding Pooling (Late Fusion)
        # =========================================================
        if self.pooling_mode == "embedding":
            if self.pooling_scheme == "average":
                if mask is not None:
                    sum_x = (x * mask.unsqueeze(-1)).sum(dim=1)
                    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    global_emb = sum_x / valid_counts
                else:
                    global_emb = x.mean(dim=1)

            elif self.pooling_scheme == "max":
                if mask is not None:
                    masked_x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
                    global_emb = masked_x.max(dim=1)[0]
                else:
                    global_emb = x.max(dim=1)[0]

            elif self.pooling_scheme == "learned_attention":
                Q_batch = self.Q.unsqueeze(0).expand(B, -1, -1)
                global_emb = self._compute_attention(Q_batch, x, mask).squeeze(1)

            elif self.pooling_scheme == "average_attention":
                if mask is not None:
                    sum_x = (x * mask.unsqueeze(-1)).sum(dim=1)
                    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    Q_avg = sum_x / valid_counts
                else:
                    Q_avg = x.mean(dim=1)
                Q_batch = Q_avg.unsqueeze(1)
                global_emb = self._compute_attention(Q_batch, x, mask).squeeze(1)

            elif self.pooling_scheme == "multilearned_attention":
                Q_batch = self.Q.unsqueeze(0).expand(B, -1, -1)
                multi_embs = self._compute_attention(Q_batch, x, mask)
                global_emb = multi_embs.mean(dim=1)

            # Final classification for embedding mode
            logits = self.classifier(global_emb)
            return logits, None

        # =========================================================
        # MODE 2: Instance Pooling (Early Classification / Logit Pooling)
        # =========================================================
        elif self.pooling_mode == "instance":

            # 1. Deep MIL Attention Logit Pooling
            if self.pooling_scheme == "attention":
                attn_scores = self.attention_net(x)
                if mask is not None:
                    attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
                attn_weights = torch.softmax(attn_scores, dim=1)

                slice_logits = self.classifier(x)  # Apply classifier to all slices
                global_logits = (slice_logits * attn_weights).sum(dim=1)
                return global_logits, None

            # 2. Standard Logit Pooling (Max, Average, Top-K)
            slice_logits = self.classifier(x)  # Shape: (B, Seq_Len, Num_Classes)

            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                if self.pooling_scheme == "max" or self.pooling_scheme.startswith("top_k"):
                    slice_logits = slice_logits.masked_fill(~mask_expanded, -1e9)
                elif self.pooling_scheme == "average":
                    slice_logits = slice_logits.masked_fill(~mask_expanded, 0.0)

            if self.pooling_scheme == "max":
                global_logits = slice_logits.max(dim=1)[0]

            elif self.pooling_scheme == "average":
                if mask is not None:
                    sum_logits = slice_logits.sum(dim=1)
                    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    global_logits = sum_logits / valid_counts
                else:
                    global_logits = slice_logits.mean(dim=1)

            elif self.pooling_scheme.startswith("top_k"):
                # Parse the K value (e.g., "top_k_5" -> 5)
                k = 5
                parts = self.pooling_scheme.split("_")
                if len(parts) >= 3:
                    k = int(parts[2])

                actual_k = min(k, N)
                top_k_vals, _ = torch.topk(slice_logits, k=actual_k, dim=1)

                if mask is not None:
                    valid_mask = top_k_vals > -1e8
                    sum_top_k = (top_k_vals * valid_mask).sum(dim=1)
                    valid_counts = valid_mask.sum(dim=1).clamp(min=1.0)
                    global_logits = sum_top_k / valid_counts
                else:
                    global_logits = top_k_vals.mean(dim=1)

            return global_logits, None
        return None

    def _compute_attention(self, Q, K, mask):
        """ Computes scaled dot-product cross attention for embedding mode. """
        D = self.input_dim
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(attn_weights, K)
        return output
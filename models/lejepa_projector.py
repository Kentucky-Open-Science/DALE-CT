from torch import nn 
import torch
from torchvision.ops import MLP

class LeJEPAProjector(nn.Module):
    """
    Projector head for LeJEPA.
    Maps backbone embeddings to a lower-dimensional space for SIGReg.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 128):
        super().__init__() 
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),   
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.proj(x)
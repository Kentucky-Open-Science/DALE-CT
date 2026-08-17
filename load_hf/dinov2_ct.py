import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel


class CTInferenceTransform:
    """
    Applies the exact HU windowing and Z-score normalization used during training.
    """

    def __init__(self):
        self.clip_min = -997.0
        self.clip_max = 888.0
        self.mean_hu = -142.39
        self.std_hu = 360.97
        self.patch_size = 14

        # Calculate 0-1 scaled mean and std
        range_val = self.clip_max - self.clip_min
        self.norm_mean = (self.mean_hu - self.clip_min) / range_val
        self.norm_std = self.std_hu / range_val

    def __call__(self, volume):
        # Expects a 2D numpy array or torch tensor (H, W) in Hounsfield Units
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).float()
        if volume.ndim == 2:
            volume = volume.unsqueeze(0)  # Add channel dim: (1, H, W)

        # 1. Clamp HU values and map strictly to [0, 1]
        volume = torch.clamp(volume, self.clip_min, self.clip_max)
        range_val = self.clip_max - self.clip_min
        volume = (volume - self.clip_min) / range_val

        # 2. Z-score standardization
        volume = (volume - self.norm_mean) / self.norm_std

        # 3. Padding/Interpolation for strict patch size alignment
        # HF DINOv2 expects dimensions to be multiples of the patch size (14)
        C, H, W = volume.shape
        target_h = int((H // self.patch_size) * self.patch_size)
        target_w = int((W // self.patch_size) * self.patch_size)

        if target_h != H or target_w != W:
            volume = volume.unsqueeze(0)  # (1, C, H, W)
            # Use nearest interpolation to prevent averaging of exact HU values
            volume = F.interpolate(volume, size=(target_h, target_w), mode='nearest')
            volume = volume.squeeze(0)

        # Returns (1, 1, H, W). For batched inference, stack these along dim=0.
        return volume.unsqueeze(0)


def load_finetuned_dinov2_ct(repo_id="Kentucky-Open-Science/Finetuned-DINOv2-Chest-CT"):
    """
    Downloads and initializes the ViT-Large backbone using Hugging Face transformers.
    """
    # The config.json in the HF repo handles architecture setup (1 in_chan, 518 native size)
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    model.eval()

    return model


if __name__ == "__main__":
    # Initialize the transform and the model
    transform = CTInferenceTransform()
    model = load_finetuned_dinov2_ct()

    # Simulate a raw CT slice (Replace this with an actual NIfTI/DICOM load in Hounsfield Units)
    raw_ct_slice = np.random.uniform(-1000, 1000, size=(512, 512))

    # Process the image to ensure correct normalization
    input_tensor = transform(raw_ct_slice)

    # Extract embeddings
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)

        # DINOv2 returns last_hidden_state, pooler_output, etc.
        # last_hidden_state includes the [CLS] token, register tokens, and spatial patch tokens
        hidden_states = outputs.last_hidden_state

        # [CLS] token is the first token
        cls_token = hidden_states[:, 0, :]

        # Register tokens (4 tokens based on config)
        register_tokens = hidden_states[:, 1:5, :]

        # Dense patch tokens (for fine-grained tasks like Segmentation)
        patch_tokens = hidden_states[:, 5:, :]

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Full hidden states shape: {hidden_states.shape}")
    print(f"CLS token shape: {cls_token.shape}")
    print(f"Register tokens shape: {register_tokens.shape}")
    print(f"Dense patch tokens shape: {patch_tokens.shape}")

import torch
import torch.nn.functional as F
import numpy as np
import timm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class CTInferenceTransform:
    """
    Applies the exact HU windowing and Z-score normalization used during LeJEPA training.
    """

    def __init__(self):
        # Calculated from the 0.5% and 99.5% foreground pixel intensities of CT-RATE-huggingface-downloads
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


def load_guided_ct_model(repo_id="Kentucky-Open-Science/DALE-CT-1S"):
    """
    Downloads and initializes the ViT-Large backbone using timm and safetensors.
    """
    # 1. Initialize the base architecture
    model = timm.create_model(
        "vit_large_patch14_dinov2",
        pretrained=False,
        num_classes=0,
        in_chans=1,  # Grayscale CT inputs
        img_size=518,  # Base initialization size
        dynamic_img_size=True  # Allows native processing of variable resolutions (e.g., 504x504)
    )

    # 2. Download and load the custom safetensors weights
    model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


if __name__ == "__main__":
    # Initialize the transform and the model
    transform = CTInferenceTransform()
    model = load_guided_ct_model()

    # Simulate a raw CT slice (Replace this with an actual NIfTI/DICOM load in Hounsfield Units)
    raw_ct_slice = np.random.uniform(-1000, 1000, size=(512, 512))

    # Process the image to ensure correct normalization and patch dimension alignment
    # A 512x512 image will be automatically resized to 504x504 (closest multiple of 14)
    input_tensor = transform(raw_ct_slice)

    # Extract embeddings
    with torch.no_grad():
        # Option A: Get the single pooled global feature for the entire slice
        global_feature = model(input_tensor)

        # Option B: Get the unpooled, dense spatial patch tokens (for fine-grained tasks like Segmentation)
        patch_tokens = model.forward_features(input_tensor)

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Extracted features shape: {global_feature.shape}")
    print(f"Dense patch tokens shape: {patch_tokens.shape}")

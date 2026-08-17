import torch
import numpy as np
import timm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class CTInferenceTransform:
    """
    Applies the exact HU windowing and Z-score normalization used during LeJEPA 2S training.
    Assumes standard 512x512 CT slice inputs.
    """

    def __init__(self):
        self.clip_min = -997.0
        self.clip_max = 888.0
        self.mean_hu = -142.39
        self.std_hu = 360.97

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

        # Returns (1, 1, H, W). For batched inference, stack these along dim=0.
        return volume.unsqueeze(0)


def load_guided_ct_model_v2(repo_id="Kentucky-Open-Science/DALE-CT-2S"):
    """
    Downloads and initializes the ViT-Large backbone using timm and safetensors.
    """
    # 1. Initialize the base architecture with 2S overrides (patch_size=16, img_size=512)
    model = timm.create_model(
        "vit_large_patch14_dinov2",
        pretrained=False,
        num_classes=0,
        in_chans=1,
        patch_size=16,
        img_size=512,
        dynamic_img_size=True
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
    model = load_guided_ct_model_v2()

    # Simulate a raw CT slice (Replace this with an actual NIfTI/DICOM load in Hounsfield Units)
    raw_ct_slice = np.random.uniform(-1000, 1000, size=(512, 512))

    # Process the image to ensure correct normalization
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

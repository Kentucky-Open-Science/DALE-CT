# CT-FM benchmark extraction image (DEPENDENCIES ONLY).
# 3D SegResEncoder (MONAI). Native foreground-crop + sliding window ->
# 1 token/patch -> (~N_patches, 512). Mirrors CT-FM's feature_extractor.py.
#
# Weights AND code are MOUNTED at runtime -- nothing is baked in. The repo is
# mounted at /app/project/ibi-staff/CT-JEPA/public; the loader reads weights
# from .../benchmark_models/CT-FM/huggingface/pretrained_segresnet.torch. See README.
FROM evandamron/lejepa:latest

RUN pip install --no-cache-dir monai

WORKDIR /app

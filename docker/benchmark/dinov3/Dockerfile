# DINOv3 benchmark extraction image (DEPENDENCIES ONLY).
# Frozen HF DINOv3ViTModel (1024-d, 224px, 4 registers). Requires transformers
# >= 4.56 for DINOv3ViTModel. Upgrading transformers is safe here: this image
# only runs benchmark extraction (error_bars probes are torch-only; DALE-CT
# embeddings are already extracted, so nothing live depends on the old version).
#
# Weights AND code are MOUNTED at runtime -- nothing is baked in. The repo is
# mounted at /app/project/ibi-staff/CT-JEPA/public; the loader reads weights
# from .../benchmark_models/dinov3-vitl16-pretrain-lvd1689m. See README.
FROM evandamron/lejepa:latest

RUN pip install --no-cache-dir -U "transformers>=4.56"

WORKDIR /app

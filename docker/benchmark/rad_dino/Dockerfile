# RAD-DINO benchmark extraction image (DEPENDENCIES ONLY).
# Frozen HF Dinov2Model (768-d, 518px). No extra deps beyond the base image
# (transformers provides AutoModel/AutoImageProcessor for Dinov2Model).
#
# Weights AND code are MOUNTED at runtime -- nothing is baked in. The repo is
# mounted at /app/project/ibi-staff/CT-JEPA/public (via the /project:/app/project
# Pyxis mount), so the loader reads weights from
# /app/project/ibi-staff/CT-JEPA/public/benchmark_models/rad-dino and the array
# script cp's scripts/dataloaders/utils into /app. See BENCHMARK_EMBEDDINGS_README.md.
FROM evandamron/lejepa:latest

WORKDIR /app

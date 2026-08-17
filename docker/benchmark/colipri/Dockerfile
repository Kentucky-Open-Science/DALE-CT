# COLIPRI benchmark extraction image (DEPENDENCIES ONLY).
# 3D Primus-M (colipri package). colipri is on PyPI (v0.1.2 == the local HF
# checkout's version, so the get_model / encode_image API the loader uses
# matches). Installing from PyPI avoids the package's uv_build backend (fragile
# under cross-arch QEMU emulation) and shrinks the image. Raw 864-d backbone
# tokens -> (13824, 864).
#
# Weights AND code are MOUNTED at runtime -- nothing is baked in. The repo is
# mounted at /app/project/ibi-staff/CT-JEPA/public; the loader reads weights
# from .../benchmark_models/colipri/model.safetensors. See README.
FROM evandamron/lejepa:latest

RUN pip install --no-cache-dir colipri==0.1.2

WORKDIR /app

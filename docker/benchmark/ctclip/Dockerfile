# CT-CLIP benchmark extraction image (DEPENDENCIES ONLY).
# 3D CTViT (transformer_maskgit). Visual encoder weights loaded from CT-CLIP_v2.pt
# (extracting the visual_transformer.* keys). Returns 13824x512 tokens.
#
# transformer_maskgit lives as a SUBDIRECTORY of the CT-CLIP repo
# (ibrahimethemhamamci/CT-CLIP/transformer_maskgit/, with its own setup.py) --
# there is no standalone kakaobrain/transformer-maskgit repo (that URL 404s).
# Install via the #subdirectory= form. CT-CLIP load is SMOKE-TEST-GATED --
# confirm the visual_transformer key extraction at --limit 2 before the full run.
#
# Weights AND code are MOUNTED at runtime -- nothing is baked in. The repo is
# mounted at /app/project/ibi-staff/CT-JEPA/public; the loader reads weights
# from .../benchmark_models/CT-CHAT/CT-RATE-huggingface-downloads/CT-CLIP_v2.pt.
FROM evandamron/lejepa:latest

# transformer_maskgit imports cv2 at package import (data.py), and opencv-python
# needs libGL.so.1 which the lejepa base lacks. Install the system lib so `from
# transformer_maskgit import CTViT` succeeds. (libglib2.0-0 is already in the base.)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "git+https://github.com/ibrahimethemhamamci/CT-CLIP.git#subdirectory=transformer_maskgit"

WORKDIR /app

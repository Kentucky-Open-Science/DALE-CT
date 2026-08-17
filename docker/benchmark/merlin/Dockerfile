# Merlin benchmark extraction image (DEPENDENCIES ONLY).
# 3D i3_resnet (merlin-vlm). The loader loads weights MANUALLY from the mounted
# .pt (NO auto-download -- compute nodes are offline). Returns a pooled 2048-d
# embedding per volume, matching the CT-FM paper's own Merlin linear-probe
# protocol (documentation/Merlin_CVPR26CTFM/extract_feat_LP.py).
#
# Weights AND code are MOUNTED at runtime -- nothing is baked in. The repo is
# mounted at /app/project/ibi-staff/CT-JEPA/public; the loader reads weights
# from .../benchmark_models/Merlin/i3_resnet_..._epoch_99.pt. See README.
FROM evandamron/lejepa:latest

RUN pip install --no-cache-dir merlin-vlm

WORKDIR /app

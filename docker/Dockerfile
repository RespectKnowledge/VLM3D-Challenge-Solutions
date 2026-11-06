# MR-RATE NeuroVFM-LoRA abnormality classification
# Follows organizer classification Docker contract:
#   /input   challenge study ZIPs
#   /output  predictions.json
#   /weights externally mounted model bundle

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
    && rm -rf /var/lib/apt/lists/*

# Keep organizer package versions wherever possible.
RUN pip install --no-cache-dir \
    nibabel SimpleITK \
    scipy \
    einops \
    timm \
    peft \
    "transformers==4.55.4" \
    pyyaml \
    pandas \
    pytorch-lightning \
    positional-encodings

# Exact NeuroVFM / MR-RATE inference source used for our trained model.
COPY src /opt/mrrate/src

# Submission application.
COPY app/run_inference.py app/labels.py app/entrypoint.sh /opt/app/

RUN chmod +x /opt/app/entrypoint.sh \
 && groupadd -r worker \
 && useradd -m -r -g worker worker \
 && mkdir -p /input /output /weights /tmp/mrrate-native \
 && chown -R worker:worker \
      /opt/app \
      /opt/mrrate \
      /input \
      /output \
      /weights \
      /tmp/mrrate-native

USER worker

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/mrrate/src:/opt/mrrate/src/scripts:/opt/mrrate/src/mr_rate:/opt/mrrate/src/vision_encoder

ENTRYPOINT ["/opt/app/entrypoint.sh"]

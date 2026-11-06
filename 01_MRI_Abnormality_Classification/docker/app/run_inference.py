#!/usr/bin/env python3

import json
import os
import sys
import time
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))

ROOT = Path("/opt/mrrate")
SRC = ROOT / "src"
SCRIPTS = SRC / "scripts"
WEIGHTS = Path(os.environ.get("FORITHMUS_WEIGHTS_DIR", "/weights"))

# Exact project source paths.
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SRC / "mr_rate"))
sys.path.insert(0, str(SRC / "vision_encoder"))
sys.path.insert(0, str(SRC))

WEIGHTS_ROOT = os.environ.get("FORITHMUS_WEIGHTS_DIR", "/weights")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = f"{WEIGHTS_ROOT}/hf-cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{WEIGHTS_ROOT}/hf-cache/hub"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from data import (
    NORMALIZERS,
    load_and_resample_nii,
    crop_or_pad,
)

from extract_features import build_encoder, _load_and_verify
from labels import LABEL_COLS


# Must match the cache used for LoRA feature extraction.
TARGET_SPACING = (1.0, 0.5, 0.5)
TARGET_SHAPE = (256, 384, 384)
POSTERIOR_SHIFT_MM = 15.0
POSTERIOR_SHIFT_VOXELS = int(
    round(POSTERIOR_SHIFT_MM / TARGET_SPACING[2])
)


BASE_NEUROVFM = WEIGHTS / "neurovfm" / "neurovfm-encoder"
MRRATE_CKPT = WEIGHTS / "neurovfm" / "MrRate.full.88540.MERGED.pt"

LINEAR_CKPT = WEIGHTS / "classifier" / "linear_head.pt"
CALIBRATION = WEIGHTS / "classifier" / "organizer_f1_calibration.json"


def discover_zip_studies(root: Path):
    zips = sorted(root.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"No study ZIP files found in {root}")
    return zips


def extract_native_images(zip_path: Path, scratch: Path):
    """
    Organizer-style ZIP handling, except our NeuroVFM uses native img/
    instead of the baseline V-JEPA atlas/ images.
    """
    study = zip_path.stem
    dest = scratch / study

    if dest.exists():
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name for name in zf.namelist()
            if "/img/" in name
            and (name.endswith(".nii.gz") or name.endswith(".nii"))
        ]

        if not members:
            raise RuntimeError(
                f"{study}: ZIP contains no native */img/*.nii[.gz] files"
            )

        zf.extractall(dest, members)

    paths = sorted(dest.rglob("*.nii.gz")) + sorted(dest.rglob("*.nii"))

    if not paths:
        raise RuntimeError(f"{study}: native image extraction produced no files")

    return paths, dest


def preprocess_volume(path: Path, normalizer):
    # Exact live raw-NIfTI pipeline used by MR-RATE.
    vol = load_and_resample_nii(str(path), TARGET_SPACING)
    vol = normalizer.normalize(vol)
    vol = crop_or_pad(
        vol,
        TARGET_SHAPE,
        POSTERIOR_SHIFT_VOXELS,
    )

    return torch.from_numpy(
        np.ascontiguousarray(vol)
    ).unsqueeze(0).to(torch.bfloat16)


def build_model(device):
    args = SimpleNamespace(
        encoder="neurovfm",
        neurovfm_checkpoint=str(BASE_NEUROVFM),
        vjepa21_checkpoint=None,
        chunk_size=64,
        fusion_mode="late",
        pooling_strategy="simple_attn",
        dim_latent=512,
        extra_latent_projection=False,
    )

    print("[model] constructing NeuroVFM + MRRATE", flush=True)

    model, dim = build_encoder(args)

    if dim != 512:
        raise RuntimeError(f"Expected latent dim 512, got {dim}")

    print(f"[model] loading {MRRATE_CKPT}", flush=True)

    _load_and_verify(
        model,
        str(MRRATE_CKPT),
        strict_missing=True,
    )

    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    model.requires_grad_(False)

    return model


def load_classifier(device):
    package = torch.load(
        LINEAR_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    names = list(package["label_names"])

    if names != list(LABEL_COLS):
        raise RuntimeError(
            "Linear-head labels do not exactly match organizer LABEL_COLS"
        )

    state = package["state_dict"]

    weight = state["weight"].float().to(device)
    bias = state["bias"].float().to(device)

    if weight.shape != (32, 512):
        raise RuntimeError(
            f"Unexpected classifier weight shape: {tuple(weight.shape)}"
        )

    calibration = json.loads(CALIBRATION.read_text())

    if calibration["label_names"] != names:
        raise RuntimeError("Calibration labels differ from classifier labels")

    thresholds = torch.tensor(
        calibration["validation_f1_logit_thresholds"],
        dtype=torch.float32,
        device=device,
    )

    if thresholds.shape != (32,):
        raise RuntimeError(
            f"Expected 32 calibration thresholds, got {tuple(thresholds.shape)}"
        )

    return names, weight, bias, thresholds


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    device = torch.device("cuda")

    print("[runtime] GPU:", torch.cuda.get_device_name(0), flush=True)

    model = build_model(device)
    label_names, W, b, thresholds = load_classifier(device)

    normalizer = NORMALIZERS["percentile"]()

    studies = discover_zip_studies(INPUT_DIR)

    print(f"[input] discovered {len(studies)} ZIP studies", flush=True)

    scratch = Path("/tmp/mrrate-native")
    scratch.mkdir(parents=True, exist_ok=True)

    predictions = []
    total_start = time.time()

    for index, zip_path in enumerate(studies, 1):
        study = zip_path.stem
        study_start = time.time()
        extracted_dir = None

        try:
            paths, extracted_dir = extract_native_images(
                zip_path,
                scratch,
            )

            print(
                f"[{index}/{len(studies)}] {study}: "
                f"{len(paths)} native volumes",
                flush=True,
            )

            volumes = [
                preprocess_volume(path, normalizer)
                for path in paths
            ]

            # [1,N,1,D,H,W]
            images = torch.stack(volumes, dim=0).unsqueeze(0)

            real_volume_mask = torch.ones(
                (1, len(volumes)),
                dtype=torch.bool,
                device=device,
            )

            images = images.to(
                device=device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )

            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                feature = model(
                    text_input=None,
                    image=images,
                    device=device,
                    real_volume_mask=real_volume_mask,
                    return_loss=False,
                    return_visual_tokens=False,
                )

            if isinstance(feature, (tuple, list)):
                raise RuntimeError(
                    f"Unexpected pooled encoder return type: {type(feature)}"
                )

            feature = feature.float()

            if feature.ndim == 2 and feature.shape[0] == 1:
                feature = feature[0]

            feature = feature.reshape(-1)

            if feature.numel() != 512:
                raise RuntimeError(
                    f"Expected pooled 512-D feature, got {tuple(feature.shape)}"
                )

            raw_logits = W @ feature + b

            # Recenter each class so organizer's probability >= 0.5 threshold
            # implements the validation-selected F1 operating point.
            calibrated_logits = raw_logits - thresholds

            probs = torch.sigmoid(
                calibrated_logits
            ).detach().cpu().numpy()

            predictions.append(
                {
                    "input_image_name": f"{study}.nii.gz",
                    "probabilities": {
                        name: float(prob)
                        for name, prob in zip(label_names, probs)
                    },
                }
            )

            elapsed = time.time() - study_start

            print(
                f"[{index}/{len(studies)}] {study}: "
                f"{elapsed:.2f}s",
                flush=True,
            )

            del images, volumes, feature, raw_logits, calibrated_logits

            torch.cuda.empty_cache()

        except Exception as exc:
            print(
                f"WARN {study}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

            # Mirror organizer baseline behavior: keep one output row
            # even when a study fails.
            predictions.append(
                {
                    "input_image_name": f"{study}.nii.gz",
                    "probabilities": {
                        name: 0.0 for name in label_names
                    },
                }
            )

        finally:
            if extracted_dir is not None:
                shutil.rmtree(extracted_dir, ignore_errors=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    document = [
        {
            "outputs": [
                {
                    "value": {
                        "predictions": predictions
                    }
                }
            ]
        }
    ]

    output_path = OUTPUT_DIR / "predictions.json"

    with output_path.open("w") as handle:
        json.dump(document, handle)

    elapsed = time.time() - total_start

    print(
        f"[done] wrote {output_path} "
        f"({len(predictions)} studies) in {elapsed:.2f}s",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

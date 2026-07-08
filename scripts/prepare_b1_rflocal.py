#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def normalize_path(path_str: str) -> Path:
    raw = path_str.strip().strip('"')
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
        if match:
            drive = match.group(1).lower()
            tail = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{tail}").resolve()
    return Path(raw).expanduser().resolve()


def strip_nii_suffix(path: Path) -> Path:
    s = str(path)
    if s.endswith(".nii.gz"):
        return Path(s[:-7])
    if s.endswith(".nii"):
        return Path(s[:-4])
    return path


def resolve_nifti(path: Path) -> Path:
    candidates = [path]
    s = str(path)
    if not s.endswith(".nii") and not s.endswith(".nii.gz"):
        candidates.extend([Path(f"{s}.nii.gz"), Path(f"{s}.nii")])
    elif s.endswith(".nii"):
        candidates.append(Path(f"{s}.gz"))
    elif s.endswith(".nii.gz"):
        candidates.append(Path(s[:-3]))
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"NIfTI not found: tried {', '.join(str(x) for x in candidates)}")


def load_flip_angle(json_path: Path, fallback: float | None) -> float:
    if json_path.exists():
        meta = json.loads(json_path.read_text())
        flip = meta.get("FlipAngle")
        if flip is not None and float(flip) > 0:
            return float(flip)
    if fallback is not None and fallback > 0:
        return float(fallback)
    raise ValueError(f"Cannot determine FlipAngle from {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an affine-aware B1 RFlocal map aligned to an MT reference grid."
    )
    parser.add_argument("--b1", required=True, help="Raw B1 NIfTI path or stem.")
    parser.add_argument("--reference", required=True, help="Reference 3D NIfTI path or stem, usually MTon.")
    parser.add_argument("--output-base", required=True, help="Output stem without extension.")
    parser.add_argument(
        "--volume",
        type=int,
        default=1,
        help="0-based B1 volume index to use. Bruker B1Map typically uses volume 1 (the second volume).",
    )
    parser.add_argument(
        "--nominal-flip-angle",
        type=float,
        default=None,
        help="Optional fallback nominal flip angle in degrees if JSON is missing FlipAngle.",
    )
    parser.add_argument("--clip-min", type=float, default=0.3, help="Lower clip bound for RFlocal.")
    parser.add_argument("--clip-max", type=float, default=2.0, help="Upper clip bound for RFlocal.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    b1_input = resolve_nifti(normalize_path(args.b1))
    ref_input = resolve_nifti(normalize_path(args.reference))
    out_base = strip_nii_suffix(normalize_path(args.output_base))

    b1_img = nib.load(str(b1_input))
    if b1_img.ndim != 4 or b1_img.shape[3] <= args.volume:
        raise ValueError(f"B1 input must be 4D with volume index {args.volume}; got shape {b1_img.shape}")

    ref_img = nib.load(str(ref_input))
    if ref_img.ndim != 3:
        raise ValueError(f"Reference image must be 3D; got shape {ref_img.shape}")

    b1_data = np.asarray(b1_img.dataobj[..., args.volume], dtype=np.float32)
    b1_vol = nib.Nifti1Image(b1_data, b1_img.affine, b1_img.header.copy())
    b1_resampled = resample_from_to(b1_vol, ref_img, order=1)

    flip_angle = load_flip_angle(b1_input.with_suffix("").with_suffix(".json"), args.nominal_flip_angle)
    b1_deg = np.asarray(b1_resampled.get_fdata(dtype=np.float32), dtype=np.float32) * (180.0 / np.pi)
    rflocal = b1_deg / float(flip_angle)
    rflocal[~np.isfinite(rflocal)] = 0
    rflocal = np.clip(rflocal, float(args.clip_min), float(args.clip_max)).astype(np.float32, copy=False)

    out_img = nib.Nifti1Image(rflocal, ref_img.affine, ref_img.header.copy())
    out_img.set_data_dtype(np.float32)
    out_img.set_qform(*ref_img.get_qform(coded=True))
    out_img.set_sform(*ref_img.get_sform(coded=True))

    out_nii = Path(f"{out_base}.nii.gz")
    out_csv = Path(f"{out_base}.csv")
    out_json = Path(f"{out_base}.summary.json")
    out_nii.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_nii))

    np.savetxt(out_csv, np.ones((rflocal.shape[2],), dtype=np.uint8), fmt="%d")
    summary = {
        "input_b1": str(b1_input),
        "reference": str(ref_input),
        "output_rflocal": str(out_nii),
        "output_csv": str(out_csv),
        "volume_index": int(args.volume),
        "flip_angle_deg": float(flip_angle),
        "clip_min": float(args.clip_min),
        "clip_max": float(args.clip_max),
        "shape": [int(x) for x in rflocal.shape],
        "min": float(rflocal.min()),
        "max": float(rflocal.max()),
        "mean": float(rflocal.mean()),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

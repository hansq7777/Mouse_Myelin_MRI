#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np


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


def default_output_path(input_path: Path) -> Path:
    return Path(f"{strip_nii_suffix(input_path)}_display.nii.gz")


def diagonal_centered_affine(shape: tuple[int, int, int], zooms: tuple[float, float, float]) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = float(zooms[0])
    affine[1, 1] = float(zooms[1])
    affine[2, 2] = float(zooms[2])
    extent = (np.asarray(shape, dtype=np.float64) - 1.0) * np.asarray(zooms, dtype=np.float64)
    affine[:3, 3] = -0.5 * extent
    return affine


def load_single_volume(img: nib.Nifti1Image, volume_index: int | None) -> nib.Nifti1Image:
    if img.ndim <= 3:
        return img
    if volume_index is None:
        raise ValueError("Input is 4D; pass --volume to choose one volume for display export.")
    if volume_index < 0 or volume_index >= img.shape[3]:
        raise ValueError(f"--volume {volume_index} is out of range for shape {img.shape}")
    data = np.asarray(img.dataobj[..., volume_index])
    hdr = img.header.copy()
    hdr.set_data_shape(data.shape)
    return nib.Nifti1Image(data, img.affine, hdr)


def save_display_upright(input_path: Path, output_path: Path, volume_index: int | None) -> dict[str, object]:
    img = nib.load(str(input_path))
    img_3d = load_single_volume(img, volume_index)
    canonical = nib.as_closest_canonical(img_3d)

    raw = np.asarray(canonical.dataobj)
    orig_dtype = canonical.header.get_data_dtype()
    if np.issubdtype(orig_dtype, np.integer):
        data = np.rint(raw).astype(orig_dtype, copy=False)
    else:
        data = raw.astype(np.float32, copy=False)

    zooms = tuple(float(x) for x in canonical.header.get_zooms()[:3])
    affine = diagonal_centered_affine(tuple(int(x) for x in canonical.shape[:3]), zooms)

    hdr = canonical.header.copy()
    hdr.set_data_dtype(data.dtype)
    hdr.set_xyzt_units("mm", "sec")
    hdr["pixdim"][1:4] = np.asarray(zooms, dtype=np.float32)

    out = nib.Nifti1Image(data, affine, hdr)
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(output_path))
    return {
        "input": str(input_path),
        "output": str(output_path),
        "shape": [int(x) for x in out.shape[:3]],
        "zooms": [float(x) for x in zooms],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create upright, centered, display-only NIfTI copies without changing quantitative canonical files."
    )
    parser.add_argument("inputs", nargs="+", help="Input NIfTI paths (.nii or .nii.gz).")
    parser.add_argument(
        "--volume",
        type=int,
        default=None,
        help="For 4D inputs, choose the 0-based volume index to export.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path when exporting a single input. Defaults to <input>_display.nii.gz.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = [normalize_path(p) for p in args.inputs]

    if args.output and len(inputs) != 1:
        raise SystemExit("--output can only be used with a single input.")

    for idx, input_path in enumerate(inputs):
        output_path = normalize_path(args.output) if args.output and idx == 0 else default_output_path(input_path)
        result = save_display_upright(input_path, output_path, args.volume)
        print(
            f"display_upright wrote {result['output']} | "
            f"shape={tuple(result['shape'])} zooms={tuple(result['zooms'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def normalize_path(path_str: str) -> Path:
    raw = path_str.strip().strip('"')
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
        if match:
            drive = match.group(1).lower()
            tail = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{tail}").resolve()
    return Path(raw).expanduser().resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def robust_range(arr: np.ndarray) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    nonzero = vals[vals != 0]
    use = nonzero if nonzero.size > 100 else vals
    lo, hi = np.percentile(use, [1, 99]).tolist()
    if hi - lo < 1e-8:
        hi = lo + 1e-6
    return float(lo), float(hi)


def normalize_image(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = (arr - lo) / max(hi - lo, 1e-8)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out


def axis_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    idx = max(0, min(idx, arr.shape[axis] - 1))
    sl = np.take(arr, idx, axis=axis)
    return np.rot90(sl)


def grayscale_rgb(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    gray = (normalize_image(arr, lo, hi) * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def put_text(rgb: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(img)


def vconcat(images: list[np.ndarray], gap: int = 6) -> np.ndarray:
    max_w = max(img.shape[1] for img in images)
    padded: list[np.ndarray] = []
    for img in images:
        if img.shape[1] < max_w:
            pad_w = max_w - img.shape[1]
            img = np.pad(img, ((0, 0), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        padded.append(img)
    spacer = np.zeros((gap, max_w, 3), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for idx, img in enumerate(padded):
        if idx > 0:
            parts.append(spacer)
        parts.append(img)
    return np.concatenate(parts, axis=0)


def save_png(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(rgb).save(path)


def load_volume(path: Path) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {arr.ndim}D: {path}")
    return arr


def volume_center(arr: np.ndarray) -> tuple[int, int, int]:
    mask = np.isfinite(arr) & (arr != 0)
    if np.count_nonzero(mask) > 100:
        coords = np.argwhere(mask)
        return tuple(int(np.median(coords[:, axis])) for axis in range(3))
    return tuple(int((arr.shape[axis] - 1) / 2) for axis in range(3))


def map_idx(idx: int, ref_dim: int, dim: int) -> int:
    if dim <= 1:
        return 0
    if ref_dim <= 1:
        return min(max(idx, 0), dim - 1)
    ratio = idx / max(ref_dim - 1, 1)
    mapped = int(round(ratio * (dim - 1)))
    return min(max(mapped, 0), dim - 1)


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return cleaned.strip("_") or "image"


def render_triview(arr: np.ndarray, center: tuple[int, int, int], *, label: str, source: Path) -> np.ndarray:
    lo, hi = robust_range(arr)
    panels: list[np.ndarray] = []
    for axis, axis_name in enumerate(("Sag", "Cor", "Axi")):
        sl = axis_slice(arr, axis, center[axis])
        panel = grayscale_rgb(sl, lo, hi)
        panel = put_text(panel, f"{label} {axis_name} @ {center[axis]}")
        panels.append(panel)
    header = np.zeros((38, max(p.shape[1] for p in panels), 3), dtype=np.uint8)
    img = Image.fromarray(header)
    draw = ImageDraw.Draw(img)
    draw.text((6, 4), label, fill=(255, 255, 255))
    draw.text((6, 20), source.name, fill=(200, 200, 200))
    return vconcat([np.asarray(img), *panels], gap=6)


def parse_item(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Item must be LABEL=PATH, got: {text}")
    label, path_text = text.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Missing label in item: {text}")
    path = normalize_path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    return label, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render tri-view QC PNGs for one or more NIfTI volumes. "
            "Each output stacks sagittal/coronal/axial views from top to bottom."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Directory where PNGs and manifest are written.")
    parser.add_argument(
        "--center-from",
        required=True,
        help="Reference image used to choose the common tri-view center.",
    )
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        help="Repeatable LABEL=PATH item. Output PNGs are ordered by appearance.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest path. Default: <output-dir>/mt_triview_qc_manifest.json.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = ensure_dir(normalize_path(args.output_dir))
    manifest_path = normalize_path(args.manifest) if args.manifest.strip() else output_dir / "mt_triview_qc_manifest.json"

    items = [parse_item(raw) for raw in args.item]
    center_ref_path = normalize_path(args.center_from)
    center_ref_arr = load_volume(center_ref_path)
    center_ref = volume_center(center_ref_arr)

    manifest: dict[str, Any] = {
        "output_dir": str(output_dir),
        "center_from": str(center_ref_path),
        "center_from_shape": list(center_ref_arr.shape),
        "center_from_index": list(center_ref),
        "items": [],
    }

    for idx, (label, source) in enumerate(items, start=1):
        arr = load_volume(source)
        mapped_center = tuple(
            map_idx(center_ref[axis], center_ref_arr.shape[axis], arr.shape[axis]) for axis in range(3)
        )
        png_name = f"{idx:02d}_{sanitize_label(label)}_triview_qc.png"
        png_path = output_dir / png_name
        save_png(render_triview(arr, mapped_center, label=label, source=source), png_path)
        manifest["items"].append(
            {
                "label": label,
                "source_path": str(source),
                "shape": list(arr.shape),
                "center_index": list(mapped_center),
                "qc_png": str(png_path),
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

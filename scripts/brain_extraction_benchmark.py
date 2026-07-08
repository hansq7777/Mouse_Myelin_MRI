#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


REPO_ROOT = Path(__file__).resolve().parents[1]
BRAIN_TOOLS_ROOT = REPO_ROOT / "third_party" / "brain_extraction_tools"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_path(path_str: str) -> Path:
    raw = path_str.strip().strip('"')
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
        if match:
            drive = match.group(1).lower()
            tail = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{tail}").resolve()
    return Path(raw).expanduser().resolve()


def to_windows_path(path: Path) -> str:
    s = str(path.resolve())
    match = re.match(r"^/mnt/([a-z])/(.*)$", s)
    if not match:
        return s
    drive = match.group(1).upper()
    tail = match.group(2).replace("/", "\\")
    return f"{drive}:\\{tail}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def robust_range(arr: np.ndarray) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    pos = vals[vals > 0]
    use = pos if pos.size > 100 else vals
    lo, hi = np.percentile(use, [1, 99.5]).tolist()
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


def hconcat(images: list[np.ndarray], gap: int = 4) -> np.ndarray:
    max_h = max(img.shape[0] for img in images)
    padded: list[np.ndarray] = []
    for img in images:
        if img.shape[0] < max_h:
            pad_h = max_h - img.shape[0]
            img = np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode="constant", constant_values=0)
        padded.append(img)
    spacer = np.zeros((max_h, gap, 3), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for idx, img in enumerate(padded):
        if idx > 0:
            parts.append(spacer)
        parts.append(img)
    return np.concatenate(parts, axis=1)


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


def grayscale_rgb(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    gray = (normalize_image(arr, lo, hi) * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def overlay_mask(base_rgb: np.ndarray, mask_sl: np.ndarray) -> np.ndarray:
    rgb = base_rgb.astype(np.float32) / 255.0
    alpha = mask_sl.astype(np.float32) * 0.45
    rgb[..., 0] = np.clip((1.0 - alpha) * rgb[..., 0] + alpha * 1.0, 0.0, 1.0)
    rgb[..., 1] = np.clip((1.0 - alpha) * rgb[..., 1] + alpha * 0.15, 0.0, 1.0)
    rgb[..., 2] = np.clip((1.0 - alpha) * rgb[..., 2] + alpha * 0.0, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def outline_mask(base_rgb: np.ndarray, mask_sl: np.ndarray) -> np.ndarray:
    edges = mask_sl ^ ndi.binary_erosion(mask_sl, iterations=1)
    rgb = base_rgb.copy()
    rgb[edges] = np.array([255, 220, 0], dtype=np.uint8)
    return rgb


def put_text(rgb: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(img)


def save_png(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(rgb).save(path)


def save_nifti_like(reference_img: nib.Nifti1Image, data: np.ndarray, path: Path) -> None:
    out = nib.Nifti1Image(data, affine=reference_img.affine, header=reference_img.header.copy())
    nib.save(out, str(path))


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return labeled == keep


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not np.any(mask):
        return mask.astype(np.uint8)
    mask = ndi.binary_fill_holes(mask)
    mask = largest_component(mask)
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> tuple[list[int], list[int]] | None:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0).astype(int).tolist()
    maxs = coords.max(axis=0).astype(int).tolist()
    return mins, maxs


def mask_stats(mask: np.ndarray, zooms: tuple[float, float, float]) -> dict[str, Any]:
    mask = mask.astype(bool)
    voxels = int(mask.sum())
    voxel_volume = float(np.prod(zooms))
    stats: dict[str, Any] = {
        "voxels": voxels,
        "voxel_volume_mm3": voxel_volume,
        "volume_mm3": float(voxels * voxel_volume),
        "num_components": 0,
        "largest_component_fraction": 0.0,
        "touches_edge": False,
        "center_of_mass_vox": None,
        "bbox_min_vox": None,
        "bbox_max_vox": None,
    }
    if voxels == 0:
        return stats
    labeled, num = ndi.label(mask)
    stats["num_components"] = int(num)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = int(counts.max())
    stats["largest_component_fraction"] = float(largest / voxels)
    com = ndi.center_of_mass(mask.astype(np.float32))
    stats["center_of_mass_vox"] = [float(x) for x in com]
    bbox = bbox_from_mask(mask.astype(np.uint8))
    if bbox is not None:
        mins, maxs = bbox
        stats["bbox_min_vox"] = mins
        stats["bbox_max_vox"] = maxs
        stats["touches_edge"] = bool(
            any(v == 0 for v in mins) or any(v == s - 1 for v, s in zip(maxs, mask.shape))
        )
    return stats


def rough_foreground_mask(arr: np.ndarray) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    vals = vals[vals > 0]
    if vals.size < 100:
        return np.zeros(arr.shape, dtype=np.uint8)
    hi = float(np.percentile(vals, 99.7))
    use = np.clip(vals, 0.0, hi)
    thr = float(threshold_otsu(use))
    mask = np.isfinite(arr) & (arr > thr)
    mask = largest_component(mask)
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(np.uint8)


def rough_center(mask: np.ndarray, fallback_shape: tuple[int, int, int]) -> list[int]:
    if np.any(mask):
        com = ndi.center_of_mass(mask.astype(np.float32))
        return [int(round(float(x))) for x in com]
    return [int((s - 1) / 2) for s in fallback_shape]


def build_qc_composite(
    image: np.ndarray,
    mask: np.ndarray,
    center: list[int],
    *,
    title: str,
    subtitle: str,
) -> np.ndarray:
    lo, hi = robust_range(image)
    rows: list[np.ndarray] = []
    row_titles = [
        ("Input", False, False),
        ("Overlay", True, False),
        ("Masked Brain", False, True),
        ("Mask Outline", False, False),
    ]
    for row_name, use_overlay, use_masked in row_titles:
        panels: list[np.ndarray] = []
        for axis, axis_name in enumerate(("Sag", "Cor", "Axi")):
            img_sl = axis_slice(image, axis, center[axis])
            mask_sl = axis_slice(mask, axis, center[axis]) > 0
            if use_masked:
                base = grayscale_rgb(img_sl * mask_sl.astype(img_sl.dtype), lo, hi)
            else:
                base = grayscale_rgb(img_sl, lo, hi)
            if use_overlay:
                base = overlay_mask(base, mask_sl)
            if row_name == "Mask Outline":
                base = outline_mask(base, mask_sl)
            panels.append(put_text(base, f"{row_name} {axis_name} @ {center[axis]}"))
        rows.append(hconcat(panels))
    canvas = vconcat(rows)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    header_h = 44
    header = Image.new("RGB", (img.width, header_h), color=(0, 0, 0))
    draw_h = ImageDraw.Draw(header)
    draw_h.text((6, 4), title, fill=(255, 255, 255))
    draw_h.text((6, 22), subtitle, fill=(200, 200, 200))
    out = vconcat([np.asarray(header), np.asarray(img)], gap=4)
    return out


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    ended = time.time()
    ensure_dir(log_path.parent)
    log_text = []
    log_text.append(f"[start] {datetime.fromtimestamp(started).isoformat(timespec='seconds')}")
    log_text.append(f"[end] {datetime.fromtimestamp(ended).isoformat(timespec='seconds')}")
    log_text.append(f"[cwd] {cwd}")
    log_text.append(f"[return_code] {completed.returncode}")
    log_text.append("[command]")
    log_text.append(" ".join(cmd))
    log_text.append("")
    log_text.append("[stdout]")
    log_text.append(completed.stdout or "")
    log_text.append("")
    log_text.append("[stderr]")
    log_text.append(completed.stderr or "")
    log_path.write_text("\n".join(log_text), encoding="utf-8")
    return {
        "command": cmd,
        "cwd": str(cwd),
        "return_code": int(completed.returncode),
        "duration_sec": float(ended - started),
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
        "log_path": str(log_path),
    }


def discover_nifti_candidates(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))])


def pick_mask_candidate(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    mask_like = [p for p in candidates if "mask" in p.name.lower()]
    if mask_like:
        mask_like.sort(key=lambda p: (len(p.name), str(p)))
        return mask_like[0]
    return candidates[0]


def download_file(url: str, dest: Path) -> None:
    ensure_dir(dest.parent)
    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        shutil.copyfileobj(response, f, length=1024 * 1024)


def ensure_mbe_weights(external_dir: Path) -> Path:
    candidates = [
        external_dir / "mbe_weights" / "MBE_weights" / "exvivo" / "checkpoint_best.pth",
        external_dir / "mbe_weights" / "mod5" / "exvivo" / "checkpoint_best.pth",
    ]
    for checkpoint in candidates:
        if checkpoint.exists():
            return checkpoint
    archive = external_dir / "mbe_weights.tar.gz"
    extract_dir = external_dir / "mbe_weights"
    if not archive.exists():
        download_file("https://data.mousesuite.org/mbe/MBE_weights.tar.gz", archive)
    ensure_dir(extract_dir)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(extract_dir)
    for checkpoint in candidates:
        if checkpoint.exists():
            return checkpoint
    raise FileNotFoundError(
        "MBE ex vivo checkpoint not found after extraction. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def make_method_dirs(root: Path, order: int, name: str) -> dict[str, Path]:
    base = root / "methods" / f"{order:02d}_{name}"
    return {
        "base": ensure_dir(base),
        "input": ensure_dir(base / "input"),
        "outputs": ensure_dir(base / "outputs"),
        "qc": ensure_dir(base / "qc"),
        "logs": ensure_dir(base / "logs"),
        "meta": ensure_dir(base / "meta"),
    }


def copy_input(src: Path, dst_dir: Path) -> Path:
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def finalize_mask_outputs(
    *,
    method_name: str,
    dirs: dict[str, Path],
    input_img: nib.Nifti1Image,
    input_arr: np.ndarray,
    mask_path: Path,
    note: str = "",
) -> dict[str, Any]:
    raw_img = nib.load(str(mask_path))
    raw_arr = np.asarray(raw_img.get_fdata())
    raw_mask = (np.isfinite(raw_arr) & (raw_arr > 0)).astype(np.uint8)
    cleaned = clean_mask(raw_mask)
    raw_mask_path = dirs["outputs"] / f"{method_name}_mask_raw.nii.gz"
    final_mask_path = dirs["outputs"] / f"{method_name}_mask_final.nii.gz"
    brain_path = dirs["outputs"] / f"{method_name}_brain_final.nii.gz"
    save_nifti_like(input_img, raw_mask.astype(np.uint8), raw_mask_path)
    save_nifti_like(input_img, cleaned.astype(np.uint8), final_mask_path)
    save_nifti_like(input_img, (input_arr * cleaned).astype(np.float32), brain_path)
    zooms = tuple(float(x) for x in input_img.header.get_zooms()[:3])
    raw_stats = mask_stats(raw_mask, zooms)
    final_stats = mask_stats(cleaned, zooms)
    if final_stats["voxels"] <= 0:
        return {
            "status": "fail",
            "reason": "empty_final_mask",
            "raw_mask_path": str(raw_mask_path),
            "final_mask_path": str(final_mask_path),
            "brain_path": str(brain_path),
            "raw_stats": raw_stats,
            "final_stats": final_stats,
            "note": note,
        }
    center = rough_center(cleaned, input_arr.shape[:3])
    qc = build_qc_composite(
        input_arr,
        cleaned,
        center,
        title=f"{method_name} COM Tri-View QC",
        subtitle=f"COM(vox)={center} | raw_vol={raw_stats['volume_mm3']:.2f} mm3 | final_vol={final_stats['volume_mm3']:.2f} mm3",
    )
    qc_path = dirs["qc"] / f"{method_name}_qc_com_triview.png"
    save_png(qc, qc_path)
    return {
        "status": "ok",
        "raw_mask_path": str(raw_mask_path),
        "final_mask_path": str(final_mask_path),
        "brain_path": str(brain_path),
        "qc_path": str(qc_path),
        "raw_stats": raw_stats,
        "final_stats": final_stats,
        "center_vox": center,
        "note": note,
    }


def run_mbe(
    output_root: Path,
    input_src: Path,
    input_img: nib.Nifti1Image,
    input_arr: np.ndarray,
    external_dir: Path,
) -> dict[str, Any]:
    dirs = make_method_dirs(output_root, 1, "mbe")
    input_copy = copy_input(input_src, dirs["input"])
    weights_path = ensure_mbe_weights(external_dir)
    raw_mask = dirs["outputs"] / "mbe_raw_mask.nii.gz"
    posenc = dirs["outputs"] / "mbe_posenc.nii.gz"
    cmd = [
        str(BRAIN_TOOLS_ROOT / "bin" / "run_mbe.sh"),
        "-i",
        str(input_copy),
        "-o",
        str(raw_mask),
        "-n",
        str(weights_path),
        "--dstype",
        "exvivo",
        "--device",
        "cpu",
        "--gen_posenc",
        str(posenc),
        "--pp",
    ]
    run_info = run_command(cmd, REPO_ROOT, dirs["logs"] / "run.log.txt")
    if run_info["return_code"] != 0:
        return {"status": "fail", "reason": "command_failed", "run": run_info}
    if not raw_mask.exists():
        return {"status": "fail", "reason": "raw_mask_missing", "run": run_info}
    out = finalize_mask_outputs(
        method_name="mbe",
        dirs=dirs,
        input_img=input_img,
        input_arr=input_arr,
        mask_path=raw_mask,
        note="Official MBE ex vivo mouse checkpoint downloaded from data.mousesuite.org.",
    )
    out["run"] = run_info
    out["weights_path"] = str(weights_path)
    return out


def run_munet(output_root: Path, input_src: Path, input_img: nib.Nifti1Image, input_arr: np.ndarray) -> dict[str, Any]:
    dirs = make_method_dirs(output_root, 2, "munet")
    input_copy = copy_input(input_src, dirs["input"])
    workdir = BRAIN_TOOLS_ROOT / "sources" / "MU-Net"
    attempts: list[dict[str, Any]] = []
    for idx, use_bbox in enumerate((True, False), start=1):
        cmd = [
            str(BRAIN_TOOLS_ROOT / "bin" / "run_munet.sh"),
            "--overwrite",
            "True",
            "--useGPU",
            "False",
            "--boundingbox",
            "True" if use_bbox else "False",
            "--multinet",
            "True",
            "--out",
            "run",
            str(input_copy),
        ]
        run_info = run_command(cmd, workdir, dirs["logs"] / f"run_attempt_{idx}.log.txt")
        candidates = discover_nifti_candidates(dirs["input"])
        mask_path = pick_mask_candidate(
            [p for p in candidates if p.name.endswith("_run_Mask.nii.gz") or "mask" in p.name.lower()]
        )
        attempts.append(
            {
                "use_boundingbox": use_bbox,
                "run": run_info,
                "mask_path": str(mask_path) if mask_path else None,
            }
        )
        if run_info["return_code"] == 0 and mask_path is not None:
            break
    else:
        return {
            "status": "fail",
            "reason": "mask_not_discovered",
            "attempts": attempts,
            "candidates": [str(p) for p in discover_nifti_candidates(dirs["input"])],
        }
    out = finalize_mask_outputs(
        method_name="munet",
        dirs=dirs,
        input_img=input_img,
        input_arr=input_arr,
        mask_path=mask_path,
        note=(
            "MU-Net used bundled Fold1-5 weights with CPU inference. "
            "If the auxiliary bounding-box network rejected the ex vivo volume, the script retried with --boundingbox False."
        ),
    )
    out["run"] = attempts[-1]["run"]
    out["attempts"] = attempts
    out["discovered_mask"] = str(mask_path)
    return out


def run_rbm(output_root: Path, input_src: Path, input_img: nib.Nifti1Image, input_arr: np.ndarray) -> dict[str, Any]:
    dirs = make_method_dirs(output_root, 3, "rbm")
    input_copy = copy_input(input_src, dirs["input"])
    raw_mask = dirs["outputs"] / "rbm_raw_mask.nii.gz"
    cmd = [
        str(BRAIN_TOOLS_ROOT / "bin" / "run_rbm.sh"),
        "-s",
        "0.08",
        str(input_copy),
        str(raw_mask),
    ]
    run_info = run_command(cmd, REPO_ROOT, dirs["logs"] / "run.log.txt")
    if run_info["return_code"] != 0:
        return {"status": "fail", "reason": "command_failed", "run": run_info}
    if not raw_mask.exists():
        return {"status": "fail", "reason": "raw_mask_missing", "run": run_info}
    out = finalize_mask_outputs(
        method_name="rbm",
        dirs=dirs,
        input_img=input_img,
        input_arr=input_arr,
        mask_path=raw_mask,
        note="RBM resampled to 0.08 mm isotropic before inference to better match mouse ex vivo scale than the rat-oriented 0.1 mm default.",
    )
    out["run"] = run_info
    return out


def run_pcnn3d(output_root: Path, input_src: Path, input_img: nib.Nifti1Image, input_arr: np.ndarray) -> dict[str, Any]:
    dirs = make_method_dirs(output_root, 4, "pcnn3d")
    input_copy = copy_input(input_src, dirs["input"])
    output_prefix = dirs["outputs"] / "pcnn3d"
    temp_dir = ensure_dir(dirs["outputs"] / "tmp")
    cmd = [
        str(BRAIN_TOOLS_ROOT / "bin" / "run_pcnn3d.sh"),
        "-i",
        str(input_copy),
        "-o",
        str(output_prefix),
        "-od",
        str(temp_dir),
        "-minv",
        "250",
        "-maxv",
        "650",
        "-bm",
        "2",
    ]
    run_info = run_command(cmd, REPO_ROOT, dirs["logs"] / "run.log.txt")
    candidates = discover_nifti_candidates(dirs["outputs"])
    mask_path = pick_mask_candidate(candidates)
    if run_info["return_code"] != 0:
        return {
            "status": "fail",
            "reason": "command_failed",
            "run": run_info,
            "candidates": [str(p) for p in candidates],
        }
    if mask_path is None:
        return {"status": "fail", "reason": "mask_not_discovered", "run": run_info}
    out = finalize_mask_outputs(
        method_name="pcnn3d",
        dirs=dirs,
        input_img=input_img,
        input_arr=input_arr,
        mask_path=mask_path,
        note="PCNN3D used a mouse-oriented brain volume prior range of 250-650 mm3 to tolerate fixed-brain ex vivo contrast while avoiding tube/background capture.",
    )
    out["run"] = run_info
    out["discovered_mask"] = str(mask_path)
    return out


def bet_score(stats: dict[str, Any]) -> float:
    if stats["voxels"] <= 0:
        return 1e9
    volume = float(stats["volume_mm3"])
    score = abs(volume - 430.0) / 100.0
    if volume < 250.0 or volume > 700.0:
        score += 12.0
    score += max(0, int(stats["num_components"]) - 1) * 2.0
    score += (1.0 - float(stats["largest_component_fraction"])) * 6.0
    if bool(stats["touches_edge"]):
        score += 2.0
    return score


def run_bet(output_root: Path, input_src: Path, input_img: nib.Nifti1Image, input_arr: np.ndarray, rough_center_vox: list[int]) -> dict[str, Any]:
    dirs = make_method_dirs(output_root, 5, "bet")
    input_copy = copy_input(input_src, dirs["input"])
    candidate_dir = ensure_dir(dirs["outputs"] / "candidates")
    zooms = tuple(float(x) for x in input_img.header.get_zooms()[:3])
    candidate_records: list[dict[str, Any]] = []
    for frac in (0.08, 0.10, 0.12, 0.15, 0.18, 0.22):
        tag = f"f{int(round(frac * 1000)):03d}"
        prefix = candidate_dir / f"bet_{tag}"
        cmd = [
            str(BRAIN_TOOLS_ROOT / "bin" / "run_bet.sh"),
            str(input_copy),
            str(prefix),
            "-m",
            "-R",
            "-f",
            f"{frac:.2f}",
            "-c",
            str(rough_center_vox[0]),
            str(rough_center_vox[1]),
            str(rough_center_vox[2]),
        ]
        run_info = run_command(cmd, REPO_ROOT, dirs["logs"] / f"bet_{tag}.log.txt")
        mask_path = prefix.with_name(prefix.name + "_mask.nii.gz")
        record = {
            "fractional_threshold": frac,
            "prefix": str(prefix),
            "mask_path": str(mask_path),
            "run": run_info,
            "status": "ok" if run_info["return_code"] == 0 and mask_path.exists() else "fail",
        }
        if mask_path.exists():
            raw_img = nib.load(str(mask_path))
            raw_mask = (np.asarray(raw_img.get_fdata()) > 0).astype(np.uint8)
            record["raw_stats"] = mask_stats(raw_mask, zooms)
            record["score"] = bet_score(record["raw_stats"])
        else:
            record["score"] = 1e9
        candidate_records.append(record)
    successful = [r for r in candidate_records if r["status"] == "ok"]
    if not successful:
        return {"status": "fail", "reason": "no_successful_candidate", "candidates": candidate_records}
    best = min(successful, key=lambda r: (float(r["score"]), float(r["fractional_threshold"])))
    best_mask = Path(str(best["mask_path"]))
    out = finalize_mask_outputs(
        method_name="bet",
        dirs=dirs,
        input_img=input_img,
        input_arr=input_arr,
        mask_path=best_mask,
        note=(
            "BET used a small parameter sweep across -f values with -R and an intensity-derived center prior. "
            "The selected candidate minimizes a mouse-brain volume heuristic while penalizing edge leakage."
        ),
    )
    out["selected_candidate"] = best
    out["candidates"] = candidate_records
    return out


def skipped_method(name: str, reason: str) -> dict[str, Any]:
    return {"status": "skip", "reason": reason}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(
    *,
    output_root: Path,
    input_info: dict[str, Any],
    records: dict[str, Any],
) -> None:
    summary_rows: list[str] = []
    summary_rows.append(
        "\t".join(
            [
                "method",
                "status",
                "final_mask_path",
                "qc_path",
                "volume_mm3",
                "num_components",
                "touches_edge",
                "note_or_reason",
            ]
        )
    )
    for name, rec in records.items():
        if rec.get("status") == "ok":
            stats = rec["final_stats"]
            summary_rows.append(
                "\t".join(
                    [
                        name,
                        "ok",
                        rec["final_mask_path"],
                        rec["qc_path"],
                        f"{stats['volume_mm3']:.3f}",
                        str(stats["num_components"]),
                        str(stats["touches_edge"]),
                        rec.get("note", ""),
                    ]
                )
            )
        else:
            summary_rows.append(
                "\t".join(
                    [
                        name,
                        rec.get("status", "unknown"),
                        "",
                        "",
                        "",
                        "",
                        "",
                        rec.get("reason", ""),
                    ]
                )
            )
    (output_root / "summary.tsv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# Brain Extraction Benchmark")
    lines.append("")
    lines.append(f"- Run time: `{now_iso()}`")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Output root (Windows): `{to_windows_path(output_root)}`")
    lines.append(f"- Input used for inference: `{input_info['main_input']}`")
    lines.append(f"- Input used for inference (Windows): `{input_info['main_input_windows']}`")
    lines.append(f"- Raw T2 reference: `{input_info['raw_input']}`")
    lines.append(f"- Acquisition summary: `{input_info['acquisition_summary']}`")
    lines.append("")
    lines.append("## Directory Layout")
    lines.append("")
    lines.append("- `metadata/`: copied JSON/method file snapshots and rough foreground QC.")
    lines.append("- `methods/<nn>_<name>/input`: method-local input copy.")
    lines.append("- `methods/<nn>_<name>/outputs`: raw outputs, final mask, brain-only NIfTI, candidate runs.")
    lines.append("- `methods/<nn>_<name>/logs`: full stdout/stderr and command lines.")
    lines.append("- `methods/<nn>_<name>/qc`: COM tri-view PNG for manual review.")
    lines.append("")
    lines.append("## Method Results")
    lines.append("")
    for name, rec in records.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Status: `{rec.get('status', 'unknown')}`")
        if rec.get("status") == "ok":
            stats = rec["final_stats"]
            lines.append(f"- Final mask: `{rec['final_mask_path']}`")
            lines.append(f"- Brain-only image: `{rec['brain_path']}`")
            lines.append(f"- QC PNG: `{rec['qc_path']}`")
            lines.append(f"- Volume: `{stats['volume_mm3']:.2f} mm3`")
            lines.append(f"- Components: `{stats['num_components']}`")
            lines.append(f"- Touches image edge: `{stats['touches_edge']}`")
            lines.append(f"- Note: {rec.get('note', '')}")
            if "run" in rec:
                lines.append(f"- Log: `{rec['run']['log_path']}`")
        else:
            lines.append(f"- Reason: `{rec.get('reason', 'unknown')}`")
        lines.append("")
    readme = "\n".join(lines).rstrip() + "\n"
    (output_root / "README.md").write_text(readme, encoding="utf-8")


@dataclass
class AcquisitionSnapshot:
    magnetic_field_strength_t: float | None
    sequence: str
    protocol: str
    echo_time_ms: float | None
    repetition_time_ms: float | None
    shape: list[int]
    zooms_mm: list[float]

    def summary(self) -> str:
        return (
            f"{self.magnetic_field_strength_t or 'NA'}T | {self.sequence} | {self.protocol} | "
            f"TE={self.echo_time_ms or 'NA'} ms | TR={self.repetition_time_ms or 'NA'} ms | "
            f"shape={self.shape} | zooms_mm={self.zooms_mm}"
        )


def load_acquisition_snapshot(t2_json_path: Path, img: nib.Nifti1Image) -> AcquisitionSnapshot:
    info = json.loads(t2_json_path.read_text(encoding="utf-8"))
    return AcquisitionSnapshot(
        magnetic_field_strength_t=info.get("MagneticFieldStrength"),
        sequence=str(info.get("SeriesDescription", "")),
        protocol=str(info.get("ProtocolName", "")),
        echo_time_ms=(float(info["EchoTime"]) * 1000.0) if info.get("EchoTime") is not None else None,
        repetition_time_ms=(float(info["RepetitionTime"]) * 1000.0) if info.get("RepetitionTime") is not None else None,
        shape=[int(x) for x in img.shape[:3]],
        zooms_mm=[float(x) for x in img.header.get_zooms()[:3]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible mouse T2 brain extraction benchmark with per-method logs, "
            "cleaned masks, and COM tri-view QC."
        )
    )
    parser.add_argument("--input", required=True, help="Main input NIfTI used for inference, e.g. T2_ungibbs.nii.gz")
    parser.add_argument("--raw-input", required=True, help="Raw T2 NIfTI for provenance, e.g. T2.nii.gz")
    parser.add_argument("--t2-json", required=True, help="T2 sidecar JSON path")
    parser.add_argument("--bruker-method", required=False, help="Optional Bruker method file path")
    parser.add_argument("--output-root", required=False, help="Output directory. Defaults to test/brain_extraction_<timestamp>")
    args = parser.parse_args()

    input_path = normalize_path(args.input)
    raw_input_path = normalize_path(args.raw_input)
    t2_json_path = normalize_path(args.t2_json)
    bruker_method_path = normalize_path(args.bruker_method) if args.bruker_method else None
    default_output = REPO_ROOT / "test" / f"brain_extraction_RD25MYELOMAP_Test9_20260216_{now_stamp()}"
    output_root = normalize_path(args.output_root) if args.output_root else default_output
    metadata_dir = ensure_dir(output_root / "metadata")
    external_dir = ensure_dir(output_root / "external")

    input_img = nib.load(str(input_path))
    input_arr = np.asarray(input_img.get_fdata(), dtype=np.float32)
    write_json(
        metadata_dir / "input_snapshot.json",
        {
            "input_path": str(input_path),
            "input_path_windows": to_windows_path(input_path),
            "raw_input_path": str(raw_input_path),
            "raw_input_path_windows": to_windows_path(raw_input_path),
            "shape": [int(x) for x in input_arr.shape],
            "zooms_mm": [float(x) for x in input_img.header.get_zooms()[:3]],
            "sha256": sha256_file(input_path),
        },
    )

    shutil.copy2(t2_json_path, metadata_dir / t2_json_path.name)
    if bruker_method_path is not None and bruker_method_path.exists():
        shutil.copy2(bruker_method_path, metadata_dir / bruker_method_path.name)

    acq = load_acquisition_snapshot(t2_json_path, input_img)
    rough_mask = rough_foreground_mask(input_arr)
    rough_mask_path = metadata_dir / "rough_foreground_mask.nii.gz"
    save_nifti_like(input_img, rough_mask.astype(np.uint8), rough_mask_path)
    rough_center_vox = rough_center(rough_mask, input_arr.shape[:3])
    rough_qc = build_qc_composite(
        input_arr,
        rough_mask,
        rough_center_vox,
        title="Rough Foreground QC",
        subtitle=f"Used for BET center prior only. COM(vox)={rough_center_vox}",
    )
    save_png(rough_qc, metadata_dir / "rough_foreground_qc.png")

    records: dict[str, Any] = {}
    try:
        records["mbe"] = run_mbe(output_root, input_path, input_img, input_arr, external_dir)
    except Exception as exc:
        records["mbe"] = {"status": "fail", "reason": f"exception: {type(exc).__name__}: {exc}"}

    try:
        records["munet"] = run_munet(output_root, input_path, input_img, input_arr)
    except Exception as exc:
        records["munet"] = {"status": "fail", "reason": f"exception: {type(exc).__name__}: {exc}"}

    try:
        records["rbm"] = run_rbm(output_root, input_path, input_img, input_arr)
    except Exception as exc:
        records["rbm"] = {"status": "fail", "reason": f"exception: {type(exc).__name__}: {exc}"}

    try:
        records["pcnn3d"] = run_pcnn3d(output_root, input_path, input_img, input_arr)
    except Exception as exc:
        records["pcnn3d"] = {"status": "fail", "reason": f"exception: {type(exc).__name__}: {exc}"}

    try:
        records["bet"] = run_bet(output_root, input_path, input_img, input_arr, rough_center_vox)
    except Exception as exc:
        records["bet"] = {"status": "fail", "reason": f"exception: {type(exc).__name__}: {exc}"}

    records["ben"] = skipped_method(
        "ben",
        "Skipped in this run because no BEN pretrained weight bundle was present locally and the source tree did not include a ready-to-use weight directory.",
    )
    records["rs2"] = skipped_method(
        "rs2",
        "Skipped in this run because RS2_predict requires RS2_pretrained_model.pt and no local checkpoint was found in the deployed environment or source tree.",
    )
    records["ants_brain_extraction"] = skipped_method(
        "ants_brain_extraction",
        "Skipped in this run because antsBrainExtraction.sh requires a template image plus probability mask prior, which were not packaged with this deployment for mouse ex vivo T2.",
    )
    records["sherm"] = skipped_method(
        "sherm",
        "Skipped in this run because neither MATLAB nor GNU Octave was available in the environment.",
    )
    records["rats"] = skipped_method(
        "rats",
        "Skipped in this run because the deployment only contains a manual-download placeholder for RATS, not the official binaries/source package.",
    )

    manifest = {
        "run_time": now_iso(),
        "repo_root": str(REPO_ROOT),
        "repo_root_windows": to_windows_path(REPO_ROOT),
        "output_root": str(output_root),
        "output_root_windows": to_windows_path(output_root),
        "input_info": {
            "main_input": str(input_path),
            "main_input_windows": to_windows_path(input_path),
            "raw_input": str(raw_input_path),
            "raw_input_windows": to_windows_path(raw_input_path),
            "t2_json": str(t2_json_path),
            "t2_json_windows": to_windows_path(t2_json_path),
            "bruker_method": str(bruker_method_path) if bruker_method_path else None,
            "bruker_method_windows": to_windows_path(bruker_method_path) if bruker_method_path else None,
            "acquisition_summary": acq.summary(),
            "rough_center_vox": rough_center_vox,
            "rough_foreground_mask": str(rough_mask_path),
        },
        "records": records,
    }
    write_json(output_root / "manifest.json", manifest)
    write_summary(output_root=output_root, input_info=manifest["input_info"], records=records)

    print(str(output_root))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List

import nibabel as nib
import numpy as np


LOCKED_PREPROCESS_MODE = "n4_locked"
LOCKED_N4_CONVERGENCE = "[50x50x40x30,1e-7]"
LOCKED_N4_BSPLINE = "[0.8]"
LOCKED_N4_SHRINK_FACTOR = "2"
LOCKED_DENOISE_NOISE_MODEL = "Rician"
LOCKED_DENOISE_SHRINK_FACTOR = "1"
LOCKED_DENOISE_PATCH_RADIUS = "1x1x1"
LOCKED_DENOISE_SEARCH_RADIUS = "2x2x2"
LOCKED_OPT_NORM_CLIP_LOW_PERCENT = 1.0
LOCKED_OPT_NORM_CLIP_HIGH_PERCENT = 99.0
LOCKED_OPT_NORM_ROBUST_SCALE = 1.4826
LOCKED_AFFINE_FALLBACK_ORDER = "moments,geometry,antsai"
LOCKED_AFFINE_MIN_DICE = 0.55
LOCKED_AFFINE_MIN_NMI = 1.0
LOCKED_AFFINE_MIN_CC = 0.0
LOCKED_AFFINE_DET_MIN = 0.25
LOCKED_AFFINE_DET_MAX = 4.0
LOCKED_AFFINE_SV_MIN = 0.25
LOCKED_AFFINE_SV_MAX = 4.0
LOCKED_JAC_MIN = 0.05
LOCKED_JAC_P01_MIN = 0.20
LOCKED_JAC_P99_MAX = 5.0
LOCKED_JAC_NEG_FRAC_MAX = 0.001
LOCKED_WARP_L2_ENERGY_MAX = 50.0
LOCKED_TIE_BREAK_DICE_EPS = 0.005
LOCKED_COVERAGE_MARGIN_MIN_VOX = 10
LOCKED_COVERAGE_MARGIN_MIN_MM = 2.5
LOCKED_ANTSAI_METRIC = "MI"
LOCKED_ANTSAI_BINS = "32"
LOCKED_ANTSAI_SEARCH_FACTOR = "[20,1.0]"
LOCKED_ANTSAI_TRANSLATION_GRID = "[20,0x0x0]"
LOCKED_DEFORMATION_GRID_DIRECTIONS = "1x1x0"
LOCKED_DEFORMATION_GRID_SPACING = "10x10x10"
LOCKED_DEFORMATION_GRID_SIGMA = "1x1x1"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: List[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if completed.stdout:
        print(completed.stdout.strip(), flush=True)
    if completed.stderr:
        print(completed.stderr.strip(), flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def run_logged(cmd: List[str], *, env: dict[str, str] | None = None, log_path: Path | None = None) -> dict[str, Any]:
    print("$ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if stdout:
        print(stdout.strip(), flush=True)
    if stderr:
        print(stderr.strip(), flush=True)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "$ " + " ".join(cmd) + "\n"
            + "[return_code] " + str(completed.returncode) + "\n"
            + "[stdout]\n" + stdout + "\n"
            + "[stderr]\n" + stderr + "\n"
        )
        log_path.write_text(text, encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")

    return {
        "command": cmd,
        "return_code": int(completed.returncode),
        "log_path": str(log_path) if log_path else "",
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
    }


def ensure_exec(ants_bin: Path, name: str) -> str:
    candidates = [
        ants_bin / name,
        ants_bin / f"{name}.exe",
    ]
    for exe in candidates:
        if exe.exists():
            return str(exe)
    raise FileNotFoundError(f"Missing executable: {candidates[-1]}")


def normalize_path(path_str: str) -> Path:
    raw = path_str.strip().strip('"')
    if os.name != "nt":
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
        if m:
            drive = m.group(1).lower()
            tail = m.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{tail}").resolve()
    return Path(raw).expanduser().resolve()


def to_ants_path(path: Path, windows_mode: bool) -> str:
    if windows_mode:
        posix = path.as_posix()
        if posix.startswith("/mnt/"):
            parts = posix.split("/")
            if len(parts) >= 4:
                drive = parts[2].upper()
                tail = "/".join(parts[3:])
                return f"{drive}:/{tail}"
    return str(path)


def append_trace(trace_path: Path, event: dict[str, Any]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cast_nifti_dtype(path: Path, dtype: np.dtype) -> None:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=dtype)
    hdr = img.header.copy()
    hdr.set_data_dtype(dtype)
    out = nib.Nifti1Image(data, img.affine, hdr)
    qf, qcode = img.get_qform(coded=True)
    sf, scode = img.get_sform(coded=True)
    out.set_qform(qf if qf is not None else img.affine, code=int(qcode))
    out.set_sform(sf if sf is not None else img.affine, code=int(scode))
    nib.save(out, str(path))


def ensure_float32(path: Path) -> None:
    cast_nifti_dtype(path, np.float32)


def compute_mi_cc(fixed_path: Path, moving_path: Path, bins: int = 64) -> dict[str, float]:
    fixed = np.asarray(nib.load(str(fixed_path)).get_fdata(), dtype=np.float32)
    moving = np.asarray(nib.load(str(moving_path)).get_fdata(), dtype=np.float32)

    if fixed.shape != moving.shape:
        raise RuntimeError(f"QC shape mismatch: {fixed.shape} vs {moving.shape}")

    mask = np.isfinite(fixed) & np.isfinite(moving)
    if np.count_nonzero(mask) == 0:
        return {"cc": float("nan"), "mi": float("nan"), "nmi": float("nan")}

    x = fixed[mask].astype(np.float64)
    y = moving[mask].astype(np.float64)

    # CC
    sx = np.std(x)
    sy = np.std(y)
    if sx <= 1e-12 or sy <= 1e-12:
        cc = float("nan")
    else:
        cc = float(np.corrcoef(x, y)[0, 1])

    # MI / NMI
    x1, x99 = np.percentile(x, [1, 99])
    y1, y99 = np.percentile(y, [1, 99])
    x = np.clip(x, x1, x99)
    y = np.clip(y, y1, y99)

    h2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = h2d / np.maximum(h2d.sum(), 1e-12)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    pxpy = np.outer(px, py)
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(np.maximum(pxy[nz], 1e-12) / np.maximum(pxpy[nz], 1e-12))))

    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))
    hxy = -float(np.sum(pxy[nz] * np.log(np.maximum(pxy[nz], 1e-12))))
    nmi = float((hx + hy) / np.maximum(hxy, 1e-12))

    return {"cc": cc, "mi": mi, "nmi": nmi}


def compute_dice(mask_a: Path, mask_b: Path) -> float:
    a = np.asarray(nib.load(str(mask_a)).get_fdata(), dtype=np.float32) > 0
    b = np.asarray(nib.load(str(mask_b)).get_fdata(), dtype=np.float32) > 0
    if a.shape != b.shape:
        raise RuntimeError(f"Dice shape mismatch: {a.shape} vs {b.shape}")
    inter = np.count_nonzero(a & b)
    denom = np.count_nonzero(a) + np.count_nonzero(b)
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def evaluate_mask_coverage_margin(
    *,
    mask_path: Path,
    reference_image_path: Path,
    min_margin_vox: int,
    min_margin_mm: float,
    label: str,
) -> dict[str, Any]:
    ref_img = nib.load(str(reference_image_path))
    ref_shape = tuple(int(x) for x in ref_img.shape[:3])
    ref_zooms = tuple(float(abs(z)) for z in ref_img.header.get_zooms()[:3])
    if len(ref_shape) != 3 or len(ref_zooms) != 3:
        raise RuntimeError(f"Unsupported reference image dimensionality for coverage check: {reference_image_path}")

    mask_data = np.asarray(nib.load(str(mask_path)).get_fdata(), dtype=np.float32)
    if mask_data.ndim > 3:
        squeeze_ok = all(int(x) == 1 for x in mask_data.shape[3:])
        if squeeze_ok:
            mask_data = np.squeeze(mask_data)
    if mask_data.shape != ref_shape:
        raise RuntimeError(
            f"Coverage check shape mismatch for {label}: "
            f"mask={mask_data.shape}, reference={ref_shape}. "
            f"Mask must be in resample-contract grid."
        )

    mask = np.isfinite(mask_data) & (mask_data > 0)
    voxels = int(np.count_nonzero(mask))
    if voxels <= 0:
        raise RuntimeError(f"Coverage check failed for {label}: mask is empty ({mask_path}).")

    coords = np.argwhere(mask)
    bb_min = coords.min(axis=0).astype(int)
    bb_max = coords.max(axis=0).astype(int)
    shape_arr = np.asarray(ref_shape, dtype=np.int64)

    low_vox = bb_min.astype(np.int64)
    high_vox = (shape_arr - 1 - bb_max.astype(np.int64))
    margins_vox_all = np.concatenate([low_vox, high_vox]).astype(np.int64)
    zoom_arr = np.asarray(ref_zooms, dtype=np.float64)
    low_mm = low_vox.astype(np.float64) * zoom_arr
    high_mm = high_vox.astype(np.float64) * zoom_arr
    margins_mm_all = np.concatenate([low_mm, high_mm]).astype(np.float64)

    min_margin_vox_value = int(np.min(margins_vox_all))
    min_margin_mm_value = float(np.min(margins_mm_all))
    pass_vox = bool(min_margin_vox_value >= int(min_margin_vox))
    pass_mm = bool(min_margin_mm_value >= float(min_margin_mm))
    passed = bool(pass_vox or pass_mm)

    margin_labels = [
        "axis0_low",
        "axis1_low",
        "axis2_low",
        "axis0_high",
        "axis1_high",
        "axis2_high",
    ]
    margin_vox_detail = {k: int(v) for k, v in zip(margin_labels, margins_vox_all.tolist())}
    margin_mm_detail = {k: float(v) for k, v in zip(margin_labels, margins_mm_all.tolist())}
    bb_size_vox = (bb_max - bb_min + 1).astype(np.int64)
    bb_size_mm = bb_size_vox.astype(np.float64) * zoom_arr

    return {
        "label": label,
        "mask_path": str(mask_path),
        "reference_image": str(reference_image_path),
        "reference_shape": [int(x) for x in ref_shape],
        "reference_spacing_mm": [float(x) for x in ref_zooms],
        "mask_voxels": voxels,
        "bbox_min_vox": [int(x) for x in bb_min.tolist()],
        "bbox_max_vox": [int(x) for x in bb_max.tolist()],
        "bbox_size_vox": [int(x) for x in bb_size_vox.tolist()],
        "bbox_size_mm": [float(x) for x in bb_size_mm.tolist()],
        "margin_vox": margin_vox_detail,
        "margin_mm": margin_mm_detail,
        "min_margin_vox": min_margin_vox_value,
        "min_margin_mm": min_margin_mm_value,
        "thresholds": {
            "min_margin_vox": int(min_margin_vox),
            "min_margin_mm": float(min_margin_mm),
            "logic": "pass_if(min_margin_vox >= threshold_vox OR min_margin_mm >= threshold_mm)",
        },
        "pass_by_vox": pass_vox,
        "pass_by_mm": pass_mm,
        "passed": passed,
    }


def normalize_intensity_robust_z(
    image_path: Path,
    out_path: Path,
    mask_path: Path | None = None,
    *,
    clip_low_percent: float = LOCKED_OPT_NORM_CLIP_LOW_PERCENT,
    clip_high_percent: float = LOCKED_OPT_NORM_CLIP_HIGH_PERCENT,
) -> dict[str, float]:
    img = nib.load(str(image_path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    mask = np.isfinite(data)
    mask_source = "finite_nonzero"
    if mask_path and mask_path.exists():
        m = np.asarray(nib.load(str(mask_path)).get_fdata(), dtype=np.float32) > 0
        if m.shape != data.shape:
            raise RuntimeError(
                f"Normalization mask shape mismatch: image {data.shape}, mask {m.shape} ({mask_path})"
            )
        mask &= m
        mask_source = "provided_mask"
    else:
        mask &= data != 0

    if np.count_nonzero(mask) < 100:
        mask = np.isfinite(data)
        mask_source = "finite_fallback"
        if np.count_nonzero(mask) < 100:
            raise RuntimeError(f"Normalization failed: too few finite voxels in {image_path}")

    vals = data[mask].astype(np.float64)
    p_lo, p_hi = np.percentile(vals, [clip_low_percent, clip_high_percent])
    if not np.isfinite(p_lo):
        p_lo = float(np.nanmin(vals))
    if not np.isfinite(p_hi):
        p_hi = float(np.nanmax(vals))
    if p_hi - p_lo < 1e-8:
        p_lo = float(np.min(vals))
        p_hi = float(np.max(vals))

    clipped_vals = np.clip(vals, p_lo, p_hi)
    median = float(np.median(clipped_vals))
    mad = float(np.median(np.abs(clipped_vals - median)))
    robust_sigma = max(float(LOCKED_OPT_NORM_ROBUST_SCALE * mad), 1e-6)

    out = np.clip(data, p_lo, p_hi)
    out = (out - median) / robust_sigma
    out[~mask] = 0.0
    out[~np.isfinite(out)] = 0.0

    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    out_img = nib.Nifti1Image(out.astype(np.float32), img.affine, hdr)
    qf, qcode = img.get_qform(coded=True)
    sf, scode = img.get_sform(coded=True)
    out_img.set_qform(qf if qf is not None else img.affine, code=int(qcode))
    out_img.set_sform(sf if sf is not None else img.affine, code=int(scode))
    nib.save(out_img, str(out_path))
    return {
        "clip_low_percent": float(clip_low_percent),
        "clip_high_percent": float(clip_high_percent),
        "clip_low_value": float(p_lo),
        "clip_high_value": float(p_hi),
        "median_after_clip": float(median),
        "mad_after_clip": float(mad),
        "robust_sigma": float(robust_sigma),
        "masked_voxels": int(np.count_nonzero(mask)),
        "mask_source": mask_source,
    }


def preprocess_image(
    *,
    image_path: Path,
    out_dir: Path,
    stem: str,
    moving_denoise: bool,
    is_moving: bool,
    mask_path: Path | None,
    ants_denoise: str,
    ants_n4: str,
    windows_mode: bool,
    env: dict[str, str],
) -> tuple[Path, Path, dict[str, Any]]:
    current = image_path
    info: dict[str, Any] = {
        "mode": LOCKED_PREPROCESS_MODE,
        "input": str(image_path),
        "is_moving": bool(is_moving),
        "moving_denoise_enabled": bool(moving_denoise if is_moving else False),
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_moving and moving_denoise:
        denoise_out = out_dir / f"{stem}_denoise.nii.gz"
        denoise_noise = out_dir / f"{stem}_denoise_noise.nii.gz"
        denoise_log = out_dir / f"{stem}_denoise.log.txt"
        cmd_denoise = [
            ants_denoise,
            "-d",
            "3",
            "-i",
            to_ants_path(current, windows_mode),
            "-o",
            f"[{to_ants_path(denoise_out, windows_mode)},{to_ants_path(denoise_noise, windows_mode)}]",
            "-n",
            LOCKED_DENOISE_NOISE_MODEL,
            "-s",
            LOCKED_DENOISE_SHRINK_FACTOR,
            "-p",
            LOCKED_DENOISE_PATCH_RADIUS,
            "-r",
            LOCKED_DENOISE_SEARCH_RADIUS,
        ]
        if mask_path and mask_path.exists():
            cmd_denoise.extend(["-x", to_ants_path(mask_path, windows_mode)])
        denoise_run = run_logged(cmd_denoise, env=env, log_path=denoise_log)
        ensure_float32(denoise_out)
        ensure_float32(denoise_noise)
        current = denoise_out
        info["denoise"] = {
            "enabled": True,
            "output": str(denoise_out),
            "noise_output": str(denoise_noise),
            "log": str(denoise_log),
            "run": denoise_run,
            "config": {
                "noise_model": LOCKED_DENOISE_NOISE_MODEL,
                "shrink_factor": LOCKED_DENOISE_SHRINK_FACTOR,
                "patch_radius": LOCKED_DENOISE_PATCH_RADIUS,
                "search_radius": LOCKED_DENOISE_SEARCH_RADIUS,
            },
        }
    else:
        info["denoise"] = {
            "enabled": False,
            "reason": "locked off for fixed image or explicitly disabled for moving",
        }

    n4_out = out_dir / f"{stem}_n4.nii.gz"
    n4_bias = out_dir / f"{stem}_n4_biasfield.nii.gz"
    n4_log = out_dir / f"{stem}_n4.log.txt"
    cmd_n4 = [
        ants_n4,
        "-d",
        "3",
        "-i",
        to_ants_path(current, windows_mode),
        "-o",
        f"[{to_ants_path(n4_out, windows_mode)},{to_ants_path(n4_bias, windows_mode)}]",
        "-c",
        LOCKED_N4_CONVERGENCE,
        "-b",
        LOCKED_N4_BSPLINE,
        "-s",
        LOCKED_N4_SHRINK_FACTOR,
        "-r",
        "1",
        "-v",
        "1",
    ]
    if mask_path and mask_path.exists():
        cmd_n4.extend(["-x", to_ants_path(mask_path, windows_mode)])
    n4_run = run_logged(cmd_n4, env=env, log_path=n4_log)
    ensure_float32(n4_out)
    ensure_float32(n4_bias)
    current = n4_out
    info["n4"] = {
        "output": str(n4_out),
        "bias_field_output": str(n4_bias),
        "log": str(n4_log),
        "run": n4_run,
        "config": {
            "convergence": LOCKED_N4_CONVERGENCE,
            "bspline_fitting": LOCKED_N4_BSPLINE,
            "shrink_factor": LOCKED_N4_SHRINK_FACTOR,
            "rescale_intensities": "1",
        },
    }

    transform_apply_output = current
    opt_norm_out = out_dir / f"{stem}_opt_norm.nii.gz"
    opt_norm_stats = normalize_intensity_robust_z(
        transform_apply_output,
        opt_norm_out,
        mask_path=mask_path,
        clip_low_percent=LOCKED_OPT_NORM_CLIP_LOW_PERCENT,
        clip_high_percent=LOCKED_OPT_NORM_CLIP_HIGH_PERCENT,
    )
    ensure_float32(opt_norm_out)
    current = opt_norm_out
    info["optimization_normalization"] = {
        "enabled": True,
        "method": "percentile_clip_then_robust_zscore",
        "output": str(opt_norm_out),
        "config": {
            "clip_low_percent": LOCKED_OPT_NORM_CLIP_LOW_PERCENT,
            "clip_high_percent": LOCKED_OPT_NORM_CLIP_HIGH_PERCENT,
            "robust_scale_factor": LOCKED_OPT_NORM_ROBUST_SCALE,
            "center": "median",
            "scale": "MAD",
            "mask_only": True,
            "outside_mask_value": 0.0,
        },
        "stats": opt_norm_stats,
    }
    info["transform_apply_output"] = str(transform_apply_output)
    info["optimization_output"] = str(current)
    info["final_output"] = str(current)
    return current, transform_apply_output, info


def parse_init_order(raw: str) -> list[str]:
    allowed = {"moments", "geometry", "antsai"}
    out: list[str] = []
    for tok in raw.split(","):
        key = tok.strip().lower()
        if not key:
            continue
        if key not in allowed:
            raise ValueError(f"Unsupported fallback init method: {key}. Allowed: {sorted(allowed)}")
        if key not in out:
            out.append(key)
    return out


def read_affine_linear_info(
    *,
    affine_path: Path,
    ants_transform_info: str,
    windows_mode: bool,
    env: dict[str, str],
    log_path: Path | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "affine_path": str(affine_path),
        "matrix": [],
        "determinant": float("nan"),
        "sv_min": float("nan"),
        "sv_max": float("nan"),
        "sv_condition": float("nan"),
        "singular_flag": None,
        "source": "unknown",
        "command": [],
        "return_code": -1,
        "log_path": str(log_path) if log_path else "",
    }
    cmd = [ants_transform_info, to_ants_path(affine_path, windows_mode)]
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    text = stdout + ("\n" + stderr if stderr else "")
    info["command"] = cmd
    info["return_code"] = int(completed.returncode)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            (
                "$ " + " ".join(cmd) + "\n"
                + "[return_code] " + str(completed.returncode) + "\n"
                + "[stdout]\n" + stdout + "\n"
                + "[stderr]\n" + stderr + "\n"
            ),
            encoding="utf-8",
        )

    if completed.returncode != 0:
        return info

    lines = text.splitlines()
    mat_rows: list[list[float]] = []
    for idx, raw in enumerate(lines):
        if raw.strip().startswith("Matrix:"):
            for j in range(idx + 1, min(idx + 10, len(lines))):
                row_txt = lines[j].strip()
                if not row_txt:
                    continue
                try:
                    row_vals = [float(x) for x in row_txt.split()]
                except Exception:
                    break
                if len(row_vals) != 3:
                    break
                mat_rows.append(row_vals)
                if len(mat_rows) == 3:
                    break
            break

    matrix = None
    if len(mat_rows) == 3:
        matrix = np.asarray(mat_rows, dtype=np.float64)
        info["matrix"] = [[float(x) for x in row] for row in matrix.tolist()]
        info["source"] = "antsTransformInfo_matrix"

    m_det = re.search(r"Determinant:\s*([\-0-9eE\.+]+)", text)
    if m_det:
        try:
            info["determinant"] = float(m_det.group(1))
        except Exception:
            info["determinant"] = float("nan")

    m_sing = re.search(r"Singular:\s*([01])", text)
    if m_sing:
        try:
            info["singular_flag"] = int(m_sing.group(1))
        except Exception:
            info["singular_flag"] = None

    if matrix is not None:
        if not np.isfinite(info["determinant"]):
            info["determinant"] = float(np.linalg.det(matrix))
        sv = np.linalg.svd(matrix, compute_uv=False)
        sv = np.asarray(sv, dtype=np.float64)
        if sv.size >= 1:
            info["sv_max"] = float(np.max(sv))
            info["sv_min"] = float(np.min(sv))
            if info["sv_min"] > 0:
                info["sv_condition"] = float(info["sv_max"] / info["sv_min"])
            elif info["sv_max"] > 0:
                info["sv_condition"] = float("inf")
            else:
                info["sv_condition"] = float("nan")

    return info


def compute_jacobian_and_warp_metrics(
    *,
    warp_field: Path,
    fixed_mask_path: Path | None,
    jacobian_exec: str,
    windows_mode: bool,
    env: dict[str, str],
    out_dir: Path,
    stem: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jac_img = out_dir / f"{stem}_jacobian_det.nii.gz"
    jac_log = out_dir / f"{stem}_jacobian.log.txt"
    jac_stats_json = out_dir / f"{stem}_jacobian_metrics.json"
    cmd_jac = [
        jacobian_exec,
        "3",
        to_ants_path(warp_field, windows_mode),
        to_ants_path(jac_img, windows_mode),
        "0",
        "1",
    ]
    jac_run = run_logged(cmd_jac, env=env, log_path=jac_log)
    ensure_float32(jac_img)

    jac_data = np.asarray(nib.load(str(jac_img)).get_fdata(), dtype=np.float32)
    jac_mask = np.isfinite(jac_data)
    jac_mask_source = "finite"
    if fixed_mask_path and fixed_mask_path.exists():
        m = np.asarray(nib.load(str(fixed_mask_path)).get_fdata(), dtype=np.float32) > 0
        if m.shape != jac_data.shape:
            raise RuntimeError(
                f"Jacobian mask shape mismatch: jacobian {jac_data.shape}, mask {m.shape} ({fixed_mask_path})"
            )
        jac_mask &= m
        jac_mask_source = "fixed_mask"
    if np.count_nonzero(jac_mask) < 100:
        jac_mask = np.isfinite(jac_data)
        jac_mask_source = "finite_fallback"
    jac_vals = jac_data[jac_mask].astype(np.float64)
    jac_stats = {
        "min": float(np.min(jac_vals)) if jac_vals.size else float("nan"),
        "p01": float(np.percentile(jac_vals, 1.0)) if jac_vals.size else float("nan"),
        "p99": float(np.percentile(jac_vals, 99.0)) if jac_vals.size else float("nan"),
        "negative_fraction": (
            float(np.count_nonzero(jac_vals <= 0) / jac_vals.size) if jac_vals.size else float("nan")
        ),
        "mean": float(np.mean(jac_vals)) if jac_vals.size else float("nan"),
        "mask_source": jac_mask_source,
        "masked_voxels": int(jac_vals.size),
    }

    warp_data = np.asarray(nib.load(str(warp_field)).get_fdata(), dtype=np.float32)
    if warp_data.ndim == 5 and warp_data.shape[3] == 1:
        warp_data = np.squeeze(warp_data, axis=3)
    if warp_data.ndim != 4 or warp_data.shape[-1] < 3:
        raise RuntimeError(f"Unexpected warp field shape: {warp_data.shape} ({warp_field})")
    warp_norm2 = np.sum(warp_data[..., :3] * warp_data[..., :3], axis=-1)
    warp_mask = np.isfinite(warp_norm2)
    warp_mask_source = "finite"
    if fixed_mask_path and fixed_mask_path.exists():
        wm = np.asarray(nib.load(str(fixed_mask_path)).get_fdata(), dtype=np.float32) > 0
        if wm.shape != warp_norm2.shape:
            raise RuntimeError(
                f"Warp energy mask shape mismatch: warp {warp_norm2.shape}, mask {wm.shape} ({fixed_mask_path})"
            )
        warp_mask &= wm
        warp_mask_source = "fixed_mask"
    if np.count_nonzero(warp_mask) < 100:
        warp_mask = np.isfinite(warp_norm2)
        warp_mask_source = "finite_fallback"
    warp_vals = warp_norm2[warp_mask].astype(np.float64)
    warp_energy = {
        "l2_energy_mean": float(np.mean(warp_vals)) if warp_vals.size else float("nan"),
        "l2_energy_rms": float(np.sqrt(np.mean(warp_vals))) if warp_vals.size else float("nan"),
        "l2_energy_p99": float(np.percentile(warp_vals, 99.0)) if warp_vals.size else float("nan"),
        "mask_source": warp_mask_source,
        "masked_voxels": int(warp_vals.size),
    }

    payload = {
        "warp_field": str(warp_field),
        "jacobian_det_image": str(jac_img),
        "jacobian_log": str(jac_log),
        "jacobian_run": jac_run,
        "jacobian": jac_stats,
        "warp_energy": warp_energy,
    }
    write_json(jac_stats_json, payload)
    payload["stats_json"] = str(jac_stats_json)
    return payload


def evaluate_attempt_gate(
    *,
    affine_result: dict[str, Any],
    affine_linear_info: dict[str, Any],
    jacobian_metrics: dict[str, Any],
    min_dice: float,
    min_nmi: float,
    min_cc: float,
    det_min: float,
    det_max: float,
    sv_min: float,
    sv_max: float,
    jac_min_threshold: float,
    jac_p01_min: float,
    jac_p99_max: float,
    jac_neg_frac_max: float,
    warp_l2_energy_max: float,
) -> dict[str, Any]:
    qc = dict(affine_result.get("qc_metrics", {}))
    dice = float(qc.get("dice")) if qc.get("dice") is not None else float("nan")
    nmi = float(qc.get("nmi")) if qc.get("nmi") is not None else float("nan")
    cc = float(qc.get("cc")) if qc.get("cc") is not None else float("nan")
    det = float(affine_linear_info.get("determinant", float("nan")))
    sv_min_val = float(affine_linear_info.get("sv_min", float("nan")))
    sv_max_val = float(affine_linear_info.get("sv_max", float("nan")))
    sv_condition = float(affine_linear_info.get("sv_condition", float("nan")))
    singular_flag = affine_linear_info.get("singular_flag", None)

    jac_stats = dict(jacobian_metrics.get("jacobian", {}))
    warp_energy = dict(jacobian_metrics.get("warp_energy", {}))
    jac_min = float(jac_stats.get("min", float("nan")))
    jac_p01 = float(jac_stats.get("p01", float("nan")))
    jac_p99 = float(jac_stats.get("p99", float("nan")))
    jac_neg_frac = float(jac_stats.get("negative_fraction", float("nan")))
    warp_l2_energy = float(warp_energy.get("l2_energy_mean", float("nan")))

    reasons: list[str] = []
    if not np.isfinite(det):
        reasons.append("affine_det_nonfinite")
    if np.isfinite(det) and (det < det_min or det > det_max):
        reasons.append(f"affine_det_out_of_range({det:.6f} not in [{det_min:.6f},{det_max:.6f}])")
    if singular_flag == 1:
        reasons.append("affine_singular_flag_true")
    if not np.isfinite(sv_min_val) or not np.isfinite(sv_max_val):
        reasons.append("affine_singular_values_nonfinite")
    else:
        if sv_min_val < sv_min:
            reasons.append(f"affine_sv_min_low({sv_min_val:.6f}<{sv_min:.6f})")
        if sv_max_val > sv_max:
            reasons.append(f"affine_sv_max_high({sv_max_val:.6f}>{sv_max:.6f})")

    if np.isfinite(dice) and dice < min_dice:
        reasons.append(f"affine_dice_low({dice:.6f}<{min_dice:.6f})")
    if np.isfinite(nmi) and nmi < min_nmi:
        reasons.append(f"affine_nmi_low({nmi:.6f}<{min_nmi:.6f})")
    if np.isfinite(cc) and cc < min_cc:
        reasons.append(f"affine_cc_low({cc:.6f}<{min_cc:.6f})")

    if not np.isfinite(jac_min):
        reasons.append("jacobian_min_nonfinite")
    elif jac_min < jac_min_threshold:
        reasons.append(f"jacobian_min_low({jac_min:.6f}<{jac_min_threshold:.6f})")
    if not np.isfinite(jac_p01):
        reasons.append("jacobian_p01_nonfinite")
    elif jac_p01 < jac_p01_min:
        reasons.append(f"jacobian_p01_low({jac_p01:.6f}<{jac_p01_min:.6f})")
    if not np.isfinite(jac_p99):
        reasons.append("jacobian_p99_nonfinite")
    elif jac_p99 > jac_p99_max:
        reasons.append(f"jacobian_p99_high({jac_p99:.6f}>{jac_p99_max:.6f})")
    if not np.isfinite(jac_neg_frac):
        reasons.append("jacobian_negative_fraction_nonfinite")
    elif jac_neg_frac > jac_neg_frac_max:
        reasons.append(f"jacobian_negative_fraction_high({jac_neg_frac:.6f}>{jac_neg_frac_max:.6f})")

    if not np.isfinite(warp_l2_energy):
        reasons.append("warp_l2_energy_nonfinite")
    elif warp_l2_energy > warp_l2_energy_max:
        reasons.append(f"warp_l2_energy_high({warp_l2_energy:.6f}>{warp_l2_energy_max:.6f})")

    # Deterministic score for tie-break/ranking among gate-qualified attempts.
    score = (
        -1.0 if not np.isfinite(dice) else float(dice),
        -1.0 if not np.isfinite(nmi) else float(nmi),
        -1.0 if not np.isfinite(cc) else float(cc),
        -1.0 if not np.isfinite(jac_min) else float(jac_min),
        -1.0 if not np.isfinite(jac_p01) else float(jac_p01),
        -1.0 if not np.isfinite(jac_p99) else -abs(float(np.log(max(jac_p99, 1e-12)))),
        -1e9 if not np.isfinite(jac_neg_frac) else -float(jac_neg_frac),
        -1e9 if not np.isfinite(warp_l2_energy) else -float(warp_l2_energy),
        -1e9 if not np.isfinite(sv_condition) else -float(sv_condition),
        -1e9 if not np.isfinite(det) else -abs(float(np.log(max(det, 1e-12)))),
    )
    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "affine_qc": qc,
        "affine_linear": {
            "determinant": det,
            "sv_min": sv_min_val,
            "sv_max": sv_max_val,
            "sv_condition": sv_condition,
            "singular_flag": singular_flag,
        },
        "jacobian": jac_stats,
        "warp_energy": warp_energy,
        "score": score,
        "thresholds": {
            "min_dice": float(min_dice),
            "min_nmi": float(min_nmi),
            "min_cc": float(min_cc),
            "det_min": float(det_min),
            "det_max": float(det_max),
            "sv_min": float(sv_min),
            "sv_max": float(sv_max),
            "jac_min_threshold": float(jac_min_threshold),
            "jac_p01_min": float(jac_p01_min),
            "jac_p99_max": float(jac_p99_max),
            "jac_neg_frac_max": float(jac_neg_frac_max),
            "warp_l2_energy_max": float(warp_l2_energy_max),
        },
    }


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _higher_better(value: Any) -> float:
    x = _safe_float(value, float("nan"))
    return x if np.isfinite(x) else -1e30


def _lower_better(value: Any) -> float:
    x = _safe_float(value, float("nan"))
    return x if np.isfinite(x) else 1e30


def _is_attempt_better(candidate: dict[str, Any], current: dict[str, Any], *, dice_eps: float) -> bool:
    cand_gate = dict(candidate.get("attempt_gate", {}))
    curr_gate = dict(current.get("attempt_gate", {}))

    cand_qc = dict(cand_gate.get("affine_qc", {}))
    curr_qc = dict(curr_gate.get("affine_qc", {}))
    cand_jac = dict(cand_gate.get("jacobian", {}))
    curr_jac = dict(curr_gate.get("jacobian", {}))
    cand_warp = dict(cand_gate.get("warp_energy", {}))
    curr_warp = dict(curr_gate.get("warp_energy", {}))

    cand_dice = _higher_better(cand_qc.get("dice"))
    curr_dice = _higher_better(curr_qc.get("dice"))
    if abs(cand_dice - curr_dice) >= dice_eps:
        return cand_dice > curr_dice

    cand_jneg = _lower_better(cand_jac.get("negative_fraction"))
    curr_jneg = _lower_better(curr_jac.get("negative_fraction"))
    if abs(cand_jneg - curr_jneg) > 1e-12:
        return cand_jneg < curr_jneg

    cand_warp_e = _lower_better(cand_warp.get("l2_energy_mean"))
    curr_warp_e = _lower_better(curr_warp.get("l2_energy_mean"))
    if abs(cand_warp_e - curr_warp_e) > 1e-12:
        return cand_warp_e < curr_warp_e

    cand_cc = _higher_better(cand_qc.get("cc"))
    curr_cc = _higher_better(curr_qc.get("cc"))
    if abs(cand_cc - curr_cc) > 1e-12:
        return cand_cc > curr_cc

    cand_name = str(candidate.get("attempt_name", ""))
    curr_name = str(current.get("attempt_name", ""))
    if cand_name != curr_name:
        return cand_name < curr_name

    cand_init = str(candidate.get("init_method", ""))
    curr_init = str(current.get("init_method", ""))
    if cand_init != curr_init:
        return cand_init < curr_init

    return False


def _select_best_attempt(attempts: list[dict[str, Any]], *, dice_eps: float) -> dict[str, Any]:
    if not attempts:
        raise RuntimeError("Cannot select best attempt from empty list.")
    best = attempts[0]
    for cand in attempts[1:]:
        if _is_attempt_better(cand, best, dice_eps=dice_eps):
            best = cand
    return best


def sanitize_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register moving template to fixed template/atlas (Rigid + Affine + SyN)."
    )
    parser.add_argument("--fixed", required=True, help="Fixed image.")
    parser.add_argument("--moving", required=True, help="Moving image.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--prefix", default="template_reg_", help="Output prefix.")
    parser.add_argument("--from-space", default="moving_template", help="Logical source space name.")
    parser.add_argument("--to-space", default="fixed_template", help="Logical target space name.")
    parser.add_argument("--edge-id", default="", help="Optional edge ID.")
    parser.add_argument("--run-id", default="", help="Optional run ID.")
    parser.add_argument("--trace-dir", default="", help="Optional trace output folder.")
    parser.add_argument(
        "--fixed-mask",
        required=True,
        help="Required fixed-space brain mask. Registration will fail without it.",
    )
    parser.add_argument(
        "--moving-mask",
        required=True,
        help="Required moving-space brain mask. Registration will fail without it.",
    )
    parser.add_argument(
        "--coverage-margin-min-vox",
        type=int,
        default=LOCKED_COVERAGE_MARGIN_MIN_VOX,
        help=(
            "Coverage hard gate: minimum bbox-to-grid margin in voxels. "
            "Passes if voxel threshold OR mm threshold is satisfied."
        ),
    )
    parser.add_argument(
        "--coverage-margin-min-mm",
        type=float,
        default=LOCKED_COVERAGE_MARGIN_MIN_MM,
        help=(
            "Coverage hard gate: minimum bbox-to-grid margin in mm. "
            "Passes if voxel threshold OR mm threshold is satisfied."
        ),
    )
    parser.add_argument(
        "--preprocess-mode",
        default=LOCKED_PREPROCESS_MODE,
        choices=[LOCKED_PREPROCESS_MODE, "n4"],
        help="Locked preprocessing profile (N4 on fixed and moving). Legacy alias: n4.",
    )
    parser.add_argument(
        "--moving-denoise",
        default="on",
        choices=["on", "off"],
        help="Moving denoise switch before N4 (default on).",
    )
    parser.add_argument(
        "--use-mask-in-optimization",
        default="on",
        choices=["on"],
        help="Registration is mask-gated; optimization always uses fixed/moving masks (-x).",
    )
    parser.add_argument(
        "--init-strategy",
        default="com_only",
        choices=["com_only", "translation_then_rigid"],
        help="Initialization strategy: COM only or add translation stage before rigid.",
    )
    parser.add_argument(
        "--affine-fallback",
        default="on",
        choices=["on", "off"],
        help=(
            "Keep com_only as first attempt. If affine sanity/QC is below thresholds, "
            "automatically retry fallback initializations."
        ),
    )
    parser.add_argument(
        "--affine-fallback-order",
        default=LOCKED_AFFINE_FALLBACK_ORDER,
        help="Comma-separated fallback init order from: moments,geometry,antsai.",
    )
    parser.add_argument(
        "--affine-min-dice",
        type=float,
        default=LOCKED_AFFINE_MIN_DICE,
        help="Affine fallback trigger threshold: minimum Dice.",
    )
    parser.add_argument(
        "--affine-min-nmi",
        type=float,
        default=LOCKED_AFFINE_MIN_NMI,
        help="Affine fallback trigger threshold: minimum NMI.",
    )
    parser.add_argument(
        "--affine-min-cc",
        type=float,
        default=LOCKED_AFFINE_MIN_CC,
        help="Affine fallback trigger threshold: minimum CC.",
    )
    parser.add_argument(
        "--affine-det-min",
        type=float,
        default=LOCKED_AFFINE_DET_MIN,
        help="Affine sanity threshold: minimum determinant of 3x3 affine linear part.",
    )
    parser.add_argument(
        "--affine-det-max",
        type=float,
        default=LOCKED_AFFINE_DET_MAX,
        help="Affine sanity threshold: maximum determinant of 3x3 affine linear part.",
    )
    parser.add_argument(
        "--affine-sv-min",
        type=float,
        default=LOCKED_AFFINE_SV_MIN,
        help="Affine sanity threshold: minimum singular value of 3x3 affine linear part.",
    )
    parser.add_argument(
        "--affine-sv-max",
        type=float,
        default=LOCKED_AFFINE_SV_MAX,
        help="Affine sanity threshold: maximum singular value of 3x3 affine linear part.",
    )
    parser.add_argument(
        "--jac-min",
        type=float,
        default=LOCKED_JAC_MIN,
        help="Jacobian hard threshold: minimum determinant value within mask.",
    )
    parser.add_argument(
        "--jac-p01-min",
        type=float,
        default=LOCKED_JAC_P01_MIN,
        help="Jacobian hard threshold: minimum p01 within mask.",
    )
    parser.add_argument(
        "--jac-p99-max",
        type=float,
        default=LOCKED_JAC_P99_MAX,
        help="Jacobian hard threshold: maximum p99 within mask.",
    )
    parser.add_argument(
        "--jac-neg-frac-max",
        type=float,
        default=LOCKED_JAC_NEG_FRAC_MAX,
        help="Jacobian hard threshold: maximum fraction of voxels with Jacobian <= 0.",
    )
    parser.add_argument(
        "--warp-l2-energy-max",
        type=float,
        default=LOCKED_WARP_L2_ENERGY_MAX,
        help="Warp hard threshold: maximum mean L2 energy (mean ||u||^2) within mask.",
    )
    parser.add_argument(
        "--tie-break-dice-eps",
        type=float,
        default=LOCKED_TIE_BREAK_DICE_EPS,
        help=(
            "Deterministic attempt tie-break epsilon on Dice. "
            "If |Dice_a-Dice_b| < eps, compare Jacobian negative fraction, warp energy, CC, then name."
        ),
    )
    parser.add_argument(
        "--winsorize-range",
        default="[0.005,0.995]",
        help="ANTs winsorize range string.",
    )
    parser.add_argument(
        "--use-histogram-matching",
        default="0",
        choices=["0", "1"],
        help="Pass through to antsRegistration --use-histogram-matching.",
    )
    parser.add_argument(
        "--rigid-affine-iterations",
        default="200x100x50x20",
        help="Iterations for Translation/Rigid/Affine stages.",
    )
    parser.add_argument(
        "--rigid-affine-sigmas",
        default="6x4x2x1vox",
        help="Smoothing sigmas for Translation/Rigid/Affine stages.",
    )
    parser.add_argument(
        "--rigid-affine-shrinks",
        default="8x4x2x1",
        help="Shrink factors for Translation/Rigid/Affine stages.",
    )
    parser.add_argument(
        "--syn-iterations",
        default="120x90x60x30",
        help="Iterations for SyN stage.",
    )
    parser.add_argument(
        "--syn-sigmas",
        default="4x3x2x1vox",
        help="Smoothing sigmas for SyN stage.",
    )
    parser.add_argument(
        "--syn-shrinks",
        default="8x4x2x1",
        help="Shrink factors for SyN stage.",
    )
    parser.add_argument(
        "--resample-interpolation",
        default="Linear",
        choices=[
            "Linear",
            "NearestNeighbor",
            "BSpline",
            "Gaussian",
            "CosineWindowedSinc",
            "WelchWindowedSinc",
            "HammingWindowedSinc",
            "LanczosWindowedSinc",
            "MultiLabel",
            "GenericLabel",
        ],
        help="Resample interpolation for moving image output.",
    )
    parser.add_argument(
        "--ants-bin",
        default="C:/tools/ANTs/ants-2.6.5/bin",
        help="ANTs bin directory (contains antsRegistration.exe).",
    )
    parser.add_argument("--threads", type=int, default=4, help="ITK/OMP threads.")
    parser.add_argument(
        "--random-seed",
        default="",
        help="Optional ANTs random seed for better reproducibility (e.g., 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fixed = normalize_path(args.fixed)
    moving = normalize_path(args.moving)
    output_dir = normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = normalize_path(args.trace_dir) if args.trace_dir.strip() else output_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "transform_trace.jsonl"

    run_id = args.run_id.strip() or datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
    run_manifest_path = trace_dir / f"run_manifest_{run_id}.json"
    edge_id = args.edge_id.strip() or sanitize_id(f"{args.from_space}_to_{args.to_space}_{args.prefix}")

    fixed_mask_path = normalize_path(args.fixed_mask)
    moving_mask_path = normalize_path(args.moving_mask)

    run_manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "step_name": "template_register",
        "started_at": now_iso(),
        "from_space": args.from_space,
        "to_space": args.to_space,
        "edge_id": edge_id,
        "fixed_image": str(fixed),
        "moving_image": str(moving),
        "resample_interpolation": args.resample_interpolation,
        "trace_path": str(trace_path),
    }

    append_trace(
        trace_path,
        {
            "ts": now_iso(),
            "run_id": run_id,
            "step_name": "RUN_START",
            "status": "started",
            "from_space": args.from_space,
            "to_space": args.to_space,
            "fixed_image": str(fixed),
            "moving_image": str(moving),
            "edge_id": edge_id,
        },
    )

    try:
        if not fixed.exists():
            raise FileNotFoundError(f"Fixed image not found: {fixed}")
        if not moving.exists():
            raise FileNotFoundError(f"Moving image not found: {moving}")
        if not fixed_mask_path.exists():
            raise FileNotFoundError(f"Fixed brain mask not found: {fixed_mask_path}")
        if not moving_mask_path.exists():
            raise FileNotFoundError(f"Moving brain mask not found: {moving_mask_path}")

        ants_bin = normalize_path(args.ants_bin)
        ants_reg = ensure_exec(ants_bin, "antsRegistration")
        ants_apply = ensure_exec(ants_bin, "antsApplyTransforms")
        ants_denoise = ensure_exec(ants_bin, "DenoiseImage")
        ants_n4 = ensure_exec(ants_bin, "N4BiasFieldCorrection")
        ants_grid = ensure_exec(ants_bin, "CreateWarpedGridImage")
        ants_jacobian = ensure_exec(ants_bin, "CreateJacobianDeterminantImage")
        ants_transform_info = ensure_exec(ants_bin, "antsTransformInfo")
        fallback_methods = parse_init_order(args.affine_fallback_order)
        fallback_on = args.affine_fallback == "on"
        ants_ai = ""
        if fallback_on and "antsai" in fallback_methods:
            ants_ai = ensure_exec(ants_bin, "antsAI")
        windows_mode = ants_reg.lower().endswith(".exe")

        env = os.environ.copy()
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(max(args.threads, 1))
        env["OMP_NUM_THREADS"] = str(max(args.threads, 1))

        if args.affine_det_min <= 0:
            raise RuntimeError("--affine-det-min must be > 0.")
        if args.affine_det_max <= args.affine_det_min:
            raise RuntimeError("--affine-det-max must be > --affine-det-min.")
        if args.affine_sv_min <= 0:
            raise RuntimeError("--affine-sv-min must be > 0.")
        if args.affine_sv_max <= args.affine_sv_min:
            raise RuntimeError("--affine-sv-max must be > --affine-sv-min.")
        if args.jac_p99_max <= 0:
            raise RuntimeError("--jac-p99-max must be > 0.")
        if args.jac_neg_frac_max < 0 or args.jac_neg_frac_max > 1:
            raise RuntimeError("--jac-neg-frac-max must be within [0,1].")
        if args.warp_l2_energy_max <= 0:
            raise RuntimeError("--warp-l2-energy-max must be > 0.")
        if args.tie_break_dice_eps <= 0:
            raise RuntimeError("--tie-break-dice-eps must be > 0.")
        if args.coverage_margin_min_vox < 0:
            raise RuntimeError("--coverage-margin-min-vox must be >= 0.")
        if args.coverage_margin_min_mm <= 0:
            raise RuntimeError("--coverage-margin-min-mm must be > 0.")

        preprocess_mode = LOCKED_PREPROCESS_MODE if args.preprocess_mode.strip() in {LOCKED_PREPROCESS_MODE, "n4"} else ""
        if preprocess_mode != LOCKED_PREPROCESS_MODE:
            raise RuntimeError(
                f"Unsupported preprocess mode: {args.preprocess_mode}. "
                f"Only {LOCKED_PREPROCESS_MODE} is allowed."
            )

        moving_denoise_on = args.moving_denoise == "on"
        preproc_dir = output_dir / "preproc"
        preproc_info: dict[str, Any] = {
            "mode": preprocess_mode,
            "moving_denoise": args.moving_denoise,
            "dtype_policy": "float32_only_for_preproc_and_registration_images",
            "locked_parameters": {
                "n4_convergence": LOCKED_N4_CONVERGENCE,
                "n4_bspline_fitting": LOCKED_N4_BSPLINE,
                "n4_shrink_factor": LOCKED_N4_SHRINK_FACTOR,
                "moving_denoise_noise_model": LOCKED_DENOISE_NOISE_MODEL,
                "moving_denoise_shrink_factor": LOCKED_DENOISE_SHRINK_FACTOR,
                "moving_denoise_patch_radius": LOCKED_DENOISE_PATCH_RADIUS,
                "moving_denoise_search_radius": LOCKED_DENOISE_SEARCH_RADIUS,
                "opt_norm_clip_low_percent": LOCKED_OPT_NORM_CLIP_LOW_PERCENT,
                "opt_norm_clip_high_percent": LOCKED_OPT_NORM_CLIP_HIGH_PERCENT,
                "opt_norm_robust_scale_factor": LOCKED_OPT_NORM_ROBUST_SCALE,
                "affine_fallback_enabled": fallback_on,
                "affine_fallback_order": fallback_methods,
                "affine_min_dice": args.affine_min_dice,
                "affine_min_nmi": args.affine_min_nmi,
                "affine_min_cc": args.affine_min_cc,
                "affine_det_min": args.affine_det_min,
                "affine_det_max": args.affine_det_max,
                "affine_sv_min": args.affine_sv_min,
                "affine_sv_max": args.affine_sv_max,
                "jac_min": args.jac_min,
                "jac_p01_min": args.jac_p01_min,
                "jac_p99_max": args.jac_p99_max,
                "jac_neg_frac_max": args.jac_neg_frac_max,
                "warp_l2_energy_max": args.warp_l2_energy_max,
                "tie_break_dice_eps": args.tie_break_dice_eps,
                "tie_break_rule": (
                    "dice_primary_with_eps_then_jacobian_negative_fraction_then_"
                    "warp_l2_energy_then_cc_then_attempt_name"
                ),
                "coverage_margin_min_vox": args.coverage_margin_min_vox,
                "coverage_margin_min_mm": args.coverage_margin_min_mm,
                "coverage_rule": "bbox_to_reference_grid_margin",
            },
        }
        fixed_opt, fixed_apply, fixed_info = preprocess_image(
            image_path=fixed,
            out_dir=preproc_dir,
            stem=f"{args.prefix}fixed",
            moving_denoise=False,
            is_moving=False,
            mask_path=fixed_mask_path,
            ants_denoise=ants_denoise,
            ants_n4=ants_n4,
            windows_mode=windows_mode,
            env=env,
        )
        moving_opt, moving_apply, moving_info = preprocess_image(
            image_path=moving,
            out_dir=preproc_dir,
            stem=f"{args.prefix}moving",
            moving_denoise=moving_denoise_on,
            is_moving=True,
            mask_path=moving_mask_path,
            ants_denoise=ants_denoise,
            ants_n4=ants_n4,
            windows_mode=windows_mode,
            env=env,
        )
        preproc_info["fixed"] = fixed_info
        preproc_info["moving"] = moving_info
        preproc_info["tools"] = {
            "DenoiseImage": ants_denoise,
            "N4BiasFieldCorrection": ants_n4,
            "CreateWarpedGridImage": ants_grid,
            "CreateJacobianDeterminantImage": ants_jacobian,
            "antsTransformInfo": ants_transform_info,
        }
        if ants_ai:
            preproc_info["tools"]["antsAI"] = ants_ai

        prefix = output_dir / args.prefix
        fixed_ants = to_ants_path(fixed, windows_mode)
        moving_ants = to_ants_path(moving, windows_mode)
        fixed_apply_ants = to_ants_path(fixed_apply, windows_mode)
        moving_apply_ants = to_ants_path(moving_apply, windows_mode)
        fixed_opt_ants = to_ants_path(fixed_opt, windows_mode)
        moving_opt_ants = to_ants_path(moving_opt, windows_mode)

        coverage_fixed = evaluate_mask_coverage_margin(
            mask_path=fixed_mask_path,
            reference_image_path=fixed_opt,
            min_margin_vox=args.coverage_margin_min_vox,
            min_margin_mm=args.coverage_margin_min_mm,
            label="fixed_mask_vs_fixed_reference",
        )
        coverage_moving = evaluate_mask_coverage_margin(
            mask_path=moving_mask_path,
            reference_image_path=moving_opt,
            min_margin_vox=args.coverage_margin_min_vox,
            min_margin_mm=args.coverage_margin_min_mm,
            label="moving_mask_vs_moving_reference",
        )
        coverage_check = {
            "thresholds": {
                "min_margin_vox": int(args.coverage_margin_min_vox),
                "min_margin_mm": float(args.coverage_margin_min_mm),
                "logic": "pass_if(min_margin_vox >= threshold_vox OR min_margin_mm >= threshold_mm)",
            },
            "checks": [coverage_fixed, coverage_moving],
        }
        preproc_info["coverage_check"] = coverage_check
        run_manifest["coverage_check"] = coverage_check

        coverage_failures = [x for x in coverage_check["checks"] if not bool(x.get("passed", False))]
        append_trace(
            trace_path,
            {
                "ts": now_iso(),
                "run_id": run_id,
                "step_name": "COVERAGE_CHECK_DONE",
                "status": "failed" if coverage_failures else "success",
                "edge_id": edge_id,
                "coverage_check": coverage_check,
                "failed_labels": [str(x.get("label", "")) for x in coverage_failures],
            },
        )
        if coverage_failures:
            details = "; ".join(
                [
                    (
                        f"{x.get('label','')}: "
                        f"min_margin_vox={x.get('min_margin_vox')} "
                        f"min_margin_mm={x.get('min_margin_mm'):.3f}"
                    )
                    for x in coverage_failures
                ]
            )
            raise RuntimeError(
                "Coverage hard gate failed after resample contract. "
                "Mask bbox is too close to reference grid boundary (possible slab-like/cropping). "
                + details
            )

        warped = output_dir / f"{args.prefix}Warped.nii.gz"
        inverse = output_dir / f"{args.prefix}InverseWarped.nii.gz"
        affine_out = output_dir / f"{args.prefix}0GenericAffine.mat"
        warp_out = output_dir / f"{args.prefix}1Warp.nii.gz"
        inv_warp_out = output_dir / f"{args.prefix}1InverseWarp.nii.gz"

        use_masks_opt = True

        append_trace(
            trace_path,
            {
                "ts": now_iso(),
                "run_id": run_id,
                "step_name": "EDGE_STARTED",
                "status": "started",
                "edge_id": edge_id,
                "from_space": args.from_space,
                "to_space": args.to_space,
                "fixed_image": str(fixed_opt),
                "moving_image": str(moving_opt),
                "use_masks_in_optimization": use_masks_opt,
                "init_strategy": args.init_strategy,
            },
        )

        mask_pair_ants = ""
        if use_masks_opt and fixed_mask_path and moving_mask_path:
            mask_pair_ants = (
                f"[{to_ants_path(fixed_mask_path, windows_mode)},"
                f"{to_ants_path(moving_mask_path, windows_mode)}]"
            )

        final_stages_dir = output_dir / "stages"
        final_stages_dir.mkdir(parents=True, exist_ok=True)

        def _base_reg_cmd(init_spec: str) -> list[str]:
            cmd = [
                ants_reg,
                "-d",
                "3",
                "--float",
                "0",
                "--interpolation",
                args.resample_interpolation,
                "--winsorize-image-intensities",
                args.winsorize_range,
                "--use-histogram-matching",
                args.use_histogram_matching,
            ]
            if args.random_seed.strip():
                cmd.extend(["--random-seed", args.random_seed.strip()])
            if mask_pair_ants:
                cmd.extend(["-x", mask_pair_ants])
            cmd.extend(["-r", init_spec])
            return cmd

        def _check_exists(path: Path, label: str) -> None:
            if not path.exists():
                raise RuntimeError(f"Expected {label} not found: {path}")

        def _run_stage(
            *,
            attempt_name: str,
            attempt_stage_dir: Path,
            attempt_stage_results: list[dict[str, Any]],
            stage_name: str,
            init_spec: str,
            stage_terms: list[str],
            stage_transforms_apply_order: list[Path],
        ) -> dict[str, Any]:
            stage_dir = attempt_stage_dir / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)

            stage_prefix = stage_dir / f"{args.prefix}{stage_name}_"
            stage_warped = stage_dir / f"{args.prefix}{stage_name}_Warped.nii.gz"
            stage_inverse = stage_dir / f"{args.prefix}{stage_name}_InverseWarped.nii.gz"
            stage_affine = stage_dir / f"{args.prefix}{stage_name}_0GenericAffine.mat"
            stage_warp = stage_dir / f"{args.prefix}{stage_name}_1Warp.nii.gz"
            stage_inv_warp = stage_dir / f"{args.prefix}{stage_name}_1InverseWarp.nii.gz"

            append_trace(
                trace_path,
                {
                    "ts": now_iso(),
                    "run_id": run_id,
                    "step_name": "REG_STAGE_STARTED",
                    "status": "started",
                    "attempt": attempt_name,
                    "stage": stage_name,
                    "edge_id": edge_id,
                },
            )

            cmd_stage = _base_reg_cmd(init_spec)
            cmd_stage.extend(stage_terms)
            cmd_stage.extend(
                [
                    "-o",
                    "["
                    + f"{to_ants_path(stage_prefix, windows_mode)},"
                    + f"{to_ants_path(stage_warped, windows_mode)},"
                    + f"{to_ants_path(stage_inverse, windows_mode)}"
                    + "]",
                ]
            )
            run(cmd_stage, env=env)

            _check_exists(stage_warped, f"{stage_name} warped image")
            _check_exists(stage_affine, f"{stage_name} affine matrix")
            ensure_float32(stage_warped)
            if stage_inverse.exists():
                ensure_float32(stage_inverse)
            if stage_warp.exists():
                ensure_float32(stage_warp)
            if stage_inv_warp.exists():
                ensure_float32(stage_inv_warp)

            qc_stage = compute_mi_cc(fixed_opt, stage_warped)
            moved_mask_stage = None
            if fixed_mask_path and moving_mask_path:
                moved_mask_stage = stage_dir / f"{args.prefix}{stage_name}_movingMaskInFixed.nii.gz"
                cmd_apply_mask = [
                    ants_apply,
                    "-d",
                    "3",
                    "-i",
                    to_ants_path(moving_mask_path, windows_mode),
                    "-r",
                    fixed_opt_ants,
                    "-o",
                    to_ants_path(moved_mask_stage, windows_mode),
                ]
                for t in stage_transforms_apply_order:
                    cmd_apply_mask.extend(["-t", to_ants_path(t, windows_mode)])
                cmd_apply_mask.extend(["-n", "NearestNeighbor"])
                run(cmd_apply_mask, env=env)
                ensure_float32(moved_mask_stage)
                qc_stage["dice"] = compute_dice(fixed_mask_path, moved_mask_stage)

            qc_stage_path = stage_dir / f"{args.prefix}{stage_name}_qc_metrics.json"
            write_json(qc_stage_path, qc_stage)

            stage_transform_files = [str(stage_affine)]
            if stage_warp.exists():
                stage_transform_files.append(str(stage_warp))
            if stage_inv_warp.exists():
                stage_transform_files.append(str(stage_inv_warp))

            result = {
                "stage": stage_name,
                "status": "success",
                "dir": str(stage_dir),
                "prefix": str(stage_prefix),
                "warped": str(stage_warped),
                "inverse_warped": str(stage_inverse),
                "affine_mat": str(stage_affine),
                "warp_field": str(stage_warp) if stage_warp.exists() else "",
                "inverse_warp_field": str(stage_inv_warp) if stage_inv_warp.exists() else "",
                "moving_mask_in_fixed": str(moved_mask_stage) if moved_mask_stage else "",
                "qc_metrics": qc_stage,
                "qc_path": str(qc_stage_path),
                "init_spec": init_spec,
                "command": cmd_stage,
                "transform_files": stage_transform_files,
            }
            attempt_stage_results.append(result)

            append_trace(
                trace_path,
                {
                    "ts": now_iso(),
                    "run_id": run_id,
                    "step_name": "REG_STAGE_DONE",
                    "status": "success",
                    "attempt": attempt_name,
                    "stage": stage_name,
                    "edge_id": edge_id,
                    "qc_metrics": qc_stage,
                    "output_path": str(stage_warped),
                },
            )
            return result

        def _build_rigid_terms() -> list[str]:
            rigid_terms: list[str] = []
            if args.init_strategy == "translation_then_rigid":
                rigid_terms.extend(
                    [
                        "-m",
                        f"MI[{fixed_opt_ants},{moving_opt_ants},1,32,Regular,0.25]",
                        "-t",
                        "Translation[0.1]",
                        "-c",
                        f"[{args.rigid_affine_iterations},1e-6,10]",
                        "-s",
                        args.rigid_affine_sigmas,
                        "-f",
                        args.rigid_affine_shrinks,
                    ]
                )
            rigid_terms.extend(
                [
                    "-m",
                    f"MI[{fixed_opt_ants},{moving_opt_ants},1,32,Regular,0.25]",
                    "-t",
                    "Rigid[0.1]",
                    "-c",
                    f"[{args.rigid_affine_iterations},1e-6,10]",
                    "-s",
                    args.rigid_affine_sigmas,
                    "-f",
                    args.rigid_affine_shrinks,
                ]
            )
            return rigid_terms

        def _run_registration_attempt(
            *,
            attempt_name: str,
            init_method: str,
            init_spec_for_rigid: str,
            init_metadata: dict[str, Any],
            attempt_stage_dir: Path,
        ) -> dict[str, Any]:
            append_trace(
                trace_path,
                {
                    "ts": now_iso(),
                    "run_id": run_id,
                    "step_name": "REG_ATTEMPT_STARTED",
                    "status": "started",
                    "attempt": attempt_name,
                    "init_method": init_method,
                    "init_spec_for_rigid": init_spec_for_rigid,
                    "edge_id": edge_id,
                },
            )

            attempt_stage_results: list[dict[str, Any]] = []
            rigid_result = _run_stage(
                attempt_name=attempt_name,
                attempt_stage_dir=attempt_stage_dir,
                attempt_stage_results=attempt_stage_results,
                stage_name="rigid",
                init_spec=init_spec_for_rigid,
                stage_terms=_build_rigid_terms(),
                stage_transforms_apply_order=[
                    Path(attempt_stage_dir / "rigid" / f"{args.prefix}rigid_0GenericAffine.mat")
                ],
            )

            affine_init = to_ants_path(Path(rigid_result["affine_mat"]), windows_mode)
            affine_terms = [
                "-m",
                f"MI[{fixed_opt_ants},{moving_opt_ants},1,32,Regular,0.25]",
                "-t",
                "Affine[0.1]",
                "-c",
                f"[{args.rigid_affine_iterations},1e-6,10]",
                "-s",
                args.rigid_affine_sigmas,
                "-f",
                args.rigid_affine_shrinks,
            ]
            affine_result = _run_stage(
                attempt_name=attempt_name,
                attempt_stage_dir=attempt_stage_dir,
                attempt_stage_results=attempt_stage_results,
                stage_name="affine",
                init_spec=affine_init,
                stage_terms=affine_terms,
                stage_transforms_apply_order=[
                    Path(attempt_stage_dir / "affine" / f"{args.prefix}affine_0GenericAffine.mat")
                ],
            )

            syn_init = to_ants_path(Path(affine_result["affine_mat"]), windows_mode)
            syn_terms = [
                "-m",
                f"CC[{fixed_opt_ants},{moving_opt_ants},1,4]",
                "-t",
                "SyN[0.05,3,0]",
                "-c",
                f"[{args.syn_iterations},1e-6,10]",
                "-s",
                args.syn_sigmas,
                "-f",
                args.syn_shrinks,
            ]
            syn_warp = Path(attempt_stage_dir / "syn" / f"{args.prefix}syn_1Warp.nii.gz")
            syn_affine = Path(attempt_stage_dir / "syn" / f"{args.prefix}syn_0GenericAffine.mat")
            syn_result = _run_stage(
                attempt_name=attempt_name,
                attempt_stage_dir=attempt_stage_dir,
                attempt_stage_results=attempt_stage_results,
                stage_name="syn",
                init_spec=syn_init,
                stage_terms=syn_terms,
                stage_transforms_apply_order=[syn_warp, syn_affine],
            )

            _check_exists(Path(syn_result["warped"]), f"{attempt_name} syn warped image")
            _check_exists(Path(syn_result["affine_mat"]), f"{attempt_name} syn affine matrix")
            _check_exists(syn_warp, f"{attempt_name} syn warp field")
            _check_exists(
                Path(attempt_stage_dir / "syn" / f"{args.prefix}syn_1InverseWarp.nii.gz"),
                f"{attempt_name} syn inverse warp field",
            )

            affine_info_log = Path(attempt_stage_dir / "affine" / f"{args.prefix}affine_transform_info.log.txt")
            affine_linear_info = read_affine_linear_info(
                affine_path=Path(str(affine_result["affine_mat"])),
                ants_transform_info=ants_transform_info,
                windows_mode=windows_mode,
                env=env,
                log_path=affine_info_log,
            )
            jacobian_metrics = compute_jacobian_and_warp_metrics(
                warp_field=syn_warp,
                fixed_mask_path=fixed_mask_path,
                jacobian_exec=ants_jacobian,
                windows_mode=windows_mode,
                env=env,
                out_dir=Path(attempt_stage_dir / "syn"),
                stem=f"{args.prefix}syn",
            )
            attempt_gate = evaluate_attempt_gate(
                affine_result=affine_result,
                affine_linear_info=affine_linear_info,
                jacobian_metrics=jacobian_metrics,
                min_dice=args.affine_min_dice,
                min_nmi=args.affine_min_nmi,
                min_cc=args.affine_min_cc,
                det_min=args.affine_det_min,
                det_max=args.affine_det_max,
                sv_min=args.affine_sv_min,
                sv_max=args.affine_sv_max,
                jac_min_threshold=args.jac_min,
                jac_p01_min=args.jac_p01_min,
                jac_p99_max=args.jac_p99_max,
                jac_neg_frac_max=args.jac_neg_frac_max,
                warp_l2_energy_max=args.warp_l2_energy_max,
            )
            append_trace(
                trace_path,
                {
                    "ts": now_iso(),
                    "run_id": run_id,
                    "step_name": "REG_ATTEMPT_DONE",
                    "status": "success",
                    "attempt": attempt_name,
                    "init_method": init_method,
                    "edge_id": edge_id,
                    "attempt_gate": attempt_gate,
                    "stage_dir": str(attempt_stage_dir),
                },
            )
            return {
                "attempt_name": attempt_name,
                "init_method": init_method,
                "init_spec_for_rigid": init_spec_for_rigid,
                "init_metadata": init_metadata,
                "stages_dir": str(attempt_stage_dir),
                "stage_results": attempt_stage_results,
                "rigid_result": rigid_result,
                "affine_result": affine_result,
                "syn_result": syn_result,
                "affine_linear_info": affine_linear_info,
                "jacobian_metrics": jacobian_metrics,
                "attempt_gate": attempt_gate,
                # Backward-compatible alias for downstream readers.
                "affine_gate": attempt_gate,
            }

        attempt_runs: list[dict[str, Any]] = []
        primary_init_spec = f"[{fixed_opt_ants},{moving_opt_ants},1]"
        primary_attempt = _run_registration_attempt(
            attempt_name="primary_com",
            init_method="com_only",
            init_spec_for_rigid=primary_init_spec,
            init_metadata={"feature": "moments_intensity", "ants_initialization_feature": 1},
            attempt_stage_dir=final_stages_dir,
        )
        attempt_runs.append(primary_attempt)

        selected_attempt = primary_attempt
        fallback_triggered = False
        fallback_trigger_reasons: list[str] = []
        fallback_root = output_dir / "fallback_attempts"
        init_signatures = {"moments"}

        if fallback_on and not primary_attempt["attempt_gate"]["passed"]:
            fallback_triggered = True
            fallback_trigger_reasons = list(primary_attempt["attempt_gate"]["reasons"])
            append_trace(
                trace_path,
                {
                    "ts": now_iso(),
                    "run_id": run_id,
                    "step_name": "AFFINE_FALLBACK_TRIGGERED",
                    "status": "started",
                    "edge_id": edge_id,
                    "reasons": fallback_trigger_reasons,
                    "order": fallback_methods,
                },
            )
            fallback_root.mkdir(parents=True, exist_ok=True)

            for idx, method in enumerate(fallback_methods, start=1):
                if method == "moments":
                    if "moments" in init_signatures:
                        continue
                    init_signatures.add("moments")
                    init_spec = f"[{fixed_opt_ants},{moving_opt_ants},1]"
                    init_metadata = {"feature": "moments_intensity", "ants_initialization_feature": 1}
                elif method == "geometry":
                    if "geometry" in init_signatures:
                        continue
                    init_signatures.add("geometry")
                    init_spec = f"[{fixed_opt_ants},{moving_opt_ants},0]"
                    init_metadata = {"feature": "geometry_center", "ants_initialization_feature": 0}
                elif method == "antsai":
                    init_dir = fallback_root / f"{idx:02d}_{method}"
                    init_dir.mkdir(parents=True, exist_ok=True)
                    antsai_mat = init_dir / f"{args.prefix}antsai_init.mat"
                    antsai_log = init_dir / f"{args.prefix}antsai_init.log.txt"
                    cmd_antsai = [
                        ants_ai,
                        "-d",
                        "3",
                        "-m",
                        (
                            f"{LOCKED_ANTSAI_METRIC}[{fixed_opt_ants},{moving_opt_ants},"
                            f"{LOCKED_ANTSAI_BINS},Regular,0.25]"
                        ),
                        "-t",
                        "Rigid[0.1]",
                        "-p",
                        "-s",
                        LOCKED_ANTSAI_SEARCH_FACTOR,
                        "-g",
                        LOCKED_ANTSAI_TRANSLATION_GRID,
                        "-x",
                        mask_pair_ants,
                        "-o",
                        to_ants_path(antsai_mat, windows_mode),
                    ]
                    if args.random_seed.strip():
                        cmd_antsai.extend(["--random-seed", args.random_seed.strip()])
                    antsai_run = run_logged(cmd_antsai, env=env, log_path=antsai_log)
                    if not antsai_mat.exists():
                        raise RuntimeError(f"antsAI init transform missing: {antsai_mat}")
                    init_spec = to_ants_path(antsai_mat, windows_mode)
                    init_metadata = {
                        "antsai_init_mat": str(antsai_mat),
                        "antsai_log": str(antsai_log),
                        "antsai_run": antsai_run,
                        "search_factor": LOCKED_ANTSAI_SEARCH_FACTOR,
                        "translation_grid": LOCKED_ANTSAI_TRANSLATION_GRID,
                    }
                else:
                    continue

                attempt_name = f"fallback_{idx:02d}_{method}"
                attempt_stage_dir = fallback_root / attempt_name / "stages"
                try:
                    attempt = _run_registration_attempt(
                        attempt_name=attempt_name,
                        init_method=method,
                        init_spec_for_rigid=init_spec,
                        init_metadata=init_metadata,
                        attempt_stage_dir=attempt_stage_dir,
                    )
                    attempt_runs.append(attempt)
                except Exception as attempt_exc:
                    append_trace(
                        trace_path,
                        {
                            "ts": now_iso(),
                            "run_id": run_id,
                            "step_name": "REG_ATTEMPT_DONE",
                            "status": "failed",
                            "attempt": attempt_name,
                            "init_method": method,
                            "edge_id": edge_id,
                            "error_message": str(attempt_exc),
                        },
                    )
                    continue

        passing_attempts = [at for at in attempt_runs if at.get("attempt_gate", {}).get("passed", False)]
        if passing_attempts:
            selected_attempt = _select_best_attempt(
                passing_attempts,
                dice_eps=args.tie_break_dice_eps,
            )
        else:
            best_attempt = _select_best_attempt(
                attempt_runs,
                dice_eps=args.tie_break_dice_eps,
            )
            selected_attempt = best_attempt
            hard_gate_error = (
                "No registration attempt passed hard gate. "
                f"best_attempt={best_attempt.get('attempt_name', '')} "
                f"init_method={best_attempt.get('init_method', '')} "
                f"reasons={best_attempt.get('attempt_gate', {}).get('reasons', [])}"
            )
            run_manifest["attempt_gate_failure"] = {
                "best_attempt": {
                    "attempt_name": best_attempt.get("attempt_name", ""),
                    "init_method": best_attempt.get("init_method", ""),
                    "attempt_gate": best_attempt.get("attempt_gate", {}),
                },
                "attempt_count": len(attempt_runs),
            }
            append_trace(
                trace_path,
                {
                    "ts": now_iso(),
                    "run_id": run_id,
                    "step_name": "ATTEMPT_HARD_GATE_FAILED",
                    "status": "failed",
                    "edge_id": edge_id,
                    "error_message": hard_gate_error,
                    "best_attempt": {
                        "attempt_name": best_attempt.get("attempt_name", ""),
                        "init_method": best_attempt.get("init_method", ""),
                        "attempt_gate": best_attempt.get("attempt_gate", {}),
                    },
                },
            )
            raise RuntimeError(hard_gate_error)

        selected_stages_dir = Path(selected_attempt["stages_dir"])
        if selected_stages_dir.resolve() != final_stages_dir.resolve():
            if final_stages_dir.exists():
                shutil.rmtree(final_stages_dir)
            shutil.copytree(selected_stages_dir, final_stages_dir)

        stages_dir = final_stages_dir
        stage_results = selected_attempt["stage_results"]
        syn_result = selected_attempt["syn_result"]
        attempt_summaries = []
        for at in attempt_runs:
            attempt_summaries.append(
                {
                    "attempt_name": at.get("attempt_name", ""),
                    "init_method": at.get("init_method", ""),
                    "init_spec_for_rigid": at.get("init_spec_for_rigid", ""),
                    "init_metadata": at.get("init_metadata", {}),
                    "stages_dir": at.get("stages_dir", ""),
                    "attempt_gate": at.get("attempt_gate", {}),
                    "affine_linear_info": at.get("affine_linear_info", {}),
                    "jacobian_metrics": at.get("jacobian_metrics", {}),
                    "affine_qc_metrics": at.get("affine_result", {}).get("qc_metrics", {}),
                    "syn_qc_metrics": at.get("syn_result", {}).get("qc_metrics", {}),
                }
            )

        promote_pairs = [
            (Path(syn_result["affine_mat"]), affine_out),
            (Path(syn_result["warp_field"]), warp_out),
            (Path(syn_result["inverse_warp_field"]), inv_warp_out),
        ]
        for src, dst in promote_pairs:
            shutil.copy2(src, dst)
        ensure_float32(warp_out)
        ensure_float32(inv_warp_out)

        cmd_apply_warped = [
            ants_apply,
            "-d",
            "3",
            "-i",
            moving_apply_ants,
            "-r",
            fixed_apply_ants,
            "-o",
            to_ants_path(warped, windows_mode),
            "-t",
            to_ants_path(warp_out, windows_mode),
            "-t",
            to_ants_path(affine_out, windows_mode),
            "-n",
            args.resample_interpolation,
        ]
        run(cmd_apply_warped, env=env)
        _check_exists(warped, "final warped image")
        ensure_float32(warped)

        cmd_apply_inverse = [
            ants_apply,
            "-d",
            "3",
            "-i",
            fixed_apply_ants,
            "-r",
            moving_apply_ants,
            "-o",
            to_ants_path(inverse, windows_mode),
            "-t",
            f"[{to_ants_path(affine_out, windows_mode)},1]",
            "-t",
            to_ants_path(inv_warp_out, windows_mode),
            "-n",
            args.resample_interpolation,
        ]
        run(cmd_apply_inverse, env=env)
        _check_exists(inverse, "final inverse warped image")
        ensure_float32(inverse)

        deformation_grid = output_dir / f"{args.prefix}deformationGrid.nii.gz"
        deformation_grid_log = output_dir / f"{args.prefix}deformationGrid.log.txt"
        cmd_grid = [
            ants_grid,
            "3",
            to_ants_path(warp_out, windows_mode),
            to_ants_path(deformation_grid, windows_mode),
            LOCKED_DEFORMATION_GRID_DIRECTIONS,
            LOCKED_DEFORMATION_GRID_SPACING,
            LOCKED_DEFORMATION_GRID_SIGMA,
        ]
        deformation_grid_run = run_logged(cmd_grid, env=env, log_path=deformation_grid_log)
        ensure_float32(deformation_grid)

        qc = dict(syn_result["qc_metrics"])
        moved_mask_path = None
        syn_mask_in_fixed = syn_result.get("moving_mask_in_fixed", "")
        if syn_mask_in_fixed:
            moved_mask_path = output_dir / f"{args.prefix}movingMaskInFixed.nii.gz"
            shutil.copy2(Path(syn_mask_in_fixed), moved_mask_path)
            ensure_float32(moved_mask_path)

        jacobian_det_out = output_dir / f"{args.prefix}JacobianDet.nii.gz"
        jacobian_log_out = output_dir / f"{args.prefix}jacobian.log.txt"
        jacobian_metrics_json_out = output_dir / f"{args.prefix}jacobian_metrics.json"
        selected_jacobian = dict(selected_attempt.get("jacobian_metrics", {}))
        src_jac_det = Path(str(selected_jacobian.get("jacobian_det_image", "")))
        if src_jac_det.exists():
            shutil.copy2(src_jac_det, jacobian_det_out)
            ensure_float32(jacobian_det_out)
        src_jac_log = Path(str(selected_jacobian.get("jacobian_log", "")))
        if src_jac_log.exists():
            shutil.copy2(src_jac_log, jacobian_log_out)
        src_jac_json = Path(str(selected_jacobian.get("stats_json", "")))
        if src_jac_json.exists():
            shutil.copy2(src_jac_json, jacobian_metrics_json_out)

        transform_files = [str(affine_out), str(warp_out), str(inv_warp_out)]

        append_trace(
            trace_path,
            {
                "ts": now_iso(),
                "run_id": run_id,
                "step_name": "EDGE_DONE",
                "status": "success",
                "edge_id": edge_id,
                "from_space": args.from_space,
                "to_space": args.to_space,
                "transform_files": transform_files,
                "edge_chain": [edge_id],
                "interpolation": args.resample_interpolation,
                "output_path": str(warped),
                "stages": [x["stage"] for x in stage_results],
                "selected_init_attempt": {
                    "attempt_name": selected_attempt.get("attempt_name", ""),
                    "init_method": selected_attempt.get("init_method", ""),
                },
            },
        )

        warped_full = None
        qc_full = None
        if moving_apply != moving:
            warped_full = output_dir / f"{args.prefix}WarpedFull.nii.gz"
            cmd_apply_full = [
                ants_apply,
                "-d",
                "3",
                "-i",
                moving_ants,
                "-r",
                fixed_ants,
                "-o",
                to_ants_path(warped_full, windows_mode),
                "-t",
                to_ants_path(warp_out, windows_mode),
                "-t",
                to_ants_path(affine_out, windows_mode),
                "-n",
                args.resample_interpolation,
            ]
            run(cmd_apply_full, env=env)
            ensure_float32(warped_full)
            qc_full = compute_mi_cc(fixed, warped_full)

        qc_json = output_dir / f"{args.prefix}qc_metrics.json"
        write_json(qc_json, qc)
        stage_qc_summary_path = output_dir / f"{args.prefix}stage_qc_summary.json"
        write_json(
            stage_qc_summary_path,
            {
                "run_id": run_id,
                "timestamp": now_iso(),
                "stage_order": [x["stage"] for x in stage_results],
                "stages": stage_results,
                "selected_final_stage": "syn",
                "final_qc_metrics": qc,
                "init_attempts": attempt_summaries,
                "selected_attempt": {
                    "attempt_name": selected_attempt.get("attempt_name", ""),
                    "init_method": selected_attempt.get("init_method", ""),
                    "stages_dir": str(stages_dir),
                    "attempt_gate": selected_attempt.get("attempt_gate", {}),
                },
                "coverage_check": coverage_check,
                "tie_break": {
                    "dice_eps": args.tie_break_dice_eps,
                    "rule": (
                        "dice_primary_with_eps_then_jacobian_negative_fraction_then_"
                        "warp_l2_energy_then_cc_then_attempt_name"
                    ),
                },
                "affine_fallback": {
                    "enabled": fallback_on,
                    "triggered": fallback_triggered,
                    "trigger_reasons": fallback_trigger_reasons,
                    "order": fallback_methods,
                },
            },
        )

        append_trace(
            trace_path,
            {
                "ts": now_iso(),
                "run_id": run_id,
                "step_name": "QC_DONE",
                "status": "success",
                "edge_id": edge_id,
                "qc_metrics": qc,
                "qc_full_raw": qc_full if qc_full else {},
                "output_path": str(qc_json),
            },
        )

        run_manifest.update(
            {
                "status": "success",
                "finished_at": now_iso(),
                "output_dir": str(output_dir),
                "prefix": args.prefix,
                "preprocess": preproc_info,
                "fixed_image_for_optimization": str(fixed_opt),
                "moving_image_for_optimization": str(moving_opt),
                "fixed_image_for_transform_application": str(fixed_apply),
                "moving_image_for_transform_application": str(moving_apply),
                "use_masks_in_optimization": use_masks_opt,
                "init_strategy": args.init_strategy,
                "affine_fallback": {
                    "enabled": fallback_on,
                    "triggered": fallback_triggered,
                    "trigger_reasons": fallback_trigger_reasons,
                    "order": fallback_methods,
                    "selected_attempt": {
                        "attempt_name": selected_attempt.get("attempt_name", ""),
                        "init_method": selected_attempt.get("init_method", ""),
                        "attempt_gate": selected_attempt.get("attempt_gate", {}),
                    },
                    "attempts": attempt_summaries,
                },
                "registration_schedule": {
                    "rigid_affine_iterations": args.rigid_affine_iterations,
                    "rigid_affine_sigmas": args.rigid_affine_sigmas,
                    "rigid_affine_shrinks": args.rigid_affine_shrinks,
                    "syn_iterations": args.syn_iterations,
                    "syn_sigmas": args.syn_sigmas,
                    "syn_shrinks": args.syn_shrinks,
                    "winsorize_range": args.winsorize_range,
                    "use_histogram_matching": args.use_histogram_matching,
                    "random_seed": args.random_seed.strip(),
                    "preprocess_mode_locked": preprocess_mode,
                    "moving_denoise": args.moving_denoise,
                    "n4_convergence": LOCKED_N4_CONVERGENCE,
                    "n4_bspline_fitting": LOCKED_N4_BSPLINE,
                    "n4_shrink_factor": LOCKED_N4_SHRINK_FACTOR,
                    "opt_norm_clip_low_percent": LOCKED_OPT_NORM_CLIP_LOW_PERCENT,
                    "opt_norm_clip_high_percent": LOCKED_OPT_NORM_CLIP_HIGH_PERCENT,
                    "opt_norm_robust_scale_factor": LOCKED_OPT_NORM_ROBUST_SCALE,
                    "opt_norm_center": "median",
                    "opt_norm_scale": "MAD",
                    "affine_fallback_enabled": fallback_on,
                    "affine_fallback_order": fallback_methods,
                    "affine_min_dice": args.affine_min_dice,
                    "affine_min_nmi": args.affine_min_nmi,
                    "affine_min_cc": args.affine_min_cc,
                    "affine_det_min": args.affine_det_min,
                    "affine_det_max": args.affine_det_max,
                    "affine_sv_min": args.affine_sv_min,
                    "affine_sv_max": args.affine_sv_max,
                    "jac_min": args.jac_min,
                    "jac_p01_min": args.jac_p01_min,
                    "jac_p99_max": args.jac_p99_max,
                    "jac_neg_frac_max": args.jac_neg_frac_max,
                    "warp_l2_energy_max": args.warp_l2_energy_max,
                    "tie_break_dice_eps": args.tie_break_dice_eps,
                    "tie_break_rule": (
                        "dice_primary_with_eps_then_jacobian_negative_fraction_then_"
                        "warp_l2_energy_then_cc_then_attempt_name"
                    ),
                    "coverage_margin_min_vox": args.coverage_margin_min_vox,
                    "coverage_margin_min_mm": args.coverage_margin_min_mm,
                    "coverage_rule": "bbox_to_reference_grid_margin",
                },
                "coverage_check": coverage_check,
                "warped": str(warped),
                "inverse_warped": str(inverse),
                "warped_full": str(warped_full) if warped_full else "",
                "deformation_grid": str(deformation_grid),
                "jacobian_det": str(jacobian_det_out) if jacobian_det_out.exists() else "",
                "jacobian_log": str(jacobian_log_out) if jacobian_log_out.exists() else "",
                "jacobian_metrics_json": (
                    str(jacobian_metrics_json_out) if jacobian_metrics_json_out.exists() else ""
                ),
                "jacobian_stats": selected_jacobian.get("jacobian", {}),
                "warp_energy_stats": selected_jacobian.get("warp_energy", {}),
                "affine_mat": transform_files[0],
                "warp_field": transform_files[1],
                "inverse_warp_field": transform_files[2],
                "transform_files": transform_files,
                "deformation_grid_config": {
                    "directions": LOCKED_DEFORMATION_GRID_DIRECTIONS,
                    "spacing": LOCKED_DEFORMATION_GRID_SPACING,
                    "sigma": LOCKED_DEFORMATION_GRID_SIGMA,
                },
                "deformation_grid_log": str(deformation_grid_log),
                "deformation_grid_run": deformation_grid_run,
                "stages_dir": str(stages_dir),
                "init_attempts": attempt_summaries,
                "selected_init_attempt": {
                    "attempt_name": selected_attempt.get("attempt_name", ""),
                    "init_method": selected_attempt.get("init_method", ""),
                    "init_spec_for_rigid": selected_attempt.get("init_spec_for_rigid", ""),
                    "init_metadata": selected_attempt.get("init_metadata", {}),
                    "attempt_gate": selected_attempt.get("attempt_gate", {}),
                },
                "registration_stage_order": [x["stage"] for x in stage_results],
                "registration_stages": stage_results,
                "stage_qc_summary_path": str(stage_qc_summary_path),
                "edge_chain": [edge_id],
                "ants_bin": str(ants_bin),
                "threads": max(args.threads, 1),
                "qc_metrics": qc,
                "qc_full_raw": qc_full if qc_full else {},
                "fixed_mask": str(fixed_mask_path) if fixed_mask_path else "",
                "moving_mask": str(moving_mask_path) if moving_mask_path else "",
                "moved_mask_in_fixed": str(moved_mask_path) if moved_mask_path else "",
                "storyboard_contract_note": (
                    "Storyboard should follow registration_stage_order from this run. "
                    "If steps are added/removed/skipped in future, keep the actual stage list for visual QC."
                ),
            }
        )

        run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

        append_trace(
            trace_path,
            {
                "ts": now_iso(),
                "run_id": run_id,
                "step_name": "RUN_END",
                "status": "success",
                "edge_id": edge_id,
                "run_manifest": str(run_manifest_path),
            },
        )

        print(f"[OK] warped: {warped}")
        print(f"[OK] deformation grid: {deformation_grid}")
        print(f"[OK] qc: {qc_json}")
        print(f"[OK] stage qc: {stage_qc_summary_path}")
        print(f"[OK] trace: {trace_path}")
        print(f"[OK] run manifest: {run_manifest_path}")

    except Exception as exc:
        run_manifest.update(
            {
                "status": "failed",
                "finished_at": now_iso(),
                "error_message": str(exc),
            }
        )
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        append_trace(
            trace_path,
            {
                "ts": now_iso(),
                "run_id": run_id,
                "step_name": "RUN_END",
                "status": "fail",
                "edge_id": edge_id,
                "error_message": str(exc),
                "run_manifest": str(run_manifest_path),
            },
        )
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Corner-background MRI SNR analysis.

This keeps the core SNR contract from scripts/snr_contract.py:
    SNR = magnitude_signal / std(background_signal)

But constrains the background to slice-wise XY corner ROIs so the sampled
background stays outside the brain more reliably for centered mouse volumes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def _dilate_connectivity(mask: np.ndarray, iters: int) -> np.ndarray:
    out = mask.astype(bool, copy=True)
    if iters <= 0:
        return out

    ndim = out.ndim
    for _ in range(iters):
        dil = out.copy()
        for axis in range(ndim):
            pos = np.roll(out, 1, axis=axis)
            pos_idx = [slice(None)] * ndim
            pos_idx[axis] = 0
            pos[tuple(pos_idx)] = False

            neg = np.roll(out, -1, axis=axis)
            neg_idx = [slice(None)] * ndim
            neg_idx[axis] = -1
            neg[tuple(neg_idx)] = False

            dil |= pos | neg
        out = dil
    return out


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(str(path), str(root))


def _safe_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "n_vox": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    q = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        "n_vox": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p50": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
    }


def _save_mask(mask: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out = nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(out, str(out_path))


def _save_float(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(out, str(out_path))


def _load_mask(mask_path: Path, shape: tuple[int, ...]) -> np.ndarray:
    m = nib.load(str(mask_path)).get_fdata()
    if m.shape != shape:
        raise ValueError(f"Mask shape mismatch: expected {shape}, got {m.shape} at {mask_path}")
    return m > 0


def _corner_xy_mask(shape: tuple[int, ...], frac_x: float, frac_y: float, min_vox: int) -> np.ndarray:
    if len(shape) < 2:
        raise ValueError(f"Need at least 2D image shape, got {shape}")
    nx, ny = shape[0], shape[1]
    wx = min(max(min_vox, int(round(nx * frac_x))), max(1, nx // 2))
    wy = min(max(min_vox, int(round(ny * frac_y))), max(1, ny // 2))

    mask2d = np.zeros((nx, ny), dtype=bool)
    mask2d[:wx, :wy] = True
    mask2d[:wx, ny - wy :] = True
    mask2d[nx - wx :, :wy] = True
    mask2d[nx - wx :, ny - wy :] = True

    if len(shape) == 2:
        return mask2d

    expand = (slice(None), slice(None)) + (None,) * (len(shape) - 2)
    return np.broadcast_to(mask2d[expand], shape).copy()


def _analyze_modality(
    *,
    mod_key: str,
    img_path: Path,
    mask_path: Path,
    output_dir: Path,
    corner_frac_x: float,
    corner_frac_y: float,
    min_corner_vox: int,
    guard_radius_vox: int,
    margin_vox: int,
    outlier_percentile: float,
) -> dict[str, Any]:
    img_nii = nib.load(str(img_path))
    img = img_nii.get_fdata().astype(np.float32)
    img_mag = np.abs(img)
    finite = np.isfinite(img_mag)

    brain_mask = _load_mask(mask_path, img.shape) & finite
    exclusion = _dilate_connectivity(brain_mask, max(0, guard_radius_vox + margin_vox))
    corner_roi = _corner_xy_mask(img.shape, corner_frac_x, corner_frac_y, min_corner_vox) & finite
    background_mask = corner_roi & (~exclusion)

    if background_mask.any() and outlier_percentile < 100.0:
        cap = float(np.percentile(img_mag[background_mask], outlier_percentile))
        background_mask = background_mask & (img_mag <= cap)
    else:
        cap = None

    bg_vals = img_mag[background_mask]
    sigma_bg = float(np.std(bg_vals)) if bg_vals.size > 0 else 0.0
    snr_map = img_mag / sigma_bg if sigma_bg > 0 else np.zeros_like(img_mag, dtype=np.float32)

    whole_bg = finite & (~exclusion)
    if whole_bg.any() and outlier_percentile < 100.0:
        whole_cap = float(np.percentile(img_mag[whole_bg], outlier_percentile))
        whole_bg = whole_bg & (img_mag <= whole_cap)
    else:
        whole_cap = None
    whole_bg_vals = img_mag[whole_bg]
    sigma_bg_whole = float(np.std(whole_bg_vals)) if whole_bg_vals.size > 0 else 0.0

    corner_roi_path = output_dir / f"{mod_key}_corner_xy_roi.nii.gz"
    bg_path = output_dir / f"{mod_key}_background_corner.nii.gz"
    snr_path = output_dir / f"{mod_key}_snr_corner.nii.gz"
    _save_mask(corner_roi, img_nii, corner_roi_path)
    _save_mask(background_mask, img_nii, bg_path)
    _save_float(snr_map, img_nii, snr_path)

    brain_signal = img_mag[brain_mask]
    brain_snr = snr_map[brain_mask]

    return {
        "modality": mod_key,
        "source_image_path": str(img_path),
        "source_mask_path": str(mask_path),
        "corner_roi_path": _rel(corner_roi_path, output_dir),
        "background_mask_path": _rel(bg_path, output_dir),
        "snr_map_path": _rel(snr_path, output_dir),
        "shape": list(img.shape),
        "voxel_size": [float(z) for z in img_nii.header.get_zooms()[: len(img.shape)]],
        "corner_fraction_xy": [corner_frac_x, corner_frac_y],
        "min_corner_vox": min_corner_vox,
        "guard_radius_vox": guard_radius_vox,
        "margin_vox": margin_vox,
        "outlier_percentile": outlier_percentile,
        "background_cap_value": cap,
        "background_stats_corner": _safe_stats(bg_vals),
        "background_stats_whole_outside_head": _safe_stats(whole_bg_vals),
        "sigma_bg_corner": sigma_bg,
        "sigma_bg_whole_outside_head": sigma_bg_whole,
        "sigma_bg_ratio_corner_to_whole": (sigma_bg / sigma_bg_whole) if sigma_bg_whole > 0 else None,
        "brain_signal_stats": _safe_stats(brain_signal),
        "brain_snr_stats": _safe_stats(brain_snr),
        "brain_mean_signal_over_sigma_bg": (float(np.mean(brain_signal)) / sigma_bg) if sigma_bg > 0 else None,
        "brain_median_signal_over_sigma_bg": (float(np.median(brain_signal)) / sigma_bg) if sigma_bg > 0 else None,
        "brain_p95_signal_over_sigma_bg": (float(np.percentile(brain_signal, 95)) / sigma_bg) if sigma_bg > 0 else None,
    }


def _write_markdown(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    md_path = output_dir / "snr_corner_report.md"
    lines: list[str] = []
    lines.append("# Corner-background SNR report")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now().isoformat()}")
    lines.append("")
    lines.append("Method:")
    lines.append("- Formula follows `scripts/snr_contract.py`: `SNR = |signal| / std(background)`.")
    lines.append("- Brain ROI comes from the provided mask.")
    lines.append("- Background uses slice-wise XY corner ROIs, then excludes a dilated brain mask.")
    lines.append("- Bright background outliers above the configured percentile are clipped, matching the project standard workflow.")
    lines.append("")
    lines.append("| Modality | Sigma_bg corner | Sigma_bg whole outside head | Ratio corner/whole | Brain SNR mean | Brain SNR median | Brain SNR p95 | n_bg corner | SNR map |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            "| {modality} | {sigma_bg_corner:.3f} | {sigma_bg_whole_outside_head:.3f} | {ratio:.3f} | {mean:.3f} | {median:.3f} | {p95:.3f} | {n_bg} | `{snr_map}` |".format(
                modality=row["modality"],
                sigma_bg_corner=row["sigma_bg_corner"],
                sigma_bg_whole_outside_head=row["sigma_bg_whole_outside_head"],
                ratio=row["sigma_bg_ratio_corner_to_whole"] if row["sigma_bg_ratio_corner_to_whole"] is not None else float("nan"),
                mean=row["brain_snr_stats"]["mean"],
                median=row["brain_snr_stats"]["p50"],
                p95=row["brain_snr_stats"]["p95"],
                n_bg=row["background_stats_corner"]["n_vox"],
                snr_map=row["snr_map_path"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Corner-background SNR analysis for processed mouse MRI volumes.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mton", required=True)
    parser.add_argument("--mtoff", required=True)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--mt-mask", required=True, help="Mask for MTon/MToff/T1 grid.")
    parser.add_argument("--t2-mask", required=True, help="Mask for T2/RARE grid.")
    parser.add_argument("--corner-frac-x", type=float, default=0.15)
    parser.add_argument("--corner-frac-y", type=float, default=0.15)
    parser.add_argument("--min-corner-vox", type=int, default=6)
    parser.add_argument("--guard-radius-vox", type=int, default=5)
    parser.add_argument("--margin-vox", type=int, default=2)
    parser.add_argument("--outlier-percentile", type=float, default=99.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append(
        _analyze_modality(
            mod_key="MTon",
            img_path=Path(args.mton).resolve(),
            mask_path=Path(args.mt_mask).resolve(),
            output_dir=output_dir,
            corner_frac_x=args.corner_frac_x,
            corner_frac_y=args.corner_frac_y,
            min_corner_vox=args.min_corner_vox,
            guard_radius_vox=args.guard_radius_vox,
            margin_vox=args.margin_vox,
            outlier_percentile=args.outlier_percentile,
        )
    )
    rows.append(
        _analyze_modality(
            mod_key="MToff_PDw",
            img_path=Path(args.mtoff).resolve(),
            mask_path=Path(args.mt_mask).resolve(),
            output_dir=output_dir,
            corner_frac_x=args.corner_frac_x,
            corner_frac_y=args.corner_frac_y,
            min_corner_vox=args.min_corner_vox,
            guard_radius_vox=args.guard_radius_vox,
            margin_vox=args.margin_vox,
            outlier_percentile=args.outlier_percentile,
        )
    )
    rows.append(
        _analyze_modality(
            mod_key="MToff_T1",
            img_path=Path(args.t1).resolve(),
            mask_path=Path(args.mt_mask).resolve(),
            output_dir=output_dir,
            corner_frac_x=args.corner_frac_x,
            corner_frac_y=args.corner_frac_y,
            min_corner_vox=args.min_corner_vox,
            guard_radius_vox=args.guard_radius_vox,
            margin_vox=args.margin_vox,
            outlier_percentile=args.outlier_percentile,
        )
    )
    rows.append(
        _analyze_modality(
            mod_key="RAREvfl",
            img_path=Path(args.t2).resolve(),
            mask_path=Path(args.t2_mask).resolve(),
            output_dir=output_dir,
            corner_frac_x=args.corner_frac_x,
            corner_frac_y=args.corner_frac_y,
            min_corner_vox=args.min_corner_vox,
            guard_radius_vox=args.guard_radius_vox,
            margin_vox=args.margin_vox,
            outlier_percentile=args.outlier_percentile,
        )
    )

    summary_json = output_dir / "snr_corner_summary.json"
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "formula_version": "paper_mt_snr_v1_corner_xy_background_v1",
        "script_path": str(Path(__file__).resolve()),
        "rows": rows,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_tsv = output_dir / "snr_corner_summary.tsv"
    header = [
        "modality",
        "sigma_bg_corner",
        "sigma_bg_whole_outside_head",
        "sigma_bg_ratio_corner_to_whole",
        "n_bg_corner",
        "brain_snr_mean",
        "brain_snr_median",
        "brain_snr_p95",
        "snr_map_path",
        "background_mask_path",
    ]
    with summary_tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write(
                "\t".join(
                    [
                        row["modality"],
                        f"{row['sigma_bg_corner']:.8f}",
                        f"{row['sigma_bg_whole_outside_head']:.8f}",
                        (
                            f"{row['sigma_bg_ratio_corner_to_whole']:.8f}"
                            if row["sigma_bg_ratio_corner_to_whole"] is not None
                            else ""
                        ),
                        str(row["background_stats_corner"]["n_vox"]),
                        f"{row['brain_snr_stats']['mean']:.8f}",
                        f"{row['brain_snr_stats']['p50']:.8f}",
                        f"{row['brain_snr_stats']['p95']:.8f}",
                        row["snr_map_path"],
                        row["background_mask_path"],
                    ]
                )
                + "\n"
            )

    md_path = _write_markdown(output_dir, rows)
    print(f"[OK] output_dir={output_dir}")
    print(f"[OK] wrote {summary_json}")
    print(f"[OK] wrote {summary_tsv}")
    print(f"[OK] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Manual air-ROI based MRI SNR analysis.

Formula:
    sigma_bg = std(abs(img[air_roi]))
    snr_map = abs(img) / sigma_bg

The brain ROI is used only for SNR summary statistics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


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


def _load_bool(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    arr = nib.load(str(path)).get_fdata() > 0
    if arr.shape != shape:
        raise ValueError(f"Mask shape mismatch for {path}: expected {shape}, got {arr.shape}")
    return arr


def _save_float(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header), str(out_path))


def _save_mask(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.uint8), ref_img.affine, ref_img.header), str(out_path))


def _world_bbox(mask: np.ndarray, affine: np.ndarray) -> dict[str, list[float]] | None:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    xyz = nib.affines.apply_affine(affine, idx)
    return {
        "min": [float(v) for v in xyz.min(axis=0)],
        "max": [float(v) for v in xyz.max(axis=0)],
    }


def _mask_bbox(mask: np.ndarray) -> dict[str, list[int]] | None:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    return {
        "min": [int(v) for v in idx.min(axis=0)],
        "max": [int(v) for v in idx.max(axis=0)],
    }


def _pick_slices(mask: np.ndarray, n_show: int = 6) -> list[int]:
    per_slice = mask.reshape((-1, mask.shape[-1])).sum(axis=0)
    nz = np.flatnonzero(per_slice > 0)
    if nz.size == 0:
        return [mask.shape[-1] // 2]
    if nz.size <= n_show:
        return [int(v) for v in nz]
    qs = np.linspace(0, nz.size - 1, n_show)
    return [int(nz[int(round(q))]) for q in qs]


def _plot_grid_qc(
    *,
    ref_img: np.ndarray,
    air_roi: np.ndarray,
    brain_mask: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    slices = _pick_slices(air_roi, n_show=6)
    vmax = float(np.percentile(ref_img[np.isfinite(ref_img)], 99))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for ax, z in zip(axes, slices):
        base = ref_img[:, :, z].T
        air = air_roi[:, :, z].T
        brain = brain_mask[:, :, z].T
        ax.imshow(base, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        ax.contour(brain.astype(float), levels=[0.5], colors=["lime"], linewidths=0.8)
        ax.contour(air.astype(float), levels=[0.5], colors=["red"], linewidths=1.0)
        ax.set_title(f"z={z}")
        ax.axis("off")
    for ax in axes[len(slices) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram(values: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=80, color="#4C78A8", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Voxel count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _analyze(
    *,
    mod_key: str,
    img_path: Path,
    brain_mask_path: Path,
    air_roi_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    img_nii = nib.load(str(img_path))
    img = np.abs(img_nii.get_fdata().astype(np.float32))
    finite = np.isfinite(img)
    brain = _load_bool(brain_mask_path, img.shape) & finite
    air = _load_bool(air_roi_path, img.shape) & finite

    air_vals = img[air]
    sigma_bg = float(np.std(air_vals)) if air_vals.size else 0.0
    snr_map = img / sigma_bg if sigma_bg > 0 else np.zeros_like(img, dtype=np.float32)
    snr_vals = snr_map[brain]

    snr_path = output_dir / f"{mod_key}_snr_manual_air_roi.nii.gz"
    _save_float(snr_map, img_nii, snr_path)

    hist_path = output_dir / f"{mod_key}_air_roi_hist.png"
    _plot_histogram(air_vals, hist_path, f"{mod_key} air ROI intensity histogram")

    per_slice_mean: list[float] = []
    per_slice_p95: list[float] = []
    per_slice_n: list[int] = []
    for z in range(air.shape[2]):
        roi_z = air[:, :, z]
        if not roi_z.any():
            continue
        vals_z = img[:, :, z][roi_z]
        per_slice_mean.append(float(np.mean(vals_z)))
        per_slice_p95.append(float(np.percentile(vals_z, 95)))
        per_slice_n.append(int(vals_z.size))

    return {
        "modality": mod_key,
        "source_image_path": str(img_path),
        "brain_mask_path": str(brain_mask_path),
        "air_roi_path": str(air_roi_path),
        "snr_map_path": str(snr_path),
        "air_hist_qc_path": str(hist_path),
        "shape": list(img.shape),
        "voxel_size": [float(v) for v in img_nii.header.get_zooms()[: len(img.shape)]],
        "air_roi_voxels": int(air.sum()),
        "air_roi_bbox_vox": _mask_bbox(air),
        "air_roi_bbox_world_mm": _world_bbox(air, img_nii.affine),
        "air_roi_stats": _safe_stats(air_vals),
        "brain_snr_stats": _safe_stats(snr_vals),
        "sigma_bg": sigma_bg,
        "bright_frac_ge_1000": float(np.mean(air_vals >= 1000.0)),
        "bright_frac_ge_2000": float(np.mean(air_vals >= 2000.0)),
        "slice_mean_std": float(np.std(per_slice_mean)) if per_slice_mean else None,
        "slice_p95_std": float(np.std(per_slice_p95)) if per_slice_p95 else None,
        "slice_mean_mean": float(np.mean(per_slice_mean)) if per_slice_mean else None,
        "slice_p95_mean": float(np.mean(per_slice_p95)) if per_slice_p95 else None,
        "slice_count_with_roi": len(per_slice_n),
    }


def _write_markdown(output_dir: Path, rows: list[dict[str, Any]], qc_paths: dict[str, str]) -> Path:
    md_path = output_dir / "manual_air_roi_report.md"
    lines = [
        "# Manual air-ROI SNR report",
        "",
        f"Generated: {dt.datetime.now().isoformat()}",
        "",
        "Method:",
        "- `sigma_bg = std(abs(img[air_roi]))`",
        "- `snr_map = abs(img) / sigma_bg`",
        "- Brain SNR summary uses the provided brain mask.",
        "",
        "| Modality | sigma_bg | Air vox | Air p95 | Bright>=1000 | Brain SNR mean | Brain SNR p50 | Brain SNR p95 | SNR map | Hist QC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    if qc_paths:
        lines.insert(8, "")
        lines.insert(8, "QC overlay files:")
        insert_at = 9
        if "mt_grid_overlay" in qc_paths:
            lines.insert(insert_at, f"- MT grid overlay: `{qc_paths['mt_grid_overlay']}`")
            insert_at += 1
        if "rare_grid_overlay" in qc_paths:
            lines.insert(insert_at, f"- RARE grid overlay: `{qc_paths['rare_grid_overlay']}`")
            insert_at += 1
        lines.insert(insert_at, "")
    for row in rows:
        lines.append(
            "| {modality} | {sigma:.3f} | {n_air} | {air_p95:.3f} | {bright:.4f} | {mean:.3f} | {p50:.3f} | {p95:.3f} | `{snr}` | `{hist}` |".format(
                modality=row["modality"],
                sigma=row["sigma_bg"],
                n_air=row["air_roi_voxels"],
                air_p95=row["air_roi_stats"]["p95"],
                bright=row["bright_frac_ge_1000"],
                mean=row["brain_snr_stats"]["mean"],
                p50=row["brain_snr_stats"]["p50"],
                p95=row["brain_snr_stats"]["p95"],
                snr=Path(row["snr_map_path"]).name,
                hist=Path(row["air_hist_qc_path"]).name,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual air-ROI based SNR analysis.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mton", default="")
    parser.add_argument("--mtoff", default="")
    parser.add_argument("--t1", default="")
    parser.add_argument("--t2", default="")
    parser.add_argument("--mt-brain-mask", default="")
    parser.add_argument("--t2-brain-mask", default="")
    parser.add_argument("--mt-air-roi", default="")
    parser.add_argument("--t2-air-roi", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    qc: dict[str, str] = {}

    mt_modalities = []
    if args.mton.strip():
        mt_modalities.append(("MTon", Path(args.mton).resolve()))
    if args.mtoff.strip():
        mt_modalities.append(("MToff_PDw", Path(args.mtoff).resolve()))
    if args.t1.strip():
        mt_modalities.append(("MToff_T1", Path(args.t1).resolve()))

    if mt_modalities:
        if not args.mt_brain_mask.strip() or not args.mt_air_roi.strip():
            raise ValueError("MT modalities were provided, but --mt-brain-mask or --mt-air-roi is missing.")
        mt_ref = nib.load(str(mt_modalities[0][1]))
        mt_img = np.abs(mt_ref.get_fdata().astype(np.float32))
        mt_air = _load_bool(Path(args.mt_air_roi).resolve(), mt_img.shape)
        mt_brain = _load_bool(Path(args.mt_brain_mask).resolve(), mt_img.shape)
        mt_overlay = output_dir / "qc_mt_grid_overlay.png"
        _plot_grid_qc(
            ref_img=mt_img,
            air_roi=mt_air,
            brain_mask=mt_brain,
            out_path=mt_overlay,
            title="MT grid QC: air ROI (red) and brain mask (green)",
        )
        qc["mt_grid_overlay"] = str(mt_overlay)
        for mod_key, img_path in mt_modalities:
            rows.append(
                _analyze(
                    mod_key=mod_key,
                    img_path=img_path,
                    brain_mask_path=Path(args.mt_brain_mask).resolve(),
                    air_roi_path=Path(args.mt_air_roi).resolve(),
                    output_dir=output_dir,
                )
            )

    if args.t2.strip():
        if not args.t2_brain_mask.strip() or not args.t2_air_roi.strip():
            raise ValueError("T2 modality was provided, but --t2-brain-mask or --t2-air-roi is missing.")
        t2_ref = nib.load(str(Path(args.t2).resolve()))
        t2_img = np.abs(t2_ref.get_fdata().astype(np.float32))
        t2_air = _load_bool(Path(args.t2_air_roi).resolve(), t2_img.shape)
        t2_brain = _load_bool(Path(args.t2_brain_mask).resolve(), t2_img.shape)
        t2_overlay = output_dir / "qc_rare_grid_overlay.png"
        _plot_grid_qc(
            ref_img=t2_img,
            air_roi=t2_air,
            brain_mask=t2_brain,
            out_path=t2_overlay,
            title="RARE grid QC: air ROI (red) and brain mask (green)",
        )
        qc["rare_grid_overlay"] = str(t2_overlay)
        rows.append(
            _analyze(
                mod_key="RAREvfl",
                img_path=Path(args.t2).resolve(),
                brain_mask_path=Path(args.t2_brain_mask).resolve(),
                air_roi_path=Path(args.t2_air_roi).resolve(),
                output_dir=output_dir,
            )
        )

    if not rows:
        raise ValueError("No modalities were provided.")

    summary_json = output_dir / "manual_air_roi_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "formula": "SNR = abs(img) / std(abs(img[air_roi]))",
                "rows": rows,
                "qc": qc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_tsv = output_dir / "manual_air_roi_summary.tsv"
    header = [
        "modality",
        "sigma_bg",
        "air_roi_voxels",
        "air_p95",
        "air_p99",
        "bright_frac_ge_1000",
        "bright_frac_ge_2000",
        "brain_snr_mean",
        "brain_snr_p50",
        "brain_snr_p95",
        "snr_map_path",
    ]
    with summary_tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write(
                "\t".join(
                    [
                        row["modality"],
                        f"{row['sigma_bg']:.8f}",
                        str(row["air_roi_voxels"]),
                        f"{row['air_roi_stats']['p95']:.8f}",
                        f"{row['air_roi_stats']['p99']:.8f}",
                        f"{row['bright_frac_ge_1000']:.8f}",
                        f"{row['bright_frac_ge_2000']:.8f}",
                        f"{row['brain_snr_stats']['mean']:.8f}",
                        f"{row['brain_snr_stats']['p50']:.8f}",
                        f"{row['brain_snr_stats']['p95']:.8f}",
                        row["snr_map_path"],
                    ]
                )
                + "\n"
            )

    md_path = _write_markdown(output_dir, rows, qc)
    print(f"[OK] output_dir={output_dir}")
    print(f"[OK] wrote {summary_json}")
    print(f"[OK] wrote {summary_tsv}")
    print(f"[OK] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

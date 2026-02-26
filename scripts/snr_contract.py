#!/usr/bin/env python3
"""
Paper-style MRI SNR map generation with a unified SNR contract.

For MT-style SNR (per P3136 definition):
    SNR = magnitude_signal / std(background_signal)

This script produces three JSON reports with a shared schema:
    - background_stats.json
    - background_stats_nobright.json
    - background_stats_t1t2.json

And writes per-modality/per-threshold:
    - head masks
    - background masks
    - SNR maps
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import nibabel as nib
import numpy as np


def _parse_percentiles(text: str) -> list[int]:
    vals: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        v = int(raw)
        if v <= 0 or v >= 100:
            raise ValueError(f"Percentile must be in (0, 100): {v}")
        vals.append(v)
    if not vals:
        raise ValueError("No valid percentiles provided.")
    return vals


def _dilate_connectivity(mask: np.ndarray, iters: int) -> np.ndarray:
    """Binary dilation with axis-neighbor connectivity (2N connectivity)."""
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


def _save_mask(mask: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out = nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(out, str(out_path))


def _save_float(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(out, str(out_path))


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


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(str(path), str(root))


def _load_mask(mask_path: Path, shape: tuple[int, ...]) -> np.ndarray:
    m = nib.load(str(mask_path)).get_fdata()
    if m.shape != shape:
        raise ValueError(f"Mask shape mismatch: expected {shape}, got {m.shape} at {mask_path}")
    return m > 0


def _run_group(
    *,
    group_name: str,
    modalities: list[tuple[str, Path | None, Path | None]],
    out_json: Path,
    output_dir: Path,
    percentiles: list[int],
    guard_radius_vox: int,
    margin_vox: int,
    outlier_percentile: float,
    nobright: bool,
    default_mask: Path | None,
    formula_version: str,
    noise_estimator: str,
    run_id: str,
    script_path: Path,
) -> list[dict[str, Any]]:
    record_images: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    used_any_input_mask = False

    for mod_key, img_path, mod_mask_path in modalities:
        if img_path is None:
            continue
        if not img_path.exists():
            raise FileNotFoundError(f"Missing input image: {img_path}")

        img_nii = nib.load(str(img_path))
        img = img_nii.get_fdata().astype(np.float32)
        img_mag = np.abs(img)
        finite = np.isfinite(img_mag)

        use_mask_as_head = False
        source_mask_path: Path | None = None
        head_mask_from_input: np.ndarray | None = None

        if mod_mask_path is not None:
            source_mask_path = mod_mask_path
            head_mask_from_input = _load_mask(mod_mask_path, img.shape)
            use_mask_as_head = True
            used_any_input_mask = True
        elif default_mask is not None:
            source_mask_path = default_mask
            head_mask_from_input = _load_mask(default_mask, img.shape)
            use_mask_as_head = True
            used_any_input_mask = True

        mod_block: dict[str, Any] = {
            "source_image_path": _rel(img_path, output_dir),
            "source_mask_path": _rel(source_mask_path, output_dir) if source_mask_path else None,
            "thresholds": {},
        }

        suffix = "_nobright" if nobright else ""
        for p in percentiles:
            pkey = f"p{p}"
            if use_mask_as_head and head_mask_from_input is not None:
                head_mask = head_mask_from_input & finite
                threshold_value = None
            else:
                nonzero = img_mag[finite & (img_mag > 0)]
                if nonzero.size == 0:
                    threshold_value = 0.0
                else:
                    threshold_value = float(np.percentile(nonzero, p))
                head_mask = finite & (img_mag >= threshold_value)

            exclusion = _dilate_connectivity(head_mask, max(0, guard_radius_vox + margin_vox))
            background_mask = finite & (~exclusion)

            if not nobright and background_mask.any() and outlier_percentile < 100.0:
                cap = float(np.percentile(img_mag[background_mask], outlier_percentile))
                background_mask = background_mask & (img_mag <= cap)

            bg_vals = img_mag[background_mask]
            sigma_bg = float(np.std(bg_vals)) if bg_vals.size > 0 else 0.0

            if sigma_bg > 0:
                snr_map = img_mag / sigma_bg
            else:
                snr_map = np.zeros_like(img_mag, dtype=np.float32)

            head_path = output_dir / f"{mod_key}_head{suffix}_{pkey}.nii.gz"
            bg_path = output_dir / f"{mod_key}_background{suffix}_{pkey}.nii.gz"
            snr_path = output_dir / f"{mod_key}_snr{suffix}_{pkey}.nii.gz"
            _save_mask(head_mask, img_nii, head_path)
            _save_mask(background_mask, img_nii, bg_path)
            _save_float(snr_map, img_nii, snr_path)

            roi_vals = snr_map[head_mask]
            thr_block = {
                "threshold_value": threshold_value,
                "formula_version": formula_version,
                "noise_estimator": noise_estimator,
                "sigma_bg": sigma_bg,
                "n_bg_vox": int(bg_vals.size),
                "snr_map_path": _rel(snr_path, output_dir),
                "source_image_path": _rel(img_path, output_dir),
                "script_path": _rel(script_path, output_dir),
                "run_id": run_id,
                "head_mask_path": _rel(head_path, output_dir),
                "background_mask_path": _rel(bg_path, output_dir),
                "background_stats": _safe_stats(bg_vals),
                "snr_roi_stats": _safe_stats(roi_vals),
            }
            mod_block["thresholds"][pkey] = thr_block

            summary_rows.append(
                {
                    "group": group_name,
                    "modality": mod_key,
                    "threshold": pkey,
                    "sigma_bg": sigma_bg,
                    "n_bg_vox": int(bg_vals.size),
                    "snr_roi_mean": thr_block["snr_roi_stats"]["mean"],
                    "snr_roi_p95": thr_block["snr_roi_stats"]["p95"],
                    "snr_map_path": thr_block["snr_map_path"],
                }
            )

        record_images[mod_key] = mod_block

    payload = {
        "schema_version": "snr_contract_v1",
        "run_id": run_id,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_path": _rel(script_path, output_dir),
        "group_name": group_name,
        "contract": {
            "formula_version": formula_version,
            "noise_estimator": noise_estimator,
        },
        "params": {
            "percentiles": percentiles,
            "guard_radius_vox": guard_radius_vox,
            "margin_vox": margin_vox,
            "outlier_percentile": outlier_percentile,
            "nobright": nobright,
            "use_mask_as_head": used_any_input_mask,
        },
        "images": record_images,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-style SNR maps with unified SNR contract.")
    parser.add_argument("--output-dir", required=True, help="Output folder for maps and JSON reports.")
    parser.add_argument("--run-id", default="", help="Optional run id. Auto-generated when empty.")
    parser.add_argument("--formula-version", default="paper_mt_snr_v1", help="Formula version tag.")
    parser.add_argument(
        "--noise-estimator",
        default="std_background_magnitude",
        help="Noise estimator tag.",
    )
    parser.add_argument("--percentiles", default="20,30,40", help="Comma-separated thresholds.")
    parser.add_argument("--guard-radius-vox", type=int, default=5)
    parser.add_argument("--margin-vox", type=int, default=2)
    parser.add_argument("--outlier-percentile", type=float, default=99.0)

    parser.add_argument("--mton", default="")
    parser.add_argument("--mtoff", default="")
    parser.add_argument("--t1", default="")
    parser.add_argument("--t2", default="")

    parser.add_argument("--mton-mask", default="")
    parser.add_argument("--mtoff-mask", default="")
    parser.add_argument("--t1-mask", default="")
    parser.add_argument("--t2-mask", default="")
    parser.add_argument("--default-mask", default="")

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or f"snr_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    percentiles = _parse_percentiles(args.percentiles)
    script_path = Path(__file__).resolve()

    def _p(x: str) -> Path | None:
        x = x.strip()
        return Path(x).resolve() if x else None

    mton = _p(args.mton)
    mtoff = _p(args.mtoff)
    t1 = _p(args.t1)
    t2 = _p(args.t2)
    mton_mask = _p(args.mton_mask)
    mtoff_mask = _p(args.mtoff_mask)
    t1_mask = _p(args.t1_mask)
    t2_mask = _p(args.t2_mask)
    default_mask = _p(args.default_mask)

    all_rows: list[dict[str, Any]] = []

    all_rows.extend(
        _run_group(
            group_name="mt_standard",
            modalities=[("mton", mton, mton_mask), ("mtoff", mtoff, mtoff_mask)],
            out_json=output_dir / "background_stats.json",
            output_dir=output_dir,
            percentiles=percentiles,
            guard_radius_vox=args.guard_radius_vox,
            margin_vox=args.margin_vox,
            outlier_percentile=args.outlier_percentile,
            nobright=False,
            default_mask=default_mask,
            formula_version=args.formula_version,
            noise_estimator=args.noise_estimator,
            run_id=run_id,
            script_path=script_path,
        )
    )
    all_rows.extend(
        _run_group(
            group_name="mt_nobright",
            modalities=[("mton", mton, mton_mask), ("mtoff", mtoff, mtoff_mask)],
            out_json=output_dir / "background_stats_nobright.json",
            output_dir=output_dir,
            percentiles=percentiles,
            guard_radius_vox=args.guard_radius_vox,
            margin_vox=args.margin_vox,
            outlier_percentile=args.outlier_percentile,
            nobright=True,
            default_mask=default_mask,
            formula_version=args.formula_version,
            noise_estimator=args.noise_estimator,
            run_id=run_id,
            script_path=script_path,
        )
    )
    all_rows.extend(
        _run_group(
            group_name="t1t2_standard",
            modalities=[("t1", t1, t1_mask), ("t2", t2, t2_mask)],
            out_json=output_dir / "background_stats_t1t2.json",
            output_dir=output_dir,
            percentiles=percentiles,
            guard_radius_vox=args.guard_radius_vox,
            margin_vox=args.margin_vox,
            outlier_percentile=args.outlier_percentile,
            nobright=False,
            default_mask=default_mask,
            formula_version=args.formula_version,
            noise_estimator=args.noise_estimator,
            run_id=run_id,
            script_path=script_path,
        )
    )

    summary_path = output_dir / "snr_contract_summary.tsv"
    header = [
        "group",
        "modality",
        "threshold",
        "sigma_bg",
        "n_bg_vox",
        "snr_roi_mean",
        "snr_roi_p95",
        "snr_map_path",
    ]
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in all_rows:
            f.write(
                "\t".join(
                    [
                        str(row.get("group", "")),
                        str(row.get("modality", "")),
                        str(row.get("threshold", "")),
                        str(row.get("sigma_bg", "")),
                        str(row.get("n_bg_vox", "")),
                        str(row.get("snr_roi_mean", "")),
                        str(row.get("snr_roi_p95", "")),
                        str(row.get("snr_map_path", "")),
                    ]
                )
                + "\n"
            )

    print(f"[OK] run_id={run_id}")
    print(f"[OK] output_dir={output_dir}")
    print(f"[OK] wrote {output_dir / 'background_stats.json'}")
    print(f"[OK] wrote {output_dir / 'background_stats_nobright.json'}")
    print(f"[OK] wrote {output_dir / 'background_stats_t1t2.json'}")
    print(f"[OK] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

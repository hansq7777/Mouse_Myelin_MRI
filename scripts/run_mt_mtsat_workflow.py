#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_SCRIPT = REPO_ROOT / "mt_pipeline" / "split_mt_dicom.py"
DCM2NII_SCRIPT = REPO_ROOT / "mt_pipeline" / "dcm2nii" / "enhDic2Nii.sh"
MATLAB_DIR = REPO_ROOT / "mt_pipeline" / "matlab"
DEFAULT_MATLAB_EXE = "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe"


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


def matlab_quote(text: str) -> str:
    return text.replace("'", "''")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def strip_nii_suffix(path: Path) -> Path:
    s = str(path)
    if s.endswith(".nii.gz"):
        return Path(s[:-7])
    if s.endswith(".nii"):
        return Path(s[:-4])
    return path


def resolve_single_dicom(path_str: str) -> Path:
    path = normalize_path(path_str)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"DICOM path not found: {path}")

    dcm_files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".dcm")
    if len(dcm_files) == 1:
        return dcm_files[0]
    if len(dcm_files) > 1:
        raise ValueError(f"Directory {path} contains multiple DICOM files; pass one file explicitly.")

    all_files = sorted(p for p in path.iterdir() if p.is_file())
    if len(all_files) == 1:
        return all_files[0]
    raise ValueError(f"Directory {path} must contain exactly one DICOM-like file.")


def run_command(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    summary = {
        "command": cmd,
        "cwd": str(cwd),
        "return_code": int(completed.returncode),
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + summary["stdout_tail"]
            + "\nstderr:\n"
            + summary["stderr_tail"]
        )
    return summary


def matlab_command(matlab_exe: str, code: str) -> list[str]:
    path = normalize_path(matlab_exe)
    if path.exists():
        return [str(path), "-batch", code]
    return [matlab_exe, "-batch", code]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standardized mouse MT workflow for MTon/MToff/T1 preparation, "
            "NIfTI conversion, orientation, and MTR/MTsat generation."
        )
    )
    parser.add_argument("--study-root", required=True, help="Study root where standardized folders are created.")
    parser.add_argument("--mtc-dicom", required=True, help="Enhanced MT DICOM file or directory.")
    parser.add_argument("--t1-dicom", required=True, help="T1w DICOM file or directory.")
    parser.add_argument(
        "--off-first-nframes",
        type=int,
        default=60,
        help="Frames assigned to MT-off when temporal indices are unavailable. Default: 60.",
    )
    parser.add_argument(
        "--gaussian-filter",
        type=int,
        choices=[0, 1],
        default=1,
        help="Pass-through to nii2mtsat. Default: 1.",
    )
    parser.add_argument(
        "--mask-stem",
        default="",
        help="Optional mask stem passed to nii2mtsat (with or without .nii.gz). Default: empty.",
    )
    parser.add_argument(
        "--b1-stem",
        default="",
        help="Optional B1/RFlocal stem passed to nii2mtsat (with or without .nii.gz). Default: empty.",
    )
    parser.add_argument(
        "--matlab-exe",
        default=os.environ.get("MATLAB_EXE", DEFAULT_MATLAB_EXE),
        help="MATLAB executable. Default: env MATLAB_EXE or R2025b matlab.exe path.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest path. Default: <study-root>/mt_mtsat_manifest.json.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    study_root = normalize_path(args.study_root)
    mtc_input = resolve_single_dicom(args.mtc_dicom)
    t1_input = resolve_single_dicom(args.t1_dicom)
    manifest_path = normalize_path(args.manifest) if args.manifest.strip() else study_root / "mt_mtsat_manifest.json"

    dirs = {
        "MTon": ensure_dir(study_root / "MTon"),
        "MToff_PDw": ensure_dir(study_root / "MToff_PDw"),
        "MToff_T1": ensure_dir(study_root / "MToff_T1"),
    }
    std_dicoms = {
        "MTon": dirs["MTon"] / "MTon.dcm",
        "MToff_PDw": dirs["MToff_PDw"] / "MToff_PDw.dcm",
        "MToff_T1": dirs["MToff_T1"] / "MToff_T1.dcm",
    }
    stems = {name: dirs[name] / name for name in dirs}
    outputs = {
        "MTR": dirs["MTon"] / "MTon_mtr.nii.gz",
        "MTC": dirs["MTon"] / "MTon_mtc.nii.gz",
        "MTsat": dirs["MTon"] / "MTon_mtsat.nii.gz",
        "MTsatRaw": dirs["MTon"] / "MTon_mtsat_raw.nii.gz",
        "MTsatClip1": dirs["MTon"] / "MTon_mtsat_clip1.nii.gz",
    }

    manifest: dict[str, Any] = {
        "run_started": now_iso(),
        "repo_root": str(REPO_ROOT),
        "study_root": str(study_root),
        "inputs": {
            "mtc_dicom": str(mtc_input),
            "t1_dicom": str(t1_input),
            "off_first_nframes": int(args.off_first_nframes),
            "gaussian_filter": int(args.gaussian_filter),
            "mask_stem": args.mask_stem,
            "b1_stem": args.b1_stem,
            "matlab_exe": args.matlab_exe,
        },
        "standardized_dicoms": {k: str(v) for k, v in std_dicoms.items()},
        "standardized_stems": {k: str(v) for k, v in stems.items()},
        "steps": [],
        "outputs": {k: str(v) for k, v in outputs.items()},
    }

    try:
        manifest["steps"].append(
            {
                "step": "split_mt_dicom",
                "result": run_command(
                    [
                        sys.executable,
                        str(SPLIT_SCRIPT),
                        str(mtc_input),
                        str(std_dicoms["MToff_PDw"]),
                        str(std_dicoms["MTon"]),
                        "--off-first-nframes",
                        str(args.off_first_nframes),
                    ],
                    cwd=REPO_ROOT,
                ),
            }
        )

        shutil.copy2(t1_input, std_dicoms["MToff_T1"])
        manifest["steps"].append(
            {
                "step": "copy_t1_dicom",
                "source": str(t1_input),
                "destination": str(std_dicoms["MToff_T1"]),
            }
        )

        for name, dicom_path in std_dicoms.items():
            manifest["steps"].append(
                {
                    "step": f"dcm2niix_{name}",
                    "result": run_command(["bash", str(DCM2NII_SCRIPT), str(dicom_path)], cwd=REPO_ROOT),
                }
            )
            nii_path = Path(f"{stems[name]}.nii.gz")
            json_path = Path(f"{stems[name]}.json")
            if not nii_path.exists():
                raise FileNotFoundError(f"Missing NIfTI output: {nii_path}")
            if not json_path.exists():
                raise FileNotFoundError(f"Missing JSON sidecar: {json_path}")

        matlab_dir = to_windows_path(MATLAB_DIR)
        stem_win = {name: to_windows_path(path) for name, path in stems.items()}
        mask_stem = str(strip_nii_suffix(normalize_path(args.mask_stem))) if args.mask_stem.strip() else ""
        b1_stem = str(strip_nii_suffix(normalize_path(args.b1_stem))) if args.b1_stem.strip() else ""
        if mask_stem:
            mask_stem = to_windows_path(Path(mask_stem))
        if b1_stem:
            b1_stem = to_windows_path(Path(b1_stem))

        matlab_code = (
            f"cd('{matlab_quote(matlab_dir)}'); "
            "addMatlabPath; "
            f"niftiOrientation('{matlab_quote(stem_win['MTon'])}'); "
            f"niftiOrientation('{matlab_quote(stem_win['MToff_PDw'])}'); "
            f"niftiOrientation('{matlab_quote(stem_win['MToff_T1'])}'); "
            f"nii2mtsat('{matlab_quote(stem_win['MTon'])}',"
            f"'{matlab_quote(stem_win['MToff_PDw'])}',"
            f"'{matlab_quote(stem_win['MToff_T1'])}',"
            f"'{matlab_quote(mask_stem)}',"
            f"{int(args.gaussian_filter)},"
            f"'{matlab_quote(b1_stem)}');"
        )
        manifest["steps"].append(
            {
                "step": "matlab_orientation_and_mtsat",
                "result": run_command(matlab_command(args.matlab_exe, matlab_code), cwd=REPO_ROOT),
            }
        )

        for out_path in outputs.values():
            if not out_path.exists():
                raise FileNotFoundError(f"Missing MT output: {out_path}")

        manifest["status"] = "success"
        manifest["run_finished"] = now_iso()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["run_finished"] = now_iso()
        manifest["error"] = str(exc)
        ensure_dir(manifest_path.parent)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_BSE_EXE = "/mnt/c/Program Files/BrainSuite23a/bin/bse.exe"
DEFAULT_MODALITIES = ("MTon", "MToff_PDw", "MToff_T1", "T2_RAREvfl")
MODALITY_ALIASES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "MTon": (("MTon",), ("MTon",)),
    "MToff_PDw": (("MToff_PDw", "MToff"), ("MToff_PDw", "MToff")),
    "MToff_T1": (("MToff_T1", "T1"), ("MToff_T1", "T1")),
    "T2_RAREvfl": (("T2_RAREvfl", "RAREvfl", "T2"), ("T2_RAREvfl", "RAREvfl", "T2")),
    "RAREvfl": (("RAREvfl", "T2_RAREvfl", "T2"), ("RAREvfl", "T2_RAREvfl", "T2")),
    "B1": (("B1",), ("B1_ph", "B1")),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Batch-run BrainSuite BSE over modality NIfTI files under one subject folder. "
            "When running from WSL, input/output paths are converted to Windows form automatically."
        )
    )
    p.add_argument("--subject-dir", required=True, help="Subject/session folder containing modality subfolders.")
    p.add_argument(
        "--modalities",
        default=",".join(DEFAULT_MODALITIES),
        help="Comma-separated modality folders to process. Default: MTon,MToff_PDw,MToff_T1,T2_RAREvfl",
    )
    p.add_argument(
        "--bse-exe",
        default=DEFAULT_BSE_EXE,
        help="Path to BrainSuite bse executable. Default assumes BrainSuite23a in Program Files.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing *_bse.nii.gz and *_mask.nii.gz files.")
    p.add_argument("--skip-missing", action="store_true", help="Skip modalities that are not present instead of failing.")
    p.add_argument("--manifest-name", default="brainsuite_bse_manifest.json", help="Manifest filename written in subject dir.")
    p.add_argument("--diffusion-iterations", type=int, default=3, help="BSE -n value. Default matches BrainSuite script.")
    p.add_argument("--diffusion-constant", type=float, default=25.0, help="BSE -d value. Default matches BrainSuite script.")
    p.add_argument("--edge-sigma", type=float, default=0.64, help="BSE -s value. Default matches BrainSuite script.")
    p.add_argument("--verbosity", type=int, default=1, help="BSE -v value.")
    p.add_argument("--no-auto", action="store_true", help="Disable BSE --auto.")
    p.add_argument("--no-trim", action="store_true", help="Disable BSE --trim.")
    p.add_argument("--no-dilate", action="store_true", help="Disable BSE -p final mask dilation.")
    return p.parse_args()


def run_checked(cmd: list[str], *, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def is_wsl() -> bool:
    if os.name == "nt":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def to_windows_path(path: Path) -> str:
    result = run_checked(["wslpath", "-w", str(path.resolve())])
    if result.returncode != 0:
        raise RuntimeError(f"wslpath failed for {path}: {result.stderr.strip()}")
    return result.stdout.strip()


def normalize_exe(exe: str) -> str:
    raw = Path(exe).expanduser()
    if raw.exists():
        return str(raw)
    if exe.startswith("C:") and is_wsl():
        converted = Path("/mnt/c") / Path(exe[3:].replace("\\", "/"))
        if converted.exists():
            return str(converted)
    return exe


def exe_exists(exe: str) -> bool:
    candidate = Path(exe)
    if candidate.exists():
        return True
    return shutil.which(exe) is not None


def modality_candidates(subject_dir: Path, modality: str) -> Iterable[tuple[Path, str]]:
    folder_names, base_names = MODALITY_ALIASES.get(modality, ((modality,), (modality,)))
    for folder_name in folder_names:
        folder = subject_dir / folder_name
        for base_name in base_names:
            yield folder, base_name


def resolve_input_nifti(subject_dir: Path, modality: str) -> tuple[Path, str]:
    suffixes = (".nii.gz", ".nii")
    for folder, base_name in modality_candidates(subject_dir, modality):
        for suffix in suffixes:
            candidate = folder / f"{base_name}{suffix}"
            if candidate.exists():
                return candidate, base_name
    searched = []
    for folder, base_name in modality_candidates(subject_dir, modality):
        for suffix in suffixes:
            searched.append(str(folder / f"{base_name}{suffix}"))
    raise FileNotFoundError(f"Could not locate input NIfTI for modality {modality}. Searched: {searched}")


def output_paths(input_nifti: Path, base_name: str) -> tuple[Path, Path]:
    out_dir = input_nifti.parent
    stripped = out_dir / f"{base_name}_bse.nii.gz"
    mask = out_dir / f"{base_name}_mask.nii.gz"
    return stripped, mask


def build_bse_command(args: argparse.Namespace, exe: str, input_nifti: Path, stripped: Path, mask: Path) -> list[str]:
    input_arg = str(input_nifti.resolve())
    stripped_arg = str(stripped.resolve())
    mask_arg = str(mask.resolve())
    if is_wsl() and exe.lower().endswith(".exe"):
        input_arg = to_windows_path(input_nifti)
        stripped_arg = to_windows_path(stripped)
        mask_arg = to_windows_path(mask)

    cmd = [
        exe,
        "-i",
        input_arg,
        "-o",
        stripped_arg,
        "--mask",
        mask_arg,
        "-n",
        str(args.diffusion_iterations),
        "-d",
        str(args.diffusion_constant),
        "-s",
        str(args.edge_sigma),
        "-v",
        str(args.verbosity),
    ]
    if not args.no_dilate:
        cmd.append("-p")
    if not args.no_trim:
        cmd.append("--trim")
    if not args.no_auto:
        cmd.append("--auto")
    return cmd


def main() -> int:
    args = parse_args()
    subject_dir = Path(args.subject_dir).expanduser().resolve()
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    exe = normalize_exe(args.bse_exe)
    if not exe_exists(exe):
        raise FileNotFoundError(f"BrainSuite bse executable not found: {args.bse_exe}")

    modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    if not modalities:
        raise RuntimeError("No modalities provided.")

    manifest: dict[str, object] = {
        "workflow": "brainsuite_bse_batch",
        "subject_dir": str(subject_dir),
        "bse_exe": exe,
        "modalities_requested": modalities,
        "runs": [],
    }
    failures: list[str] = []

    for modality in modalities:
        try:
            input_nifti, base_name = resolve_input_nifti(subject_dir, modality)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"[skip] {exc}", file=sys.stderr)
                manifest["runs"].append({"modality": modality, "status": "skipped_missing", "reason": str(exc)})
                continue
            failures.append(str(exc))
            manifest["runs"].append({"modality": modality, "status": "missing", "reason": str(exc)})
            continue

        stripped, mask = output_paths(input_nifti, base_name)
        if not args.overwrite and stripped.exists() and mask.exists():
            print(f"[skip] {modality}: outputs already exist", file=sys.stderr)
            manifest["runs"].append(
                {
                    "modality": modality,
                    "status": "skipped_existing",
                    "input_nifti": str(input_nifti),
                    "stripped_output": str(stripped),
                    "mask_output": str(mask),
                }
            )
            continue

        cmd = build_bse_command(args, exe, input_nifti, stripped, mask)
        print("$ " + shlex.join(cmd))
        result = run_checked(cmd)
        run_entry = {
            "modality": modality,
            "input_nifti": str(input_nifti),
            "stripped_output": str(stripped),
            "mask_output": str(mask),
            "command": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
        if result.returncode != 0:
            failures.append(f"{modality}: bse failed with exit code {result.returncode}")
            run_entry["status"] = "failed"
        else:
            run_entry["status"] = "completed"
        manifest["runs"].append(run_entry)

    manifest_path = subject_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")

    if failures:
        for failure in failures:
            print(f"[error] {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

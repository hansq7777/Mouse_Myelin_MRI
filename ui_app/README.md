# MT pipeline UI

PySide6 UI that wraps the MT processing steps. Everything needed to run the UI lives in this `ui_app` folder.

## Install

```
python -m venv .venv
.venv/Scripts/activate  # Windows
# or source .venv/bin/activate on Linux/WSL
pip install -r requirements.txt
```

## Run

```
python -m ui_app
```

The UI reads defaults from `ui_app/configs/presets.json` (MATLAB path setup, dcm2niix path, etc.) and saves project context to `ui_app/project_context.json`.

## Steps covered

- Project/subject: select project root and subject folder.
- MT split: run `mt_pipeline/split_mt_dicom.py` to split MTon/MToff.
- DICOM → NIfTI: call `dcm2niix` or `mt_pipeline/dcm2nii/enhDic2Nii.sh`.
- Recenter: MATLAB `recenterNifti`.
- Orientation: MATLAB `niftiOrientation`.
- Gibbs: MRtrix `mrdegibbs`.
- Brain extraction: open BrainSuite.
- Coregistration: SPM `coreg_est_write_weighted`.
- B1 preprocess: MATLAB `prepareB1RFlocal`.
- MT metrics: MATLAB `nii2mtsat`.
- T1/T2w/r: MATLAB `nii2t1wt2wr`.
- SNR contract maps: Python `scripts/snr_contract.py` (paper-style MT SNR, unified `background_stats*.json` schema).
- Template registration (ANTs): Python `scripts/template_register_multistart.py` (default 3-seed multi-start, includes seed `42`, chooses best QC run) built on `scripts/template_register.py` tuned defaults (`preprocess=n4`, `mask-opt=on`, `init=com_only`).
- Registration storyboard (visual QC): Python `scripts/registration_storyboard.py` composes a long tri-view image from raw/resample/preprocess plus per-stage (`rigid/affine/syn`) outputs; it follows the actual stage list recorded in each run manifest so added/removed/skipped steps are reflected in the storyboard.
- Compare: quick side-by-side NIfTI slice viewer.

## External tools (not bundled in repo)

- MATLAB (command-line `matlab` available) and SPM for coregistration (`coreg_est_write_weighted`). Set `"spm.path"` in `ui_app/configs/presets.json` (e.g., `C:/tools/spm`) so the UI automatically adds it to the MATLAB path before running SPM jobs.
- dcm2niix binary for DICOM → NIfTI (set path in `ui_app/configs/presets.json` or install to PATH).
- MRtrix3 (`mrdegibbs`) for Gibbs ringing correction.
- ANTs binaries for template registration (`antsRegistration`, `antsApplyTransforms`), default expected under `C:/tools/ANTs/ants-2.6.5/bin`.
- BrainSuite23a for manual brain mask creation (set `brainsuite.exe` in presets).
- Optional dMRI steps require FSL (TOPUP/EDDY) if you run the diffusion pipeline; not used by the MT-only UI steps.
- Large imaging datasets and masks under `Data/` are not versioned to GitHub; sync from your storage/share as needed.

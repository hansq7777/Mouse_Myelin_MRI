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
- Compare: quick side-by-side NIfTI slice viewer.

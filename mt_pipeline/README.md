# MT pipeline tools

All MT-processing utilities now live under `mt_pipeline/`:

- `split_mt_dicom.py`: split enhanced DICOM into MT-off/MT-on files with frame selection.
- `dcm2nii/enhDic2Nii.sh` (+ `brukHead.py`): convert enhanced DICOMs to NIfTI and save JSON sidecars.
- `matlab/addMatlabPath.m`: add the MATLAB utilities in this folder to the path (used by Snakemake/UI).
- `matlab/recenterNifti.m`: zero out translation in the NIfTI `Transform.T` and resave.
- `matlab/niftiOrientation.m`: flip volumes to keep cortex superior for downstream alignment.
- `matlab/prepareB1RFlocal.m`: convert a Bruker B1 map into an RFlocal map aligned to MTon.
- `matlab/nii2mtsat.m` + `calcMTsat.m`: generate MTR/MTsat maps (uses optional B1/mask); `bfact_water.txt` holds the B1 correction lookup.

Both the Snakemake pipeline and the UI call these tools relative to the repo root. In MATLAB calls, change into `mt_pipeline/matlab` then run `addMatlabPath` before invoking `recenterNifti`, `niftiOrientation`, or `nii2mtsat`.

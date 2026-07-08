# Mouse MT / MTsat Project Note

Date: 2026-04-10

This note records the current working practice for mouse MT processing in this repository after standardizing the MT partial-registration reference to `MToff_PDw`.

## Scope

- Study folders live under `Data/<study>/`
- Standard modality folders:
  - `MTon/`
  - `MToff_PDw/`
  - `MToff_T1/`
  - `T2_RAREvfl/`
  - `B1/`
- Quantitative MT outputs are generated in MT space.
- `T2` remains the target space for downstream whole-brain export when needed, but the MT quantification itself is run in `MToff_PDw` space.

## Standard raw-to-MTsat workflow

1. Select scans from the raw Bruker study.
   - Prefer the formal multi-average acquisitions, not `1avg` tests.
   - Map the chosen scans to:
     - `MTon` / `MToff_PDw` from the enhanced MTC DICOM
     - `MToff_T1`
     - `T2_RAREvfl`
     - `B1`

2. Split the enhanced MT DICOM.
   - Use `mt_pipeline/split_mt_dicom.py`
   - Write:
     - `MTon/MTon.dcm`
     - `MToff_PDw/MToff_PDw.dcm`

3. Convert the chosen DICOMs to NIfTI + JSON.
   - Use `mt_pipeline/dcm2nii/enhDic2Nii.sh`
   - Expected stems:
     - `MTon/MTon`
     - `MToff_PDw/MToff_PDw`
     - `MToff_T1/MToff_T1`
     - `T2_RAREvfl/T2_RAREvfl`
     - `B1/B1_ph`

4. Normalize slice direction only.
   - Run `niftiOrientation` on all converted NIfTI files.
   - This flips the voxel array so cortex is superior, but it does not remove oblique scanner angulation from the affine.

5. Generate and QC masks.
   - Brain extraction is done per modality.
   - The MT workflow now uses the `MToff_PDw` mask as the primary MT-space mask.
   - If the saved mask is not already on the exact MT grid, resample it first and save a grid-matched derivative such as:
     - `MToff_PDw/MToff_brain_mask_mtgrid.nii.gz`

6. Partial registration in MT space.
   - Register `MToff_T1` to `MToff_PDw`.
   - Preserve the resliced output as:
     - `MToff_T1/r2MToff_MToff_T1.nii.gz`
   - `MTon` and `MToff_PDw` are usually already paired from the same split acquisition and are used directly.

7. Generate MT quantitative maps without B1 correction.
   - Run `nii2mtsat` with:
     - `MTon`
     - `MToff_PDw`
     - `r2MToff_MToff_T1`
     - `MToff_brain_mask_mtgrid`
     - `gaussianFilter=1`
     - empty B1 input
   - Outputs appear under `MTon/`:
     - `*_mtr.nii.gz`
     - `*_mtc.nii.gz`
     - `*_mtsat.nii.gz`
     - `*_mtsat_raw.nii.gz`
     - `*_mtsat_clip1.nii.gz`

## Current display practice

The canonical quantitative files preserve the scanner affine. Some studies therefore look tilted in image viewers even though the voxel data are valid. This is expected when the affine still contains oblique rotation.

- `niftiOrientation` is not enough to make the volume look upright in a viewer.
- `recenterNifti` only changes the origin and also does not remove oblique angulation.

For viewing and screenshots, create display-only derivatives with:

- `scripts/make_display_upright.py`

This script:

- reorients to a canonical axis order,
- replaces the affine with an orthogonal diagonal affine,
- recenters the volume around the origin,
- writes `<stem>_display.nii.gz`

These display files are for visualization only and must not replace the canonical quantitative files used for registration or MT quantification.

## B1 correction practice

### Formula

`calcMTsat.m` computes:

- Helms 2008 apparent MT saturation:
  - `R1app = 0.5*(refT1*alphT1/TRT1 - refPD*alphPD/TRPD) ./ (refPD/alphPD - refT1/alphT1)`
  - `Aapp = (TRPD*alphT1/alphPD - TRT1*alphPD/alphT1) * refPD .* refT1 ./ (refT1*TRPD*alphT1 - refPD*TRT1*alphPD)`
  - `MTsatApp = (Aapp*alphPD./MTon - 1) .* R1app * TRPD - alphPD^2/2`
- Optional Hagiwara 2018 B1 correction:
  - `MTsat = MTsatApp * (1 - 0.4) / (1 - 0.4 * RFlocal)`

### Local B1 handling

Raw Bruker `B1_ph.nii.gz` is a 4D file. The working assumption for these studies is:

- use volume 2 (0-based index 1),
- values are local flip angle in radians,
- nominal flip angle comes from `B1_ph.json`

Recommended preprocessing:

- use `scripts/prepare_b1_rflocal.py`

This helper:

- extracts B1 volume 2,
- resamples it to the MT reference grid using affine-aware resampling,
- converts radians to degrees,
- divides by nominal flip angle to form `RFlocal`,
- clips to a conservative range,
- writes:
  - `<out>.nii.gz`
  - `<out>.csv`
  - `<out>.summary.json`

Recommended B1-corrected run:

1. Prepare `B1_RFlocal`.
2. Run `nii2mtsat` again with the same MTon / MToff / T1 / mask inputs and pass the `B1_RFlocal` stem as the sixth argument.
3. To avoid overwriting the non-B1 run, use a copied MTon stem named `MTon_b1corr`.
4. The resulting B1-corrected outputs are:
   - `MTon_b1corr_mtr.nii.gz`
   - `MTon_b1corr_mtc.nii.gz`
   - `MTon_b1corr_mtsat.nii.gz`
   - `MTon_b1corr_mtsat_raw.nii.gz`
   - `MTon_b1corr_mtsat_clip1.nii.gz`

## Display derivatives in practice

For current studies, display-only upright derivatives are generated for:

- core MT-space inputs:
  - `MTon_display.nii.gz`
  - `MToff_PDw_display.nii.gz`
  - `r2MToff_MToff_T1_display.nii.gz`
- `T2_RAREvfl_display.nii.gz`
- `B1_RFlocal_display.nii.gz`
- non-B1 MT outputs:
  - `MTon_mtr_display.nii.gz`
  - `MTon_mtc_display.nii.gz`
  - `MTon_mtsat_display.nii.gz`
  - `MTon_mtsat_raw_display.nii.gz`
  - `MTon_mtsat_clip1_display.nii.gz`
- B1-corrected MT outputs:
  - `MTon_b1corr_mtr_display.nii.gz`
  - `MTon_b1corr_mtc_display.nii.gz`
  - `MTon_b1corr_mtsat_display.nii.gz`
  - `MTon_b1corr_mtsat_raw_display.nii.gz`
  - `MTon_b1corr_mtsat_clip1_display.nii.gz`

## Known caveat

`coreg_est_write_weighted.m` currently tries to set `spm.spatial.coreg.estwrite.wref`, but the local `SPM25` batch config does not expose that field. As a result, current runs complete, but the coregistration is effectively unweighted unless this function is revised.

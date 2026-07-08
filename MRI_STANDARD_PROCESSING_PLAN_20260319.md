# MRI Standard Processing Plan Draft

Date: 2026-03-19

## Purpose

This note consolidates the MRI-side workflows already present in the workspace, then proposes a standard processing plan for review before execution or skill creation.

## Confirmed existing workflow sources

The workspace already contains a prior MRI processing workflow. It is not a single script; it is a stack:

- `mouse_mt_pipeline/README.txt`
  - MT and dMRI Snakemake workflow overview
- `mouse_mt_pipeline/mt_pipeline/README.md`
  - core MT tools: split, DICOM to NIfTI, orientation, B1 prep, MTsat
- `mouse_mt_pipeline/ui_app/README.md`
  - step-by-step interactive MT processing UI
- `mouse_mt_pipeline/ui_app/windows/main_window.py`
  - actual runnable step order and default paths
- `mouse_mt_pipeline/scripts/*.py`
  - SNR, template build, template registration, QC helpers
- `registration_pipeline/mri/docs/benchmark_default_2026-02-26.md`
  - frozen MRI registration defaults
- `registration_pipeline/mri/docs/mouse_mt_data_cleanup_policy_2026-03-06.md`
  - final-vs-intermediate output guidance

## Current workflow structure in the repo

### 1. MT processing

Primary repo: `mouse_mt_pipeline`

Confirmed steps already implemented:

1. Select project root / subject folder
2. Split combined MT enhanced DICOM into MT-off and MT-on
3. Convert DICOM to NIfTI with `dcm2niix`
4. Apply MATLAB orientation normalization
5. Optional recenter
6. Gibbs correction
7. Manual brain masking in BrainSuite
8. SPM coregistration to MT-on space
9. B1 preprocessing
10. MT metric generation with `nii2mtsat`
11. T1/T2 ratio generation with `nii2t1wt2wr`
12. Optional SNR contract generation
13. Optional template build / template registration
14. Compare and QC

### 2. MRI registration

Two identical script entry sets currently exist:

- `mouse_mt_pipeline/scripts`
- `registration_pipeline/mri/scripts`

As of 2026-03-19, `template_register.py` and `template_register_multistart.py` are byte-identical in both locations. The active default should remain `mouse_mt_pipeline/scripts` because the UI already calls that path.

### 3. Existing registration default

From `registration_pipeline/mri/docs/benchmark_default_2026-02-26.md`:

- inputs must already share one reference grid
- fixed and moving brain masks are required
- preprocess with `N4` on both images
- denoise moving image before `N4`
- default init: `translation_then_rigid`
- keep affine fallback enabled
- nonlinear chain: `Affine -> SyN[0.05,3,0]`
- fixed stage order: `rigid -> affine -> syn`
- always keep QC outputs and transform provenance

## Observed folder conventions

Two conventions exist in prior data:

### Legacy / phantom-style convention

Observed in `Data/RD25MYELOMAP_Test9_20260216/`

- raw Bruker study root kept as one folder
- derived folders created beside it:
  - `MTon/`
  - `MToff/`
  - `T1/`
  - `T2/`

### More modern mouse-side convention

Observed in `Data/mouse4test6_20251113/`

- derived modality folders:
  - `MTon/`
  - `MToff_PDw/`
  - `MToff_T1/`
  - `T2_RAREvfl/`
  - `B1/`
- final outputs kept in-place beside modality raw/intermediate files

### Recommended standard for future reuse

For future automation and skill reuse, prefer the modern naming:

- `MTon`
- `MToff_PDw`
- `MToff_T1`
- `T2_RAREvfl`
- `B1`

Keep legacy `MToff/T1/T2/RAREvfl` only as compatibility aliases when handling older examples.

## Final vs intermediate outputs

Per the cleanup policy dated 2026-03-06:

Keep as durable outputs when produced:

- main modality NIfTI files
- `MTon_mtc.nii.gz`
- `MTon_mtr.nii.gz`
- `MTon_mtsat.nii.gz`
- `B1_RFlocal.nii.gz`
- masks needed by downstream workflow

Treat as intermediate / non-canonical:

- `*_recenter.nii.gz`
- `*_RS.nii.gz`
- `*_vol2*.nii.gz`
- transient retry artifacts

## Processing notes to freeze

These rules should be treated as defaults for future reuse:

1. `1avg` acquisitions are usually test sequences, not primary analysis inputs.
2. When both `1avg` and multi-average versions exist for the same modality, prefer the multi-average acquisition as the formal processing input.
3. `B1` correction is optional and should be off by default.
4. Enable `B1` correction only when a trusted `B1Map` acquisition exists and passes QC.
5. Automatic BrainSuite skull stripping is allowed as the first-pass mask generator, but downstream processing must stop for manual QC before using that mask.

## Mapping of the newly added datasets

### Dataset A

Raw study folder:

- `Data/20260318_111530_RyanDilger_RD25MYELOMAP_2602_MT_Exp1_1_12`

Protocol mapping extracted from Bruker metadata:

- scan `4`: `T1w_16avg (E4)` -> T1 candidate
- scan `5`: `MTC_5uT_16avg (E5)` -> combined MT dataset to split into MT-off and MT-on
- scan `6`: `RAREvfl_16avg (E6)` -> T2_RAREvfl candidate
- scan `7`: `B1Map (E7)` -> B1 candidate

This dataset matches the modern MT workflow well and appears suitable for the full chain including B1 correction.

### Dataset B

Raw study folder:

- `Data/20260317_122544_RyanDilger_RD25MYELOMAP_Test10_mousebrainphantom_1_11`

Protocol mapping extracted from Bruker metadata:

- scan `4`: `T1w_125um_1avg (E4)` -> lower-SNR T1 candidate
- scan `5`: `T1w_125um_75avg (E5)` -> preferred T1 candidate
- scan `6`: `MTC_5uT_125um__1avg (E6)` -> MT candidate
- scan `7`: `RAREvfl_16avg (E7)` -> T2_RAREvfl candidate
- scan `9`: `RAREvfl_150um_10avg (E9)` -> alternate higher-resolution T2_RAREvfl candidate
- scan `10`: `MTC_5uT_125um__1avg (E10)` -> second MT candidate

No `B1Map` scan was found in this study.

This dataset needs one QC decision before standard processing:

- choose the preferred T1 source: scan `5` is the likely default because `75avg` should be preferred over `1avg`
- choose the preferred T2 source: scan `7` vs `9`
- choose the preferred MTC source: scan `6` vs `10`

## Proposed standard processing plan

This is the plan I recommend using going forward, pending review.

### Phase 0. Intake and normalization

1. Create one subject/session wrapper directory per study under `Data/`.
2. Keep the original numbered Bruker study folder untouched inside that wrapper.
3. Create derived modality folders as siblings of the raw study folder.
4. Use modern names by default:
   - `MTon`
   - `MToff_PDw`
   - `MToff_T1`
   - `T2_RAREvfl`
   - `B1`
5. Record scan-number to modality mapping in a small note before running processing.

### Phase 1. Raw-to-modality extraction

1. Identify the MTC enhanced DICOM under the chosen MT scan `pdata/1/dicom`.
2. Run `mt_pipeline/split_mt_dicom.py` on the MTC DICOM.
3. Write outputs to:
   - `MToff_PDw/mtoff.dcm`
   - `MTon/mton.dcm`
4. For T1, T2_RAREvfl, and B1, point `dcm2niix` at the chosen modality folder or DICOM folder.

### Phase 2. NIfTI conversion and geometric normalization

1. Convert each modality with `dcm2niix`.
2. Output names should match folder names:
   - `MTon/MTon.nii.gz`
   - `MToff_PDw/MToff_PDw.nii.gz`
   - `MToff_T1/MToff_T1.nii.gz`
   - `T2_RAREvfl/T2_RAREvfl.nii.gz`
   - `B1/B1_ph.nii.gz`
3. Run `niftiOrientation` on all converted NIfTI files.
4. Use recenter only when needed; do not treat recentered outputs as canonical finals.

### Phase 3. Artifact reduction and mask creation

1. Run Gibbs correction on MTon, MToff_PDw, MToff_T1, and T2_RAREvfl.
2. Run BrainSuite BSE in batch mode first for the available modalities.
3. Save per-modality mask outputs as `<modality>/<modality>_mask.nii.gz`.
4. Stop for manual QC after automatic mask generation. No downstream processing should continue until a human has accepted the chosen mask.
5. Use manual BrainSuite GUI review only for modalities whose automatic BSE result is unsatisfactory.
6. Use MTon or MToff_T1 as the primary masking reference for downstream alignment, depending on QC quality.

### Phase 4. Alignment into MT-on space

1. Use SPM coregistration to align:
   - `MToff_PDw` to `MTon`
   - `MToff_T1` to `MTon`
   - `T2_RAREvfl` to `MTon`
   - `B1` to `MTon` when B1 exists
2. Use the MTon brain mask as the weighting mask.
3. Preserve the final aligned files as the canonical versions used for quantification.

### Phase 5. Quantitative map generation

1. If B1 exists:
   - run `prepareB1RFlocal`
   - save `B1/B1_RFlocal.nii.gz`
2. Run `nii2mtsat` with:
   - MTon
   - MToff_PDw
   - MToff_T1
   - MTon mask
   - optional `B1_RFlocal`
3. Run `nii2t1wt2wr` when T1, T2_RAREvfl, and mask are available.

### Phase 6. QC and optional downstream analysis

1. Visual compare:
   - MTon vs MToff_PDw
   - MTon vs MToff_T1
   - MTon vs T2_RAREvfl
2. Generate SNR contract outputs if paper-style quantitative QC is needed.
3. Only start template registration after:
   - images are on a shared grid
   - fixed and moving masks exist
   - modality QC is acceptable

### Phase 7. MRI template registration

If registration is needed:

1. Use `mouse_mt_pipeline/scripts/template_register_multistart.py`
2. Require fixed and moving masks
3. Keep default registration settings from the 2026-02-26 benchmark
4. Keep full QC and provenance outputs

## Study-specific recommendation

### Recommended plan for 2602

Process with the full standard chain:

- `MToff_T1` from scan `4`
- combined MT from scan `5`
- `T2_RAREvfl` from scan `6`
- `B1` from scan `7`

### Recommended plan for Test10

Process with one QC gate before standardization:

- default T1 candidate: scan `5` because `1avg` should be treated as a test sequence and `75avg` as the formal acquisition
- inspect scan `7` vs `9` before fixing the T2 source
- inspect scan `6` vs `10` before fixing the MT source
- proceed without B1 correction unless an external B1 source exists and passes QC

If MTsat is generated without B1, label that run clearly as "no B1 correction".

## Skillization plan after review

After the workflow is approved, create a reusable Codex skill with:

- skill name: `mouse-mri-standard-processing`
- `SKILL.md`
  - trigger conditions
  - concise processing workflow
  - decision rules for B1 / no-B1 / phantom cases
- `references/`
  - `workflow.md`
  - `naming_conventions.md`
  - `registration_defaults.md`
  - `scan_mapping_examples.md`

The skill should default to:

- modern folder naming
- non-destructive handling of raw Bruker folders
- `mouse_mt_pipeline/scripts` as the active registration entry
- explicit QC checkpoints before registration or final metric export

## Review items needing confirmation

1. Wrap each raw study in a subject/session folder before processing?
2. For Test10, which T2 and MTC repeat should be treated as primary?
3. Should no-B1 MTsat be accepted as a standard deliverable, or only MTR in no-B1 cases?

"""Step metadata for the MT/RARE pipeline UI.

This keeps the UI declarative: names, rough descriptions,
and default commands/placeholders that the UI can call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Step:
    name: str
    description: str
    command: Optional[str] = None  # shell command template; UI supplements with paths
    enabled: bool = True


def default_steps() -> List[Step]:
    return [
        Step(
            name="Split MT ref (MTon/MToff)",
            description="Run split_mt_dicom.py to split enhanced DICOM into MTon/MToff",
            command=(
                'python mt_pipeline/split_mt_dicom.py "<src_dcm>" "<mtoff_out>" "<mton_out>" '
                "--off-first-nframes 60"
            ),
        ),
        Step(
            name="DICOM → NIfTI",
            description="Use dcm2niix/enhDic2Nii.sh to create NIfTI files (MTon/MToff/RARE/B1)",
            command='C:\\tools\\dcm2niix\\dcm2niix.exe -z y -o "<out_dir>" -f "<name>" "<dicom_dir>"',
        ),
        Step(
            name="Recenter NIfTI",
            description="MATLAB: reset Transform.T translation to center the volume",
            command="matlab -batch \"cd('C:/work/mouse_mt_pipeline/mt_pipeline/matlab'); addMatlabPath; recenterNifti('<input>','<output>');\"",
        ),
        Step(
            name="Orientation unify",
            description="MATLAB: run niftiOrientation to standardize slice orientation",
            command="matlab -batch \"cd('C:/work/mouse_mt_pipeline/mt_pipeline/matlab'); addMatlabPath; niftiOrientation('<stem>');\"",
        ),
        Step(
            name="Gibbs correction",
            description="Run MRtrix mrdegibbs on NIfTI (axes 0,1; K=[1,3]; nshifts=20).",
            command="mrdegibbs -axes 0,1 -nshifts 20 -minW 1 -maxW 3 \"<input.nii.gz>\" \"<output.nii.gz>\"",
        ),
        Step(
            name="Brain extraction",
            description="Launch BrainSuite23a and save mask; mark done after saving",
            command=None,
        ),
        Step(
            name="Coregistration (SPM)",
            description="MATLAB: coreg_est_write_weighted with mask weighting",
            command="matlab -batch \"addpath('C:/tools/spm'); spm('Defaults','fMRI'); spm_jobman('initcfg'); coreg_est_write_weighted('<src>','<ref>','<mask>','r2T2_',4);\"",
        ),
        Step(
            name="Compute MTR / MTsat",
            description="MATLAB: nii2mtsat with MTon, MToff_PDw, optional T1/B1, mask",
            command="matlab -batch \"cd('C:/work/mouse_mt_pipeline/mt_pipeline/matlab'); addMatlabPath; nii2mtsat('<mton>','<mtoff>','<t1>','<mask>',1,'<b1>');\"",
        ),
        Step(
            name="Compute T1w/T2w/r",
            description="MATLAB: nii2t1wt2wr with mask, gaussian filter, intensity match",
            command="matlab -batch \"cd('C:/work/mouse_mt_pipeline/mt_pipeline/matlab'); addMatlabPath; nii2t1wt2wr('<t1>','<t2>','', '<mask>',1,1);\"",
        ),
    ]

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class StepStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    DONE = auto()


@dataclass
class PipelineStep:
    key: str
    title: str
    description: str
    status: StepStatus = StepStatus.PENDING


@dataclass
class PipelineState:
    steps: List[PipelineStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.steps:
            self.steps = self._default_steps()

    def _default_steps(self) -> List[PipelineStep]:
        return [
            PipelineStep("project", "Project / Subject", "Set project root, subject/date, modalities."),
            PipelineStep("split", "Step 0 – MT split", "Use split_mt_dicom.py to split enhanced MT DICOM into MTon/MToff."),
            PipelineStep("dcm2nii", "Step 1 – DICOM → NIfTI", "Run dcm2niix or enhDic2Nii.sh to generate .nii.gz and JSON."),
            PipelineStep("recenter", "Step 2 – Recenter", "MATLAB recenterNifti: zero translation in Transform.T."),
            PipelineStep("orient", "Step 3 – Orientation", "MATLAB niftiOrientation: flip so cortex is superior."),
            PipelineStep("gibbs", "Step 3.5 – Gibbs (mrdegibbs)", "MRtrix mrdegibbs ringing correction on NIfTI."),
            PipelineStep("brainsuite", "Step 4 – Brain extraction", "Launch BrainSuite23a to draw/save brain mask."),
            PipelineStep("coreg", "Step 5 – Coregistration", "SPM coreg_est_write_weighted with mask weighting."),
            PipelineStep("b1prep", "Step 6 – B1 preprocess", "Convert Bruker B1 map to RFlocal aligned to MTon."),
            PipelineStep(
                "mtsat",
                "Step 7 – MT metrics",
                "MTR=(MToff-MTon)/MToff (clip 0-1); MTsat (Helms) from PDw+T1w with optional B1 correction and mask.",
            ),
            PipelineStep("t1t2", "Step 8 – T1/T2w/r", "MATLAB nii2t1wt2wr to compute T1w/T2w/T1wT2w ratio."),
            PipelineStep(
                "snr",
                "Step 9 – SNR contract maps",
                "Paper-style SNR maps + unified JSON contract (background_stats*.json).",
            ),
            PipelineStep("compare", "Step 10 – Compare volumes", "Side-by-side slice viewer with synchronization."),
            PipelineStep(
                "template_build",
                "Step 11 – Group template build",
                "Build in-group MT/T2 template from multiple subjects (ANTs pairwise + averaging).",
            ),
            PipelineStep(
                "template_register",
                "Step 12 – Template registration",
                "Register template-to-template or template-to-atlas (ANTs: Rigid + Affine + SyN).",
            ),
        ]

    def get_step(self, key: str) -> Optional[PipelineStep]:
        return next((s for s in self.steps if s.key == key), None)

    def set_status(self, key: str, status: StepStatus) -> None:
        step = self.get_step(key)
        if step:
            step.status = status

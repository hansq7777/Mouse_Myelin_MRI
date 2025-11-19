from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import imageio
from PySide6 import QtCore, QtGui, QtWidgets

from ui_app.state.pipeline_state import PipelineState, StepStatus
from ui_app.widgets.run_log_widget import RunLogWidget
from ui_app.services.commands import run_command
from ui_app.widgets.nii_viewer import NiiSliceViewer
from ui_app.widgets.scale_bar import DiffScaleBar


class MainWindow(QtWidgets.QMainWindow):
    """
    Minimal scaffold with a left step list and right panel placeholder.
    Hooks are ready to wire commands (split, dcm2niix, MATLAB) later.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mouse MT Pipeline")
        self.resize(1100, 700)

        self.repo_root = Path(__file__).resolve().parents[2]
        self.presets = self._load_presets()
        self.state = PipelineState()
        self.python_exe = self.presets.get("python_launcher") or sys.executable or "python"
        self.last_browse_dir = self._load_last_dir()
        self.fixed_root: Path | None = Path(self.presets.get("default_project_root")) if self.presets.get("default_project_root") else None
        self.subject_folder_path: Path | None = None
        self._compare_orient_options = [("Axial", 2), ("Coronal", 1), ("Sagittal", 0)]
        self._compare_right_orient_offset = 0  # 0 = same as left, 1/2 cycle right view
        self._compare_data_left = None
        self._compare_data_right = None
        self._left_rot = 0
        self._right_rot = 0
        self._left_flip_h = False
        self._left_flip_v = False
        self._right_flip_h = False
        self._right_flip_v = False
        self._init_ui()

    def _init_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Step list
        self.steps_list = QtWidgets.QListWidget()
        self.steps_list.setFixedWidth(240)
        self.steps_list.itemSelectionChanged.connect(self._on_step_changed)
        layout.addWidget(self.steps_list)

        # Right panel
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel("Select a step")
        self.title_label.setFont(QtGui.QFont("", 14, QtGui.QFont.Bold))
        right_layout.addWidget(self.title_label)

        self.description = QtWidgets.QLabel(
            "This UI will guide you through MT pipeline steps. "
            "Fill parameters and click run for each step."
        )
        self.description.setWordWrap(True)
        right_layout.addWidget(self.description)

        self.form_stack = QtWidgets.QStackedWidget()
        right_layout.addWidget(self.form_stack, stretch=1)

        # Log/output panel
        self.run_log = RunLogWidget()
        right_layout.addWidget(self.run_log, stretch=1)

        layout.addWidget(right_panel, stretch=1)
        self.setCentralWidget(central)

        self.step_forms: dict[str, QtWidgets.QWidget] = {}
        self._populate_steps()
        self._build_step_forms()
        # Ensure the first step widget is visible
        if self.steps_list.currentItem():
            self._show_form(self.steps_list.currentItem().data(QtCore.Qt.UserRole))

    def _populate_steps(self) -> None:
        for step in self.state.steps:
            item = QtWidgets.QListWidgetItem(step.title)
            item.setData(QtCore.Qt.UserRole, step.key)
            self._update_step_item(item, step.status)
            self.steps_list.addItem(item)
        if self.steps_list.count() > 0:
            self.steps_list.setCurrentRow(0)

    def _update_step_item(self, item: QtWidgets.QListWidgetItem, status: StepStatus) -> None:
        color = {
            StepStatus.PENDING: QtGui.QColor("#666"),
            StepStatus.IN_PROGRESS: QtGui.QColor("#d68b00"),
            StepStatus.DONE: QtGui.QColor("#2b8a3e"),
        }[status]
        item.setForeground(QtGui.QBrush(color))

    def _on_step_changed(self) -> None:
        item = self.steps_list.currentItem()
        if not item:
            return
        key = item.data(QtCore.Qt.UserRole)
        step = self.state.get_step(key)
        if not step:
            return
        self.title_label.setText(step.title)
        self.description.setText(step.description)
        self._show_form(step.key)
        self.run_log.clear_log()

    def _show_form(self, step_key: str) -> None:
        widget = self.step_forms.get(step_key)
        if widget:
            self.form_stack.setCurrentWidget(widget)

    # Placeholder: call this when a step finishes to update status
    def mark_done(self, step_key: str) -> None:
        self.state.set_status(step_key, StepStatus.DONE)
        for i in range(self.steps_list.count()):
            item = self.steps_list.item(i)
            if item.data(QtCore.Qt.UserRole) == step_key:
                self._update_step_item(item, StepStatus.DONE)
                break

    # Path selection helper
    def pick_file(self, start_dir: Path | None = None, for_save: bool = False) -> Path | None:
        start = str(start_dir or self.last_browse_dir or Path.cwd())
        if for_save:
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select output", start)
        else:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", start)
        if file_path:
            p = Path(file_path)
            self._save_last_dir(p.parent)
            return p
        return None

    def pick_folder(self, start_dir: Path | None = None) -> Path | None:
        start = str(start_dir or self.last_browse_dir or Path.cwd())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
        if folder:
            p = Path(folder)
            self._save_last_dir(p)
            return p
        return None

    # ---- builders ----
    def _build_step_forms(self) -> None:
        builders = {
            "project": self._make_project_form,
            "split": self._make_split_form,
            "dcm2nii": self._make_dcm2nii_form,
            "recenter": self._make_recenter_form,
            "orient": self._make_orient_form,
            "gibbs": self._make_gibbs_form,
            "brainsuite": self._make_brainsuite_form,
            "coreg": self._make_coreg_form,
            "b1prep": self._make_b1prep_form,
            "mtsat": self._make_mtsat_form,
            "t1t2": self._make_t1t2_form,
            "compare": self._make_compare_form,
        }
        for step in self.state.steps:
            builder = builders.get(step.key)
            widget = builder() if builder else QtWidgets.QWidget()
            self.step_forms[step.key] = widget
            self.form_stack.addWidget(widget)

    def _make_project_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        default_root = self.presets.get("default_project_root", "C:/work/mouse_mt_pipeline/Data")
        self.project_root = QtWidgets.QLineEdit(default_root)
        if default_root:
            self.fixed_root = Path(default_root)
        btn_root = QtWidgets.QPushButton("Browse")
        btn_root.clicked.connect(self._pick_project_root)
        form.addRow("Project root", self._hbox(self.project_root, btn_root))

        self.project_subject_folder = QtWidgets.QLineEdit()
        btn_subject_folder = QtWidgets.QPushButton("Pick subject folder")
        btn_subject_folder.clicked.connect(self._pick_subject_folder)
        form.addRow("Subject folder", self._hbox(self.project_subject_folder, btn_subject_folder))

        self.project_subject = QtWidgets.QLineEdit()
        form.addRow("Subject / session", self.project_subject)

        save_btn = QtWidgets.QPushButton("Save context")
        save_btn.clicked.connect(self._save_project_context)
        form.addRow(save_btn)
        return widget

    def _make_split_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self.split_src = QtWidgets.QLineEdit()
        btn_src = QtWidgets.QPushButton("Pick DICOM")
        btn_src.clicked.connect(lambda: self._set_line_from_file(self.split_src))
        form.addRow("MT combined DICOM", self._hbox(self.split_src, btn_src))

        self.split_off = QtWidgets.QLineEdit()
        self.split_off.setPlaceholderText("Leave empty: auto-create MToff/mtoff.dcm under subject folder")
        btn_off = QtWidgets.QPushButton("MToff folder")
        btn_off.clicked.connect(self._pick_split_off_folder)
        form.addRow("MToff output", self._hbox(self.split_off, btn_off))

        self.split_on = QtWidgets.QLineEdit()
        self.split_on.setPlaceholderText("Leave empty: auto-create MTon/mton.dcm under subject folder")
        btn_on = QtWidgets.QPushButton("MTon folder")
        btn_on.clicked.connect(self._pick_split_on_folder)
        form.addRow("MTon output", self._hbox(self.split_on, btn_on))

        self.split_frames = QtWidgets.QSpinBox()
        self.split_frames.setRange(1, 5000)
        self.split_frames.setValue(60)
        form.addRow("Off first n frames", self.split_frames)

        run_btn = QtWidgets.QPushButton("Run split_mt_dicom")
        run_btn.clicked.connect(self._run_split)
        form.addRow(run_btn)
        return widget

    def _make_dcm2nii_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self.dcm_input = QtWidgets.QLineEdit()
        btn_in = QtWidgets.QPushButton("Pick DICOM folder")
        btn_in.clicked.connect(self._select_dcm_input)
        form.addRow("Input DICOM folder", self._hbox(self.dcm_input, btn_in))

        self.dcm_outdir = QtWidgets.QLineEdit()
        btn_outdir = QtWidgets.QPushButton("Pick output folder")
        btn_outdir.clicked.connect(lambda: self._set_line_from_folder(self.dcm_outdir))
        form.addRow("Output folder (-o)", self._hbox(self.dcm_outdir, btn_outdir))

        self.dcm_outname = QtWidgets.QLineEdit("MTon")
        form.addRow("Output name (-f)", self.dcm_outname)

        default_args = " ".join(self.presets.get("dcm2niix", {}).get("default_args", ["-z", "y"]))
        self.dcm_args = QtWidgets.QLineEdit(default_args)
        form.addRow("Extra args", self.dcm_args)

        run_btn = QtWidgets.QPushButton("Run dcm2niix")
        run_btn.clicked.connect(self._run_dcm2niix)
        form.addRow(run_btn)
        return widget

    def _make_recenter_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self.recenter_input = QtWidgets.QLineEdit()
        btn_in = QtWidgets.QPushButton("Pick NIfTI")
        btn_in.clicked.connect(lambda: self._set_line_from_file(self.recenter_input))
        form.addRow("Input NIfTI", self._hbox(self.recenter_input, btn_in))

        self.recenter_output = QtWidgets.QLineEdit()
        self.recenter_output.setPlaceholderText("Leave empty: auto-create *_recenter.nii.gz in the same folder")
        btn_out = QtWidgets.QPushButton("Save recentered")
        btn_out.clicked.connect(lambda: self._set_line_from_save(self.recenter_output))
        form.addRow("Output", self._hbox(self.recenter_output, btn_out))

        run_btn = QtWidgets.QPushButton("Run recenter (MATLAB)")
        run_btn.clicked.connect(self._run_recenter)
        form.addRow(run_btn)
        return widget

    def _make_orient_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self.orient_input = QtWidgets.QLineEdit()
        btn_in = QtWidgets.QPushButton("Pick NIfTI")
        btn_in.clicked.connect(lambda: self._set_line_from_file(self.orient_input))
        form.addRow("Input NIfTI", self._hbox(self.orient_input, btn_in))

        run_btn = QtWidgets.QPushButton("Run niftiOrientation (MATLAB)")
        run_btn.clicked.connect(self._run_orientation)
        form.addRow(run_btn)
        return widget

    def _make_gibbs_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        default_exec = self.presets.get("mrdegibbs", {}).get("exe", "deGibbs3D")
        self.gibbs_exec = QtWidgets.QLineEdit(default_exec)
        form.addRow("mrdegibbs exe", self.gibbs_exec)

        self.gibbs_input = QtWidgets.QLineEdit()
        btn_in = QtWidgets.QPushButton("Pick input NIfTI")
        btn_in.clicked.connect(lambda: self._set_line_from_file(self.gibbs_input))
        form.addRow("Input (.nii/.nii.gz)", self._hbox(self.gibbs_input, btn_in))

        self.gibbs_outdir = QtWidgets.QLineEdit()
        self.gibbs_outdir.setPlaceholderText("Leave empty: use the input directory")
        btn_out = QtWidgets.QPushButton("Pick output folder")
        btn_out.clicked.connect(lambda: self._set_line_from_folder(self.gibbs_outdir))
        form.addRow("Output folder", self._hbox(self.gibbs_outdir, btn_out))

        self.gibbs_suffix = QtWidgets.QLineEdit("_ungibbs")
        form.addRow("Filename suffix", self.gibbs_suffix)

        self.gibbs_axes = QtWidgets.QLineEdit("0,1,2")
        form.addRow("Axes (phase-encode dir first)", self.gibbs_axes)

        self.gibbs_nshifts = QtWidgets.QSpinBox()
        self.gibbs_nshifts.setRange(2, 200)
        self.gibbs_nshifts.setValue(20)
        form.addRow("nshifts", self.gibbs_nshifts)

        self.gibbs_minW = QtWidgets.QSpinBox()
        self.gibbs_minW.setRange(0, 20)
        self.gibbs_minW.setValue(1)
        form.addRow("minW", self.gibbs_minW)

        self.gibbs_maxW = QtWidgets.QSpinBox()
        self.gibbs_maxW.setRange(1, 30)
        self.gibbs_maxW.setValue(3)
        form.addRow("maxW", self.gibbs_maxW)

        self.gibbs_use_wsl = QtWidgets.QCheckBox("Use WSL bridge (Windows→WSL)")
        self.gibbs_use_wsl.setChecked(sys.platform.startswith("win"))
        form.addRow(self.gibbs_use_wsl)

        btn_guess = QtWidgets.QPushButton("Guess axes from method")
        btn_guess.clicked.connect(self._guess_gibbs_axes)
        form.addRow(btn_guess)

        note = QtWidgets.QLabel("MRtrix3 mrdegibbs assumes symmetric fully-sampled data; axes follow NIfTI storage order.")
        note.setWordWrap(True)
        form.addRow(note)

        run_btn = QtWidgets.QPushButton("Run mrdegibbs")
        run_btn.clicked.connect(self._run_gibbs)
        form.addRow(run_btn)
        return widget

    def _make_brainsuite_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.brainsuite_input = QtWidgets.QLineEdit()
        btn_in = QtWidgets.QPushButton("Pick volume")
        btn_in.clicked.connect(lambda: self._set_line_from_file(self.brainsuite_input))
        btn_launch = QtWidgets.QPushButton("Launch BrainSuite")
        btn_launch.clicked.connect(self._run_brainsuite)
        layout.addWidget(self._hbox(self.brainsuite_input, btn_in))
        layout.addWidget(btn_launch)
        layout.addWidget(QtWidgets.QLabel("Use BrainSuite GUI to skull-strip and save mask."))
        layout.addStretch(1)
        return widget

    def _make_coreg_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self.coreg_ref = QtWidgets.QLineEdit()
        btn_ref = QtWidgets.QPushButton("Pick ref")
        btn_ref.clicked.connect(lambda: self._set_line_from_file(self.coreg_ref))
        form.addRow("Reference image", self._hbox(self.coreg_ref, btn_ref))

        self.coreg_src = QtWidgets.QLineEdit()
        btn_src = QtWidgets.QPushButton("Pick source")
        btn_src.clicked.connect(lambda: self._set_line_from_file(self.coreg_src))
        form.addRow("Source image", self._hbox(self.coreg_src, btn_src))

        self.coreg_mask = QtWidgets.QLineEdit()
        btn_mask = QtWidgets.QPushButton("Pick mask")
        btn_mask.clicked.connect(lambda: self._set_line_from_file(self.coreg_mask))
        form.addRow("Weight/mask", self._hbox(self.coreg_mask, btn_mask))

        self.coreg_prefix = QtWidgets.QLineEdit()
        self.coreg_prefix.setPlaceholderText("Leave empty + check overwrite to replace source; otherwise enter a prefix")
        form.addRow("Prefix", self.coreg_prefix)

        self.coreg_overwrite = QtWidgets.QCheckBox("Overwrite source (no prefix)")
        form.addRow(self.coreg_overwrite)

        # Strategy menu (only controls interpolation)
        self.coreg_strategy = QtWidgets.QComboBox()
        self.coreg_strategy.addItem("Spline degree 3 (interp=3)", {"interp": 3})
        self.coreg_strategy.addItem("Trilinear fast (interp=1)", {"interp": 1})
        self.coreg_strategy.addItem("Nearest neighbor (labels, interp=0)", {"interp": 0})
        form.addRow("Interpolation", self.coreg_strategy)

        run_btn = QtWidgets.QPushButton("Run SPM coreg (MATLAB)")
        run_btn.clicked.connect(self._run_coreg)
        form.addRow(run_btn)
        return widget

    def _make_b1prep_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self.b1prep_b1 = QtWidgets.QLineEdit()
        btn_b1 = QtWidgets.QPushButton("Pick raw B1 (phase)")
        btn_b1.clicked.connect(lambda: self._set_line_from_file(self.b1prep_b1))
        form.addRow("Raw B1 NIfTI", self._hbox(self.b1prep_b1, btn_b1))

        self.b1prep_mton = QtWidgets.QLineEdit()
        btn_mton = QtWidgets.QPushButton("Pick MTon")
        btn_mton.clicked.connect(lambda: self._set_line_from_file(self.b1prep_mton))
        form.addRow("MTon (target grid)", self._hbox(self.b1prep_mton, btn_mton))

        self.b1prep_mask = QtWidgets.QLineEdit()
        btn_mask = QtWidgets.QPushButton("Pick mask (optional)")
        btn_mask.clicked.connect(lambda: self._set_line_from_file(self.b1prep_mask))
        form.addRow("Mask (opt)", self._hbox(self.b1prep_mask, btn_mask))

        self.b1prep_out = QtWidgets.QLineEdit()
        btn_out = QtWidgets.QPushButton("Save B1_RFlocal")
        btn_out.clicked.connect(lambda: self._set_line_from_save(self.b1prep_out))
        form.addRow("Output base (no ext)", self._hbox(self.b1prep_out, btn_out))

        run_btn = QtWidgets.QPushButton("Run B1 preprocess (MATLAB)")
        run_btn.clicked.connect(self._run_b1prep)
        form.addRow(run_btn)
        return widget

    def _make_mtsat_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self.mtsat_mton = QtWidgets.QLineEdit()
        btn_mton = QtWidgets.QPushButton("Pick MTon")
        btn_mton.clicked.connect(lambda: self._set_line_from_file(self.mtsat_mton))
        form.addRow("MTon", self._hbox(self.mtsat_mton, btn_mton))

        self.mtsat_mtoff = QtWidgets.QLineEdit()
        btn_mtoff = QtWidgets.QPushButton("Pick MToff/PDw")
        btn_mtoff.clicked.connect(lambda: self._set_line_from_file(self.mtsat_mtoff))
        form.addRow("MToff / PDw", self._hbox(self.mtsat_mtoff, btn_mtoff))

        self.mtsat_t1 = QtWidgets.QLineEdit()
        btn_t1 = QtWidgets.QPushButton("Pick T1 (optional)")
        btn_t1.clicked.connect(lambda: self._set_line_from_file(self.mtsat_t1))
        form.addRow("T1 (opt)", self._hbox(self.mtsat_t1, btn_t1))

        self.mtsat_mask = QtWidgets.QLineEdit()
        btn_mask = QtWidgets.QPushButton("Pick mask (opt)")
        btn_mask.clicked.connect(lambda: self._set_line_from_file(self.mtsat_mask))
        form.addRow("Mask (opt)", self._hbox(self.mtsat_mask, btn_mask))

        self.mtsat_gauss = QtWidgets.QCheckBox("Gaussian filter")
        self.mtsat_gauss.setChecked(True)
        form.addRow(self.mtsat_gauss)

        self.mtsat_b1 = QtWidgets.QLineEdit()
        btn_b1 = QtWidgets.QPushButton("Pick B1_RFlocal (opt)")
        btn_b1.clicked.connect(lambda: self._set_line_from_file(self.mtsat_b1))
        form.addRow("B1_RFlocal (opt)", self._hbox(self.mtsat_b1, btn_b1))

        run_btn = QtWidgets.QPushButton("Run nii2mtsat (MATLAB)")
        run_btn.clicked.connect(self._run_mtsat)
        form.addRow(run_btn)
        return widget

    def _make_t1t2_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self.t1t2_t1 = QtWidgets.QLineEdit()
        btn_t1 = QtWidgets.QPushButton("Pick T1")
        btn_t1.clicked.connect(lambda: self._set_line_from_file(self.t1t2_t1))
        form.addRow("T1 input", self._hbox(self.t1t2_t1, btn_t1))

        self.t1t2_t2 = QtWidgets.QLineEdit()
        btn_t2 = QtWidgets.QPushButton("Pick T2/RAREvfl")
        btn_t2.clicked.connect(lambda: self._set_line_from_file(self.t1t2_t2))
        form.addRow("T2 / RAREvfl", self._hbox(self.t1t2_t2, btn_t2))

        self.t1t2_mask = QtWidgets.QLineEdit()
        btn_mask = QtWidgets.QPushButton("Pick mask")
        btn_mask.clicked.connect(lambda: self._set_line_from_file(self.t1t2_mask))
        form.addRow("Mask", self._hbox(self.t1t2_mask, btn_mask))

        self.t1t2_gauss = QtWidgets.QCheckBox("Gaussian filter")
        self.t1t2_gauss.setChecked(True)
        form.addRow(self.t1t2_gauss)

        self.t1t2_match = QtWidgets.QCheckBox("Intensity match")
        self.t1t2_match.setChecked(True)
        form.addRow(self.t1t2_match)

        run_btn = QtWidgets.QPushButton("Run nii2t1wt2wr (MATLAB)")
        run_btn.clicked.connect(self._run_t1t2)
        form.addRow(run_btn)
        return widget

    def _make_compare_form(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # file pickers
        pick_layout = QtWidgets.QHBoxLayout()
        self.compare_left = QtWidgets.QLineEdit()
        btn_left = QtWidgets.QPushButton("Pick left")
        btn_left.clicked.connect(lambda: self._set_line_from_file(self.compare_left))
        pick_layout.addWidget(self.compare_left)
        pick_layout.addWidget(btn_left)
        layout.addLayout(pick_layout)

        pick_layout_r = QtWidgets.QHBoxLayout()
        self.compare_right = QtWidgets.QLineEdit()
        btn_right = QtWidgets.QPushButton("Pick right")
        btn_right.clicked.connect(lambda: self._set_line_from_file(self.compare_right))
        pick_layout_r.addWidget(self.compare_right)
        pick_layout_r.addWidget(btn_right)
        layout.addLayout(pick_layout_r)

        # orientation and slider
        orient_layout = QtWidgets.QHBoxLayout()
        orient_layout.addWidget(QtWidgets.QLabel("View"))
        self.compare_orient = QtWidgets.QComboBox()
        for text, axis in self._compare_orient_options:
            self.compare_orient.addItem(text, axis)
        self.compare_right_orient_label = QtWidgets.QLabel()
        btn_cycle_right_view = QtWidgets.QPushButton("Cycle right view")
        btn_cycle_right_view.setToolTip("Cycle the right view; left view stays as selected.")
        btn_cycle_right_view.clicked.connect(self._cycle_right_orient)
        self.compare_orient.currentIndexChanged.connect(self._update_compare_slider_range)
        orient_layout.addWidget(self.compare_orient)
        orient_layout.addStretch(1)
        orient_layout.addWidget(self.compare_right_orient_label)
        orient_layout.addWidget(btn_cycle_right_view)
        layout.addLayout(orient_layout)

        self.compare_info = QtWidgets.QLabel("Slice: -")
        layout.addWidget(self.compare_info)

        # Independent sliders
        self.compare_slider_left = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.compare_slider_left.setMinimum(0)
        self.compare_slider_left.valueChanged.connect(self._update_compare_view)
        self.compare_slider_right = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.compare_slider_right.setMinimum(0)
        self.compare_slider_right.valueChanged.connect(self._update_compare_view)
        slider_box = QtWidgets.QVBoxLayout()
        slider_box.addWidget(QtWidgets.QLabel("Left slice"))
        slider_box.addWidget(self.compare_slider_left)
        slider_box.addWidget(QtWidgets.QLabel("Right slice"))
        slider_box.addWidget(self.compare_slider_right)
        layout.addLayout(slider_box)

        # viewers
        viewer_layout = QtWidgets.QHBoxLayout()
        self.viewer_left = NiiSliceViewer()
        self.viewer_right = NiiSliceViewer()
        viewer_layout.addWidget(self.viewer_left, 1)
        viewer_layout.addWidget(self.viewer_right, 1)
        self.viewer_right.hide()
        self.diff_scale = DiffScaleBar()
        self.diff_scale.hide()
        viewer_layout.addWidget(self.diff_scale, 0)
        layout.addLayout(viewer_layout)

        # Intensity controls
        inten_layout = QtWidgets.QHBoxLayout()
        inten_layout.addWidget(QtWidgets.QLabel("Left min/max"))
        self.compare_min_left = QtWidgets.QDoubleSpinBox()
        self.compare_min_left.setDecimals(6)
        self.compare_min_left.setRange(-1e6, 1e6)
        self.compare_min_left.setValue(0.0)
        self.compare_min_left.setSingleStep(0.0001)
        self.compare_max_left = QtWidgets.QDoubleSpinBox()
        self.compare_max_left.setDecimals(6)
        self.compare_max_left.setRange(-1e6, 1e6)
        self.compare_max_left.setValue(1.0)
        self.compare_max_left.setSingleStep(0.0001)
        inten_layout.addWidget(self.compare_min_left)
        inten_layout.addWidget(self.compare_max_left)

        inten_layout.addWidget(QtWidgets.QLabel("Right min/max"))
        self.compare_min_right = QtWidgets.QDoubleSpinBox()
        self.compare_min_right.setDecimals(6)
        self.compare_min_right.setRange(-1e6, 1e6)
        self.compare_min_right.setValue(0.0)
        self.compare_min_right.setSingleStep(0.0001)
        self.compare_max_right = QtWidgets.QDoubleSpinBox()
        self.compare_max_right.setDecimals(6)
        self.compare_max_right.setRange(-1e6, 1e6)
        self.compare_max_right.setValue(1.0)
        self.compare_max_right.setSingleStep(0.0001)
        inten_layout.addWidget(self.compare_min_right)
        inten_layout.addWidget(self.compare_max_right)
        layout.addLayout(inten_layout)

        # Overlay option
        self.compare_overlay = QtWidgets.QCheckBox("Overlay diff (Comparison vs Reference)")
        layout.addWidget(self.compare_overlay)

        scheme_layout = QtWidgets.QHBoxLayout()
        scheme_layout.addWidget(QtWidgets.QLabel("Color scheme"))
        self.compare_scheme = QtWidgets.QComboBox()
        self.compare_scheme.addItem("Red / Blue", ((255, 0, 0), (0, 0, 255)))
        self.compare_scheme.addItem("Purple / Green", ((128, 0, 128), (0, 128, 0)))
        self.compare_scheme.addItem("Orange / Cyan", ((255, 140, 0), (0, 160, 200)))
        scheme_layout.addWidget(self.compare_scheme)
        scheme_layout.addStretch(1)
        layout.addLayout(scheme_layout)

        # Transform controls
        transforms_layout = QtWidgets.QHBoxLayout()
        left_controls = QtWidgets.QVBoxLayout()
        left_controls.addWidget(QtWidgets.QLabel("Left transforms"))
        btn_l_rot = QtWidgets.QPushButton("Rotate 90°")
        btn_l_rot.clicked.connect(lambda: self._rotate("left"))
        btn_l_flip_h = QtWidgets.QPushButton("Flip H")
        btn_l_flip_h.clicked.connect(lambda: self._flip("left", horizontal=True))
        btn_l_flip_v = QtWidgets.QPushButton("Flip V")
        btn_l_flip_v.clicked.connect(lambda: self._flip("left", horizontal=False))
        left_controls.addWidget(btn_l_rot)
        left_controls.addWidget(btn_l_flip_h)
        left_controls.addWidget(btn_l_flip_v)

        right_controls = QtWidgets.QVBoxLayout()
        right_controls.addWidget(QtWidgets.QLabel("Right transforms"))
        btn_r_rot = QtWidgets.QPushButton("Rotate 90°")
        btn_r_rot.clicked.connect(lambda: self._rotate("right"))
        btn_r_flip_h = QtWidgets.QPushButton("Flip H")
        btn_r_flip_h.clicked.connect(lambda: self._flip("right", horizontal=True))
        btn_r_flip_v = QtWidgets.QPushButton("Flip V")
        btn_r_flip_v.clicked.connect(lambda: self._flip("right", horizontal=False))
        right_controls.addWidget(btn_r_rot)
        right_controls.addWidget(btn_r_flip_h)
        right_controls.addWidget(btn_r_flip_v)

        transforms_layout.addLayout(left_controls)
        transforms_layout.addLayout(right_controls)
        layout.addLayout(transforms_layout)

        btn_load = QtWidgets.QPushButton("Load and compare")
        btn_load.clicked.connect(self._load_and_compare)
        layout.addWidget(btn_load)

        btn_export = QtWidgets.QPushButton("Export video (MP4)")
        btn_export.clicked.connect(self._export_compare_video)
        layout.addWidget(btn_export)

        # Step overrides (physical step per slice, unit from header; 0=auto)
        step_layout = QtWidgets.QHBoxLayout()
        step_layout.addWidget(QtWidgets.QLabel("Left step override (units, 0=auto)"))
        self.compare_step_left = QtWidgets.QDoubleSpinBox()
        self.compare_step_left.setDecimals(4)
        self.compare_step_left.setRange(0, 1e6)
        self.compare_step_left.setValue(0.0)
        step_layout.addWidget(self.compare_step_left)
        step_layout.addWidget(QtWidgets.QLabel("Right step override"))
        self.compare_step_right = QtWidgets.QDoubleSpinBox()
        self.compare_step_right.setDecimals(4)
        self.compare_step_right.setRange(0, 1e6)
        self.compare_step_right.setValue(0.0)
        step_layout.addWidget(self.compare_step_right)
        layout.addLayout(step_layout)

        # Immediate refresh when intensity/scheme/overlay changes
        self.compare_min_left.valueChanged.connect(self._update_compare_view)
        self.compare_max_left.valueChanged.connect(self._update_compare_view)
        self.compare_min_right.valueChanged.connect(self._update_compare_view)
        self.compare_max_right.valueChanged.connect(self._update_compare_view)
        self.compare_overlay.stateChanged.connect(self._update_compare_view)
        self.compare_scheme.currentIndexChanged.connect(self._update_compare_view)

        self._refresh_right_orient_label()
        layout.addStretch(1)
        return widget

    # ---- actions ----
    def _run_split(self) -> None:
        if not self.split_src.text():
            self._warn("Please set input DICOM.")
            return
        src_path = Path(self.split_src.text())
        base_dir = src_path.parent
        subject_base = self._guess_subject_base(src_path)
        def ensure_file(path_str: str, fname: str) -> str:
            p = Path(path_str)
            if p.suffix.lower() != ".dcm":
                p = p / fname
            return str(p)
        if not self.split_off.text():
            auto_off = (subject_base or base_dir) / "MToff" / "mtoff.dcm"
            auto_off.parent.mkdir(parents=True, exist_ok=True)
            self.split_off.setText(str(auto_off))
        else:
            self.split_off.setText(ensure_file(self.split_off.text(), "mtoff.dcm"))
        if not self.split_on.text():
            auto_on = (subject_base or base_dir) / "MTon" / "mton.dcm"
            auto_on.parent.mkdir(parents=True, exist_ok=True)
            self.split_on.setText(str(auto_on))
        else:
            self.split_on.setText(ensure_file(self.split_on.text(), "mton.dcm"))
        cmd = [
            self.python_exe,
            str(self.repo_root / "mt_pipeline" / "split_mt_dicom.py"),
            self.split_src.text(),
            self.split_off.text(),
            self.split_on.text(),
            "--off-first-nframes",
            str(self.split_frames.value()),
        ]
        self._execute(cmd, "split")

    def _run_dcm2niix(self) -> None:
        exe = self.presets.get("dcm2niix", {}).get("exe", "dcm2niix")
        if not self.dcm_input.text():
            self._warn("Please select DICOM folder or modality folder.")
            return
        # If user gave a modality folder, try to locate dicom under pdata/1/dicom
        in_path = Path(self.dcm_input.text())
        dicom_path = self._resolve_dicom_folder(in_path)
        if not dicom_path.exists():
            self._warn(f"Cannot locate DICOM folder under {in_path}")
            return

        # Auto-fill output dir/name if missing
        if not self.dcm_outdir.text():
            suggested = self._suggest_dcm_outdir(in_path)
            if suggested:
                self.dcm_outdir.setText(str(suggested))
                self.dcm_outdir.setPlaceholderText(str(suggested))
        if not self.dcm_outname.text():
            mod_name = (self.dcm_outdir.text() and Path(self.dcm_outdir.text()).name) or dicom_path.name
            self.dcm_outname.setText(mod_name)
            self.dcm_outname.setPlaceholderText(mod_name)
        if not self.dcm_outdir.text() or not self.dcm_outname.text():
            self._warn("Please fill output dir and name.")
            return

        extra_args = shlex.split(self.dcm_args.text())
        cmd = [exe, *extra_args, "-o", self.dcm_outdir.text(), "-f", self.dcm_outname.text(), str(dicom_path)]
        self._execute(cmd, "dcm2nii")

    def _run_recenter(self) -> None:
        if not self.recenter_input.text():
            self._warn("Select input NIfTI.")
            return
        out = self.recenter_output.text()
        if not out:
            p = Path(self.recenter_input.text())
            # always write .nii.gz in same folder
            out = str(p.with_name(p.stem + "_recenter").with_suffix(".nii.gz"))
            self.recenter_output.setText(out)
        code = self._matlab_batch(
            f"recenterNifti('{self._matlab_str(self.recenter_input.text())}', "
            f"'{self._matlab_str(out)}');"
        )
        cmd = ["matlab", "-batch", code]
        self._execute(cmd, "recenter", done_message=f"Recenter done: {out}")

    def _run_orientation(self) -> None:
        if not self.orient_input.text():
            self._warn("Select input NIfTI.")
            return
        p = Path(self.orient_input.text())
        path_no_ext = str(p.with_suffix("").with_suffix(""))
        code = self._matlab_batch(f"niftiOrientation('{self._matlab_str(path_no_ext)}');")
        cmd = ["matlab", "-batch", code]
        self._execute(cmd, "orient")

    def _run_gibbs(self) -> None:
        exe = self.gibbs_exec.text().strip() or "mrdegibbs"
        if not exe:
            self._warn("Provide mrdegibbs executable path or name.")
            return
        if not self.gibbs_input.text():
            self._warn("Select input NIfTI.")
            return
        in_path = Path(self.gibbs_input.text())
        if not in_path.exists():
            self._warn("Input file does not exist.")
            return
        out_dir = Path(self.gibbs_outdir.text()) if self.gibbs_outdir.text() else in_path.parent
        suffix = self.gibbs_suffix.text().strip() or "_ungibbs"
        base = in_path.name
        if base.endswith(".nii.gz"):
            base = base[:-7]
        elif base.endswith(".nii"):
            base = base[:-4]
        out_path = out_dir / f"{base}{suffix}.nii.gz"
        self.gibbs_outdir.setText(str(out_dir))
        # Prevent accidental overwrite
        if out_path.resolve() == in_path.resolve():
            self._warn("Output path matches input; change suffix or output folder.")
            return
        axes = self.gibbs_axes.text().strip()
        if not axes:
            self._warn("Enter axes (e.g., 0,1 or 1,2).")
            return
        cmd: list[str]
        if self.gibbs_use_wsl.isChecked() and sys.platform.startswith("win"):
            exe_wsl = self._to_wsl_path(Path(exe)) if exe.startswith(("/", "\\", "C:", "D:", "E:", "F:")) else exe
            parts = [
                shlex.quote(exe_wsl),
                "-axes",
                axes,
                "-nshifts",
                str(self.gibbs_nshifts.value()),
                "-minW",
                str(self.gibbs_minW.value()),
                "-maxW",
                str(self.gibbs_maxW.value()),
                shlex.quote(self._to_wsl_path(in_path)),
                shlex.quote(self._to_wsl_path(Path(out_path))),
            ]
            # Don't rely on ~/.bashrc (non-interactive shells may early-return).
            wsl_paths = self.presets.get("mrdegibbs", {}).get(
                "wsl_path_prefix", "/mnt/c/work/mrtrix3/bin:/mnt/c/work/mrdegibbs3D/bin"
            )
            wsl_cmd = f'PATH="{wsl_paths}:$PATH"; ' + " ".join(parts)
            cmd = ["wsl", "bash", "-lc", wsl_cmd]
        else:
            cmd = [
                exe,
                "-axes",
                axes,
                "-nshifts",
                str(self.gibbs_nshifts.value()),
                "-minW",
                str(self.gibbs_minW.value()),
                "-maxW",
                str(self.gibbs_maxW.value()),
                str(in_path),
                out_path,
            ]
        self._execute(cmd, "gibbs")

    def _guess_gibbs_axes(self) -> None:
        if not self.gibbs_input.text():
            self._warn("Select input NIfTI.")
            return
        nifti_path = Path(self.gibbs_input.text())
        if not nifti_path.exists():
            self._warn("Input file does not exist.")
            return
        shape = nib.load(str(nifti_path)).shape[:3]
        method_path = None
        for cand in (nifti_path.parent / "method", nifti_path.parent.parent / "method"):
            if cand.exists():
                method_path = cand
                break
        if not method_path:
            self._warn("method file not found (same or parent folder).")
            return
        method_dims = self._parse_bruker_matrix(method_path)
        if not method_dims:
            self._warn("No PVM_Matrix=3 entry parsed in method.")
            return
        # map method dims (read, phase1, phase2) to nifti axes by matching sizes
        nid_for_method = [-1] * len(method_dims)
        used = set()
        for n_axis, n_size in enumerate(shape):
            for m_axis, m_size in enumerate(method_dims):
                if m_size == n_size and m_axis not in used:
                    nid_for_method[m_axis] = n_axis
                    used.add(m_axis)
                    break
        if any(idx == -1 for idx in nid_for_method[:2]):
            self._warn(f"Shape matching failed: NIfTI {shape} vs method {method_dims}")
            return
        read_axis = nid_for_method[0]
        phase1_axis = nid_for_method[1]
        axes = ",".join(str(x) for x in sorted({read_axis, phase1_axis}))
        self.gibbs_axes.setText(axes)
        msg = (
            f"Guessed axes={axes} (read→axis{read_axis}, phase→axis{phase1_axis}, "
            f"matched from PVM_Matrix {method_dims} and NIfTI shape {shape})."
        )
        self.run_log.append(msg)

    def _run_brainsuite(self) -> None:
        exe = self.presets.get("brainsuite", {}).get("exe", "BrainSuite.exe")
        volume = self.brainsuite_input.text()
        cmd = [exe] + ([volume] if volume else [])
        self._execute(cmd, "brainsuite")

    def _run_coreg(self) -> None:
        if not (self.coreg_ref.text() and self.coreg_src.text()):
            self._warn("Set ref and source.")
            return
        strategy = self.coreg_strategy.currentData()
        interp = strategy.get("interp", 4)
        prefix = self.coreg_prefix.text()
        overwrite = self.coreg_overwrite.isChecked()
        if not prefix and not overwrite:
            self._warn("When prefix is empty, check overwrite or provide a prefix to create new files.")
            return
        code = self._matlab_batch(
            "coreg_est_write_weighted("
            f"'{self._matlab_str(self.coreg_src.text())}', "
            f"'{self._matlab_str(self.coreg_ref.text())}', "
            f"'{self._matlab_str(self.coreg_mask.text())}', "
            f"'{self._matlab_str(prefix)}', "
            f"{interp});"
        )
        cmd = ["matlab", "-batch", code]
        self._execute(cmd, "coreg")

    def _run_mtsat(self) -> None:
        if not (self.mtsat_mton.text() and self.mtsat_mtoff.text()):
            self._warn("Set MTon and MToff/PDw.")
            return
        mton = self._strip_nii(self.mtsat_mton.text())
        mtoff = self._strip_nii(self.mtsat_mtoff.text())
        t1 = self._strip_nii(self.mtsat_t1.text()) if self.mtsat_t1.text() else ""
        mask = self._strip_nii(self.mtsat_mask.text()) if self.mtsat_mask.text() else ""
        b1 = self._strip_nii(self.mtsat_b1.text()) if self.mtsat_b1.text() else ""
        gauss = 1 if self.mtsat_gauss.isChecked() else 0
        code = self._matlab_batch(
            "nii2mtsat("
            f"'{self._matlab_str(mton)}', "
            f"'{self._matlab_str(mtoff)}', "
            f"'{self._matlab_str(t1)}', "
            f"'{self._matlab_str(mask)}', "
            f"{gauss}, "
            f"'{self._matlab_str(b1)}');"
        )
        cmd = ["matlab", "-batch", code]
        self._execute(cmd, "mtsat")

    def _run_t1t2(self) -> None:
        if not (self.t1t2_t1.text() and self.t1t2_t2.text() and self.t1t2_mask.text()):
            self._warn("Set T1, T2/RAREvfl, and mask.")
            return
        gauss = 1 if self.t1t2_gauss.isChecked() else 0
        match = 1 if self.t1t2_match.isChecked() else 0
        code = self._matlab_batch(
            "nii2t1wt2wr("
            f"'{self._matlab_str(self.t1t2_t1.text())}', "
            f"'{self._matlab_str(self.t1t2_t2.text())}', "
            "'' , "
            f"'{self._matlab_str(self.t1t2_mask.text())}', "
            f"{gauss}, "
            f"{match});"
        )
        cmd = ["matlab", "-batch", code]
        self._execute(cmd, "t1t2")
        self.mark_done("t1t2")

    def _load_and_compare(self) -> None:
        if not (self.compare_left.text() or self.compare_right.text()):
            self._warn("Select at least one NIfTI file.")
            return
        try:
            self._compare_data_left = nib.load(self.compare_left.text()).get_fdata() if self.compare_left.text() else None
            self._compare_data_right = nib.load(self.compare_right.text()).get_fdata() if self.compare_right.text() else None
        except Exception as exc:
            self._warn(f"Failed to load NIfTI: {exc}")
            return
        def _auto_range(arr: np.ndarray) -> tuple[float, float]:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return 0.0, 1.0
            nz = finite[finite != 0]
            if nz.size > 0:
                p1, p99 = np.percentile(nz, [1, 99])
                lo = 0.0 if finite.min() >= 0 else float(p1)
                hi = float(p99)
            else:
                lo, hi = np.percentile(finite, [1, 99]).tolist()
                if lo >= 0:
                    lo = 0.0
            if hi - lo < 1e-6:
                hi = lo + 1e-3
            return float(lo), float(hi)

        if self._compare_data_left is not None:
            lmin, lmax = _auto_range(self._compare_data_left)
            self.compare_min_left.setValue(lmin)
            self.compare_max_left.setValue(lmax)
        if self._compare_data_right is not None:
            rmin, rmax = _auto_range(self._compare_data_right)
            self.compare_min_right.setValue(rmin)
            self.compare_max_right.setValue(rmax)
        self._update_compare_slider_range()
        self._update_compare_view()
        self.mark_done("compare")

    def _export_compare_video(self) -> None:
        if not (hasattr(self, "_compare_data_left") or hasattr(self, "_compare_data_right")):
            self._warn("Load at least one volume first.")
            return
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save video", str(self.repo_root / "compare.mp4"), "MP4 files (*.mp4)")
        if not out_path:
            return
        axis_left = self.compare_orient.currentData()
        axis_right = self._right_orient_axis()
        if not hasattr(self, "_compare_data_left") and not hasattr(self, "_compare_data_right"):
            self._warn("No data.")
            return

        # compute aligned coordinate grid
        step_left = self._slice_step(self._compare_data_left, axis_left, self.compare_left.text())
        step_right = self._slice_step(self._compare_data_right, axis_right, self.compare_right.text())
        if self.compare_step_left.value() > 0:
            step_left = self.compare_step_left.value()
        if self.compare_step_right.value() > 0:
            step_right = self.compare_step_right.value()

        idx_left_anchor = self.compare_slider_left.value()
        idx_right_anchor = self.compare_slider_right.value()

        coords_left = None
        coords_right = None
        if self._compare_data_left is not None:
            coords_left = (np.arange(self._compare_data_left.shape[axis_left]) - idx_left_anchor) * step_left
        if self._compare_data_right is not None:
            coords_right = (np.arange(self._compare_data_right.shape[axis_right]) - idx_right_anchor) * step_right

        # align anchor coordinate at 0 with optional shift if both exist
        delta = 0.0
        if coords_left is not None and coords_right is not None:
            delta = coords_right[idx_right_anchor] - coords_left[idx_left_anchor]
            coords_right = coords_right - delta

        # coordinate range
        min_coord = 0.0
        max_coord = 0.0
        if coords_left is not None:
            min_coord = min(min_coord, coords_left.min())
            max_coord = max(max_coord, coords_left.max())
        if coords_right is not None:
            min_coord = min(min_coord, coords_right.min())
            max_coord = max(max_coord, coords_right.max())

        if min_coord == max_coord:
            self._warn("Slice range is invalid.")
            return

        nominal_step = min([s for s in [step_left, step_right] if s is not None])
        coord_grid = np.arange(min_coord, max_coord + nominal_step * 0.5, nominal_step)

        # Collect frames
        frames = []
        for c in coord_grid:
            left_slice = None
            right_slice = None
            if coords_left is not None:
                idx = int(round(c / step_left)) + idx_left_anchor
                if 0 <= idx < self._compare_data_left.shape[axis_left]:
                    left_slice = self._apply_transforms(self._extract_slice(self._compare_data_left, axis_left, idx), "left")
            if coords_right is not None:
                idx = int(round((c + delta) / step_right)) + idx_right_anchor
                if 0 <= idx < self._compare_data_right.shape[axis_right]:
                    right_slice = self._apply_transforms(self._extract_slice(self._compare_data_right, axis_right, idx), "right")
            left_img, right_img = self._render_compare_frame_from_slices(left_slice, right_slice)
            if left_img is None and right_img is None:
                continue
            if left_img is not None and right_img is not None:
                frame = self._concat_with_padding(left_img, right_img)
            elif left_img is not None:
                frame = left_img
            else:
                frame = right_img
            frames.append(frame)

        # Pad all frames to a common size (even dims, width multiple of 16)
        if frames:
            max_h = max(f.shape[0] for f in frames)
            max_w = max(f.shape[1] for f in frames)
            max_h = max_h + (max_h % 2)
            max_w = max_w + ((16 - (max_w % 16)) % 16)
            norm_frames = []
            for f in frames:
                pad_h = max_h - f.shape[0]
                pad_w = max_w - f.shape[1]
                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left
                norm_frames.append(np.pad(f, ((top, bottom), (left, right), (0, 0)), mode="constant"))
            frames = norm_frames

        if not frames:
            self._warn("No frames to write.")
            return

        fps = max(0.1, min(30.0, len(frames) / 300.0))  # ~5 min playback, capped 0.1-30 fps
        try:
            try:
                import imageio_ffmpeg  # type: ignore
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = None

            writer_kwargs = dict(mode="I", fps=fps, format="FFMPEG", macro_block_size=1)
            if ffmpeg_exe:
                os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe
            else:
                # if using Windows native ffmpeg.exe, try C:\\tools\\ffmpeg.exe
                win_ffmpeg = Path("C:/tools/ffmpeg.exe")
                if win_ffmpeg.exists():
                    os.environ["IMAGEIO_FFMPEG_EXE"] = str(win_ffmpeg)
                else:
                    try:
                        imageio.plugins.ffmpeg.get_exe()
                    except Exception:
                        raise RuntimeError("FFMPEG plugin missing; try: pip install imageio-ffmpeg or place ffmpeg.exe at C:\\tools")

            if not out_path.lower().endswith(".mp4"):
                out_path = out_path + ".mp4"
            with imageio.get_writer(out_path, **writer_kwargs) as writer:
                for frame in frames:
                    writer.append_data(frame)
            self.run_log.append(f"Video saved: {out_path} (fps={fps:.2f}, frames={len(frames)})")
        except Exception as exc:
            self._warn(f"Failed to export video: {exc}")

    # ---- helpers ----
    def _execute(self, cmd: list[str], step_key: str, done_message: str | None = None) -> None:
        self.state.set_status(step_key, StepStatus.IN_PROGRESS)
        for i in range(self.steps_list.count()):
            item = self.steps_list.item(i)
            if item.data(QtCore.Qt.UserRole) == step_key:
                self._update_step_item(item, StepStatus.IN_PROGRESS)
                break
        self.run_log.append(f"$ {' '.join(cmd)}")
        result = run_command(cmd, cwd=self.repo_root)
        if result.stdout:
            self.run_log.append(result.stdout.strip())
        if result.stderr:
            self.run_log.append(result.stderr.strip())
        if result.ok:
            self.mark_done(step_key)
            if done_message:
                self.run_log.append(done_message)
        else:
            self.run_log.append(f"Command failed with code {result.code}")

    def _warn(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Missing fields", message)

    def _hbox(self, left: QtWidgets.QWidget, right: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(left, 1)
        layout.addWidget(right)
        return box

    def _set_line_from_file(self, line: QtWidgets.QLineEdit) -> None:
        p = self.pick_file()
        if p:
            line.setText(str(p))

    def _set_line_from_save(self, line: QtWidgets.QLineEdit) -> None:
        p = self.pick_file(for_save=True)
        if p:
            line.setText(str(p))

    def _set_line_from_folder(self, line: QtWidgets.QLineEdit) -> None:
        p = self.pick_folder()
        if p:
            line.setText(str(p))

    def _save_project_context(self) -> None:
        data = {
            "project_root": self.project_root.text(),
            "subject": self.project_subject.text(),
            "subject_folder": self.project_subject_folder.text(),
        }
        dest = self.repo_root / "ui_app" / "project_context.json"
        dest.write_text(json.dumps(data, indent=2))
        self.run_log.append(f"Saved context to {dest}")
        self.mark_done("project")

    def _matlab_batch(self, code: str) -> str:
        prefix = self.presets.get("matlab", {}).get(
            "path_setup", "cd('mt_pipeline/matlab'); addMatlabPath;"
        )
        return f"{prefix} {code}"

    @staticmethod
    def _matlab_str(path_str: str) -> str:
        return path_str.replace("'", "''")

    @staticmethod
    def _to_wsl_path(path: Path) -> str:
        """
        Convert Windows path to WSL (/mnt/...) form if needed.
        """
        p = Path(path)
        try:
            p = p.resolve()
        except Exception:
            pass
        # If already a POSIX path starting with /mnt, return as-is
        s = str(p)
        if s.startswith("/"):
            return s
        if len(s) > 1 and s[1] == ":":
            drive = s[0].lower()
            rest = s[2:].replace("\\", "/")
            return f"/mnt/{drive}/{rest.lstrip('/')}"
        return s.replace("\\", "/")

    @staticmethod
    def _parse_bruker_matrix(method_path: Path) -> list[int] | None:
        """
        Parse PVM_Matrix=(3) values from a Bruker method file.
        Returns [read, phase1, phase2] or None if not found.
        """
        try:
            lines = method_path.read_text(errors="ignore").splitlines()
        except Exception:
            return None
        vals: list[int] = []
        collecting = False
        for line in lines:
            if line.startswith("##$PVM_Matrix"):
                collecting = True
                continue
            if collecting:
                if line.startswith("##"):
                    break
                for token in line.strip().split():
                    try:
                        vals.append(int(token))
                    except ValueError:
                        continue
                if len(vals) >= 3:
                    return vals[:3]
        return None

    def _load_presets(self) -> dict:
        cfg_path = self.repo_root / "ui_app" / "configs" / "presets.json"
        if cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text())
            except Exception:
                return {}
        return {}

    def _load_last_dir(self) -> Path | None:
        ctx = self.repo_root / "ui_app" / "browse_context.json"
        if ctx.exists():
            try:
                data = json.loads(ctx.read_text())
                if "last_dir" in data:
                    return Path(data["last_dir"])
            except Exception:
                return None
        return None

    def _save_last_dir(self, path: Path) -> None:
        self.last_browse_dir = path
        ctx = self.repo_root / "ui_app" / "browse_context.json"
        try:
            ctx.write_text(json.dumps({"last_dir": str(path)}, indent=2))
        except Exception:
            pass

    def _pick_project_root(self) -> None:
        folder = self.pick_folder(start_dir=self.project_root.text() or self.last_browse_dir or Path.cwd())
        if folder:
            self.project_root.setText(str(folder))
            self.fixed_root = folder
            self._save_last_dir(folder)
            self.run_log.append(f"Project root set to {folder}")

    def _pick_subject_folder(self) -> None:
        prefer = self.fixed_root or (Path(self.project_root.text()) if self.project_root.text() else None)
        folder = self.pick_folder(start_dir=prefer)
        if folder:
            self.subject_folder_path = folder
            self.project_subject_folder.setText(str(folder))
            self.project_subject.setText(folder.name)
            # optional: set project root to parent of subject
            if not self.project_root.text():
                self.project_root.setText(str(folder.parent))
                self.fixed_root = folder.parent
            self.run_log.append(f"Selected subject folder: {folder}")
            self._save_last_dir(folder)
            self._apply_subject_suggestions()

    def _pick_split_off_folder(self) -> None:
        base = self.subject_folder_path or self.fixed_root or Path.cwd()
        folder = self.pick_folder(start_dir=base)
        if folder:
            path = folder / "mtoff.dcm"
            self.split_off.setText(str(path))

    def _pick_split_on_folder(self) -> None:
        base = self.subject_folder_path or self.fixed_root or Path.cwd()
        folder = self.pick_folder(start_dir=base)
        if folder:
            path = folder / "mton.dcm"
            self.split_on.setText(str(path))

    def _run_b1prep(self) -> None:
        if not self.b1prep_b1.text() or not self.b1prep_mton.text():
            self._warn("Set raw B1 and MTon.")
            return
        out_base = self.b1prep_out.text()
        if not out_base:
            p = Path(self.b1prep_b1.text())
            out_base = str(p.with_name("B1_RFlocal"))
            self.b1prep_out.setText(out_base)
        mask = self.b1prep_mask.text() or ""
        code = self._matlab_batch(
            "prepareB1RFlocal("
            f"'{self._matlab_str(self.b1prep_b1.text())}', "
            f"'{self._matlab_str(self.b1prep_mton.text())}', "
            f"'{self._matlab_str(mask)}', "
            f"'{self._matlab_str(out_base)}');"
        )
        cmd = ["matlab", "-batch", code]
        self._execute(cmd, "b1prep", done_message=f"B1 RFlocal saved to {out_base}.nii.gz")

    def _guess_subject_base(self, src_path: Path) -> Path | None:
        if self.subject_folder_path:
            # ensure we point to subject root (not modality)
            known_modalities = {"MTon&off", "MTon", "MToff", "MToff_PDw", "MToff_T1", "T1", "T2", "RAREvfl", "B1"}
            if self.subject_folder_path.name in known_modalities:
                return self.subject_folder_path.parent
            return self.subject_folder_path
        # Attempt to find subject folder as parent of the DICOM path that sits under project root/Data
        parts = src_path.parts
        if "Data" in parts:
            idx = parts.index("Data")
            if idx + 2 < len(parts):
                return Path(*parts[: idx + 2])
            if idx + 1 < len(parts):
                return Path(*parts[: idx + 1])
        return src_path.parent

    def _apply_subject_suggestions(self) -> None:
        """Pre-fill common paths based on selected subject folder, without overriding user-entered values."""
        if not self.subject_folder_path:
            return
        known_modalities = {
            "MTon&off",
            "MTon",
            "MToff",
            "MToff_PDw",
            "MToff_T1",
            "T1",
            "T2",
            "RAREvfl",
            "B1",
        }
        base = self.subject_folder_path
        if base.name in known_modalities:
            base = base.parent  # user picked a modality folder; go one level up

        def set_if_empty(line_edit: QtWidgets.QLineEdit, path: Path) -> None:
            if not line_edit.text():
                line_edit.setText(str(path))
            else:
                # update placeholder to show suggested path without overwriting
                line_edit.setPlaceholderText(str(path))
        # Split combined DICOM
        set_if_empty(self.split_src, base / "MTon&off" / "pdata" / "1" / "dicom")
        # Split outputs
        self.split_off.setPlaceholderText(str(base / "MToff" / "mtoff.dcm"))
        self.split_on.setPlaceholderText(str(base / "MTon" / "mton.dcm"))
        # dcm2niix defaults
        set_if_empty(self.dcm_input, base / "MTon&off" / "pdata" / "1" / "dicom")
        set_if_empty(self.dcm_outdir, base / "MTon&off")
        if not self.dcm_outname.text():
            self.dcm_outname.setText("MTon")
        # Recenter / orientation suggestions
        set_if_empty(self.recenter_input, base / "MTon" / "MTon.nii.gz")
        set_if_empty(self.orient_input, base / "MTon" / "MTon.nii.gz")
        # BrainSuite input suggestion
        set_if_empty(self.brainsuite_input, base / "MTon" / "MTon.nii.gz")
        # Coreg suggestions
        set_if_empty(self.coreg_ref, base / "MTon" / "MTon.nii.gz")
        set_if_empty(self.coreg_src, base / "MToff_PDw" / "MToff_PDw.nii.gz")
        set_if_empty(self.coreg_mask, base / "MTon" / "MTon_mask.nii.gz")
        # MTsat inputs
        set_if_empty(self.mtsat_mton, base / "MTon" / "MTon.nii.gz")
        set_if_empty(self.mtsat_mtoff, base / "MToff_PDw" / "MToff_PDw.nii.gz")
        set_if_empty(self.mtsat_t1, base / "MToff_T1" / "MToff_T1.nii.gz")
        set_if_empty(self.mtsat_mask, base / "MTon" / "MTon_mask.nii.gz")
        set_if_empty(self.mtsat_b1, base / "B1" / "B1_RFlocal.nii.gz")
        # B1 preprocess defaults
        set_if_empty(self.b1prep_b1, base / "B1" / "B1_ph.nii.gz")
        set_if_empty(self.b1prep_mton, base / "MTon" / "MTon.nii.gz")
        set_if_empty(self.b1prep_mask, base / "MTon" / "MTon_mask.nii.gz")
        set_if_empty(self.b1prep_out, base / "B1" / "B1_RFlocal")
        # T1/T2 step suggestions
        set_if_empty(self.t1t2_t1, base / "T1" / "T1.nii.gz")
        set_if_empty(self.t1t2_t2, base / "RAREvfl" / "RAREvfl.nii.gz")
        set_if_empty(self.t1t2_mask, base / "RAREvfl" / "RAREvfl.mask.nii.gz")
        # Compare defaults: left as MTon, right as MToff_PDw
        set_if_empty(self.compare_left, base / "MTon" / "MTon.nii.gz")
        set_if_empty(self.compare_right, base / "MToff_PDw" / "MToff_PDw.nii.gz")

    def _toggle_intensity_fields(self) -> None:
        pass

    def _rotate(self, side: str) -> None:
        if side == "left":
            self._left_rot = (self._left_rot + 1) % 4
        else:
            self._right_rot = (self._right_rot + 1) % 4
        self._update_compare_view()

    def _flip(self, side: str, horizontal: bool) -> None:
        if side == "left":
            if horizontal:
                self._left_flip_h = not self._left_flip_h
            else:
                self._left_flip_v = not self._left_flip_v
        else:
            if horizontal:
                self._right_flip_h = not self._right_flip_h
            else:
                self._right_flip_v = not self._right_flip_v
        self._update_compare_view()

    # ---- window close handling ----
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Exit UI")
        msg.setText("Close UI?")
        checkbox = QtWidgets.QCheckBox("Close console on exit (skip pause)")
        msg.setCheckBox(checkbox)
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msg.setDefaultButton(QtWidgets.QMessageBox.No)
        ret = msg.exec()
        if ret == QtWidgets.QMessageBox.Yes:
            if checkbox.isChecked():
                flag_path = self.repo_root / "ui_app" / "close_console.flag"
                try:
                    flag_path.write_text("close")
                except Exception:
                    pass
            event.accept()
        else:
            event.ignore()

    # ---- compare view helpers ----
    def _right_orient_axis(self) -> int:
        if self.compare_orient.count() == 0:
            return 2
        base_idx = max(0, self.compare_orient.currentIndex())
        right_idx = (base_idx + self._compare_right_orient_offset) % self.compare_orient.count()
        axis = self.compare_orient.itemData(right_idx)
        return axis if axis is not None else 2

    def _right_orient_text(self) -> str:
        if self.compare_orient.count() == 0:
            return ""
        base_idx = max(0, self.compare_orient.currentIndex())
        right_idx = (base_idx + self._compare_right_orient_offset) % self.compare_orient.count()
        return self.compare_orient.itemText(right_idx)

    def _refresh_right_orient_label(self) -> None:
        text = self._right_orient_text()
        self.compare_right_orient_label.setText(f"Right view: {text}" if text else "Right view")

    def _cycle_right_orient(self) -> None:
        if self.compare_orient.count() == 0:
            return
        self._compare_right_orient_offset = (self._compare_right_orient_offset + 1) % self.compare_orient.count()
        self._refresh_right_orient_label()
        self._update_compare_slider_range()
        self._update_compare_view()

    def _update_compare_slider_range(self) -> None:
        axis_left = self.compare_orient.currentData()
        axis_right = self._right_orient_axis()
        max_left = 0
        max_right = 0
        if hasattr(self, "_compare_data_left") and self._compare_data_left is not None:
            max_left = max(0, self._compare_data_left.shape[axis_left] - 1)
        if hasattr(self, "_compare_data_right") and self._compare_data_right is not None:
            max_right = max(0, self._compare_data_right.shape[axis_right] - 1)
        self.compare_slider_left.setMaximum(max_left)
        self.compare_slider_left.setValue(min(self.compare_slider_left.value(), max_left))
        self.compare_slider_right.setMaximum(max_right)
        self.compare_slider_right.setValue(min(self.compare_slider_right.value(), max_right))
        self._refresh_right_orient_label()

    def _update_compare_view(self) -> None:
        if not hasattr(self, "_compare_data_left") and not hasattr(self, "_compare_data_right"):
            return
        axis_left = self.compare_orient.currentData()
        axis_right = self._right_orient_axis()
        idx_left = self.compare_slider_left.value()
        idx_right = self.compare_slider_right.value()

        pos_color, neg_color = self.compare_scheme.currentData() or ((255, 0, 0), (0, 0, 255))

        left_slice = None
        right_slice = None
        if hasattr(self, "_compare_data_left") and self._compare_data_left is not None:
            left_slice = self._apply_transforms(self._extract_slice(self._compare_data_left, axis_left, idx_left), "left")
        if hasattr(self, "_compare_data_right") and self._compare_data_right is not None:
            right_slice = self._apply_transforms(self._extract_slice(self._compare_data_right, axis_right, idx_right), "right")

        def _norm_bounds(vmin_val: float, vmax_val: float) -> tuple[float, float]:
            if vmax_val <= vmin_val:
                vmax_val = vmin_val + max(abs(vmin_val) * 1e-3, 1e-6)
            return vmin_val, vmax_val

        vmin_left, vmax_left = _norm_bounds(self.compare_min_left.value(), self.compare_max_left.value())
        vmin_right, vmax_right = _norm_bounds(self.compare_min_right.value(), self.compare_max_right.value())

        overlay_ok = (
            left_slice is not None
            and right_slice is not None
            and self.compare_overlay.isChecked()
            and left_slice.shape == right_slice.shape
        )

        if overlay_ok:
            overlay, scale_val = self._build_overlay(
                left_slice, right_slice, vmin=vmin_left, vmax=vmax_left, pos_color=pos_color, neg_color=neg_color
            )
            self.viewer_left.set_slice(overlay)
            self.viewer_right.set_slice(right_slice, vmin=vmin_right, vmax=vmax_right)
            self.viewer_right.show()
            self.diff_scale.set_max_abs(scale_val)
            self.diff_scale.set_colors(QtGui.QColor(*pos_color), QtGui.QColor(*neg_color))
            self.diff_scale.show()
        elif left_slice is not None and right_slice is not None:
            self.viewer_left.set_slice(left_slice, vmin=vmin_left, vmax=vmax_left)
            self.viewer_right.set_slice(right_slice, vmin=vmin_right, vmax=vmax_right)
            self.viewer_right.show()
            self.diff_scale.hide()
        elif left_slice is not None:
            self.viewer_left.set_slice(left_slice, vmin=vmin_left, vmax=vmax_left)
            self.viewer_right.hide()
            self.diff_scale.hide()
        elif right_slice is not None:
            self.viewer_left.set_slice(right_slice, vmin=vmin_right, vmax=vmax_right)
            self.viewer_right.hide()
            self.diff_scale.hide()
        self.compare_info.setText(
            f"L:{idx_left} {self.compare_orient.currentText()} | R:{idx_right} {self._right_orient_text()}"
        )

    def _render_compare_frame(
        self, axis_left: int, idx_left: int, idx_right: int, axis_right: int | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        pos_color, neg_color = self.compare_scheme.currentData() or ((255, 0, 0), (0, 0, 255))
        axis_right = axis_left if axis_right is None else axis_right

        left_slice = None
        right_slice = None
        if (
            hasattr(self, "_compare_data_left")
            and self._compare_data_left is not None
            and idx_left < self._compare_data_left.shape[axis_left]
        ):
            left_slice = self._apply_transforms(self._extract_slice(self._compare_data_left, axis_left, idx_left), "left")
        if (
            hasattr(self, "_compare_data_right")
            and self._compare_data_right is not None
            and idx_right < self._compare_data_right.shape[axis_right]
        ):
            right_slice = self._apply_transforms(self._extract_slice(self._compare_data_right, axis_right, idx_right), "right")

        def _norm_bounds(vmin_val: float, vmax_val: float) -> tuple[float, float]:
            if vmax_val <= vmin_val:
                vmax_val = vmin_val + max(abs(vmin_val) * 1e-3, 1e-6)
            return vmin_val, vmax_val

        vmin_left, vmax_left = _norm_bounds(self.compare_min_left.value(), self.compare_max_left.value())
        vmin_right, vmax_right = _norm_bounds(self.compare_min_right.value(), self.compare_max_right.value())

        def _to_uint8(arr: np.ndarray, vmin_val: float, vmax_val: float) -> np.ndarray:
            norm = np.clip((arr - vmin_val) / (vmax_val - vmin_val), 0, 1)
            img = np.ascontiguousarray((norm * 255).astype(np.uint8))
            return np.repeat(img[:, :, None], 3, axis=2)

        overlay_ok = (
            left_slice is not None
            and right_slice is not None
            and self.compare_overlay.isChecked()
            and left_slice.shape == right_slice.shape
        )

        if overlay_ok:
            overlay, _ = self._build_overlay(
                left_slice, right_slice, vmin=vmin_left, vmax=vmax_left, pos_color=pos_color, neg_color=neg_color
            )
            left_img = np.ascontiguousarray((overlay * 255).astype(np.uint8))
            right_img = _to_uint8(right_slice, vmin_right, vmax_right)
            return left_img, right_img
        if left_slice is not None and right_slice is not None:
            return _to_uint8(left_slice, vmin_left, vmax_left), _to_uint8(right_slice, vmin_right, vmax_right)
        if left_slice is not None:
            return _to_uint8(left_slice, vmin_left, vmax_left), None
        if right_slice is not None:
            return _to_uint8(right_slice, vmin_right, vmax_right), None
        return None, None

    def _render_compare_frame_from_slices(self, left_slice: np.ndarray | None, right_slice: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        pos_color, neg_color = self.compare_scheme.currentData() or ((255, 0, 0), (0, 0, 255))
        def _norm_bounds(vmin_val: float, vmax_val: float) -> tuple[float, float]:
            if vmax_val <= vmin_val:
                vmax_val = vmin_val + max(abs(vmin_val) * 1e-3, 1e-6)
            return vmin_val, vmax_val
        vmin_left, vmax_left = _norm_bounds(self.compare_min_left.value(), self.compare_max_left.value())
        vmin_right, vmax_right = _norm_bounds(self.compare_min_right.value(), self.compare_max_right.value())
        def _to_uint8(arr: np.ndarray, vmin_val: float, vmax_val: float) -> np.ndarray:
            norm = np.clip((arr - vmin_val) / (vmax_val - vmin_val), 0, 1)
            img = np.ascontiguousarray((norm * 255).astype(np.uint8))
            return np.repeat(img[:, :, None], 3, axis=2)

        overlay_ok = (
            left_slice is not None
            and right_slice is not None
            and self.compare_overlay.isChecked()
            and left_slice.shape == right_slice.shape
        )
        if overlay_ok:
            overlay, _ = self._build_overlay(
                left_slice, right_slice, vmin=vmin_left, vmax=vmax_left, pos_color=pos_color, neg_color=neg_color
            )
            left_img = np.ascontiguousarray((overlay * 255).astype(np.uint8))
            right_img = _to_uint8(right_slice, vmin_right, vmax_right)
            return left_img, right_img
        if left_slice is not None and right_slice is not None:
            return _to_uint8(left_slice, vmin_left, vmax_left), _to_uint8(right_slice, vmin_right, vmax_right)
        if left_slice is not None:
            return _to_uint8(left_slice, vmin_left, vmax_left), None
        if right_slice is not None:
            return _to_uint8(right_slice, vmin_right, vmax_right), None
        return None, None

    def _concat_with_padding(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        h = max(left_img.shape[0], right_img.shape[0])
        def _pad(img, target_h):
            if img.shape[0] == target_h:
                out = img
            else:
                pad_top = (target_h - img.shape[0]) // 2
                pad_bottom = target_h - img.shape[0] - pad_top
                out = np.pad(img, ((pad_top, pad_bottom), (0,0), (0,0)), mode='constant')
            return out
        left_p = _pad(left_img, h)
        right_p = _pad(right_img, h)
        return np.concatenate([left_p, right_p], axis=1)

    def _slice_step(self, data: np.ndarray | None, axis: int, path: str | None) -> float | None:
        if data is None:
            return None
        if path:
            try:
                info = nib.load(path).header
                pixdim = info.get_zooms()
                if len(pixdim) > axis:
                    return abs(pixdim[axis])
            except Exception:
                pass
        # fallback: unit step
        return 1.0

    @staticmethod
    def _extract_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
        axis = max(0, min(axis, volume.ndim - 1))
        idx = max(0, min(idx, volume.shape[axis] - 1))
        slc = volume.take(idx, axis=axis)
        if slc.ndim == 1:
            slc = slc[np.newaxis, :]
        return slc

    def _apply_transforms(self, slc: np.ndarray, side: str) -> np.ndarray:
        rot = self._left_rot if side == "left" else self._right_rot
        flip_h = self._left_flip_h if side == "left" else self._right_flip_h
        flip_v = self._left_flip_v if side == "left" else self._right_flip_v
        out = np.rot90(slc, k=rot) if rot else slc
        if flip_h:
            out = np.fliplr(out)
        if flip_v:
            out = np.flipud(out)
        return out

    def _suggest_dcm_outdir(self, in_path: Path) -> Path | None:
        """
        If input is .../<modality>/pdata/1/dicom, suggest .../<modality> as output.
        If input already is a modality folder, keep it. Otherwise suggest parent.
        """
        parts = in_path.parts
        # look for pattern .../<modality>/pdata/1/dicom
        if len(parts) >= 3 and parts[-3].lower() == "pdata" and parts[-1].lower() == "dicom":
            return in_path.parent.parent.parent
        known_modalities = {"MTon&off", "MTon", "MToff", "MToff_PDw", "MToff_T1", "T1", "T2", "RAREvfl", "B1"}
        if in_path.name in known_modalities:
            return in_path
        if len(parts) >= 1:
            return in_path.parent
        return None

    def _resolve_dicom_folder(self, in_path: Path) -> Path:
        """
        If a modality folder is given, return its pdata/1/dicom if it exists. Otherwise return the path itself.
        """
        candidate = in_path / "pdata" / "1" / "dicom"
        if candidate.exists():
            return candidate
        return in_path

    @staticmethod
    def _strip_nii(path: str) -> str:
        if not path:
            return ""
        p = Path(path)
        if p.suffix.lower() == ".gz":
            p = p.with_suffix("")  # drop .gz
        if p.suffix.lower() == ".nii":
            p = p.with_suffix("")
        return str(p)

    def _build_overlay(
        self, ref: np.ndarray, cmp: np.ndarray, vmin=None, vmax=None, pos_color=(255, 0, 0), neg_color=(0, 0, 255)
    ) -> tuple[np.ndarray, float]:
        # Normalize reference base to 0-1 grayscale
        lo, hi = NiiSliceViewer._compute_range(ref, vmin, vmax)
        base = np.clip((ref - lo) / (hi - lo), 0, 1)
        if base.ndim != 2:
            return ref, 1.0
        base_rgb = np.stack([base, base, base], axis=-1)
        diff = cmp - ref
        absdiff = np.abs(diff)
        scale = np.percentile(absdiff, 99) if np.any(absdiff) else 1.0
        if scale < 1e-6:
            scale = 1.0
        norm = np.clip(diff / scale, -1, 1)
        overlay = base_rgb.copy()

        pos_mask = norm > 0
        neg_mask = norm < 0
        if np.any(pos_mask):
            pos = norm[pos_mask]
            pr, pg, pb = [c / 255.0 for c in pos_color]
            r = (1 - pos) * overlay[..., 0][pos_mask] + pos * pr
            g = (1 - pos) * overlay[..., 1][pos_mask] + pos * pg
            b = (1 - pos) * overlay[..., 2][pos_mask] + pos * pb
            overlay[..., 0][pos_mask] = r
            overlay[..., 1][pos_mask] = g
            overlay[..., 2][pos_mask] = b
        if np.any(neg_mask):
            neg = -norm[neg_mask]
            nr, ng, nb = [c / 255.0 for c in neg_color]
            r = (1 - neg) * overlay[..., 0][neg_mask] + neg * nr
            g = (1 - neg) * overlay[..., 1][neg_mask] + neg * ng
            b = (1 - neg) * overlay[..., 2][neg_mask] + neg * nb
            overlay[..., 0][neg_mask] = r
            overlay[..., 1][neg_mask] = g
            overlay[..., 2][neg_mask] = b
        return overlay, float(scale)

    def _select_dcm_input(self) -> None:
        start = Path(self.dcm_input.text() or self.subject_folder_path or self.fixed_root or self.last_browse_dir or Path.cwd())
        folder = self.pick_folder(start_dir=start)
        if not folder:
            return
        self.dcm_input.setText(str(folder))
        suggested_out = self._suggest_dcm_outdir(folder)
        if suggested_out:
            # Always refresh suggestion
            self.dcm_outdir.setText(str(suggested_out))
            self.dcm_outdir.setPlaceholderText(str(suggested_out))
            mod_name = suggested_out.name
            self.dcm_outname.setText(mod_name)
            self.dcm_outname.setPlaceholderText(mod_name)

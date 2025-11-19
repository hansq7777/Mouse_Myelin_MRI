from __future__ import annotations

import numpy as np
from PySide6 import QtGui, QtWidgets


class NiiSliceViewer(QtWidgets.QLabel):
    """Simple slice viewer with linear normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(256, 256)
        self.setAlignment(QtGui.Qt.AlignCenter)  # type: ignore[attr-defined]
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

    def set_slice(self, arr: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> None:
        if arr is None:
            self.clear()
            return
        arr = np.nan_to_num(arr)
        if arr.ndim != 2:
            self.clear()
            return
        lo, hi = self._compute_range(arr, vmin, vmax)
        norm = np.clip((arr - lo) / (hi - lo), 0, 1)
        img = np.ascontiguousarray((norm * 255).astype(np.uint8))
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.size(), QtGui.Qt.KeepAspectRatio, QtGui.Qt.SmoothTransformation  # type: ignore[attr-defined]
        )
        self.setPixmap(pix)

    @staticmethod
    def _compute_range(arr: np.ndarray, vmin: float | None, vmax: float | None) -> tuple[float, float]:
        if vmin is None or vmax is None:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return 0.0, 1.0
            nz = finite[finite != 0]
            if nz.size > 0:
                lo, hi = np.percentile(nz, [1, 99])
                if lo >= 0:
                    lo = 0.0
            else:
                lo, hi = np.percentile(finite, [1, 99])
                if lo >= 0:
                    lo = 0.0
            if hi - lo < 1e-9:
                hi = lo + 1e-3
        else:
            lo, hi = vmin, vmax
            if hi - lo <= 0:
                hi = lo + max(abs(lo) * 1e-3, 1e-6)
        return float(lo), float(hi)


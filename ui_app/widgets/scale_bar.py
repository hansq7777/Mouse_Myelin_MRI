from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class DiffScaleBar(QtWidgets.QWidget):
    """Vertical scale bar for overlay diff colors."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(60)
        self.setMinimumHeight(200)
        self.max_abs = 1.0
        self.pos_color = QtGui.QColor(255, 0, 0)
        self.neg_color = QtGui.QColor(0, 0, 255)

    def set_max_abs(self, value: float) -> None:
        self.max_abs = max(value, 1e-6)
        self.update()

    def set_colors(self, pos: QtGui.QColor, neg: QtGui.QColor) -> None:
        self.pos_color = pos
        self.neg_color = neg
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        rect = self.rect().adjusted(20, 10, -20, -10)
        grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        # Top: pos color, middle: gray zero, bottom: neg color
        grad.setColorAt(0.0, self.pos_color)
        grad.setColorAt(0.5, QtGui.QColor(200, 200, 200))
        grad.setColorAt(1.0, self.neg_color)
        painter.fillRect(rect, grad)
        painter.setPen(QtCore.Qt.black)
        painter.drawRect(rect)

        fm = painter.fontMetrics()
        labels = [
            (rect.top(), f"+{self.max_abs:.3g}"),
            (rect.center().y(), "0"),
            (rect.bottom(), f"-{self.max_abs:.3g}"),
        ]
        for y, text in labels:
            painter.drawText(rect.right() + 4, y + fm.ascent() / 2, text)

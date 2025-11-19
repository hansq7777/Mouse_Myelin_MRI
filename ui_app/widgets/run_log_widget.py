from __future__ import annotations

from PySide6 import QtWidgets


class RunLogWidget(QtWidgets.QWidget):
    """Simple text area to display command logs/output."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("Run outputs will appear here.")
        layout.addWidget(self.text)

    def append(self, line: str) -> None:
        self.text.appendPlainText(line)

    def clear_log(self) -> None:
        self.text.clear()

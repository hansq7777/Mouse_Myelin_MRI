"""
Entry point for the MT pipeline UI.
Run with: python3 -m ui_app or python3 ui_app/main.py
"""

import sys

from PySide6 import QtWidgets, QtGui

from ui_app.windows.main_window import MainWindow


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    # Windows 有些环境缺少 MS Sans Serif，设置一个常见字体避免 DirectWrite 报错
    app.setFont(QtGui.QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

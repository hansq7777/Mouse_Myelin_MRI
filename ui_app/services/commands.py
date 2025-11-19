from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


class CommandResult:
    def __init__(self, cmd: List[str], code: int, stdout: str, stderr: str):
        self.cmd = cmd
        self.code = code
        self.stdout = stdout
        self.stderr = stderr

    @property
    def ok(self) -> bool:
        return self.code == 0


def run_command(cmd: Iterable[str], cwd: Optional[Path] = None, env=None) -> CommandResult:
    """
    Minimal wrapper to run external tools (MATLAB, dcm2niix, shell scripts, etc).
    """
    process = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )
    return CommandResult(
        cmd=list(cmd),
        code=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )

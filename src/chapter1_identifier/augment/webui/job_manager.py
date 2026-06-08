from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.chapter1_identifier.augment._bootstrap import project_root, resolve_path
from src.chapter1_identifier.augment.loop.job_state import read_job_state, write_job_state

STILL_ACTIVE = 259


class JobManager:
    def __init__(self, job_state_path: str, python_executable: Optional[str] = None):
        self.job_state_path = resolve_path(job_state_path)
        self.python_executable = python_executable or sys.executable
        self.log_dir = self.job_state_path.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_handles: dict[int, object] = {}

    def _read_log_tail(self, log_path: Optional[str], max_chars: int = 800) -> str:
        if not log_path:
            return ""
        path = Path(log_path)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) <= max_chars:
            return text.strip()
        return text[-max_chars:].strip()

    def _launch(self, phase: str, round_idx: int, config_path: Optional[str]) -> dict:
        state = read_job_state(str(self.job_state_path))
        if state.get("status") == "running":
            raise RuntimeError("已有任务在运行")

        cmd = [
            self.python_executable,
            "-m",
            "src.chapter1_identifier.augment",
            phase,
            "--round",
            str(round_idx),
        ]
        if config_path:
            cmd.extend(["--config", config_path])

        log_path = self.log_dir / f"{phase}_round_{round_idx:02d}.log"
        log_handle = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root()),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._log_handles[proc.pid] = log_handle

        write_job_state(
            str(self.job_state_path),
            {
                "status": "running",
                "phase": phase,
                "round": round_idx,
                "pid": proc.pid,
                "log_path": str(log_path),
                "error": None,
            },
        )
        return {"pid": proc.pid, "phase": phase, "round": round_idx, "log_path": str(log_path), "python": self.python_executable}

    def start_train(self, round_idx: int, config_path: Optional[str] = None) -> dict:
        return self._launch("train", round_idx, config_path)

    def start_infer(self, round_idx: int, config_path: Optional[str] = None) -> dict:
        return self._launch("infer", round_idx, config_path)

    def _process_exit_code(self, pid: int) -> Optional[int]:
        import os

        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return -1
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            if not ok:
                return -1
            if exit_code.value == STILL_ACTIVE:
                return None
            return int(exit_code.value)

        import os as os_mod

        waited_pid, status = os_mod.waitpid(int(pid), os_mod.WNOHANG)
        if waited_pid == 0:
            return None
        if os_mod.WIFEXITED(status):
            return int(os_mod.WEXITSTATUS(status))
        return -1

    def poll(self) -> dict:
        state = read_job_state(str(self.job_state_path))
        pid = state.get("pid")
        if state.get("status") != "running" or pid is None:
            return state

        exit_code = self._process_exit_code(int(pid))
        if exit_code is None:
            return state

        state["pid"] = None
        log_handle = self._log_handles.pop(int(pid), None)
        if log_handle is not None:
            log_handle.close()
        if exit_code == 0:
            state["status"] = "done"
            state["error"] = None
        else:
            state["status"] = "error"
            tail = self._read_log_tail(state.get("log_path"))
            state["error"] = f"进程退出码 {exit_code}" + (f": {tail}" if tail else "")

        write_job_state(str(self.job_state_path), state)
        return state

    def read_log_tail(self, round_idx: int, phase: str = "train", max_chars: int = 6000) -> str:
        log_path = self.log_dir / f"{phase}_round_{round_idx:02d}.log"
        return self._read_log_tail(str(log_path), max_chars=max_chars)

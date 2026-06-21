from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.chapter3_identifier.regression_forecast._bootstrap import project_root, resolve_path
from src.chapter3_identifier.regression_forecast.webui.jobs.log_tail import read_log_tail_text
from src.chapter3_identifier.regression_forecast.webui.jobs.state import read_job_state, write_job_state

STILL_ACTIVE = 259
_SUCCESS_MARKERS = ("缓存完成", "训练完成", "预测完成")


class JobManager:
    def __init__(
        self,
        job_state_path: str,
        python_executable: Optional[str] = None,
        module_name: str = "src.chapter3_identifier.regression_forecast",
    ) -> None:
        self.job_state_path = resolve_path(job_state_path)
        self.python_executable = python_executable or sys.executable
        self.module_name = module_name
        self.log_dir = self.job_state_path.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[int, subprocess.Popen] = {}
        self._log_handles: dict[int, object] = {}
        self.reconcile_running_job()

    def _pid_alive(self, pid: int) -> bool:
        proc = self._processes.get(int(pid))
        if proc is not None:
            return proc.poll() is None
        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return bool(ok) and exit_code.value == STILL_ACTIVE
        return False

    def _detect_exit_from_log(self, log_path: Optional[str]) -> tuple[int, str]:
        if not log_path:
            return 1, "进程已终止"
        path = Path(log_path)
        if not path.exists():
            return 1, "进程已终止"
        tail = read_log_tail_text(log_path, max_chars=8000)
        if any(marker in tail for marker in _SUCCESS_MARKERS):
            return 0, ""
        if "Traceback" in tail:
            lines = [line.strip() for line in tail.splitlines() if line.strip()]
            hint = lines[-1] if lines else "Traceback"
            return 1, f"任务异常退出：{hint[:240]}"
        return 1, "进程已终止（未正常完成）"

    def _process_exit_code(self, pid: int) -> Optional[int]:
        proc = self._processes.get(int(pid))
        if proc is not None:
            return proc.poll()
        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return None
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            if not ok or exit_code.value == STILL_ACTIVE:
                return None
            return int(exit_code.value)
        return None

    def _finalize_job(self, state: dict, pid: int, exit_code: int, error: str = "") -> dict:
        state["pid"] = None
        self._processes.pop(int(pid), None)
        log_handle = self._log_handles.pop(int(pid), None)
        if log_handle is not None:
            log_handle.close()
        if exit_code == 0:
            state["status"] = "done"
            state["error"] = None
        else:
            state["status"] = "error"
            state["error"] = error or f"进程异常退出，退出码 {exit_code}"
        write_job_state(str(self.job_state_path), state)
        return state

    def reconcile_running_job(self) -> dict:
        state = read_job_state(str(self.job_state_path))
        pid = state.get("pid")
        if state.get("status") != "running":
            return state
        if pid is None:
            state["status"] = "error"
            state["error"] = state.get("error") or "任务状态异常（running 但无 pid）"
            write_job_state(str(self.job_state_path), state)
            return state
        pid = int(pid)
        if self._pid_alive(pid):
            return state
        exit_code, err = self._detect_exit_from_log(state.get("log_path"))
        return self._finalize_job(state, pid, exit_code, err)

    def reset_job(self) -> dict:
        state = read_job_state(str(self.job_state_path))
        pid = state.get("pid")
        if pid is not None:
            self._processes.pop(int(pid), None)
            log_handle = self._log_handles.pop(int(pid), None)
            if log_handle is not None:
                log_handle.close()
        cleared = {
            "status": "idle",
            "phase": None,
            "round": state.get("round", 0),
            "pid": None,
            "log_path": state.get("log_path"),
            "error": None,
        }
        write_job_state(str(self.job_state_path), cleared)
        return cleared

    def _launch(self, phase: str, round_idx: int, config_path: Optional[str]) -> dict:
        self.reconcile_running_job()
        state = read_job_state(str(self.job_state_path))
        if state.get("status") == "running":
            raise RuntimeError("已有任务在运行，请等待完成或释放后台状态")

        cmd = [self.python_executable, "-m", self.module_name, phase, "--round", str(round_idx)]
        if config_path:
            cmd.extend(["--config", config_path])

        log_path = self.log_dir / f"{phase}_round_{round_idx:02d}.log"
        log_handle = open(log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root()),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        self._processes[proc.pid] = proc
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

    def start_build_cache(self, round_idx: int, config_path: Optional[str] = None) -> dict:
        return self._launch("build-cache", round_idx, config_path)

    def start_train(self, round_idx: int, config_path: Optional[str] = None) -> dict:
        return self._launch("train", round_idx, config_path)

    def start_infer(self, round_idx: int, config_path: Optional[str] = None) -> dict:
        return self._launch("infer", round_idx, config_path)

    def poll(self) -> dict:
        state = read_job_state(str(self.job_state_path))
        pid = state.get("pid")
        if state.get("status") != "running":
            return state
        if pid is None:
            return self.reconcile_running_job()
        pid = int(pid)
        exit_code = self._process_exit_code(pid)
        if exit_code is None and self._pid_alive(pid):
            return state
        if exit_code is None:
            exit_code, err = self._detect_exit_from_log(state.get("log_path"))
            return self._finalize_job(state, pid, exit_code, err)
        if exit_code == 0:
            return self._finalize_job(state, pid, 0, "")
        _, err = self._detect_exit_from_log(state.get("log_path"))
        return self._finalize_job(state, pid, exit_code, err)

    def read_log_tail(self, round_idx: int, phase: str = "train", max_chars: int = 6000) -> str:
        log_path = self.log_dir / f"{phase}_round_{round_idx:02d}.log"
        return read_log_tail_text(str(log_path), max_chars=max_chars)


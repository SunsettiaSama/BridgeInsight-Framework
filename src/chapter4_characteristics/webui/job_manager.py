from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from src.chapter4_characteristics._bootstrap import project_root, resolve_path
from src.chapter4_characteristics.job_state import read_job_state, write_job_state

STILL_ACTIVE = 259
_SUCCESS_MARKERS = ("推理完成", "归档完成", "Copula 拟合完成")


class JobManager:
    def __init__(self, job_state_path: str, python_executable: Optional[str] = None):
        self.job_state_path = resolve_path(job_state_path)
        self.python_executable = python_executable or sys.executable
        self.log_dir = self.job_state_path.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[int, subprocess.Popen] = {}
        self._log_handles: dict[int, object] = {}
        self.reconcile_running_job()

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

    def _pid_alive(self, pid: int) -> bool:
        proc = self._processes.get(int(pid))
        if proc is not None:
            return proc.poll() is None
        import os
        if os.name == "nt":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            if not ok:
                return False
            return exit_code.value == STILL_ACTIVE
        import os
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _detect_exit_from_log(self, log_path: Optional[str]) -> Tuple[int, str]:
        if not log_path:
            return 1, "进程已终止"
        path = Path(log_path)
        if not path.exists():
            return 1, "进程已终止"
        tail = path.read_text(encoding="utf-8", errors="replace")[-8000:]
        if any(marker in tail for marker in _SUCCESS_MARKERS):
            return 0, ""
        if "KeyboardInterrupt" in tail:
            return 130, "任务被中断（KeyboardInterrupt）"
        if "Traceback" in tail:
            lines = [line.strip() for line in tail.splitlines() if line.strip()]
            hint = lines[-1] if lines else "Traceback"
            return 1, f"任务异常退出：{hint[:240]}"
        return 1, "进程已终止（未正常完成）"

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
        proc_exit = self._process_exit_code(pid)
        if proc_exit is not None and proc_exit == 0:
            return self._finalize_job(state, pid, 0)
        log_exit, err = self._detect_exit_from_log(state.get("log_path"))
        exit_code = proc_exit if proc_exit is not None else log_exit
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
            "pid": None,
            "log_path": state.get("log_path"),
            "error": None,
        }
        write_job_state(str(self.job_state_path), cleared)
        return cleared

    def _launch(self, phase: str, config_path: Optional[str], extra_args: Optional[list] = None) -> dict:
        self.reconcile_running_job()
        state = read_job_state(str(self.job_state_path))
        if state.get("status") == "running":
            raise RuntimeError("已有任务在运行，请等待完成或点击释放后台")

        cmd = [
            self.python_executable,
            "-m",
            "src.chapter4_characteristics",
            phase,
        ]
        if config_path:
            cmd.extend(["--config", config_path])
        if extra_args:
            cmd.extend(extra_args)

        log_path = self.log_dir / f"{phase}.log"
        log_handle = open(log_path, "w", encoding="utf-8")
        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["AUGMENT_PLAIN_LOG"] = "1"
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
                "pid": proc.pid,
                "log_path": str(log_path),
                "error": None,
            },
        )
        return {"pid": proc.pid, "phase": phase, "log_path": str(log_path)}

    def start_infer(self, config_path: Optional[str] = None, limit: Optional[int] = None) -> dict:
        extra = ["--limit", str(limit)] if limit else None
        return self._launch("infer", config_path, extra)

    def start_enrich(self, config_path: Optional[str] = None, limit: Optional[int] = None) -> dict:
        extra = ["--limit", str(limit)] if limit else None
        return self._launch("enrich", config_path, extra)

    def start_copula(self, class_id: int, config_path: Optional[str] = None) -> dict:
        return self._launch(
            "copula",
            config_path,
            ["run", "--class-id", str(class_id)],
        )

    def _process_exit_code(self, pid: int) -> Optional[int]:
        proc = self._processes.get(int(pid))
        if proc is not None:
            return proc.poll()
        import os
        if os.name == "nt":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return None
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            if not ok:
                return None
            if exit_code.value == STILL_ACTIVE:
                return None
            return int(exit_code.value)
        import os as os_mod
        waited_pid, status = os_mod.waitpid(int(pid), os_mod.WNOHANG)
        if waited_pid == 0:
            return None
        if os_mod.WIFEXITED(status):
            return int(os_mod.WEXITSTATUS(status))
        return None

    def poll(self) -> dict:
        state = read_job_state(str(self.job_state_path))
        pid = state.get("pid")
        if state.get("status") != "running":
            return state
        if pid is None:
            return self.reconcile_running_job()
        pid = int(pid)
        if not self._pid_alive(pid):
            log_exit, err = self._detect_exit_from_log(state.get("log_path"))
            proc_exit = self._process_exit_code(pid)
            exit_code = proc_exit if proc_exit is not None else log_exit
            return self._finalize_job(state, pid, exit_code, err)
        exit_code = self._process_exit_code(pid)
        if exit_code is None:
            return state
        if exit_code == 0:
            return self._finalize_job(state, pid, 0, "")
        _, err = self._detect_exit_from_log(state.get("log_path"))
        return self._finalize_job(state, pid, exit_code, err or f"进程异常退出，退出码 {exit_code}")

    def read_log_tail(self, phase: str = "infer", max_chars: int = 6000) -> str:
        log_path = self.log_dir / f"{phase}.log"
        return self._read_log_tail(str(log_path), max_chars=max_chars)

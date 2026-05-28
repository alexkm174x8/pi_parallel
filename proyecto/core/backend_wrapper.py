from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
import tempfile
from typing import Callable
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "c_backend"
BACKEND_BIN_DIR = BACKEND_DIR / "bin"
BACKEND_EXECUTABLE = BACKEND_BIN_DIR / ("para_image_parra.exe" if sys.platform.startswith("win") else "para_image_parra")
BACKEND_EXECUTABLE_FALLBACK = BACKEND_DIR / ("para_image_parra.exe" if sys.platform.startswith("win") else "para_image_parra")
MPI_BACKEND_EXECUTABLE = BACKEND_BIN_DIR / ("para_image_parra_mpi.exe" if sys.platform.startswith("win") else "para_image_parra_mpi")
MPI_BACKEND_EXECUTABLE_FALLBACK = BACKEND_DIR / ("para_image_parra_mpi.exe" if sys.platform.startswith("win") else "para_image_parra_mpi")
TIME_PATTERN = re.compile(r"TOTAL_TIME=([0-9]*\.?[0-9]+)")


class BackendError(Exception):
    pass


@dataclass(slots=True)
class ProcessingRequest:
    image_paths: list[str]
    output_dir: str
    filters: list[str]
    kernel_gray: int | None = None
    kernel_color: int | None = None
    executable: Path | None = None
    use_mpi: bool = False
    mpi_processes: int = 4
    mpi_hostfile: str = "machinefile"
    mpi_oversubscribe: bool = True
    mpi_map_by: str = "node"
    mpi_shared_root: str = "/shared/proyecto"
    mpi_auto_detect_nodes: bool = True


@dataclass(slots=True)
class ProcessingResult:
    execution_time: float
    output_dir: str
    stdout: str
    stderr: str
    command: list[str]


class BackendRunner:
    def __init__(self, executable: Path | None = None) -> None:
        self.executable = executable or BACKEND_EXECUTABLE
        self._process: subprocess.Popen[str] | None = None

    def cancel(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()

    def run(self, request: ProcessingRequest, on_output: Callable[[str], None] | None = None) -> ProcessingResult:
        shared_root = (request.mpi_shared_root or "").strip()
        if request.use_mpi and not shared_root:
            raise BackendError("En modo MPI debes indicar una carpeta compartida (shared root).")

        def to_mpi_shared_path(path_value: str) -> str:
            path_obj = Path(path_value)
            if path_obj.is_absolute() and str(path_obj).startswith(shared_root):
                return str(path_obj)

            candidate_prefixes: list[str] = [str(PROJECT_ROOT)]
            parent = PROJECT_ROOT.parent
            candidate_prefixes.append(str(parent))

            if "/shared/" in str(PROJECT_ROOT):
                base = str(PROJECT_ROOT)
                shared_index = base.find("/shared/")
                if shared_index > 0:
                    candidate_prefixes.append(base[:shared_index] + base[shared_index:])

            normalized = str(path_obj)
            for prefix in sorted(set(candidate_prefixes), key=len, reverse=True):
                if normalized.startswith(prefix + "/"):
                    suffix = normalized[len(prefix):].lstrip("/")
                    return f"{shared_root.rstrip('/')}/{suffix}"

            if normalized.startswith("/home/") and "/shared/" in normalized:
                shared_pos = normalized.find("/shared/")
                suffix = normalized[shared_pos + len("/shared/"):]
                return f"{shared_root.rstrip('/')}/{suffix}"

            if not path_obj.is_absolute():
                return f"{shared_root.rstrip('/')}/{normalized.lstrip('./')}"

            return normalized

        if request.use_mpi:
            executable = Path(f"{shared_root.rstrip('/')}/c_backend/bin/{MPI_BACKEND_EXECUTABLE.name}")
        else:
            executable = Path(request.executable or self.executable)
        if not executable.exists():
            executable = MPI_BACKEND_EXECUTABLE_FALLBACK if request.use_mpi else BACKEND_EXECUTABLE_FALLBACK
        if not executable.exists():
            if request.use_mpi:
                raise BackendError("No se encontro el ejecutable MPI. Compila c_backend/src/bmp_processor_mpi.c con mpicc.")
            raise BackendError("No se encontro el ejecutable del backend. Compila primero c_backend/src/bmp_processor.c.")

        output_dir_value = to_mpi_shared_path(request.output_dir) if request.use_mpi else request.output_dir
        output_dir = Path(output_dir_value)
        output_dir.mkdir(parents=True, exist_ok=True)

        backend_command = [
            str(executable),
            "--output",
            str(output_dir),
            "--filters",
            ",".join(request.filters),
        ]

        if request.kernel_gray is not None:
            backend_command.extend(["--kernel-gray", str(request.kernel_gray)])
        if request.kernel_color is not None:
            backend_command.extend(["--kernel-color", str(request.kernel_color)])

        image_paths = [to_mpi_shared_path(path) for path in request.image_paths] if request.use_mpi else request.image_paths
        backend_command.extend(image_paths)

        if request.use_mpi:
            if request.mpi_processes < 1:
                raise BackendError("El numero de procesos MPI debe ser mayor a 0.")

            hostfile = Path(request.mpi_hostfile)
            if not hostfile.is_absolute():
                hostfile = Path(to_mpi_shared_path(str(hostfile)))
            if not hostfile.exists():
                raise BackendError(f"No se encontro el hostfile MPI: {hostfile}")

            alive_hostfile = hostfile
            adjusted_np = request.mpi_processes
            tmp_hostfile_path: Path | None = None

            def log_line(text: str) -> None:
                if on_output is not None:
                    on_output(text)

            def parse_hosts(file_path: Path) -> list[tuple[str, int, str]]:
                entries: list[tuple[str, int, str]] = []
                for raw in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    host = parts[0]
                    slots = 1
                    for part in parts[1:]:
                        if part.startswith("slots="):
                            try:
                                slots = max(1, int(part.split("=", 1)[1]))
                            except ValueError:
                                slots = 1
                    entries.append((host, slots, line))
                return entries

            def node_available(host: str) -> tuple[bool, str]:
                check_cmd = (
                    f"test -x {shared_root.rstrip('/')}/c_backend/bin/{MPI_BACKEND_EXECUTABLE.name} "
                    f"&& test -d {shared_root.rstrip('/')}/img "
                    f"&& test -d {shared_root.rstrip('/')}/output"
                )
                local_hostnames = {
                    "localhost",
                    "127.0.0.1",
                    socket.gethostname(),
                    socket.getfqdn(),
                    socket.gethostname().split(".")[0],
                    socket.getfqdn().split(".")[0],
                }
                if host in local_hostnames:
                    cmd = ["sh", "-lc", check_cmd]
                    mode = "local"
                else:
                    cmd = [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=3",
                        host,
                        check_cmd,
                    ]
                    mode = "ssh"
                try:
                    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=6, check=False)
                except subprocess.TimeoutExpired:
                    return False, "timeout"
                if completed.returncode == 0:
                    return True, mode
                err = (completed.stderr or completed.stdout or "failed").strip().replace("\n", " ")
                return False, err[:120]

            if request.mpi_auto_detect_nodes:
                log_line("Detectando nodos disponibles...")
                host_entries = parse_hosts(hostfile)
                if not host_entries:
                    raise BackendError(f"El hostfile esta vacio o invalido: {hostfile}")

                alive_entries: list[tuple[str, int, str]] = []
                dead_entries: list[tuple[str, str]] = []

                for host, slots, original in host_entries:
                    ok, reason = node_available(host)
                    if ok:
                        alive_entries.append((host, slots, original))
                        log_line(f"CHECK {host} OK slots={slots} via={reason}")
                    else:
                        dead_entries.append((host, reason))
                        log_line(f"CHECK {host} FAIL {reason}")

                if not alive_entries:
                    raise BackendError("No hay nodos disponibles para ejecutar MPI.")

                total_alive_slots = sum(slots for _, slots, _ in alive_entries)
                adjusted_np = min(request.mpi_processes, total_alive_slots)
                if adjusted_np < 1:
                    raise BackendError("No hay slots disponibles en nodos activos.")

                fd, tmp_path = tempfile.mkstemp(prefix="mpi_alive_hosts_", text=True)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
                        for _, _, original in alive_entries:
                            tmpf.write(original + "\n")
                except Exception:
                    Path(tmp_path).unlink(missing_ok=True)
                    raise

                tmp_hostfile_path = Path(tmp_path)
                alive_hostfile = tmp_hostfile_path

                alive_names = ", ".join(host for host, _, _ in alive_entries)
                log_line(f"Nodos disponibles: {alive_names}")
                if dead_entries:
                    ignored_names = ", ".join(host for host, _ in dead_entries)
                    log_line(f"Nodos ignorados: {ignored_names}")
                log_line(f"Slots disponibles: {total_alive_slots}")
                log_line(f"Procesos solicitados: {request.mpi_processes}")
                log_line(f"Procesos MPI ajustados: {adjusted_np}")
                log_line(f"Hostfile temporal: {alive_hostfile}")
                log_line("Iniciando procesamiento MPI...")

            command = ["mpirun"]
            if request.mpi_oversubscribe:
                command.append("--oversubscribe")
            command.extend(["-np", str(adjusted_np), "--hostfile", str(alive_hostfile)])
            map_by = (request.mpi_map_by or "").strip()
            if map_by:
                command.extend(["--map-by", map_by])
            command += backend_command
        else:
            command = backend_command

        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self._process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )

        output_lines: list[str] = []
        if self._process.stdout is not None:
            for line in self._process.stdout:
                output_lines.append(line)
                if on_output is not None:
                    on_output(line.rstrip("\n"))

        return_code = self._process.wait()
        stdout_text = "".join(output_lines)
        self._process = None

        if request.use_mpi and "tmp_hostfile_path" in locals() and tmp_hostfile_path is not None:
            tmp_hostfile_path.unlink(missing_ok=True)

        if return_code != 0:
            error_text = stdout_text.strip() or "El backend devolvio un error."
            command_text = " ".join(command)
            raise BackendError(f"Comando ejecutado:\n{command_text}\n\n{error_text}")

        match = TIME_PATTERN.search(stdout_text)
        if not match:
            raise BackendError("No se pudo leer el tiempo total desde la salida del backend.")

        return ProcessingResult(
            execution_time=float(match.group(1)),
            output_dir=str(output_dir),
            stdout=stdout_text,
            stderr="",
            command=command,
        )

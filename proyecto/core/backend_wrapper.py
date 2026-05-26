from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    executable: Path = BACKEND_EXECUTABLE
    use_mpi: bool = False
    mpi_processes: int = 4
    mpi_hostfile: str = "machinefile"
    mpi_oversubscribe: bool = True
    mpi_map_by: str = "node"


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

    def run(self, request: ProcessingRequest) -> ProcessingResult:
        if request.use_mpi:
            executable = Path(request.executable or MPI_BACKEND_EXECUTABLE)
        else:
            executable = Path(request.executable or self.executable)
        if not executable.exists():
            executable = MPI_BACKEND_EXECUTABLE_FALLBACK if request.use_mpi else BACKEND_EXECUTABLE_FALLBACK
        if not executable.exists():
            if request.use_mpi:
                raise BackendError("No se encontro el ejecutable MPI. Compila c_backend/src/bmp_processor_mpi.c con mpicc.")
            raise BackendError("No se encontro el ejecutable del backend. Compila primero c_backend/src/bmp_processor.c.")

        output_dir = Path(request.output_dir)
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

        backend_command.extend(request.image_paths)

        if request.use_mpi:
            if request.mpi_processes < 1:
                raise BackendError("El numero de procesos MPI debe ser mayor a 0.")

            hostfile = Path(request.mpi_hostfile)
            if not hostfile.is_absolute():
                hostfile = PROJECT_ROOT / hostfile
            if not hostfile.exists():
                raise BackendError(f"No se encontro el hostfile MPI: {hostfile}")

            command = ["mpirun"]
            if request.mpi_oversubscribe:
                command.append("--oversubscribe")
            command.extend(["-np", str(request.mpi_processes), "--hostfile", str(hostfile)])
            map_by = (request.mpi_map_by or "").strip()
            if map_by:
                command.extend(["--map-by", map_by])
            command += backend_command
        else:
            command = backend_command

        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            check=False,
        )

        if completed.returncode != 0:
            error_text = completed.stderr.strip() or completed.stdout.strip() or "El backend devolvio un error."
            raise BackendError(error_text)

        match = TIME_PATTERN.search(completed.stdout)
        if not match:
            raise BackendError("No se pudo leer el tiempo total desde la salida del backend.")

        return ProcessingResult(
            execution_time=float(match.group(1)),
            output_dir=str(output_dir),
            stdout=completed.stdout,
            stderr=completed.stderr,
            command=command,
        )

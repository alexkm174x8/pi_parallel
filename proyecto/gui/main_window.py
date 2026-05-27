from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.backend_wrapper import BACKEND_DIR, BackendError, BackendRunner, ProcessingRequest, ProcessingResult
from gui.about_dialog import AboutDialog


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MACHINEFILE = Path("/shared/proyecto/hosts")
DEFAULT_SHARED_ROOT = "/shared/proyecto"
MAX_IMAGES = 10
FILTER_LABELS = {
    "vg": "Inversion vertical gris",
    "vc": "Inversion vertical color",
    "hg": "Espejo horizontal gris",
    "hc": "Espejo horizontal color",
    "dg": "Desenfoque gris",
    "dc": "Desenfoque color",
}


class DropArea(QFrame):
    files_dropped = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropArea")

        label = QLabel("Arrastra hasta 10 imagenes BMP aqui\n\no usa el boton Agregar BMP")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addStretch(1)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        self.files_dropped.emit(paths)
        event.acceptProposedAction()


class ProcessingWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log_line = Signal(str)

    def __init__(self, request: ProcessingRequest) -> None:
        super().__init__()
        self.request = request
        self.runner = BackendRunner()

    def run(self) -> None:
        try:
            result = self.runner.run(self.request, on_output=self.log_line.emit)
        except BackendError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover
            self.failed.emit(f"Error inesperado: {exc}")
            return

        self.finished.emit(result)

    def cancel(self) -> None:
        self.runner.cancel()


@dataclass(slots=True)
class ValidationResult:
    request: ProcessingRequest | None
    error: str | None = None


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BMP Parallel Studio")
        self.resize(1080, 680)

        self._thread: QThread | None = None
        self._worker: ProcessingWorker | None = None
        self.image_paths: list[str] = []
        self.filter_checkboxes: dict[str, QCheckBox] = {}

        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self._build_menu()
        self._build_ui()
        self.statusBar().showMessage(f"Backend esperado en: {BACKEND_DIR}")

    def _build_menu(self) -> None:
        about_action = QAction("Acerca de", self)
        about_action.triggered.connect(self.show_about_dialog)
        self.menuBar().addMenu("Ayuda").addAction(about_action)

    def _build_ui(self) -> None:
        container = QWidget()
        root = QHBoxLayout(container)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        logo_label = QLabel()
        logo_path = PROJECT_ROOT / "logo_tec.png"
        pixmap = QPixmap(str(logo_path))

        logo_label.setPixmap(pixmap.scaledToWidth(220, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignLeft)

        about_button = QPushButton("Acerca de")
        about_button.clicked.connect(self.show_about_dialog)

        logo_row = QHBoxLayout()
        logo_row.addWidget(logo_label)
        logo_row.addStretch(1)  # empuja el botón hacia la derecha
        logo_row.addWidget(about_button)

        left_panel = self._build_left_panel()

        left_layout.addLayout(logo_row)
        left_layout.addWidget(left_panel)

        root.addWidget(left_container, 5)
        root.addWidget(self._build_right_panel(), 4)

        self.setCentralWidget(container)

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(14)

        title = QLabel("Imagenes BMP")
        title.setObjectName("sectionTitle")

        subtitle = QLabel("Carga archivos BMP con arrastrar y soltar o con el selector.")
        subtitle.setObjectName("mutedLabel")

        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.add_images)

        buttons = QHBoxLayout()
        add_button = QPushButton("Agregar BMP")
        add_button.clicked.connect(self.select_images)

        remove_button = QPushButton("Quitar seleccion")
        remove_button.clicked.connect(self.remove_selected_images)

        clear_button = QPushButton("Limpiar lista")
        clear_button.clicked.connect(self.clear_images)

        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addWidget(clear_button)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list.setAlternatingRowColors(True)

        self.loaded_count_label = QLabel("0 / 10 imagenes cargadas")
        self.loaded_count_label.setObjectName("mutedLabel")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.drop_area, 4)
        layout.addLayout(buttons)
        layout.addWidget(self.image_list, 5)
        layout.addWidget(self.loaded_count_label)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)

        title = QLabel("Opciones de procesamiento")
        title.setObjectName("sectionTitle")

        transform_group = QGroupBox("Transformaciones")
        transform_layout = QGridLayout(transform_group)

        self.all_checkbox = QCheckBox("Todas")
        self.all_checkbox.toggled.connect(self.toggle_all_filters)
        transform_layout.addWidget(self.all_checkbox, 0, 0, 1, 2)

        for index, (filter_code, label) in enumerate(FILTER_LABELS.items(), start=1):
            checkbox = QCheckBox(label)
            checkbox.toggled.connect(self.sync_all_checkbox)
            self.filter_checkboxes[filter_code] = checkbox
            row = (index + 1) // 2
            column = (index - 1) % 2
            transform_layout.addWidget(checkbox, row, column)

        kernel_group = QGroupBox("Kernels de desenfoque")
        kernel_layout = QFormLayout(kernel_group)

        self.kernel_gray_spin = self._create_kernel_spinbox()
        self.kernel_color_spin = self._create_kernel_spinbox()
        kernel_layout.addRow("Kernel gris:", self.kernel_gray_spin)
        kernel_layout.addRow("Kernel color:", self.kernel_color_spin)

        mpi_group = QGroupBox("Ejecucion MPI")
        mpi_layout = QFormLayout(mpi_group)
        self.use_mpi_checkbox = QCheckBox("Usar cluster MPI")
        self.use_mpi_checkbox.setChecked(True)
        self.mpi_processes_spin = QSpinBox()
        self.mpi_processes_spin.setRange(1, 128)
        self.mpi_processes_spin.setValue(6)
        self.mpi_hostfile_edit = QLineEdit(str(DEFAULT_MACHINEFILE))
        self.mpi_shared_root_edit = QLineEdit(DEFAULT_SHARED_ROOT)
        self.mpi_oversubscribe_checkbox = QCheckBox("Activar oversubscribe")
        self.mpi_oversubscribe_checkbox.setChecked(True)
        self.mpi_map_by_edit = QLineEdit("node")
        mpi_layout.addRow("Modo:", self.use_mpi_checkbox)
        mpi_layout.addRow("Procesos:", self.mpi_processes_spin)
        mpi_layout.addRow("Hostfile:", self.mpi_hostfile_edit)
        mpi_layout.addRow("Shared root:", self.mpi_shared_root_edit)
        mpi_layout.addRow("Oversubscribe:", self.mpi_oversubscribe_checkbox)
        mpi_layout.addRow("Map by:", self.mpi_map_by_edit)

        output_group = QGroupBox("Salida y resultado")
        output_layout = QFormLayout(output_group)

        output_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        self.output_dir_edit.setReadOnly(True)
        browse_output_button = QPushButton("Elegir carpeta")
        browse_output_button.clicked.connect(self.select_output_dir)
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(browse_output_button)

        self.execution_time_edit = QLineEdit()
        self.execution_time_edit.setReadOnly(True)
        self.execution_time_edit.setPlaceholderText("Aun sin ejecutar")

        output_layout.addRow("Ruta de salida:", self._wrap_layout(output_row))
        output_layout.addRow("Tiempo total:", self.execution_time_edit)
        self.logs_edit = QTextEdit()
        self.logs_edit.setReadOnly(True)
        self.logs_edit.setPlaceholderText("Aqui se muestran logs DISPATCH/COMPLETE del backend MPI.")
        self.summary_edit = QTextEdit()
        self.summary_edit.setReadOnly(True)
        self.summary_edit.setPlaceholderText("Resumen por nodo (tareas, imagenes, tiempo, errores).")
        output_layout.addRow("Logs:", self.logs_edit)
        output_layout.addRow("Resumen por nodo:", self.summary_edit)

        action_row = QHBoxLayout()
        self.execute_button = QPushButton("Ejecutar")
        self.execute_button.setObjectName("primaryButton")
        self.execute_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_processing)

        action_row.addWidget(self.execute_button, 1)
        action_row.addWidget(self.cancel_button)

        backend_hint = QLabel("El boton Ejecutar se bloquea mientras corre el backend en C.")
        backend_hint.setObjectName("mutedLabel")

        layout.addWidget(title)
        layout.addWidget(transform_group)
        layout.addWidget(kernel_group)
        layout.addWidget(mpi_group)
        layout.addWidget(output_group)
        layout.addWidget(backend_hint)
        layout.addStretch(1)
        layout.addLayout(action_row)
        return panel

    def _create_kernel_spinbox(self) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, 999)
        spin.setSingleStep(2)
        spin.setValue(27)
        return spin

    def _wrap_layout(self, layout) -> QWidget:
        wrapper = QWidget()
        wrapper.setLayout(layout)
        return wrapper

    def select_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar imagenes BMP",
            str(PROJECT_ROOT),
            "Imagenes BMP (*.bmp)",
        )
        if files:
            self.add_images(files)

    def add_images(self, paths: list[str]) -> None:
        normalized_paths = [str(Path(path).resolve()) for path in paths if path]
        bmp_paths = [path for path in normalized_paths if Path(path).suffix.lower() == ".bmp"]

        if len(bmp_paths) != len(normalized_paths):
            QMessageBox.warning(self, "Archivos no validos", "Solo se aceptan archivos con extension .bmp.")
            return

        unique_new_paths = [path for path in bmp_paths if path not in self.image_paths]
        if len(self.image_paths) + len(unique_new_paths) > MAX_IMAGES:
            QMessageBox.warning(self, "Limite excedido", "Solo puedes cargar un maximo de 10 imagenes BMP.")
            return

        self.image_paths.extend(unique_new_paths)
        self.refresh_image_list()

    def remove_selected_images(self) -> None:
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            return

        selected_paths = {item.data(Qt.UserRole) for item in selected_items}
        self.image_paths = [path for path in self.image_paths if path not in selected_paths]
        self.refresh_image_list()

    def clear_images(self) -> None:
        self.image_paths.clear()
        self.refresh_image_list()

    def refresh_image_list(self) -> None:
        self.image_list.clear()
        for path in self.image_paths:
            item = QListWidgetItem(f"{Path(path).name}\n{path}")
            item.setData(Qt.UserRole, path)
            self.image_list.addItem(item)

        self.loaded_count_label.setText(f"{len(self.image_paths)} / {MAX_IMAGES} imagenes cargadas")

    def select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta de salida",
            self.output_dir_edit.text() or str(DEFAULT_OUTPUT_DIR),
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def toggle_all_filters(self, checked: bool) -> None:
        for checkbox in self.filter_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

    def sync_all_checkbox(self) -> None:
        all_selected = all(checkbox.isChecked() for checkbox in self.filter_checkboxes.values())
        self.all_checkbox.blockSignals(True)
        self.all_checkbox.setChecked(all_selected)
        self.all_checkbox.blockSignals(False)

    def selected_filters(self) -> list[str]:
        return [code for code, checkbox in self.filter_checkboxes.items() if checkbox.isChecked()]

    def validate_request(self) -> ValidationResult:
        if not self.image_paths:
            return ValidationResult(None, "Carga al menos una imagen BMP antes de ejecutar.")

        if len(self.image_paths) > MAX_IMAGES:
            return ValidationResult(None, "El maximo permitido es de 10 imagenes BMP.")

        selected_filters = self.selected_filters()
        if not selected_filters:
            return ValidationResult(None, "Selecciona al menos una transformacion.")

        kernel_gray = self.kernel_gray_spin.value()
        kernel_color = self.kernel_color_spin.value()

        if "dg" in selected_filters and not self._is_valid_kernel(kernel_gray):
            return ValidationResult(None, "El kernel para desenfoque gris debe ser un entero positivo e impar.")

        if "dc" in selected_filters and not self._is_valid_kernel(kernel_color):
            return ValidationResult(None, "El kernel para desenfoque color debe ser un entero positivo e impar.")

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            return ValidationResult(None, "Selecciona una carpeta de salida.")

        request = ProcessingRequest(
            image_paths=list(self.image_paths),
            output_dir=output_dir,
            filters=selected_filters,
            kernel_gray=kernel_gray if "dg" in selected_filters else None,
            kernel_color=kernel_color if "dc" in selected_filters else None,
            use_mpi=self.use_mpi_checkbox.isChecked(),
            mpi_processes=self.mpi_processes_spin.value(),
            mpi_hostfile=self.mpi_hostfile_edit.text().strip(),
            mpi_oversubscribe=self.mpi_oversubscribe_checkbox.isChecked(),
            mpi_map_by=self.mpi_map_by_edit.text().strip(),
            mpi_shared_root=self.mpi_shared_root_edit.text().strip(),
        )

        if request.use_mpi:
            if not request.mpi_hostfile:
                return ValidationResult(None, "En MPI, debes indicar un hostfile.")
            if not request.mpi_map_by.strip():
                return ValidationResult(None, "En MPI, el campo Map by no puede estar vacio.")
            if not request.mpi_shared_root.strip().startswith("/"):
                return ValidationResult(None, "En MPI, Shared root debe ser una ruta absoluta (ej: /shared/proyecto).")
        return ValidationResult(request)

    def _is_valid_kernel(self, value: int) -> bool:
        return value > 0 and value % 2 == 1

    def start_processing(self) -> None:
        validation = self.validate_request()
        if validation.error:
            QMessageBox.warning(self, "Validacion", validation.error)
            return

        request = validation.request
        if request is None:
            QMessageBox.warning(self, "Validacion", "No se pudo construir la solicitud.")
            return

        self.execute_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.execution_time_edit.setText("Procesando...")
        self.logs_edit.clear()
        self.summary_edit.clear()
        self.statusBar().showMessage("Ejecutando backend...")
        self.logs_edit.append(f"$ {' '.join([str(x) for x in self._preview_command(request)])}")
        self.logs_edit.append("")

        self._thread = QThread(self)
        self._worker = ProcessingWorker(request)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log_line.connect(self.on_log_line)
        self._worker.finished.connect(self.on_processing_finished)
        self._worker.failed.connect(self.on_processing_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def on_processing_finished(self, result: ProcessingResult) -> None:
        self.execute_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.execution_time_edit.setText(f"{result.execution_time:.3f} s")
        self.output_dir_edit.setText(result.output_dir)
        self.statusBar().showMessage("Procesamiento finalizado.")
        logs = "\n".join([
            f"$ {' '.join(result.command)}",
            "",
            "[STDOUT]",
            result.stdout.strip(),
            "",
            "[STDERR]",
            result.stderr.strip(),
        ]).strip()
        self.logs_edit.setPlainText(logs)
        self.summary_edit.setPlainText(self._build_node_summary(result.stdout, result.execution_time))

        total_outputs = len(self.image_paths) * len(self.selected_filters())
        QMessageBox.information(
            self,
            "Proceso completado",
            "Las imagenes se procesaron correctamente.\n\n"
            f"Tiempo total: {result.execution_time:.3f} s\n"
            f"Ruta de salida: {result.output_dir}\n"
            f"Archivos esperados: {total_outputs}",
        )

    def on_processing_failed(self, error_message: str) -> None:
        self.execute_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.execution_time_edit.clear()
        self.logs_edit.setPlainText(error_message)
        self.summary_edit.clear()
        self.statusBar().showMessage("Fallo la ejecucion del backend.")
        QMessageBox.critical(self, "Error de procesamiento", error_message)

    def on_log_line(self, line: str) -> None:
        self.logs_edit.append(line)

    def cancel_processing(self) -> None:
        if self._worker is None:
            return
        self.statusBar().showMessage("Cancelando ejecucion...")
        self._worker.cancel()

    def _preview_command(self, request: ProcessingRequest) -> list[str]:
        shared_root = request.mpi_shared_root.strip() if request.use_mpi else ""

        def to_shared(path: str) -> str:
            if not request.use_mpi:
                return path
            if path.startswith(shared_root + "/"):
                return path
            if path.startswith("/home/") and "/shared/" in path:
                suffix = path[path.find("/shared/") + len("/shared/"):]
                return f"{shared_root.rstrip('/')}/{suffix}"
            if path.startswith(str(PROJECT_ROOT) + "/"):
                suffix = path[len(str(PROJECT_ROOT)):].lstrip("/")
                return f"{shared_root.rstrip('/')}/{suffix}"
            if path.startswith(str(PROJECT_ROOT.parent) + "/"):
                suffix = path[len(str(PROJECT_ROOT.parent)):].lstrip("/")
                return f"{shared_root.rstrip('/')}/{suffix}"
            if path.startswith("/"):
                return path
            return f"{shared_root.rstrip('/')}/{path.lstrip('./')}"

        executable = f"{shared_root.rstrip('/')}/c_backend/bin/para_image_parra_mpi" if request.use_mpi else "c_backend/bin/para_image_parra"
        backend = [
            executable,
            "--output",
            to_shared(request.output_dir),
            "--filters",
            ",".join(request.filters),
        ]
        if request.kernel_gray is not None:
            backend.extend(["--kernel-gray", str(request.kernel_gray)])
        if request.kernel_color is not None:
            backend.extend(["--kernel-color", str(request.kernel_color)])
        backend.extend([to_shared(path) for path in request.image_paths])
        if not request.use_mpi:
            return backend

        command = ["mpirun"]
        if request.mpi_oversubscribe:
            command.append("--oversubscribe")
        command.extend(["-np", str(request.mpi_processes), "--hostfile", to_shared(request.mpi_hostfile)])
        if request.mpi_map_by.strip():
            command.extend(["--map-by", request.mpi_map_by.strip()])
        return command + backend

    def _build_node_summary(self, stdout: str, total_time: float) -> str:
        dispatch_re = re.compile(r"^DISPATCH\s+source_rank=(\d+)\s+source_node=(\S+)\s+target_rank=(\d+)\s+target_node=(\S+)\s+image=(\S+)\s+filter=(\S+)\s+output=(\S+)")
        complete_re = re.compile(r"^COMPLETE\s+rank=(\d+)\s+node=(\S+)\s+(.+)$")
        seconds_re = re.compile(r"seconds=([0-9]*\.?[0-9]+)")
        image_re = re.compile(r"image=([^\t\s]+)")
        filter_re = re.compile(r"filter=([^\t\s]+)")

        by_node: dict[str, dict[str, object]] = {}

        def get_node_entry(node: str) -> dict[str, object]:
            if node not in by_node:
                by_node[node] = {
                    "rank": "?",
                    "sent": 0,
                    "completed": 0,
                    "errors": 0,
                    "seconds": 0.0,
                    "images": set(),
                    "filters": set(),
                }
            return by_node[node]

        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            d = dispatch_re.match(line)
            if d:
                target_rank = d.group(3)
                target_node = d.group(4)
                image = Path(d.group(5)).name
                flt = d.group(6)
                entry = get_node_entry(target_node)
                entry["rank"] = target_rank
                entry["sent"] = int(entry["sent"]) + 1
                cast_images = entry["images"]
                cast_filters = entry["filters"]
                if isinstance(cast_images, set):
                    cast_images.add(image)
                if isinstance(cast_filters, set):
                    cast_filters.add(flt)
                continue

            c = complete_re.match(line)
            if c:
                rank = c.group(1)
                node = c.group(2)
                payload = c.group(3)
                entry = get_node_entry(node)
                entry["rank"] = rank
                entry["completed"] = int(entry["completed"]) + 1
                if "\tERR\t" in payload or payload.startswith("ERR\t"):
                    entry["errors"] = int(entry["errors"]) + 1

                s = seconds_re.search(payload)
                if s:
                    entry["seconds"] = float(entry["seconds"]) + float(s.group(1))

                im = image_re.search(payload)
                flt = filter_re.search(payload)
                cast_images = entry["images"]
                cast_filters = entry["filters"]
                if im and isinstance(cast_images, set):
                    cast_images.add(Path(im.group(1)).name)
                if flt and isinstance(cast_filters, set):
                    cast_filters.add(flt.group(1))

        if not by_node:
            return "No se detectaron lineas DISPATCH/COMPLETE en la salida del backend."

        lines: list[str] = []
        lines.append(f"Tiempo total del cluster: {total_time:.3f} s")
        lines.append("")

        def rank_sort_value(value: object) -> int:
            text = str(value)
            return int(text) if text.isdigit() else 9999

        ordered_nodes = sorted(by_node.items(), key=lambda item: (rank_sort_value(item[1]["rank"]), item[0]))
        for node, stats in ordered_nodes:
            images = sorted(stats["images"]) if isinstance(stats["images"], set) else []
            filters = sorted(stats["filters"]) if isinstance(stats["filters"], set) else []
            lines.append(f"Nodo {node} (rank {stats['rank']})")
            lines.append(f"- Tareas enviadas: {stats['sent']}")
            lines.append(f"- Tareas completadas: {stats['completed']}")
            lines.append(f"- Errores: {stats['errors']}")
            lines.append(f"- Tiempo acumulado de tareas: {float(stats['seconds']):.3f} s")
            lines.append(f"- Imagenes: {', '.join(images) if images else '(sin datos)'}")
            lines.append(f"- Filtros: {', '.join(filters) if filters else '(sin datos)'}")
            lines.append("")

        return "\n".join(lines).strip()

    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def show_about_dialog(self) -> None:
        dialog = AboutDialog(self)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

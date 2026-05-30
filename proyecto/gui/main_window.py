"""
main_window.py 
==========================================================
Que hace:
Contiene toda la lógica de la interfaz gráfica:
    - Carga y validación de imágenes BMP (drag & drop y selector).
    - Selección de filtros y configuración de kernels de desenfoque.
    - Construcción del ProcessingRequest y ejecución del backend C en un hilo separado.
    - Presentación del resultado (tiempo de ejecución, errores).

Relación con otros módulos:
    - Importa ProcessingRequest, BackendRunner y ProcessingResult de core/backend_wrapper.py.
    - Importa AboutDialog de gui/about_dialog.py.
    - El backend C es invocado indirectamente a través de BackendRunner dentro de ProcessingWorker.
"""

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
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.backend_wrapper import (
    BACKEND_DIR,
    BackendError,
    BackendRunner,
    ProcessingRequest,
    ProcessingResult,
)
from gui.about_dialog import AboutDialog


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MACHINEFILE = Path("/shared/proyecto/hosts")
DEFAULT_SHARED_ROOT = "/shared/proyecto"
MAX_IMAGES = 10

# Mapeo código de filtro → etiqueta visible en la UI.
# El código (clave) es exactamente el argumento que recibirá el backend C
# en su flag --filters (p.ej. --filters vg,hc,dg).
FILTER_LABELS = {
    "vg": "Inversion vertical gris",
    "vc": "Inversion vertical color",
    "hg": "Espejo horizontal gris",
    "hc": "Espejo horizontal color",
    "dg": "Desenfoque gris",
    "dc": "Desenfoque color",
}


# ===========================================================================
# DropArea — zona de arrastrar y soltar archivos BMP
# ===========================================================================

class DropArea(QFrame):
    """
    Widget de área de arrastre (drag & drop) para archivos BMP.

    Emite la señal files_dropped con la lista de rutas locales cada vez
    que el usuario suelta archivos sobre el área. La señal está conectada
    a MainWindow.add_images() para validar y registrar las imágenes.

    Objeto en la UI:
        Panel izquierdo, zona central con borde punteado verde.
        Muestra el texto "Arrastra hasta 10 imagenes BMP aqui".

    Señales:
        files_dropped(list[str]): Emitida en dropEvent con las rutas de los
                                  archivos soltados. Solo incluye archivos
                                  locales (descarta URLs remotas).
    """

    files_dropped = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropArea")     # referenciado en styles.qss

        label = QLabel("Arrastra hasta 10 imagenes BMP aqui\n\no usa el boton Agregar BMP")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        layout.addWidget(label)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        """Acepta la operación de arrastre solo si contiene URLs (archivos)."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        """Mantiene la aceptación mientras el usuario arrastra sobre el área."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        """
        Maneja el evento de soltar archivos.

        Extrae las rutas locales de las URLs del evento MIME y emite
        files_dropped con esa lista. La validación (.bmp, duplicados,
        límite) ocurre en MainWindow.add_images(), no aquí.

        Flujo de datos:
            event.mimeData().urls()     → lista de QUrl
            url.toLocalFile()           → str con ruta del sistema de archivos
            self.files_dropped.emit()   → dispara MainWindow.add_images(paths)
        """
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        self.files_dropped.emit(paths)
        event.acceptProposedAction()


# ===========================================================================
# ProcessingWorker — ejecuta el backend C en un hilo separado
# ===========================================================================

class ProcessingWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log_line = Signal(str)

    def __init__(self, request: ProcessingRequest) -> None:
        """
        Args:
            request: Solicitud de procesamiento construida por MainWindow.validate_request().
        """
        super().__init__()
        self.request = request
        self.runner = BackendRunner()

    def run(self) -> None:
        """
        Método ejecutado por el QThread al iniciar.

        Llama BackendRunner.run(self.request) y emite finished o failed
        según el resultado. No lanza excepciones: las captura todas y las
        convierte en la señal failed para que la GUI las muestre.
        """
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


# ===========================================================================
# ValidationResult — resultado interno de validate_request()
# ===========================================================================

@dataclass(slots=True)
class ValidationResult:
    """
    Resultado de MainWindow.validate_request().

    Si la validación pasó, request contiene el ProcessingRequest listo
    para enviarse al backend. Si falló, request es None y error contiene
    el mensaje a mostrar al usuario.

    Atributos:
        request: ProcessingRequest construido, o None si hay error.
        error:   Mensaje de error para QMessageBox, o None si todo está bien.
    """
    request: ProcessingRequest | None
    error: str | None = None


# ===========================================================================
# MainWindow — ventana principal
# ===========================================================================

class MainWindow(QMainWindow):
    """
    Ventana principal de BMP Parallel Studio.

    Construye y gestiona todos los widgets de la interfaz, coordina
    la validación de entradas y orquesta la ejecución del backend C.

    Atributos de estado interno:
        image_paths (list[str]):          Lista maestra de rutas absolutas de los BMP
                                          cargados. Es la fuente de verdad; el QListWidget
                                          es solo su representación visual.
        filter_checkboxes (dict[str, QCheckBox]): Mapeo código→checkbox para los 6 filtros.
        _thread (QThread | None):         Hilo de procesamiento activo (None si está inactivo).
        _worker (ProcessingWorker | None): Worker activo dentro del hilo.

    Layout:
        Panel izquierdo (proporción 5): carga de imágenes.
        Panel derecho (proporción 4): opciones de procesamiento y ejecución.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BMP Parallel Studio")
        self.resize(1100, 650)
        self.setMinimumSize(760, 520)

        self._thread: QThread | None = None
        self._worker: ProcessingWorker | None = None
        self.image_paths: list[str] = []
        self.filter_checkboxes: dict[str, QCheckBox] = {}

        # Crea la carpeta de salida por defecto si no existe
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self._build_menu()
        self._build_ui()
        # Muestra en la barra de estado la ruta donde se espera el ejecutable C
        self.statusBar().showMessage(f"Backend esperado en: {BACKEND_DIR}")

    # -----------------------------------------------------------------------
    # Construcción de la interfaz
    # -----------------------------------------------------------------------

    def _build_menu(self) -> None:
        """
        Construye la barra de menú con la opción Ayuda > Acerca de.
        Al activarla se invoca show_about_dialog().
        """
        about_action = QAction("Acerca de", self)
        about_action.triggered.connect(self.show_about_dialog)
        self.menuBar().addMenu("Ayuda").addAction(about_action)

    def _build_ui(self) -> None:
        """
        Construye el layout raíz vertical con:
        - Header superior: logo del tec y botón Acerca de
        - Contenedor principal: los dos paneles (imágenes y opciones)
        """
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(14)

        # Header con logo y botón Acerca de
        header_layout = QHBoxLayout()
        header_layout.setSpacing(14)
        
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = PROJECT_ROOT / "logo_tec.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                logo_label.setPixmap(pixmap.scaledToWidth(160, Qt.SmoothTransformation))
        
        header_layout.addWidget(logo_label)
        header_layout.addStretch()
        
        about_button = QPushButton("Acerca de")
        about_button.clicked.connect(self.show_about_dialog)
        header_layout.addWidget(about_button)
        
        main_layout.addLayout(header_layout)
        
        # Contenedor con los dos paneles principales
        panels_container = QWidget()
        panels_layout = QHBoxLayout(panels_container)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.setSpacing(12)
        panels_layout.addWidget(self._build_left_panel(), 3)
        panels_layout.addWidget(self._build_right_panel(), 5)
        
        main_layout.addWidget(panels_container)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(container)
        self.setCentralWidget(scroll_area)

    def _build_left_panel(self) -> QWidget:
        """
        Construye el panel izquierdo de carga de imágenes.

        Widgets creados (en orden vertical):
            - Título "Imagenes BMP" (QLabel, objectName="sectionTitle")
            - Subtítulo descriptivo (QLabel, objectName="mutedLabel")
            - DropArea: zona de drag & drop, señal files_dropped → add_images()
            - Botón "Agregar BMP"    → select_images()
            - Botón "Quitar seleccion" → remove_selected_images()
            - Botón "Limpiar lista"  → clear_images()
            - image_list (QListWidget): muestra nombre y ruta de cada BMP cargado.
                                        Modo de selección ExtendedSelection para
                                        permitir eliminar múltiples ítems a la vez.
            - loaded_count_label (QLabel): muestra "N / 10 imagenes cargadas".
                                           Se actualiza en refresh_image_list().

        Returns:
            QFrame con objectName="card" (estilizado en styles.qss).
        """
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(14)

        title = QLabel("Imagenes BMP")
        title.setObjectName("sectionTitle")

        subtitle = QLabel("Carga archivos BMP con arrastrar y soltar o con el selector.")
        subtitle.setObjectName("mutedLabel")

        # DropArea: conecta su señal files_dropped a add_images()
        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.add_images)

        # Fila de botones de gestión de imágenes
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

        # Lista de imágenes cargadas
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list.setAlternatingRowColors(True)
        self.image_list.setMaximumHeight(280)

        # Contador de imágenes
        self.loaded_count_label = QLabel("0 / 10 imagenes cargadas")
        self.loaded_count_label.setObjectName("mutedLabel")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.drop_area, 0)
        layout.addLayout(buttons)
        layout.addWidget(self.image_list, 0)
        layout.addWidget(self.loaded_count_label)
        layout.addStretch()
        return panel

    def _build_right_panel(self) -> QWidget:
        """
        Construye el panel derecho de opciones de procesamiento.

        Grupos y widgets creados:
            Grupo "Transformaciones":
                - all_checkbox (QCheckBox "Todas")  → toggle_all_filters()
                - 6 checkboxes individuales en grid 2×3 (vg, vc, hg, hc, dg, dc)
                  Cada uno → sync_all_checkbox()

            Grupo "Kernels de desenfoque":
                - kernel_gray_spin  (QSpinBox, rango 1-999, paso 2, valor inicial 27)
                  Controla el kernel para el filtro "dg" (desenfoque gris).
                - kernel_color_spin (QSpinBox, misma configuración)
                  Controla el kernel para el filtro "dc" (desenfoque color).

            Grupo "Salida y resultado":
                - output_dir_edit (QLineEdit, readOnly) muestra la carpeta de salida activa.
                - Botón "Elegir carpeta" → select_output_dir()
                - execution_time_edit (QLineEdit, readOnly) muestra el tiempo al terminar.

            Fila de acciones:
                - execute_button (QPushButton "Ejecutar", objectName="primaryButton")
                  → start_processing()
                - Botón "Acerca de" → show_about_dialog()

        Returns:
            QFrame con objectName="card".
        """
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)

        title = QLabel("Opciones de procesamiento")
        title.setObjectName("sectionTitle")

        # --- Grupo de transformaciones ---
        transform_group = QGroupBox("Transformaciones")
        transform_layout = QGridLayout(transform_group)

        # Checkbox "Todas": marca/desmarca todos los filtros a la vez
        self.all_checkbox = QCheckBox("Todas")
        self.all_checkbox.toggled.connect(self.toggle_all_filters)
        transform_layout.addWidget(self.all_checkbox, 0, 0, 1, 2)

        # Checkboxes individuales: se crean dinámicamente desde FILTER_LABELS
        for index, (filter_code, label) in enumerate(FILTER_LABELS.items(), start=1):
            checkbox = QCheckBox(label)
            checkbox.toggled.connect(self.sync_all_checkbox)
            self.filter_checkboxes[filter_code] = checkbox
            row = (index + 1) // 2
            column = (index - 1) % 2
            transform_layout.addWidget(checkbox, row, column)

        # --- Grupo de kernels ---
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
        self.mpi_auto_detect_checkbox = QCheckBox("Auto-detectar nodos disponibles")
        self.mpi_auto_detect_checkbox.setChecked(True)
        self.mpi_oversubscribe_checkbox = QCheckBox("Activar oversubscribe")
        self.mpi_oversubscribe_checkbox.setChecked(True)
        self.mpi_map_by_edit = QLineEdit("node")
        mpi_layout.addRow("Modo:", self.use_mpi_checkbox)
        mpi_layout.addRow("Procesos:", self.mpi_processes_spin)
        mpi_layout.addRow("Hostfile:", self.mpi_hostfile_edit)
        mpi_layout.addRow("Shared root:", self.mpi_shared_root_edit)
        mpi_layout.addRow("Deteccion:", self.mpi_auto_detect_checkbox)
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
        self.logs_edit.setMinimumHeight(120)
        self.logs_edit.setPlaceholderText("Aqui se muestran logs DISPATCH/COMPLETE del backend MPI.")
        self.summary_edit = QTextEdit()
        self.summary_edit.setReadOnly(True)
        self.summary_edit.setMinimumHeight(120)
        self.summary_edit.setPlaceholderText("Resumen por nodo (tareas, imagenes, tiempo, errores).")
        output_layout.addRow("Logs:", self.logs_edit)
        output_layout.addRow("Resumen por nodo:", self.summary_edit)

        # --- Fila de ejecución ---
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
        """
        Crea y configura un QSpinBox para el tamaño del kernel de desenfoque.

        Configuración:
            - Rango: 1 a 999.
            - Paso: 2 (para facilitar mantener valores impares).
            - Valor inicial: 27.

        El valor debe ser un entero positivo e impar para que el algoritmo
        de box blur en el backend C pueda calcular k = kernel_size // 2
        de forma simétrica.

        Returns:
            QSpinBox configurado.
        """
        spin = QSpinBox()
        spin.setRange(1, 999)
        spin.setSingleStep(2)
        spin.setValue(27)
        return spin

    def _wrap_layout(self, layout) -> QWidget:
        """
        Envuelve un QLayout en un QWidget.
        Necesario para insertar un layout en un QFormLayout como widget de campo.
        """
        wrapper = QWidget()
        wrapper.setLayout(layout)
        return wrapper

    # -----------------------------------------------------------------------
    # Gestión de imágenes
    # -----------------------------------------------------------------------

    def select_images(self) -> None:
        """
        Abre el selector de archivos nativo del sistema operativo filtrado a *.bmp.

        Invocado por: botón "Agregar BMP" (clicked).
        Resultado: llama add_images() con la lista de archivos seleccionados.
        """
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar imagenes BMP",
            str(PROJECT_ROOT),
            "Imagenes BMP (*.bmp)",
        )
        if files:
            self.add_images(files)

    def add_images(self, paths: list[str]) -> None:
        """
        Valida y agrega rutas BMP a la lista maestra self.image_paths.

        Es el destino común tanto del drag & drop (DropArea.files_dropped)
        como del selector de archivos (select_images).

        Validaciones en orden:
            1. Normaliza todas las rutas a absolutas con Path.resolve().
            2. Rechaza cualquier archivo que no tenga extensión .bmp (case-insensitive).
               Muestra QMessageBox.warning y aborta si hay al menos uno inválido.
            3. Filtra duplicados (rutas ya presentes en self.image_paths).
            4. Verifica que el total no supere MAX_IMAGES (10).
               Muestra QMessageBox.warning y aborta si se excedería el límite.
            5. Extiende self.image_paths con los nuevos paths únicos.
            6. Llama refresh_image_list() para actualizar el QListWidget y el contador.

        Args:
            paths: Lista de rutas (pueden ser relativas o absolutas, se normalizan aquí).
        """
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
        """
        Elimina de self.image_paths las imágenes seleccionadas en image_list.

        Invocado por: botón "Quitar seleccion" (clicked).

        La ruta de cada ítem se recupera de su metadata (Qt.UserRole),
        guardada en refresh_image_list(). Tras eliminar, actualiza la lista.
        """
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            return

        selected_paths = {item.data(Qt.UserRole) for item in selected_items}
        self.image_paths = [path for path in self.image_paths if path not in selected_paths]
        self.refresh_image_list()

    def clear_images(self) -> None:
        """
        Vacía completamente self.image_paths y actualiza la lista visual.

        Invocado por: botón "Limpiar lista" (clicked).
        """
        self.image_paths.clear()
        self.refresh_image_list()

    def refresh_image_list(self) -> None:
        """
        Sincroniza el QListWidget image_list con el estado actual de self.image_paths.

        Para cada ruta en image_list crea un QListWidgetItem que muestra:
            - Línea 1: nombre del archivo (Path.name).
            - Línea 2: ruta absoluta completa.
        La ruta completa se guarda en el rol Qt.UserRole del ítem para que
        remove_selected_images() pueda recuperarla sin parsear el texto.

        También actualiza loaded_count_label con el conteo actual (N / 10).
        """
        self.image_list.clear()
        for path in self.image_paths:
            item = QListWidgetItem(f"{Path(path).name}\n{path}")
            item.setData(Qt.UserRole, path)     # metadato: ruta completa
            self.image_list.addItem(item)

        self.loaded_count_label.setText(f"{len(self.image_paths)} / {MAX_IMAGES} imagenes cargadas")

    # -----------------------------------------------------------------------
    # Gestión de la carpeta de salida
    # -----------------------------------------------------------------------

    def select_output_dir(self) -> None:
        """
        Abre el selector de carpetas nativo y actualiza output_dir_edit.

        Invocado por: botón "Elegir carpeta" (clicked).
        El directorio inicial del diálogo es el valor actual de output_dir_edit,
        o DEFAULT_OUTPUT_DIR si el campo está vacío.
        """
        directory = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta de salida",
            self.output_dir_edit.text() or str(DEFAULT_OUTPUT_DIR),
        )
        if directory:
            self.output_dir_edit.setText(directory)

    # -----------------------------------------------------------------------
    # Gestión de filtros
    # -----------------------------------------------------------------------

    def toggle_all_filters(self, checked: bool) -> None:
        """
        Marca o desmarca todos los checkboxes de filtros individuales.

        Invocado por: all_checkbox.toggled (señal booleana).
        Usa blockSignals para evitar que cada checkbox dispare sync_all_checkbox
        y cause un bucle de señales.

        Args:
            checked: True para marcar todos, False para desmarcar todos.
        """
        for checkbox in self.filter_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

    def sync_all_checkbox(self) -> None:
        """
        Sincroniza all_checkbox con el estado de los checkboxes individuales.

        Invocado por: toggled de cualquier checkbox individual.
        Marca all_checkbox solo si los 6 filtros están activos; lo desmarca
        en cualquier otro caso. Usa blockSignals para no disparar toggle_all_filters.
        """
        all_selected = all(checkbox.isChecked() for checkbox in self.filter_checkboxes.values())
        self.all_checkbox.blockSignals(True)
        self.all_checkbox.setChecked(all_selected)
        self.all_checkbox.blockSignals(False)

    def selected_filters(self) -> list[str]:
        """
        Devuelve la lista de códigos de los filtros actualmente seleccionados.

        Itera filter_checkboxes e incluye la clave solo si el checkbox está marcado.
        La clave es el código exacto que el backend C espera (p.ej. "vg", "hc").

        Returns:
            Lista de códigos, p.ej. ["vg", "hg", "dc"]. Vacía si ninguno está marcado.
        """
        return [code for code, checkbox in self.filter_checkboxes.items() if checkbox.isChecked()]

    # -----------------------------------------------------------------------
    # Validación y ejecución
    # -----------------------------------------------------------------------

    def validate_request(self) -> ValidationResult:
        """
        Valida los valores actuales de la UI y construye un ProcessingRequest.

        Validaciones en orden:
            1. Al menos una imagen cargada en self.image_paths.
            2. No más de MAX_IMAGES (10) imágenes.
            3. Al menos un filtro seleccionado.
            4. Si "dg" está seleccionado, kernel_gray_spin debe ser positivo e impar.
            5. Si "dc" está seleccionado, kernel_color_spin debe ser positivo e impar.
            6. output_dir_edit no debe estar vacío.

        Si todas las validaciones pasan, construye y devuelve un ProcessingRequest
        con los valores actuales de todos los widgets relevantes.

        Returns:
            ValidationResult con request=ProcessingRequest si todo es válido,
            o request=None y error=<mensaje> si alguna validación falla.
        """
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

        # Construye el ProcessingRequest con los valores actuales de la UI:
        #   image_paths  ← self.image_paths (lista maestra)
        #   output_dir   ← output_dir_edit.text()
        #   filters      ← selected_filters() → códigos de checkboxes activos
        #   kernel_gray  ← kernel_gray_spin.value()  (solo si "dg" está activo)
        #   kernel_color ← kernel_color_spin.value() (solo si "dc" está activo)
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
            mpi_auto_detect_nodes=self.mpi_auto_detect_checkbox.isChecked(),
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
        """
        Verifica que un valor de kernel sea positivo e impar.

        El backend C necesita kernels impares para que el box blur tenga
        un centro simétrico (k = kernel_size // 2 píxeles a cada lado).

        Args:
            value: Valor entero del QSpinBox a validar.

        Returns:
            True si value > 0 y value es impar.
        """
        return value > 0 and value % 2 == 1

    def start_processing(self) -> None:
        """
        Punto de entrada del botón "Ejecutar". Orquesta la ejecución del backend.

        Flujo:
            1. Llama validate_request(); si hay error, muestra QMessageBox y retorna.
            2. Deshabilita execute_button para evitar ejecuciones simultáneas.
            3. Actualiza execution_time_edit a "Procesando...".
            4. Crea un QThread y mueve un ProcessingWorker a él.
            5. Conecta señales:
                   _thread.started       → _worker.run
                   _worker.finished      → on_processing_finished
                   _worker.failed        → on_processing_failed
                   _worker.finished/failed → _thread.quit
                   _thread.finished      → _cleanup_thread
            6. Inicia el hilo.

        El backend C corre completamente en el hilo separado, por lo que
        la GUI permanece responsiva durante el procesamiento.

        Invocado por: execute_button.clicked.
        """
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
        """
        Callback invocado cuando el backend C termina con éxito.

        Actualiza la UI con los resultados y muestra un diálogo informativo.

        Args:
            result: ProcessingResult con el tiempo de ejecución y la carpeta de salida.
                    Viene de BackendRunner.run() vía ProcessingWorker.finished.
        """
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
        """
        Callback invocado cuando el backend C falla o lanza BackendError.

        Reactiva el botón, limpia el campo de tiempo y muestra el error.

        Args:
            error_message: Mensaje de error proveniente de BackendError o excepción inesperada.
                           Viene de ProcessingWorker.failed.
        """
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
        image_assign_re = re.compile(r"^IMAGE_ASSIGN\s+rank=(\d+)\s+node=(\S+)\s+image=(\S+)\s+filters=(\S*)")
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
                    "assigned_images": 0,
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

            ia = image_assign_re.match(line)
            if ia:
                rank = ia.group(1)
                node = ia.group(2)
                image = Path(ia.group(3)).name
                filters_text = ia.group(4)
                entry = get_node_entry(node)
                entry["rank"] = rank
                entry["assigned_images"] = int(entry["assigned_images"]) + 1
                cast_images = entry["images"]
                cast_filters = entry["filters"]
                if isinstance(cast_images, set):
                    cast_images.add(image)
                if filters_text and isinstance(cast_filters, set):
                    for flt in filters_text.split(","):
                        if flt:
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
            lines.append(f"- Imagenes asignadas: {stats['assigned_images']}")
            lines.append(f"- Tareas enviadas (legacy): {stats['sent']}")
            lines.append(f"- Tareas completadas: {stats['completed']}")
            lines.append(f"- Errores: {stats['errors']}")
            lines.append(f"- Suma de tiempos de tareas: {float(stats['seconds']):.3f} s")
            avg = float(stats['seconds']) / int(stats['completed']) if int(stats['completed']) > 0 else 0.0
            lines.append(f"- Promedio por tarea: {avg:.3f} s")
            lines.append(f"- Imagenes: {', '.join(images) if images else '(sin datos)'}")
            lines.append(f"- Filtros: {', '.join(filters) if filters else '(sin datos)'}")
            lines.append("")

        return "\n".join(lines).strip()

    def _cleanup_thread(self) -> None:
        """
        Libera los recursos del hilo y el worker al finalizar.

        Invocado por: _thread.finished (señal Qt).
        Llama deleteLater() para que Qt destruya los objetos de forma segura
        desde el hilo principal, evitando accesos a memoria ya liberada.
        """
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def show_about_dialog(self) -> None:
        """
        Abre el diálogo modal "Acerca de" con información del equipo.

        Invocado por: botón "Acerca de" y menú Ayuda > Acerca de.
        El diálogo bloquea la interacción con MainWindow hasta cerrarse.
        """
        dialog = AboutDialog(self)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

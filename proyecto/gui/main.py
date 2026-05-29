"""
main.py 
==================================================
Que hace: 
    - Inicializa la aplicación Qt (QApplication).
    - Carga la hoja de estilos (styles.qss).
    - Instancia y muestra la ventana principal (MainWindow).

No contiene lógica. 

Uso:
    python gui/main.py          (desde la carpeta proyecto/)
"""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.main_window import MainWindow


def load_stylesheet(app: QApplication) -> None:
    """
    Carga y aplica el archivo de estilos QSS a la aplicación.

    El archivo styles.qss define colores, fuentes, bordes y apariencia
    general de todos los widgets Qt. Si no existe, la app corre con
    el tema por defecto del sistema operativo sin errores.

    Args:
        app: Instancia de QApplication a la que se aplican los estilos.
    """
    stylesheet_path = CURRENT_DIR / "styles.qss"
    if stylesheet_path.exists():
        app.setStyleSheet(stylesheet_path.read_text(encoding="utf-8"))


def main() -> int:
    """
    Función principal la cual orquesta el arranque completo de la aplicación.

    Cómo es el flujo:
        1. Crea QApplication (requerida antes de cualquier widget Qt).
        2. Carga estilos desde styles.qss.
        3. Instancia MainWindow (construye toda la UI internamente).
        4. Muestra la ventana.
        5. Entra al event loop de Qt — bloquea hasta que el usuario cierre.

    Returns:
        Código de salida del proceso (0 = éxito).
    """
    app = QApplication(sys.argv)
    load_stylesheet(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
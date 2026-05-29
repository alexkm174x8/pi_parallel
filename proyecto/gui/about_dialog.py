"""
about_dialog.py 
===================================================
Muestra un modal con la información del equipo y la materia.
Se invoca desde MainWindow a través del botón "Acerca de" o el menú Ayuda.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class AboutDialog(QDialog):
    """
    Diálogo modal que presenta la información institucional del proyecto.

    Elementos visuales:
        - Título: etiqueta grande con estilo "aboutTitle" (definido en styles.qss).
        - Cuerpo: etiqueta de texto plano con materia, campus, fecha y equipo.
        - Botón OK: cierra el diálogo.

    Se abre con dialog.exec() desde MainWindow.show_about_dialog(),
    lo que bloquea la ventana principal hasta que el usuario lo cierre.
    """

    def __init__(self, parent=None) -> None:
        """
        Construye y configura el diálogo.

        Args:
            parent: Widget padre (MainWindow). Qt usa esto para centrar
                    el diálogo sobre la ventana principal.
        """
        super().__init__(parent)
        self.setObjectName("aboutDialog")       # referenciado en styles.qss
        self.setWindowTitle("Acerca de:")
        self.setModal(True)                     # bloquea interacción con la ventana padre
        self.resize(460, 380)

        # --- Título ---
        title = QLabel("Acerca de")
        title.setObjectName("aboutTitle")       # estilo especial en styles.qss

        # --- Cuerpo con información del equipo ---
        body = QLabel(
            "TC3003B\n"
            "Tecnológico de Monterrey\n"
            "Campus Puebla\n"
            "Abril 2026\n"
            "Equipo:\n"
            "Estefanía Antonio Villaseca A01736897\n"
            "Miranda Eugenia Colorado Arróniz A01737023\n"
            "Alejandro Kong Montoya A01734271\n"
            "Sofía Zugasti Delgado A00837478"
        )
        body.setObjectName("aboutBody")
        body.setWordWrap(True)
        body.setTextFormat(Qt.PlainText)        # evita interpretación HTML accidental

        # --- Logo institucional inferior ---
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = Path(__file__).resolve().parent.parent / "logo_tec.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                logo_label.setPixmap(pixmap.scaledToWidth(260, Qt.SmoothTransformation))

        # --- Botón de cierre ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.setObjectName("aboutButtons")
        buttons.accepted.connect(self.accept)   # OK → cierra y devuelve Accepted

        # --- Layout vertical ---
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addWidget(logo_label)
        layout.addStretch(1)                    # empuja el botón hacia abajo
        layout.addWidget(buttons)

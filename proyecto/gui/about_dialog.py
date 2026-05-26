from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("aboutDialog")
        self.setWindowTitle("Acerca de:")
        self.setModal(True)
        self.resize(420, 240)

        title = QLabel("Acerca de")
        title.setObjectName("aboutTitle")

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
        body.setTextFormat(Qt.PlainText)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.setObjectName("aboutButtons")
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addStretch(1)
        layout.addWidget(buttons)
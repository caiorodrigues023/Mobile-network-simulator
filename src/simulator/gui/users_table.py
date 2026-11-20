# ---------------------------------------------------------------------------
# src/simulator/gui/users_table.py — Tabela Detalhada de Usuários (UEs)
# ---------------------------------------------------------------------------

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QComboBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


class UsersTableView(QWidget):
    """Widget de tabela para inspeção de todos os usuários (UEs) do cenário."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ues_data = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Barra superior de filtros e busca
        top_bar = QHBoxLayout()
        top_bar.setSpacing(10)

        lbl_search = QLabel("🔍 Buscar:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filtrar por UE ID, AP, Canal...")
        self.search_input.textChanged.connect(self._filter_table)

        lbl_ap_filter = QLabel("Filtrar por AP:")
        self.combo_ap = QComboBox()
        self.combo_ap.addItem("Todos os APs")
        self.combo_ap.currentIndexChanged.connect(self._filter_table)

        btn_export = QPushButton("📥 Exportar CSV")
        btn_export.clicked.connect(self._export_csv)

        top_bar.addWidget(lbl_search)
        top_bar.addWidget(self.search_input, stretch=2)
        top_bar.addWidget(lbl_ap_filter)
        top_bar.addWidget(self.combo_ap, stretch=1)
        top_bar.addWidget(btn_export)

        layout.addLayout(top_bar)

        # Tabela principal
        self.table = QTableWidget()
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "UE ID", "Posição (X, Y)", "AP Servidor", "Distância",
            "Azimute", "Feixe", "Canal", "Potência (W)",
            "Potência (dBm)", "SINR (dB)", "Capacidade", "Status"
        ])

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)

        layout.addWidget(self.table)

        # Rodapé com resumo
        self.lbl_summary = QLabel("Nenhum dado carregado.")
        self.lbl_summary.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_summary)

    def update_data(self, ues: list, aps: list):
        """Atualiza a tabela com a lista de dicionários de UEs."""
        self._ues_data = ues

        # Atualiza opções de AP no combo
        self.combo_ap.blockSignals(True)
        self.combo_ap.clear()
        self.combo_ap.addItem("Todos os APs")
        for ap in aps:
            self.combo_ap.addItem(f"AP {ap['id']}")
        self.combo_ap.blockSignals(False)

        self._populate_table(ues)

    def _populate_table(self, ues: list):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(ues))

        for row, u in enumerate(ues):
            items = [
                QTableWidgetItem(f"UE {u['id']}"),
                QTableWidgetItem(f"({u['x']:.0f}, {u['y']:.0f}) m"),
                QTableWidgetItem(f"AP {u['ap_id']}"),
                QTableWidgetItem(f"{u['dist']:.1f} m"),
                QTableWidgetItem(f"{u['angle_deg']:.1f}°"),
                QTableWidgetItem(str(u["beam"])),
                QTableWidgetItem(str(u["channel"])),
                QTableWidgetItem(f"{u['power_w']:.4f}"),
                QTableWidgetItem(f"{u['power_dbm']:.1f}"),
                QTableWidgetItem(f"{u['sinr_db']:+.1f}"),
                QTableWidgetItem(f"{u['cap_mbps']:.2f} Mbps"),
                QTableWidgetItem(u["status"]),
            ]

            # Alinhamento centralizado
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Destaque de cor no SINR
                if col == 9:
                    s_db = u["sinr_db"]
                    if s_db >= 10:
                        item.setForeground(QColor("#34d399"))  # Verde
                    elif s_db >= 0:
                        item.setForeground(QColor("#fbbf24"))  # Amarelo
                    else:
                        item.setForeground(QColor("#f87171"))  # Vermelho

                self.table.setItem(row, col, item)

        self.table.setSortingEnabled(True)

        # Atualiza resumo
        total = len(ues)
        if total > 0:
            avg_sinr = sum(u["sinr_db"] for u in ues) / total
            sum_cap = sum(u["cap_mbps"] for u in ues)
            self.lbl_summary.setText(
                f"Total de {total} Usuários exibidos | SINR Médio: {avg_sinr:+.1f} dB | Capacidade Agregada: {sum_cap:.1f} Mbps"
            )
        else:
            self.lbl_summary.setText("Nenhum usuário no cenário atual.")

    def _filter_table(self):
        query = self.search_input.text().strip().lower()
        selected_ap = self.combo_ap.currentText()

        filtered = []
        for u in self._ues_data:
            if selected_ap != "Todos os APs":
                ap_str = f"AP {u['ap_id']}"
                if ap_str != selected_ap:
                    continue

            if query:
                row_text = f"ue {u['id']} ap {u['ap_id']} canal {u['channel']} feixe {u['beam']} {u['status']}".lower()
                if query not in row_text:
                    continue

            filtered.append(u)

        self._populate_table(filtered)

    def _export_csv(self):
        if not self._ues_data:
            QMessageBox.warning(self, "Exportar", "Não há dados para exportar.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Salvar Usuários em CSV", "usuarios_simulacao.csv", "CSV (*.csv)")
        if not path:
            return

        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "UE_ID", "X_m", "Y_m", "AP_ID", "Distancia_m", "Azimute_deg",
                    "Feixe", "Canal", "Potencia_W", "Potencia_dBm", "SINR_dB",
                    "Capacidade_Mbps", "Status"
                ])
                for u in self._ues_data:
                    writer.writerow([
                        u["id"], u["x"], u["y"], u["ap_id"], u["dist"], u["angle_deg"],
                        u["beam"], u["channel"], u["power_w"], u["power_dbm"],
                        u["sinr_db"], u["cap_mbps"], u["status"]
                    ])
            QMessageBox.information(self, "Sucesso", f"Dados exportados com sucesso em:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao exportar CSV: {str(e)}")

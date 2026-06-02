import logging
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from pymol import cmd
from .pymol_interface import extract_point_cloud, create_pymol_cloud
from .siteomix import full_alignment


logger = logging.getLogger(__name__)

def make_unique_name(base_name: str) -> str:
    existing = cmd.get_object_list()
    name = base_name
    counter = 2
    while name in existing:
        name = f"{base_name}_{counter}"
        counter += 1
    return name
 

BS_DEFAULT_SETTINGS = {
    # PDB interpretation
    "keep_hydrogens": (False, bool),
    "keep_hetatms": (False, bool),
    "keep_waters": (False, bool),
    "resn_water": (("HOH", "WAT", "H2O"), tuple),
    "alternate": ("A", str),
    # Grid setup
    "grid_spacing": (0.6, float),
    "probe_radius": (1.4, float),
    "softness": (0.5, float),
    "cushion": (0.0, float),
    "radii": ("UA", str),
    "radius_factor": (1.0, float),
    # LigSite parameters
    "ligsite_cutoff": (5, int),
    "gap": (0, int),
    "original_ligsite": (False, bool),
    "max_dist": (1.22, float),
    "vol_resol": (3.0, float),
    "min_size": (4, int),
    "max_size": (99999, int),
    "split_files": (False, bool),
    # Shaping
    "shape_dmax": (3.0, float),
    "shape_object": ("sele", str),
}


BS_TOOLTIPS = {
    "keep_hydrogens": "Keep hydrogen atoms",
    "keep_hetatms": "Keep hetero atoms",
    "keep_waters": "Keep water molecules",
    "resn_water": "Residue name for water",
    "alternate": "Alternate conformation (e.g., A, B, '')",
    "grid_spacing": "Grid spacing in Å",
    "probe_radius": "Probe radius in Å",
    "softness": "Softness factor for cavity detection",
    "cushion": "Cushion distance around the protein",
    "radii": "Atom radii set or 'all'",
    "radius_factor": "Scale factor for radii",
}

#-------------------------------------------------------------------------------------

class BsWorker(QThread):
    """Поток для выполнения расчёта BS, чтобы не блокировать GUI."""
    finished = pyqtSignal(object)
    error = pyqtSignal(tuple)
    progress = pyqtSignal(str)

    def __init__(self, pdb_string, settings, parent=None):
        super().__init__(parent)
        self.pdb_string = pdb_string
        self.settings = settings

    def run(self):
        try:
            import numpy as np
            from .cavfind import CavFind, default_settings
            import logging
            logger = logging.getLogger(__name__)

            cavfind = CavFind(name="binding_site", obj_name="")
            merged_settings = default_settings.copy()
            merged_settings.update(self.settings)
            cavfind.settings = merged_settings
            cavfind.struct_from_pdb(self.pdb_string)
            cavfind.detect_binding_site()
            self.finished.emit(cavfind)
        except Exception as e:
            import traceback
            self.error.emit((type(e), e, traceback.format_exc()))

#-------------------------------------------------------------------------------------

class BSTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QtWidgets.QVBoxLayout())

        # --- Верхняя часть: выбор объекта и имя результата ---
        top_layout = QtWidgets.QHBoxLayout()
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(QtWidgets.QLabel("PyMOL Object:"))
        self.object_list = QtWidgets.QListWidget(self)
        self.object_list.setToolTip("Select a protein structure for BS detection")
        self.object_list.setMaximumHeight(120)
        left_layout.addWidget(self.object_list)

        self.name_label = QtWidgets.QLabel("Result object name:")
        self.name_edit = QtWidgets.QLineEdit(self)
        self.name_edit.setPlaceholderText("auto-generated")
        left_layout.addWidget(self.name_label)
        left_layout.addWidget(self.name_edit)

        top_layout.addLayout(left_layout)

        # --- Кнопки управления ---
        btn_layout = QtWidgets.QVBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh List")
        self.refresh_btn.setToolTip("Update object list from PyMOL")
        self.run_btn = QtWidgets.QPushButton("Detect Binding Site")
        self.run_btn.setToolTip("Start BS detection")
        self.settings_btn = QtWidgets.QPushButton("Edit Settings...")
        self.settings_btn.setToolTip("Open settings dialog")
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_label.setWordWrap(True)

        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.settings_btn)
        btn_layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))
        btn_layout.addWidget(self.progress_label)

        top_layout.addLayout(btn_layout)
        self.layout().addLayout(top_layout)

        # --- Хранение настроек (текущих) ---
        self.current_settings = {k: v[0] for k, v in BS_DEFAULT_SETTINGS.items()}

        # Инициализация состояния
        self.refresh_object_list()
        self.run_btn.setEnabled(False)

        # Сигналы
        self.refresh_btn.clicked.connect(self.refresh_object_list)
        self.object_list.currentItemChanged.connect(self.on_object_selected)
        self.run_btn.clicked.connect(self.run_detection)
        self.settings_btn.clicked.connect(self.open_settings_dialog)

    def refresh_object_list(self):
        """Заполнить список объектами из PyMOL."""
        self.object_list.clear()
        objects = cmd.get_object_list()
        self.object_list.addItems(objects)
        if self.object_list.count() > 0:
            self.object_list.setCurrentRow(0)
        else:
            self.run_btn.setEnabled(False)
            self.name_edit.clear()

    def on_object_selected(self, item):
        if item is not None:
            self.run_btn.setEnabled(True)
            obj_name = item.text()            
            self.name_edit.setText(f"{obj_name}_bs")
        else:
            self.run_btn.setEnabled(False)

    def open_settings_dialog(self):
        """Открыть диалоговое окно для редактирования параметров."""
        dlg = BsSettingsDialog(self.current_settings, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.current_settings = dlg.get_settings()
            logger.info("BS settings updated")

    def run_detection(self):
        """Запуск обнаружения сайта связывания в отдельном потоке."""
        if not self.object_list.currentItem():
            return
        obj = self.object_list.currentItem().text()
        if obj.startswith("(") and obj.endswith(")"):
            obj = obj[1:-1]   # если выделение, убрать скобки

        # Имя результата
        base_name = self.name_edit.text().strip()
        if not base_name:
            base_name = f"{obj}_bs"
        result_name = make_unique_name(base_name)
        self.name_edit.setText(result_name)

        logger.info(f"Starting BS detection on object '{obj}' -> result '{result_name}'")
        logger.info(f"Settings: {self.current_settings}")

        # Получаем PDB-строку из PyMOL 
        try:
            pdb_str = get_pdbstr(obj)   # см. ниже реализацию
        except Exception as e:
            logger.error(f"Failed to get structure: {e}")
            return

        # Блокируем кнопки
        self.run_btn.setEnabled(False)
        self.progress_label.setText("Running...")

        # Запускаем поток
        self.worker = BsWorker(pdb_str, self.current_settings)
        self.worker.finished.connect(self.on_bs_finished)
        self.worker.error.connect(self.on_bs_error)
        self.worker.start()

    def on_bs_finished(self, cavfind):
        """После успешной детекции открываем диалог выбора полостей."""
        result_name = self.name_edit.text().strip()
        if not result_name:
            result_name = f"{cavfind.name}_cav"

        # Открываем диалог выбора полостей
        dlg = CavitySelectionDialog(cavfind, result_name, self)
        dlg.exec_()   # модальный диалог

        self.progress_label.setText("Finished")
        self.run_btn.setEnabled(True)
        self.refresh_object_list()
    
    def on_bs_error(self, error_info):
        """Обработка ошибки в потоке."""
        exc_type, exc_val, tb = error_info
        logger.error(f"BS detection failed: {exc_type.__name__}: {exc_val}")
        logger.error(tb)
        self.progress_label.setText("Error – see log")
        self.run_btn.setEnabled(True)



def get_pdbstr(obj_name: str) -> str:
    pdb_str = cmd.get_pdbstr(obj_name)
    if not pdb_str or not pdb_str.strip():
        raise ValueError(f"Object '{obj_name}' returned empty PDB string")
    return pdb_str

#-------------------------------------------------------------------------------------

class BsSettingsDialog(QtWidgets.QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BS Detection Settings")
        self.settings = settings.copy()
        self.widgets = {}

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        for key, (default_val, typ) in BS_DEFAULT_SETTINGS.items():
            if typ == bool:
                w = QtWidgets.QCheckBox()
                w.setChecked(self.settings.get(key, default_val))
            elif typ == int:
                w = QtWidgets.QSpinBox()
                w.setRange(-1000000, 1000000)        # подберите диапазон под свои параметры
                w.setValue(int(self.settings.get(key, default_val)))
            elif typ == float:
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(3)
                w.setRange(0.0, 1000.0)
                w.setValue(float(self.settings.get(key, default_val)))
            elif typ == tuple:
                w = QtWidgets.QLineEdit()
                default_val = BS_DEFAULT_SETTINGS[key][0]
                w.setText(", ".join(default_val))
            else:  # str
                w = QtWidgets.QLineEdit()
                w.setText(str(self.settings.get(key, default_val)))
            tooltip = BS_TOOLTIPS.get(key, "")
            if tooltip:
                w.setToolTip(tooltip)
            form.addRow(key, w)
            self.widgets[key] = (w, typ)

        layout.addLayout(form)

        # Кнопки OK/Cancel
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_settings(self) -> dict:
        new_settings = {}
        for key, (w, typ) in self.widgets.items():
            if typ == bool:
                new_settings[key] = w.isChecked()
            elif typ == int:
                new_settings[key] = w.value()
            elif typ == float:
                new_settings[key] = w.value()
            elif typ == tuple:
                text = w.text()
                items = [item.strip() for item in text.split(",") if item.strip()]
                new_settings[key] = tuple(items) if items else ()
            else:
                new_settings[key] = w.text()
        return new_settings

#-------------------------------------------------------------------------------------

class CavitySelectionDialog(QtWidgets.QDialog):
    """Диалог выбора полостей для отображения."""
    def __init__(self, cavfind, base_name, parent=None):
        super().__init__(parent)
        self.cavfind = cavfind
        self.base_name = base_name
        self.setWindowTitle("Select Cavities to Display")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        layout = QtWidgets.QVBoxLayout(self)

        # Информационная метка
        info = QtWidgets.QLabel(f"Object: {cavfind.name}\n"
                                f"Total cavities: {len(cavfind.cavities)}")
        layout.addWidget(info)

        # Список полостей с чекбоксами (QListWidget с флагами)
        self.cavity_list = QtWidgets.QListWidget()
        self.cavity_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for i, cav in enumerate(cavfind.cavities):
            item_text = f"Cavity #{i+1}: {cav.size} points, {cav.volume:.0f} Å³"
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, i)   # храним индекс полости
            self.cavity_list.addItem(item)
        layout.addWidget(self.cavity_list)

        # Кнопка "Select all large" (например, > 50 точек)
        self.select_large_btn = QtWidgets.QPushButton("Select large (>100 points)")
        self.select_large_btn.clicked.connect(self.select_large_cavities)
        layout.addWidget(self.select_large_btn)

        # Кнопки OK/Cancel
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.show_selected_cavities)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def select_large_cavities(self):
        """Выбирает все полости с количеством точек > 100."""
        self.cavity_list.clearSelection()
        for i in range(self.cavity_list.count()):
            item = self.cavity_list.item(i)
            cav_index = item.data(QtCore.Qt.UserRole)
            if self.cavfind.cavities[cav_index].size > 100:
                item.setSelected(True)

    def show_selected_cavities(self):
        """Собирает точки из выбранных полостей и создаёт облако."""
        selected_indices = []
        for item in self.cavity_list.selectedItems():
            cav_idx = item.data(QtCore.Qt.UserRole)
            selected_indices.append(cav_idx)

        if not selected_indices:
            QtWidgets.QMessageBox.warning(self, "No selection",
                                          "Please select at least one cavity.")
            return

        # Собираем координаты и оценки LigSite для выбранных полостей
        all_points = []
        all_scores = []
        for idx in selected_indices:
            cav = self.cavfind.cavities[idx]
            all_points.append(cav.coords)
            all_scores.append(cav.annotations["LIG"])

        if all_points:
            import numpy as np
            cloud = np.vstack(all_points)
            scores = np.hstack(all_scores)

            # Генерируем уникальное имя объекта
            from .pymol_interface import create_pymol_cloud
            obj_name = make_unique_name(self.base_name)   # используем вашу функцию уникальности
            create_pymol_cloud(cloud, obj_name, scores)
            logger.info(f"Displayed {len(cloud)} points from {len(selected_indices)} cavities as '{obj_name}'")
        else:
            logger.warning("No points to display")

        self.accept()   # закрываем диалог

#-------------------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Siteomix Plugin")
        self.resize(650, 420)
        self.setGeometry(100, 100, 650, 420)

        # --- Логотип ---
        logo_widget = QtWidgets.QWidget()
        logo_layout = QtWidgets.QHBoxLayout(logo_widget)
        logo_layout.setContentsMargins(0, 0, 0, 0)

        logo_layout.addItem(
            QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        )

        logo_label = QtWidgets.QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "fig/sitomix_logo.png")
        pixmap = QtGui.QPixmap(logo_path)
        if not pixmap.isNull():
            target_height = 80              
            scaled_pixmap = pixmap.scaledToHeight(target_height, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setFixedSize(scaled_pixmap.size())
        else:
            logo_label.setText("Logo not found")
        logo_label.setAlignment(QtCore.Qt.AlignCenter)

        logo_layout.addWidget(logo_label)

        logo_layout.addItem(
            QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        )

        # --- Основной layout ---
        central_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central_widget)
        central_layout.addWidget(logo_widget)

        tabs = QtWidgets.QTabWidget()
        self.bs_tab = BSTab()
        self.run_tab = RunTab()
        self.log_tab = LogTab()
        tabs.addTab(self.bs_tab, "Binding Site")
        tabs.addTab(self.run_tab, "Run Siteomix")
        tabs.addTab(self.log_tab, "Log")
        central_layout.addWidget(tabs)

        self.setCentralWidget(central_widget)

        # --- Цвет фона (пример) ---
        self.setStyleSheet("background-color: #FFFFFF;")  # белый

    def closeEvent(self, event):
        if hasattr(self, '_log_handler'):
            logger.removeHandler(self._log_handler)
        # Сбрасываем глобальную переменную dialog
        from . import reset_dialog
        reset_dialog()
        super().closeEvent(event)

#-------------------------------------------------------------------------------------

class RunTab(QtWidgets.QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ref_combo = QtWidgets.QComboBox(self)
        self.target_combo = QtWidgets.QComboBox(self)
        self.run_btn = QtWidgets.QPushButton("Run", self)
        self.refresh_btn = QtWidgets.QPushButton("Refresh lists", self)
        self.run_btn.setEnabled(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Reference cloud:"))
        layout.addWidget(self.ref_combo)
        layout.addWidget(QtWidgets.QLabel("Target cloud:"))
        layout.addWidget(self.target_combo)
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.run_btn)

        self.ref_combo.currentTextChanged.connect(self.on_ref_selected)
        self.target_combo.currentTextChanged.connect(self.on_target_selected)
        self.run_btn.clicked.connect(self.run_algorithm)
        self.refresh_btn.clicked.connect(self.refresh_object_list)

        self.ref_obj_name = None
        self.target_obj_name = None
        self.refresh_object_list()


    def refresh_object_list(self):
        self.ref_combo.blockSignals(True)
        self.target_combo.blockSignals(True)
        self.ref_combo.clear()
        self.target_combo.clear()

        objects = cmd.get_object_list()
        for obj in objects:
            self.ref_combo.addItem(obj)
            self.target_combo.addItem(obj)

        self.ref_combo.blockSignals(False)
        self.target_combo.blockSignals(False)

        self.ref_obj_name = self.ref_combo.currentText()
        self.target_obj_name = self.target_combo.currentText()
        self.update_run_state()

    def on_ref_selected(self, text):
        self.ref_obj_name = text
        self.update_run_state()

    def on_target_selected(self, text):
        self.target_obj_name = text
        self.update_run_state()

    def update_run_state(self):
        if self.ref_obj_name and self.target_obj_name:
            if self.ref_obj_name != self.target_obj_name:
                self.run_btn.setEnabled(True)
                return
        self.run_btn.setEnabled(False)

    def run_algorithm(self):
        try:
            logger.info(f"Loading reference cloud from '{self.ref_obj_name}'")
            ref_cloud = extract_point_cloud(self.ref_obj_name)
            if len(ref_cloud) == 0:
                raise ValueError(f"Reference cloud '{self.ref_obj_name}' is empty.")
            logger.info(f"Loaded {len(ref_cloud)} points.")

            logger.info(f"Loading target cloud from '{self.target_obj_name}'")
            target_cloud = extract_point_cloud(self.target_obj_name)  
            if len(target_cloud) == 0:                                 
                raise ValueError(f"Target cloud '{self.target_obj_name}' is empty.")
            logger.info(f"Loaded {len(target_cloud)} points.")

            result = full_alignment(ref_cloud, target_cloud)
            aligned_cloud = result["aligned_cloud"]
            overlap = result.get("overlap", None)
            icp_error = result.get("icp_error", None)      
            fine_rmsd = result.get("fine_rmsd", None)

            if overlap is not None and fine_rmsd is not None:
                logger.info(f"Alignment done. Overlap = {overlap:.4f}")
                if icp_error is not None:
                    logger.info(f"ICP error (mean distance) = {icp_error:.4f} Å")
                logger.info(f"Fine RMSD (after optimization) = {fine_rmsd:.4f} Å")
            else:
                logger.info("Alignment done.")

            self.visualize_result(aligned_cloud)

        except Exception as e:
            logger.error(f"Alignment failed: {e} :( )", exc_info=True)

    def visualize_result(self, aligned_cloud):
        name = f"{self.target_obj_name}_aligned"
        create_pymol_cloud(aligned_cloud, name)
        logger.info(f"Aligned cloud saved as object '{name}'.")

#-------------------------------------------------------------------------------------

class LogTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.log_browser = QtWidgets.QTextEdit(self)
        self.log_browser.setReadOnly(True)
        layout.addWidget(self.log_browser)

        
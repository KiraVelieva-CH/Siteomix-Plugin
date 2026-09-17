import logging
import os
import sys

# Настройка базового логгера (убираем префикс INFO позже, здесь только для консоли)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Добавляем путь для импортов (на случай, если нужно)
dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
sys.path.append(dir_path)


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("Siteomix", command=run_plugin_gui, menuName="Plugin")


dialog = None

from .interface import MainWindow


def reset_dialog():
    """Сбрасывает ссылку на диалог и удаляет обработчик логов."""
    global dialog
    if dialog is not None:
        if hasattr(dialog, '_log_handler'):
            logger.removeHandler(dialog._log_handler)
        dialog = None


def run_plugin_gui():
    global dialog
    # Если диалог уже существует и жив, показываем его
    if dialog is not None:
        try:
            if dialog.isVisible():
                dialog.raise_()
                dialog.activateWindow()
                return
            else:
                dialog.show()
                return
        except RuntimeError:
            # Старый диалог удалён – сбрасываем
            dialog = None

    # Создаём новое главное окно
    dialog = MainWindow()
    handler = QTextEditLoggingHandler(dialog.log_tab.log_browser)

    # Кастомный форматтер: для INFO не выводим "INFO: ", только сообщение
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                return record.getMessage()
            else:
                return f"{record.levelname}: {record.getMessage()}"

    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    dialog._log_handler = handler

    # Приветственное сообщение с ASCII-артом (можно оставить или изменить)
    ascii_art = """


Siteomix plugin - binding site detection and calculating the similarity score between them
Hello, thank you for using our Siteomix plugin!
If you use it in your research, please cite us: https://doi.org/10.1007/s10822-026-00913-3
                    (•‿•)   

"""
    logger.info(ascii_art)

    dialog.show()


class QTextEditLoggingHandler(logging.Handler):
    """Обработчик логов, который пишет в QTextEdit (вкладка Log) с цветами."""
    COLORS = {logging.WARNING: "orange", logging.INFO: "blue", logging.DEBUG: "black"}

    def __init__(self, text_edit):
        super().__init__()            # вызываем конструктор родителя
        self.text_edit = text_edit

    def emit(self, record):
        # Проверяем, существует ли ещё виджет (C++ объект не удалён)
        if self.text_edit is None:
            return
        try:
            color = self.COLORS.get(record.levelno, "red")
            text = self.format(record)   # форматируем с помощью установленного форматтера
            self.text_edit.append(f'<span style="color:{color};">{text}</span>')
        except RuntimeError:
            # Виджет уже удалён – удаляем себя из логгера, чтобы не спамить ошибками
            logger.removeHandler(self)

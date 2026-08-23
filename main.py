#!/usr/bin/env python3
"""
ollama-batch-processor - run text files through an Ollama pipeline (translate / audiobook prep /
book cleanup / paraphrase), chunk by chunk, with progressive saving.

Set OLLAMA_BATCH_SELFTEST=<text file> for a headless smoke test (exits 0 when the server is unreachable
but everything else works, and only after a real run when a server with a model is available).
"""
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QIcon, QTextCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from config import (APP_NAME, APP_VERSION, CHUNK_PRESETS, DEFAULT_SETTINGS, TEXT_EXTENSIONS,  # noqa: E402
                    WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH, app_dir, resource_dir)
from processor import OllamaError, OllamaProcessor, ProcessingStopped, load_operations  # noqa: E402
from utils import childproc  # noqa: E402
from utils.naturalsort import natural_key, natural_sorted  # noqa: E402

_BTN = """
    QPushButton { font-size: 14px; font-weight: bold; padding: 8px; background-color: %s; color: %s; border-radius: 5px; }
    QPushButton:hover { background-color: %s; }
    QPushButton:disabled { background-color: #cccccc; color: #666; }
"""
_BAR = """
    QProgressBar { border: 1px solid #c8c8c8; border-radius: 4px; background: #f0f0f0;
                   text-align: center; height: 18px; font-weight: bold; color: #333; }
    QProgressBar::chunk { background-color: %s; border-radius: 3px; }
"""
_STATUS_COLORS = {"pending": "#ffffff", "running": "#fff3cd", "done": "#d4edda", "failed": "#f8d7da",
                  "stopped": "#e2e3e5", "skipped": "#e2e3e5"}


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# =============================================================================== threads
class Worker(QThread):
    log = Signal(str, str)
    status = Signal(str)
    file_started = Signal(int)
    chunk_progress = Signal(int, int, str)
    file_done = Signal(int, str, dict)
    finished_all = Signal(int, int)

    def __init__(self, files: List[str], settings: Dict, operations: Dict, pipeline):
        super().__init__()
        self.files, self.settings, self.operations, self.pipeline = files, settings, operations, pipeline
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        proc = OllamaProcessor(self.settings, self.operations, log=self.log.emit, status=self.status.emit,
                               progress=self.chunk_progress.emit, should_stop=lambda: self._stop)
        try:
            proc.connect()
            version = OllamaProcessor.server_version(self.settings["host"])
            self.log.emit("INFO", f"Ollama {version} at {self.settings['host']}")
        except Exception as exc:  # noqa: BLE001
            self.log.emit("ERROR", f"Ollama server not reachable at {self.settings['host']}: {exc}")
            for i in range(len(self.files)):
                self.file_done.emit(i, "failed", {})
            self.finished_all.emit(0, len(self.files))
            return
        ok = 0
        for i, f in enumerate(self.files):
            if self._stop:
                self.file_done.emit(i, "stopped", {})
                continue
            self.file_started.emit(i)
            try:
                stats = proc.process_file(f, self.pipeline)
                ok += 1
                self.file_done.emit(i, "done", stats)
            except ProcessingStopped:
                self.log.emit("WARNING", f"{os.path.basename(f)}: stopped (no partial output left behind)")
                self.file_done.emit(i, "stopped", {})
            except OllamaError as exc:
                self.log.emit("ERROR", f"{os.path.basename(f)}: {exc}")
                self.file_done.emit(i, "skipped" if "Output exists" in str(exc) else "failed", {})
            except Exception as exc:  # noqa: BLE001
                self.log.emit("ERROR", f"{os.path.basename(f)}: {exc}")
                self.log.emit("DEBUG", traceback.format_exc())
                self.file_done.emit(i, "failed", {})
        self.finished_all.emit(ok, len(self.files))


class ProbeThread(QThread):
    done = Signal(str, list, str)        # version, models, error

    def __init__(self, host: str):
        super().__init__()
        self.host = host

    def run(self):
        try:
            version = OllamaProcessor.server_version(self.host)
            models = OllamaProcessor.list_models(self.host)
            self.done.emit(version, models, "")
        except Exception as exc:  # noqa: BLE001
            self.done.emit("", [], str(exc))


# =============================================================================== widgets
class QueueTable(QTableWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels(["File", "Status", "Chunks"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAcceptDrops(True)
        self.setStyleSheet("QTableWidget { border: 2px solid #aaa; border-radius: 5px; background: #f9f9f9; }")

    def dragEnterEvent(self, e: QDragEnterEvent):
        e.accept() if e.mimeData().hasUrls() else e.ignore()

    def dragMoveEvent(self, e):
        e.accept() if e.mimeData().hasUrls() else e.ignore()

    def dropEvent(self, e: QDropEvent):
        if e.mimeData().hasUrls():
            e.accept()
            self.files_dropped.emit([u.toLocalFile() for u in e.mimeData().urls()])


# =============================================================================== main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(1180, 760)
        self.qsettings = QSettings(APP_NAME, "Settings")
        self.presets_dir = os.path.join(app_dir(), "presets")
        os.makedirs(self.presets_dir, exist_ok=True)
        self.operations = load_operations()
        self.files: List[str] = []
        self.worker: Optional[Worker] = None
        self.probe: Optional[ProbeThread] = None
        self.controls: Dict[str, QWidget] = {}          # global settings
        self.op_controls: Dict[str, Dict[str, QWidget]] = {}   # per operation
        self.models: List[str] = []
        self._run_start = 0.0
        self._chunks = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._build_menu()
        self.apply_settings(dict(DEFAULT_SETTINGS))
        self._load_persisted()
        self.refresh_server()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._left_panel())
        split.addWidget(self._right_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([640, 520])
        root.addWidget(split, 1)
        self.statusBar().showMessage("Ready")

    def _left_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("<b>Queue</b> — drop text files or folders here"))
        self.table = QueueTable()
        self.table.files_dropped.connect(self.add_paths)
        lay.addWidget(self.table, 3)
        row = QHBoxLayout()
        for text, fn in (("Add files", self.add_files), ("Add folder", self.add_folder),
                         ("Remove", self.remove_selected), ("Clear", self.clear_queue)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        open_out = QPushButton("Open output folder")
        open_out.clicked.connect(self.open_output)
        row.addWidget(open_out)
        lay.addLayout(row)

        prog = QGroupBox("Progress")
        pl = QVBoxLayout(prog)
        top = QHBoxLayout()
        self.phase_label = QLabel("Idle")
        self.phase_label.setStyleSheet("font-weight: bold;")
        self.counter_label = QLabel("")
        self.counter_label.setStyleSheet("color: #666;")
        top.addWidget(self.phase_label, 1)
        top.addWidget(self.counter_label)
        pl.addLayout(top)
        self.file_bar = QProgressBar()
        self.file_bar.setStyleSheet(_BAR % "#4a90d9")
        self.file_bar.setFormat("%v / %m chunks")
        pl.addWidget(self.file_bar)
        self.total_bar = QProgressBar()
        self.total_bar.setStyleSheet(_BAR % "#28a745")
        self.total_bar.setFormat("%v / %m files")
        pl.addWidget(self.total_bar)
        stats = QHBoxLayout()
        self.stat_chunks = self._stat(stats, "chunks")
        self.stat_speed = self._stat(stats, "speed")
        self.stat_elapsed = self._stat(stats, "elapsed")
        self.stat_server = self._stat(stats, "server")
        pl.addLayout(stats)
        lay.addWidget(prog)

        lay.addWidget(QLabel("<b>Log</b>"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("QTextEdit { font-family: monospace; font-size: 12px; background: #fcfcfc; }")
        lay.addWidget(self.log, 2)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setStyleSheet(_BTN % ("#28a745", "white", "#218838"))
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setStyleSheet(_BTN % ("#dc3545", "white", "#c82333"))
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        btns.addWidget(self.start_btn, 2)
        btns.addWidget(self.stop_btn, 1)
        lay.addLayout(btns)
        return w

    @staticmethod
    def _stat(layout, caption) -> QLabel:
        box = QVBoxLayout()
        c = QLabel(caption)
        c.setStyleSheet("color: #888; font-size: 10px;")
        v = QLabel("--")
        v.setStyleSheet("font-weight: bold; font-size: 13px;")
        box.addWidget(c)
        box.addWidget(v)
        layout.addLayout(box)
        return v

    @staticmethod
    def _wrap(widget: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(widget)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return sa

    @staticmethod
    def _hint(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color: #666; font-size: 11px;")
        return lab

    def _right_panel(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._wrap(self._pipeline_tab()), "Pipeline")
        for op_id, op in self.operations["operations"].items():
            self.tabs.addTab(self._wrap(self._operation_tab(op_id, op)), op.get("tab_name", op_id.title()))
        self.tabs.addTab(self._wrap(self._server_tab()), "Server")
        all_controls = list(self.controls.values()) + [w for d in self.op_controls.values() for w in d.values()]
        for w in all_controls:
            if isinstance(w, QComboBox):
                w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                w.setMinimumContentsLength(8)
            if isinstance(w, (QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox)):
                w.setMinimumWidth(0)
                w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        return self.tabs

    def _pipeline_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        c = self.controls
        g = QGroupBox("Operations (checked ones run, top to bottom)")
        gl = QVBoxLayout(g)
        self.pipeline_list = QListWidget()
        self.pipeline_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.pipeline_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        for op_id, op in self.operations["operations"].items():
            it = QListWidgetItem(op.get("tab_name", op_id.title()))
            it.setData(Qt.ItemDataRole.UserRole, op_id)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
            self.pipeline_list.addItem(it)
        self.pipeline_list.setFixedHeight(self.pipeline_list.sizeHintForRow(0) * self.pipeline_list.count() + 8)
        gl.addWidget(self.pipeline_list)
        br = QHBoxLayout()
        up, down = QPushButton("Move up"), QPushButton("Move down")
        up.clicked.connect(lambda: self._move_op(-1))
        down.clicked.connect(lambda: self._move_op(1))
        br.addWidget(up)
        br.addWidget(down)
        br.addStretch()
        gl.addLayout(br)
        gl.addWidget(self._hint("Each operation's settings and model are on its own tab. Every file goes through the "
                                "whole pipeline; each step is saved to its own file."))
        lay.addWidget(g)

        g2 = QGroupBox("Chunking")
        f2 = QFormLayout(g2)
        c["whole_file"] = QCheckBox("Process each file as a single chunk")
        f2.addRow("", c["whole_file"])
        self.chunk_preset = QComboBox()
        self.chunk_preset.addItem("custom", None)
        for name, (size, ov) in CHUNK_PRESETS.items():
            self.chunk_preset.addItem(name, (size, ov))
        self.chunk_preset.currentIndexChanged.connect(self._apply_chunk_preset)
        f2.addRow("Preset:", self.chunk_preset)
        c["chunk_size"] = QSpinBox()
        c["chunk_size"].setRange(200, 100000)
        c["chunk_size"].setSingleStep(100)
        c["chunk_size"].setSuffix(" chars")
        f2.addRow("Chunk size:", c["chunk_size"])
        c["overlap"] = QSpinBox()
        c["overlap"].setRange(0, 5000)
        c["overlap"].setSingleStep(50)
        c["overlap"].setSuffix(" chars")
        c["overlap"].setToolTip("Tail of the previous result shown to the model for consistency")
        c["chunk_size"].valueChanged.connect(self._sync_chunk_preset)
        c["overlap"].valueChanged.connect(self._sync_chunk_preset)
        f2.addRow("Context overlap:", c["overlap"])
        c["deduplicate"] = QCheckBox("Remove duplicate paragraphs at chunk boundaries")
        f2.addRow("", c["deduplicate"])
        f2.addRow(self._hint("Chunks break at paragraph or sentence ends. Bigger chunks = more context but slower "
                             "and needs more VRAM; the context window (num_ctx) is sized automatically."))
        lay.addWidget(g2)

        g3 = QGroupBox("Output")
        f3 = QFormLayout(g3)
        c["output_mode"] = QComboBox()
        c["output_mode"].addItem("next to the source file", "same")
        c["output_mode"].addItem("custom folder", "custom")
        f3.addRow("Write to:", c["output_mode"])
        c["output_dir"] = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(c["output_dir"], 1)
        b = QPushButton("…")
        b.setFixedWidth(32)
        b.clicked.connect(lambda: self._browse_dir(c["output_dir"]))
        row.addWidget(b)
        f3.addRow("Folder:", row)
        c["suffix"] = QLineEdit()
        f3.addRow("Suffix:", c["suffix"])
        c["save_steps"] = QCheckBox("Save every pipeline step to its own file")
        f3.addRow("", c["save_steps"])
        c["overwrite"] = QCheckBox("Overwrite existing output")
        f3.addRow("", c["overwrite"])
        lay.addWidget(g3)
        lay.addStretch()
        return w

    def _operation_tab(self, op_id: str, op: Dict) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        ctl: Dict[str, QWidget] = {}
        self.op_controls[op_id] = ctl
        if op.get("description"):
            lay.addWidget(self._hint(op["description"]))
        g = QGroupBox("Model")
        f = QFormLayout(g)
        ctl["model"] = QComboBox()
        ctl["model"].setEditable(True)
        ctl["model"].setToolTip("Any model installed on the server (Server tab → Refresh). Type a name to use one that is not listed yet.")
        f.addRow("Model:", ctl["model"])
        lay.addWidget(g)

        g2 = QGroupBox("Options")
        f2 = QFormLayout(g2)
        for oid, o in op.get("options", {}).items():
            t = o.get("type")
            label = o.get("label", oid)
            if t == "text":
                wd = QLineEdit(str(o.get("default", "")))
            elif t == "combo":
                wd = QComboBox()
                for item in o.get("options", []):
                    wd.addItem(item.get("name", str(item)), item.get("value") if isinstance(item, dict) else item)
                wd.setCurrentIndex(int(o.get("default_index", 0)))
            elif t == "spinbox":
                if o.get("decimal"):
                    wd = QDoubleSpinBox()
                    wd.setDecimals(int(o.get("decimals", 2)))
                    wd.setSingleStep(float(o.get("single_step", 0.1)))
                else:
                    wd = QSpinBox()
                    wd.setSingleStep(int(o.get("step", 1)))
                wd.setRange(o.get("min", 0), o.get("max", 100000))
                wd.setValue(o.get("default", 0))
                if o.get("suffix"):
                    wd.setSuffix(o["suffix"])
            elif t == "checkbox":
                wd = QCheckBox(label)
                wd.setChecked(bool(o.get("default", False)))
                label = ""
            else:
                continue
            if o.get("tooltip"):
                wd.setToolTip(o["tooltip"])
            f2.addRow(label, wd)
            ctl[oid] = wd
        lay.addWidget(g2)
        if op.get("sub_operations"):
            lay.addWidget(self._hint("All checked tasks are merged into one prompt, so the text passes through the model once."))
        lay.addStretch()
        return w

    def _server_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        c = self.controls
        g = QGroupBox("Ollama server")
        f = QFormLayout(g)
        c["host"] = QLineEdit()
        c["host"].setPlaceholderText("http://localhost:11434")
        c["host"].editingFinished.connect(self.refresh_server)
        f.addRow("URL:", c["host"])
        row = QHBoxLayout()
        self.server_label = self._hint("")
        self.server_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        refresh = QPushButton("Test connection / refresh models")
        refresh.clicked.connect(self.refresh_server)
        row.addWidget(self.server_label, 1)
        row.addWidget(refresh)
        f.addRow(row)
        c["timeout"] = QSpinBox()
        c["timeout"].setRange(0, 7200)
        c["timeout"].setSuffix(" s")
        c["timeout"].setSpecialValueText("none")
        c["timeout"].setToolTip("Per-request timeout; 0 = wait forever (slow models on CPU can take minutes per chunk)")
        f.addRow("Request timeout:", c["timeout"])
        c["keep_alive"] = QLineEdit()
        c["keep_alive"].setToolTip("How long the model stays loaded after the last request, e.g. 10m, 1h, -1 = forever")
        f.addRow("Keep model loaded:", c["keep_alive"])
        lay.addWidget(g)

        g2 = QGroupBox("Generation")
        f2 = QFormLayout(g2)
        c["num_ctx"] = QSpinBox()
        c["num_ctx"].setRange(0, 262144)
        c["num_ctx"].setSingleStep(1024)
        c["num_ctx"].setSpecialValueText("auto")
        c["num_ctx"].setToolTip("Context window in tokens. auto = sized from the chunk (prompt + answer)")
        f2.addRow("Context (num_ctx):", c["num_ctx"])
        c["num_predict"] = QSpinBox()
        c["num_predict"].setRange(-1, 262144)
        c["num_predict"].setSpecialValueText("auto")
        c["num_predict"].setToolTip("Max answer tokens per chunk. auto = generous cap derived from the chunk size")
        f2.addRow("Max tokens:", c["num_predict"])
        c["top_p"] = QDoubleSpinBox()
        c["top_p"].setRange(0.0, 1.0)
        c["top_p"].setSingleStep(0.05)
        f2.addRow("top_p:", c["top_p"])
        c["strip_thinking"] = QCheckBox("Strip <think>…</think> blocks from reasoning models")
        f2.addRow("", c["strip_thinking"])
        lay.addWidget(g2)
        lay.addStretch()
        return w

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        fm.addAction("Add files…", self.add_files, "Ctrl+O")
        fm.addAction("Add folder…", self.add_folder, "Ctrl+Shift+O")
        fm.addAction("Open output folder", self.open_output)
        fm.addAction("Edit operations / prompts (config.json)", self.open_config)
        fm.addSeparator()
        fm.addAction("Exit", self.close, "Ctrl+Q")
        self.presets_menu = mb.addMenu("&Presets")
        self._rebuild_presets_menu()
        hm = mb.addMenu("&Help")
        hm.addAction("About", self._about)

    def _rebuild_presets_menu(self):
        m = self.presets_menu
        m.clear()
        names = sorted((f[:-5] for f in os.listdir(self.presets_dir) if f.endswith(".json") and f != "defaults.json"),
                       key=natural_key)
        for name in names:
            m.addAction(name, lambda n=name: self._load_preset_file(os.path.join(self.presets_dir, n + ".json")))
        if names:
            m.addSeparator()
        m.addAction("Save preset…", self._save_preset)
        m.addAction("Delete preset…", self._delete_preset)
        m.addAction("Import preset…", self._import_preset)
        m.addAction("Export preset…", self._export_preset)
        m.addSeparator()
        m.addAction("Save current settings as defaults", self._save_defaults)
        m.addAction("Reset to factory defaults", lambda: self.apply_settings(dict(DEFAULT_SETTINGS)))

    # ------------------------------------------------------------- settings io
    @staticmethod
    def _read_widget(w):
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            data = w.currentData()
            return w.currentText().strip() if w.isEditable() or data is None else data
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            return w.value()
        if isinstance(w, QLineEdit):
            return w.text().strip()
        return None

    @staticmethod
    def _write_widget(w, v):
        try:
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QComboBox):
                idx = w.findData(v)
                if idx < 0:
                    idx = w.findText(str(v))
                if idx >= 0:
                    w.setCurrentIndex(idx)
                elif w.isEditable():
                    w.setCurrentText(str(v))
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.setValue(v)
            elif isinstance(w, QLineEdit):
                w.setText("" if v is None else str(v))
        except Exception:
            pass

    def get_settings(self) -> Dict:
        s = {k: self._read_widget(w) for k, w in self.controls.items()}
        order, enabled = [], []
        for i in range(self.pipeline_list.count()):
            it = self.pipeline_list.item(i)
            order.append(it.data(Qt.ItemDataRole.UserRole))
            if it.checkState() == Qt.CheckState.Checked:
                enabled.append(order[-1])
        s["pipeline"], s["enabled"] = order, enabled
        s["ops"] = {op: {k: self._read_widget(w) for k, w in ctl.items()} for op, ctl in self.op_controls.items()}
        return s

    def apply_settings(self, s: Dict):
        for k, w in self.controls.items():
            if k in s:
                self._write_widget(w, s[k])
        if "pipeline" in s:
            items = {}
            for i in range(self.pipeline_list.count()):
                it = self.pipeline_list.item(i)
                items[it.data(Qt.ItemDataRole.UserRole)] = it
            order = [o for o in s["pipeline"] if o in items] + [o for o in items if o not in s["pipeline"]]
            for it in items.values():
                self.pipeline_list.takeItem(self.pipeline_list.row(it))
            for o in order:
                self.pipeline_list.addItem(items[o])
        if "enabled" in s:
            for i in range(self.pipeline_list.count()):
                it = self.pipeline_list.item(i)
                on = it.data(Qt.ItemDataRole.UserRole) in s["enabled"]
                it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        for op, vals in (s.get("ops") or {}).items():
            for k, v in vals.items():
                if k in self.op_controls.get(op, {}):
                    self._write_widget(self.op_controls[op][k], v)
        self._sync_chunk_preset()

    def _load_persisted(self):
        d = os.path.join(self.presets_dir, "defaults.json")
        if os.path.isfile(d):
            try:
                with open(d, encoding="utf-8") as fh:
                    self.apply_settings(json.load(fh))
            except Exception:
                pass
        raw = self.qsettings.value("settings")
        if raw:
            try:
                self.apply_settings(json.loads(raw))
            except Exception:
                pass
        geo = self.qsettings.value("geometry")
        if geo:
            self.restoreGeometry(geo)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Quit", "Processing is running. Stop and quit?") != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.request_stop()
            self.worker.wait(5000)
        self.qsettings.setValue("settings", json.dumps(self.get_settings()))
        self.qsettings.setValue("geometry", self.saveGeometry())
        event.accept()

    # ---------------------------------------------------------------- presets
    def _write_json(self, path: str, data: Dict) -> bool:
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Preset", f"Could not write {path}:\n{exc}")
            return False

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok and name.strip() and self._write_json(os.path.join(self.presets_dir, name.strip() + ".json"),
                                                    self.get_settings()):
            self._rebuild_presets_menu()

    def _delete_preset(self):
        names = sorted((f[:-5] for f in os.listdir(self.presets_dir) if f.endswith(".json") and f != "defaults.json"),
                       key=natural_key)
        if not names:
            QMessageBox.information(self, "Presets", "No saved presets.")
            return
        name, ok = QInputDialog.getItem(self, "Delete preset", "Preset:", names, 0, False)
        if ok and name:
            os.remove(os.path.join(self.presets_dir, name + ".json"))
            self._rebuild_presets_menu()

    def _load_preset_file(self, path: str):
        try:
            with open(path, encoding="utf-8") as fh:
                self.apply_settings(json.load(fh))
            self.statusBar().showMessage(f"Preset loaded: {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.warning(self, "Preset", f"Could not read preset:\n{exc}")

    def _import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import preset", "", "JSON (*.json)")
        if path:
            self._load_preset_file(path)

    def _export_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export preset", "preset.json", "JSON (*.json)")
        if path:
            self._write_json(path, self.get_settings())

    def _save_defaults(self):
        if self._write_json(os.path.join(self.presets_dir, "defaults.json"), self.get_settings()):
            self.statusBar().showMessage("Current settings saved as defaults")

    # ------------------------------------------------------------------ helpers
    def _browse_dir(self, line: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Output folder", line.text() or os.path.expanduser("~"))
        if d:
            line.setText(d)
            self.controls["output_mode"].setCurrentIndex(1)

    def _apply_chunk_preset(self, idx: int):
        data = self.chunk_preset.itemData(idx)
        if data:
            self.controls["chunk_size"].setValue(data[0])
            self.controls["overlap"].setValue(data[1])

    def _sync_chunk_preset(self):
        cur = (self.controls["chunk_size"].value(), self.controls["overlap"].value())
        self.chunk_preset.blockSignals(True)
        idx = next((i for i in range(self.chunk_preset.count()) if tuple(self.chunk_preset.itemData(i) or ()) == cur), 0)
        self.chunk_preset.setCurrentIndex(idx)
        self.chunk_preset.blockSignals(False)

    def _move_op(self, delta: int):
        row = self.pipeline_list.currentRow()
        if row < 0 or not 0 <= row + delta < self.pipeline_list.count():
            return
        it = self.pipeline_list.takeItem(row)
        self.pipeline_list.insertItem(row + delta, it)
        self.pipeline_list.setCurrentRow(row + delta)

    def open_config(self):
        path = os.path.join(app_dir(), "config.json")
        if not os.path.isfile(path):
            import shutil
            shutil.copy2(os.path.join(resource_dir(), "config.json"), path)
        self._open_path(path)
        self._append_log("INFO", f"Edit {path} and restart the app to pick up changes.")

    def _open_path(self, path: str):
        try:
            if sys.platform == "win32":
                os.startfile(path)  # noqa
            elif sys.platform == "darwin":
                childproc.popen(["open", path])
            else:
                childproc.popen(["xdg-open", path])
        except Exception as exc:
            QMessageBox.information(self, "Open", f"{path}\n\n({exc})")

    def open_output(self):
        s = self.get_settings()
        if s["output_mode"] == "custom" and s["output_dir"]:
            out = s["output_dir"]
        elif self.files:
            out = os.path.dirname(self.files[0])
        else:
            out = os.getcwd()
        os.makedirs(out, exist_ok=True)
        self._open_path(out)

    # ------------------------------------------------------------------ server
    def refresh_server(self, *_):
        if self.probe and self.probe.isRunning():
            return
        host = self.controls["host"].text().strip() or DEFAULT_SETTINGS["host"]
        self.server_label.setText(f"Connecting to {host}…")
        self.server_label.setStyleSheet("color: #666; font-size: 11px;")
        self.probe = ProbeThread(host)
        self.probe.done.connect(self._server_probed)
        self.probe.start()

    def _server_probed(self, version: str, models: List[str], err: str):
        if err:
            self.server_label.setText(f"Not reachable: {err}. Start it with `ollama serve`.")
            self.server_label.setStyleSheet("color: #856404; font-size: 11px;")
            self.stat_server.setText("offline")
            return
        self.models = models
        self.stat_server.setText(f"Ollama {version}")
        if not models:
            self.server_label.setText(f"Ollama {version} connected, but no models installed (`ollama pull qwen2.5`).")
            self.server_label.setStyleSheet("color: #856404; font-size: 11px;")
        else:
            self.server_label.setText(f"Ollama {version} — {len(models)} model(s): " + ", ".join(models[:6])
                                      + (" …" if len(models) > 6 else ""))
            self.server_label.setStyleSheet("color: #155724; font-size: 11px;")
        for ctl in self.op_controls.values():
            combo: QComboBox = ctl["model"]
            current = combo.currentText().strip()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(models)
            if current and combo.findText(current) < 0:
                combo.addItem(current)
            combo.setCurrentText(current or (models[0] if models else ""))
            combo.blockSignals(False)

    # ------------------------------------------------------------------ queue
    def add_paths(self, paths: List[str]):
        added = dupes = 0
        for p in paths:
            if os.path.isdir(p):
                found = []
                for root, dirs, files in os.walk(p):
                    dirs.sort(key=natural_key)
                    found += [os.path.join(root, f) for f in files if os.path.splitext(f)[1].lower() in TEXT_EXTENSIONS]
                for f in natural_sorted(found):
                    n = self._add_one(f)
                    added += n
                    dupes += 1 - n
            elif os.path.isfile(p):
                n = self._add_one(p)
                added += n
                dupes += 1 - n
        msg = f"{added} file(s) added — {len(self.files)} in queue"
        if dupes:
            msg += f" ({dupes} already in queue)"
        if added or dupes:
            self.statusBar().showMessage(msg)

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def _add_one(self, path: str) -> int:
        key = self._key(path)
        if any(self._key(f) == key for f in self.files):
            return 0
        self.files.append(path)
        r = self.table.rowCount()
        self.table.insertRow(r)
        item = QTableWidgetItem(os.path.basename(path))
        try:
            item.setToolTip(f"{path}\n{os.path.getsize(path) / 1024:.1f} KB")
        except OSError:
            item.setToolTip(path)
        self.table.setItem(r, 0, item)
        self.table.setItem(r, 1, QTableWidgetItem("pending"))
        self.table.setItem(r, 2, QTableWidgetItem(""))
        self._color_row(r, "pending")
        return 1

    def _color_row(self, row: int, state: str):
        for col in range(3):
            it = self.table.item(row, col)
            if it:
                it.setBackground(QColor(_STATUS_COLORS.get(state, "#ffffff")))

    def add_files(self):
        exts = " ".join(f"*{e}" for e in TEXT_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(self, "Add text files", "", f"Text files ({exts});;All files (*)")
        self.add_paths(files)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder")
        if d:
            self.add_paths([d])

    def remove_selected(self):
        if self.worker and self.worker.isRunning():
            return
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
            del self.files[r]

    def clear_queue(self):
        if self.worker and self.worker.isRunning():
            return
        self.files.clear()
        self.table.setRowCount(0)

    # -------------------------------------------------------------- processing
    def build_pipeline(self, s: Dict) -> Tuple[List[Tuple[str, Dict, List[str]]], List[str]]:
        """-> (pipeline, problems)"""
        pipeline, problems = [], []
        ops = self.operations["operations"]
        for op_id in s["pipeline"]:
            if op_id not in s["enabled"]:
                continue
            values = dict(s["ops"].get(op_id, {}))
            if not values.get("model"):
                problems.append(f"{ops[op_id].get('tab_name', op_id)}: no model selected.")
            enabled: List[str] = []
            if op_id != "translation":
                subs = ops[op_id].get("sub_operations", {})
                for sid in subs:
                    if values.get(sid) is True:
                        enabled.append(sid)
                tone = values.get("target_tone")
                if tone and tone != "original" and f"adjust_tone_{tone}" in subs:
                    enabled.append(f"adjust_tone_{tone}")
                if not enabled:
                    problems.append(f"{ops[op_id].get('tab_name', op_id)} is enabled but no task is checked.")
            pipeline.append((op_id, values, enabled))
        if not pipeline:
            problems.append("No operation is checked in the Pipeline tab.")
        return pipeline, problems

    def start(self):
        if not self.files:
            QMessageBox.information(self, "Queue", "Add some text files first.")
            return
        s = self.get_settings()
        pipeline, problems = self.build_pipeline(s)
        if problems:
            QMessageBox.warning(self, "Check settings", "\n".join(problems))
            return
        if s["output_mode"] == "custom" and not s["output_dir"]:
            QMessageBox.warning(self, "Check settings", "Choose an output folder or write next to the source files.")
            return
        for r in range(self.table.rowCount()):
            self.table.item(r, 1).setText("pending")
            self.table.item(r, 2).setText("")
            self._color_row(r, "pending")
        self.log.clear()
        self._chunks = 0
        self._run_start = time.time()
        self.total_bar.setRange(0, len(self.files))
        self.total_bar.setValue(0)
        self.file_bar.setRange(0, 1)
        self.file_bar.setValue(0)
        self.stat_chunks.setText("0")
        self.stat_speed.setText("--")
        self._timer.start(1000)
        names = [self.operations["operations"][p[0]].get("tab_name", p[0]) for p in pipeline]
        self._append_log("INFO", "Pipeline: " + " → ".join(names))
        self.worker = Worker(list(self.files), s, self.operations, pipeline)
        w = self.worker
        w.log.connect(self._append_log)
        w.status.connect(self._on_status)
        w.file_started.connect(self._on_file_started)
        w.chunk_progress.connect(self._on_chunk)
        w.file_done.connect(self._on_file_done)
        w.finished_all.connect(self._on_finished)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tabs.setEnabled(False)
        w.start()

    def stop(self):
        if self.worker:
            self.worker.request_stop()
            self.phase_label.setText("Stopping after the current chunk…")
            self.stop_btn.setEnabled(False)

    def _tick(self):
        el = time.time() - self._run_start
        self.stat_elapsed.setText(format_duration(el))
        if self._chunks and el > 0:
            self.stat_speed.setText(f"{self._chunks / el * 60:.1f} chunks/min")

    def _on_status(self, text: str):
        self.phase_label.setText(text)
        self.statusBar().showMessage(text)

    def _on_file_started(self, i: int):
        self.table.item(i, 1).setText("running")
        self._color_row(i, "running")
        self.table.scrollToItem(self.table.item(i, 0))
        self.counter_label.setText(f"file {i + 1} / {len(self.files)} — {os.path.basename(self.files[i])}")
        self.file_bar.setRange(0, 1)
        self.file_bar.setValue(0)

    def _on_chunk(self, done: int, total: int, phase: str):
        self.file_bar.setRange(0, max(1, total))
        self.file_bar.setValue(done)
        self._chunks += 1
        self.stat_chunks.setText(str(self._chunks))
        row = self.total_bar.value()
        if row < self.table.rowCount():
            self.table.item(row, 2).setText(f"{done}/{total}")

    def _on_file_done(self, i: int, state: str, stats: Dict):
        self.table.item(i, 1).setText(state)
        self._color_row(i, state)
        if state == "done":
            self.table.item(i, 2).setText(str(stats.get("chunks", "")))
        self.total_bar.setValue(self.total_bar.value() + 1)

    def _on_finished(self, ok: int, total: int):
        self._timer.stop()
        self._tick()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.tabs.setEnabled(True)
        self.phase_label.setText("Finished" if ok == total else "Finished with problems")
        self.counter_label.setText("")
        self.statusBar().showMessage(f"Done: {ok}/{total} files")
        if self.worker:
            self.worker.wait(10000)
            self.worker.deleteLater()
        self.worker = None

    def _append_log(self, level: str, msg: str):
        color = {"DEBUG": "#888", "INFO": "#2c3e50", "WARNING": "#b8860b", "ERROR": "#c0392b"}.get(level, "#2c3e50")
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.log.append(f'<span style="color:{color};">{safe}</span>')
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<b>{APP_NAME} {APP_VERSION}</b><br>Batch text processing with local LLMs via Ollama: "
                          "translation, audiobook preparation, book cleanup, paraphrasing.<br><br>"
                          "<a href='https://github.com/hclivess/ollama-batch-processor'>github.com/hclivess/ollama-batch-processor</a>")


# =============================================================================== entry
def main():
    if sys.platform == "win32":
        # The taskbar button takes its icon from the process's Application User Model ID, not from the
        # window: with none of its own the process is grouped under whatever launched it and shows that
        # program's icon. Must happen before any window exists; no version in the id, so a pinned button
        # survives an upgrade.
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"hclivess.{APP_NAME}")
        except Exception:
            pass

    app = QApplication(sys.argv)
    childproc.install_qt_hook(app)
    app.setStyle("Fusion")
    icon = os.path.join(resource_dir(), "icon.ico")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    win = MainWindow()
    win.show()
    selftest = os.environ.get("OLLAMA_BATCH_SELFTEST")
    if selftest:
        _selftest(win, selftest)
    sys.exit(app.exec())


def _selftest(win: MainWindow, path: str):
    """CI smoke test. With a reachable server + model: real translation run must succeed.
    Without a server: the run must fail cleanly with 'not reachable' (exit 0)."""
    from processor import TextChunker
    assert len(TextChunker.chunk_text("Hello world. " * 400, 500, 50)) > 5
    win.apply_settings({"enabled": ["translation"], "chunk_size": 600, "overlap": 60, "overwrite": True,
                        "timeout": 300, "num_predict": 200, "suffix": "_selftest"})
    win.op_controls["translation"]["source_language"].setText("English")
    win.op_controls["translation"]["target_language"].setText("German")
    win.add_paths([path])
    mode = {"server": None}

    def probed(version, models, err):
        mode["server"] = bool(version) and bool(models)
        if mode["server"]:
            win.op_controls["translation"]["model"].setCurrentText(models[0])
            print(f"selftest: server {version}, model {models[0]}", flush=True)
        else:
            print(f"selftest: no server ({err or 'no models'}) - expecting a clean failure", flush=True)
        QTimer.singleShot(200, go)

    def finished(ok, total):
        expected = 1 if mode["server"] else 0
        print(f"selftest: finished {ok}/{total} (expected {expected})", flush=True)
        code = 0 if ok == expected else 1
        QTimer.singleShot(0, lambda: (win.worker and win.worker.wait(10000), QApplication.exit(code)))

    def go():
        if not mode["server"]:
            win.op_controls["translation"]["model"].setCurrentText("selftest-model")
        win.start()
        if not win.worker:
            print("selftest: start refused", flush=True)
            QApplication.exit(1)
            return
        win.worker.log.connect(lambda lvl, m: print(f"selftest[{lvl}]: {m}", flush=True))
        win.worker.status.connect(lambda m: print(f"selftest: {m}", flush=True))
        win.worker.finished_all.connect(finished)

    def wait_probe():
        if win.probe and win.probe.isRunning():
            QTimer.singleShot(200, wait_probe)
            return
        probed(win.stat_server.text().replace("Ollama ", "") if win.models else "", win.models,
               "" if win.models else win.server_label.text())
    QTimer.singleShot(300, wait_probe)


if __name__ == "__main__":
    main()

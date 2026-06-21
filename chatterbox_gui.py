#!/usr/bin/env python3
"""
ChatterboxTTS GUI Interface
A proper GUI wrapper for the main_launcher.py functionality
"""

import sys
import os

# Set HuggingFace environment variables before any imports
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
if 'TRANSFORMERS_CACHE' not in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')

import subprocess
import threading
import io
import contextlib
import logging
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QGridLayout, QWidget, QPushButton, QLabel, QLineEdit,
                            QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit,
                            QFileDialog, QProgressBar, QGroupBox, QCheckBox,
                            QMessageBox, QSplitter, QFrame, QListWidget, QListWidgetItem, QTabWidget,
                            QFormLayout, QSlider, QSpacerItem, QSizePolicy, QScrollArea, QDialog,
                            QDialogButtonBox, QProgressDialog,
                            QButtonGroup, QRadioButton, QMenu)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, pyqtSlot, Qt, QSettings
from PyQt5.QtGui import QFont, QPixmap, QIcon
# Import the existing modules
from config.config import *

# Ensure allocator config from config is applied to environment before any CUDA allocations
try:
    if 'PYTORCH_CUDA_ALLOC_CONF' in globals():
        if isinstance(PYTORCH_CUDA_ALLOC_CONF, str) and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = PYTORCH_CUDA_ALLOC_CONF
except Exception:
    pass

# Import TTS engine only AFTER allocator env is set
from modules.tts_engine import _release_global_tts_model
from interface import main as interface_main
from modules.resume_handler import find_incomplete_books
from tools.combine_only import run_combine_only_mode
from wrapper.chunk_tool import run_chunk_repair_tool
from utils.generate_from_json import main as generate_from_json_main
from modules.tts_engine import process_book_folder, _release_global_tts_model
from modules.realtime_status_manager import RealTimeStatusManager, get_status_manager, set_status_manager
import importlib
import config.config as config_mod

# Import voice analyzer
try:
    from voice_analyzer.analyzer import analyze_voice_sample
    from voice_analyzer.audio_processor import process_voice_sample
    VOICE_ANALYZER_AVAILABLE = True
    print("✅ Voice analyzer available")
except ImportError as e:
    VOICE_ANALYZER_AVAILABLE = False
    print(f"Warning: Missing dependencies for voice analysis: {e}")
    print("Install with: pip install -r voice_analyzer/requirements.txt")

class NoScrollSpinBox(QSpinBox):
    """Ignore wheel events to prevent scrolling and value changes in spin boxes.
    Custom QGroupBox for displaying structured status information during TTS operations.
    """
    def wheelEvent(self, event):
        """Structured status panel widget for TTS operations.
        Args:
        title (str): Title of the status panel.
        Returns:
        None
        """
        # Ignore wheel so parent scroll area handles scrolling; prevents value changes
        event.ignore()

class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """NoScrollDoubleSpinBox: Ignores wheel events to prevent value changes and allow parent scroll area to handle scrolling.
    StructuredStatusPanel: Widget for displaying structured status information in TTS operations. Initializes with a title and sets up a QFormLayout for UI components.
    """
    def wheelEvent(self, event):
        """Handle wheel events to prevent scrolling in the parent scroll area.
        Args:
        event (QWheelEvent): The wheel event to ignore.
        Returns: None
        """
        # Ignore wheel so parent scroll area handles scrolling; prevents value changes
        event.ignore()


class StructuredStatusPanel(QGroupBox):
    """Structured status panel widget for TTS operations"""

    def __init__(self, title="Processing Status"):
        """Initializes the user interface components for a processing status display.
        Args:
        title (str): The title of the UI component.
        Returns:
        None
        """
        super().__init__(title)
        self.setup_ui()

    def setup_ui(self):
        """Initializes and sets up the user interface components. Args: None Returns: None"""
        layout = QFormLayout(self)

        # Status fields
        self.operation_label = QLabel("⏸ Ready")
        self.device_label = QLabel("🔍 Detecting...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v/%m")
        self.elapsed_label = QLabel("0:00:00")

# Audio controls moved to main button area next to Regenerate M4B

        self.remaining_label = QLabel("--:--:--")
        self.realtime_label = QLabel("--")
        self.vram_label = QLabel("-- GB")

        # Add fields to layout
        layout.addRow("Current Operation:", self.operation_label)
        layout.addRow("Device Status:", self.device_label)
        layout.addRow("Progress:", self.progress_bar)
        layout.addRow("Elapsed Time:", self.elapsed_label)
# Audio controls row removed from status panel
        layout.addRow("Time Remaining:", self.remaining_label)
        layout.addRow("Realtime Factor:", self.realtime_label)
        layout.addRow("VRAM Usage:", self.vram_label)

        # Styling - Dark blue background with white text
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                background-color: #1e3a8a;
                color: white;
                border: 2px solid #3b82f6;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: white;
            }
            QLabel {
                font-family: monospace;
                padding: 2px;
                color: white;
                background-color: transparent;
            }
            QProgressBar {
                border: 1px solid #3b82f6;
                border-radius: 3px;
                background-color: #1e40af;
                color: white;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #60a5fa;
                border-radius: 2px;
            }
        """)

    def update_status(self, operation=None, progress=None, elapsed=None, eta=None,
                     remaining=None, realtime=None, vram=None, chunk_info=None, avg_its=None):
        """Update status panel fields"""
        if operation:
            self.operation_label.setText(operation)
        if progress is not None:
            if isinstance(progress, tuple):  # (current, total)
                current, total = progress
                self.progress_bar.setVisible(True)
                self.progress_bar.setMaximum(total)
                self.progress_bar.setValue(current)
            else:
                self.progress_bar.setVisible(False)
        if elapsed:
            self.elapsed_label.setText(elapsed)
        # ETA removed - replaced with audio controls
        if remaining:
            self.remaining_label.setText(remaining)
        if realtime and realtime != "Calculating...":
            self.realtime_label.setText(realtime)
        if vram:
            self.vram_label.setText(vram)

    def reset(self):
        """Reset all fields to default state"""
        self.operation_label.setText("⏸ Ready")
        self.progress_bar.setVisible(False)
        self.elapsed_label.setText("0:00:00")
        self.remaining_label.setText("--:--:--")
        self.realtime_label.setText("--")
        self.vram_label.setText("-- GB")

    def update_realtime(self, elapsed_seconds, remaining_seconds, realtime_factor, completed_chunks, total_chunks, vram_usage):
        """Update status panel with real-time metrics - called directly by RealTimeStatusManager"""
        # Format and update elapsed time
        self.elapsed_label.setText(self.format_time(elapsed_seconds))
        
        # Format and update remaining time
        if remaining_seconds is not None and remaining_seconds >= 0:
            self.remaining_label.setText(self.format_time(remaining_seconds))
        else:
            self.remaining_label.setText("--:--:--")
        
        # Format and update realtime factor
        if realtime_factor is not None and realtime_factor > 0:
            self.realtime_label.setText(f"{realtime_factor:.2f}x")
        else:
            self.realtime_label.setText("--")
        
        # Update progress bar
        if total_chunks > 0:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(total_chunks)
            self.progress_bar.setValue(completed_chunks)
        
        # Update VRAM usage
        if vram_usage is not None:
            self.vram_label.setText(f"{vram_usage:.2f} GB")
    
    @staticmethod
    def format_time(seconds):
        """Convert seconds to HH:MM:SS format"""
        if seconds is None or seconds < 0:
            return "--:--:--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"

# Audio controls are now in main button area, not status panel


class ChunkingTestWindow(QDialog):
    """Popup window to display chunking test results"""

    def __init__(self, parent=None):
        """Initializes the Chunking Test Results window.
        Args:
        parent (QWidget): The parent widget of this dialog.
        Returns:
        None
        """
        super().__init__(parent)
        self.setWindowTitle("Chunking Test Results")
        self.setGeometry(200, 200, 800, 600)
        self.setModal(False)  # Allow interaction with main window

        layout = QVBoxLayout(self)

        # Title label
        title_label = QLabel("📝 Text Chunking Analysis")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2E7D32; padding: 10px;")
        layout.addWidget(title_label)

        # Text display area
        self.text_display = QTextEdit()
        self.text_display.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 11px;
                background-color: #fafafa;
                border: 1px solid #ccc;
                padding: 10px;
            }
        """)
        self.text_display.setReadOnly(True)
        layout.addWidget(self.text_display)

        # Button layout
        button_layout = QHBoxLayout()

        # Copy button
        copy_btn = QPushButton("📋 Copy to Clipboard")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(copy_btn)

        # Clear button
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self.text_display.clear)
        button_layout.addWidget(clear_btn)

        # Close button
        close_btn = QPushButton("✖️ Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def set_chunking_results(self, results_text):
        """Set the chunking test results"""
        self.text_display.setPlainText(results_text)
        # Scroll to top
        cursor = self.text_display.textCursor()
        cursor.movePosition(cursor.Start)
        self.text_display.setTextCursor(cursor)

    def copy_to_clipboard(self):
        """Copy results to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_display.toPlainText())
        QMessageBox.information(self, "Copied", "Chunking results copied to clipboard!")


class ProcessThread(QThread):
    """Thread to run background processes without blocking GUI"""
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)  # Added missing progress signal
    structured_status_signal = pyqtSignal(dict)  # For structured status panel updates

    def __init__(self, target_function, *args, **kwargs):
        """Initialize a parser for handling status updates.
        Args:
        target_function (callable): The function to be called after parsing.
        *args: Positional arguments for the target function.
        **kwargs: Keyword arguments for the target function.
        Returns:
        None
        """
        super().__init__()
        self.target_function = target_function
        self.args = args
        self.kwargs = kwargs

    def parse_and_emit_status(self, status_text):
        """Parse clean status text and emit structured data"""
        import re

        # Emit raw status for old-style updates
        self.status_signal.emit(status_text)

        # Parse structured data from status text
        # Format: "Elapsed: 0:00:40 | ETA: 0:01:12 | Remaining: 0:00:32 | Realtime: 1.42x | VRAM: 3.4GB | Audio: 0:00:57"
        status_data = {}

        # Extract chunk progress
        chunk_match = re.search(r'Chunk:\s*(\d+)\/(\d+)', status_text)
        if chunk_match:
            current = int(chunk_match.group(1))
            total = int(chunk_match.group(2))
            status_data['progress'] = (current, total)
            status_data['chunk_info'] = f"Chunk {current}/{total}"

        # Extract elapsed time
        elapsed_match = re.search(r'Elapsed: ([0-9:]+)', status_text)
        if elapsed_match:
            status_data['elapsed'] = elapsed_match.group(1)

        # Extract ETA
        eta_match = re.search(r'ETA: ([0-9:]+)', status_text)
        if eta_match:
            status_data['eta'] = eta_match.group(1)

        # Extract remaining time
        remaining_match = re.search(r'Remaining: ([0-9:]+)', status_text)
        if remaining_match:
            status_data['remaining'] = remaining_match.group(1)

        # Extract realtime factor
        realtime_match = re.search(r'Realtime: ([0-9.]+x|Calculating\.\.\.)', status_text)
        if realtime_match:
            rt = realtime_match.group(1)
            if rt != "Calculating...":
                status_data['realtime'] = rt

        # Extract VRAM usage
        vram_match = re.search(r'VRAM: ([0-9.]+GB)', status_text)
        if vram_match:
            status_data['vram'] = vram_match.group(1)

        # Extract audio duration
        audio_match = re.search(r'Audio: ([0-9:]+)', status_text)
        if audio_match:
            status_data['audio'] = audio_match.group(1)


        # Emit structured data if we found any
        if status_data:
            self.structured_status_signal.emit(status_data)

    def parse_chunk_progress(self, output_text):
        """Parse chunk progress from stdout text"""
        import re

        # Look for chunk completion messages: "✅ Completed chunk X/Y"
        completed_match = re.search(r'✅ Completed chunk (\d+)/(\d+)', output_text)
        if completed_match:
            current = int(completed_match.group(1))
            total = int(completed_match.group(2))

            # Update status panels with progress and operation
            status_data = {
                'operation': f'Processing chunk {current} of {total}',
                'progress': (current, total)
            }
            self.structured_status_signal.emit(status_data)

        # Look for operation messages like "🔇 Added period silence to chunk"
        operation_match = re.search(r'🔇 Added (.+) to chunk (\d+)', output_text)
        if operation_match:
            silence_type = operation_match.group(1)
            chunk_num = operation_match.group(2)

            status_data = {
                'chunk_info': f'Adding {silence_type} (chunk {int(chunk_num):05d})'
            }
            self.structured_status_signal.emit(status_data)

    def run(self):
        """Redirects stdout to a custom GUIOutput class that emits text to a GUI signal.
        Args:
        output_signal: A signal object used to send text to the GUI.
        thread_instance: The instance of the thread where this method is called.
        Returns:
        None.
        """
        try:
            # Redirect stdout to capture print statements
            import sys
            from io import StringIO

            # Create a custom stdout that emits to GUI
            class GUIOutput:
                """Manages output redirection and signal emission for a GUI application thread. Captures non-empty lines of text, emits them via a signal, and passes them to a chunk progress parser. Also writes output to original stdout for debugging purposes."""
                def __init__(self, output_signal, thread_instance):
                    """Initializes a logger that captures and processes output.
                    Args:
                    output_signal: A signal emitter for non-empty lines.
                    thread_instance: An instance of ThreadParser to handle chunk progress.
                    Returns:
                    None
                    """
                    self.output_signal = output_signal
                    self.thread_instance = thread_instance
                    self.original_stdout = sys.stdout

                def write(self, text):
                    """Redirects `sys.stdout` to a custom output handler that filters and processes text before writing it to both a signal for GUI updates and the original stdout for debugging purposes. The `flush` method ensures that any buffered data is written out.
                    Args:
                    text (str): Text to be processed and redirected.
                    Returns:
                    None
                    """
                    if text.strip():  # Only emit non-empty lines
                        self.output_signal.emit(text.strip())
                        # Also parse for chunk progress
                        self.thread_instance.parse_chunk_progress(text.strip())
                    # Also write to original stdout for debugging
                    self.original_stdout.write(text)

                def flush(self):
                    """Flushes the original stdout buffer.
                    Args:
                    None
                    Returns:
                    None
                    """
                    self.original_stdout.flush()

            # Temporarily redirect stdout
            gui_output = GUIOutput(self.output_signal, self)
            original_stdout = sys.stdout
            sys.stdout = gui_output

            try:
                # Set up status callback for progress tracking
                from modules.progress_tracker import log_chunk_progress
                log_chunk_progress._status_callback = self.parse_and_emit_status

                result = self.target_function(*self.args, **self.kwargs)
                self.finished_signal.emit(True, "Process completed successfully")
            finally:
                # Clean up status callback
                from modules.progress_tracker import log_chunk_progress
                if hasattr(log_chunk_progress, '_status_callback'):
                    log_chunk_progress._status_callback = None

                # Always restore original stdout
                sys.stdout = original_stdout

        except Exception as e:
            self.output_signal.emit(f"❌ Error: {str(e)}")
            self.finished_signal.emit(False, str(e))


class ChatterboxMainWindow(QMainWindow):
    """A main window for the ChatterboxTTS audiobook generator application, providing a graphical interface for user interaction and configuration management."""
    def __init__(self):
        """Initialize the main window for the ChatterboxTTS application. Sets up the window title, geometry, and initializes settings for persistent folder memory. Tracks unsaved configuration changes. Central widget is set to a tabbed interface.
        Args:
        None
        Returns:
        None
        """
        super().__init__()
        self.setWindowTitle("ChatterboxTTS - Audiobook Generator")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize settings for persistent folder memory
        self.settings = QSettings("ChatterboxTTS", "GUI")

        # Track unsaved config changes
        self.config_has_unsaved_changes = False
        self.original_config_values = {}

        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Create vertical splitter for tabs and output log
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # Create scroll area for tabs (top part of splitter)
        self.main_scroll_area = QScrollArea()
        self.main_scroll_area.setWidgetResizable(True)
        self.main_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.main_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create tab widget inside scroll area
        tab_container = QWidget()
        tab_layout = QVBoxLayout(tab_container)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        tab_layout.addWidget(self.tab_widget)

        # Connect tab change signal to check for unsaved config changes
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        self.main_scroll_area.setWidget(tab_container)
        splitter.addWidget(self.main_scroll_area)

        # Create output area widget (bottom part of splitter)
        output_widget = QWidget()
        self.create_output_area_widget(output_widget)
        splitter.addWidget(output_widget)

        # Set splitter proportions (tabs get more space, but output log always visible)
        splitter.setSizes([500, 200])  # Tabs: 500px, Output: 200px
        splitter.setCollapsible(1, False)  # Don't allow output log to collapse

        # Create tabs for each main menu option
        self.create_convert_book_tab()
        self.create_config_tab()
        self.create_resume_tab()
        self.create_combine_tab()
        self.create_prepare_text_tab()
        self.create_test_chunking_tab()
        self.create_repair_tool_tab()
        self.create_json_generate_tab()
        self.create_voice_analyzer_tab()
        self.create_audio_output_analyzer_tab()

        # Hide tabs as requested - keep only Convert Book, Config, Combine Audio, Repair Tool
        self.tab_widget.setTabVisible(2, False)  # 3. Resume Processing - HIDE
        self.tab_widget.setTabVisible(3, True)   # 4. Combine Audio - SHOW
        self.tab_widget.setTabVisible(4, False)  # 5. Prepare Text - HIDE
        self.tab_widget.setTabVisible(5, False)  # 6. Test Chunking - HIDE
        self.tab_widget.setTabVisible(6, True)   # 7. Repair Tool - SHOW
        self.tab_widget.setTabVisible(7, False)  # 8. Generate from JSON - HIDE
        self.tab_widget.setTabVisible(8, False)  # 9. Voice Analyzer - HIDE
        self.tab_widget.setTabVisible(9, False)  # 10. Audio Output Analyzer - HIDE

        # Status bar
        self.statusBar().showMessage("Ready")

        # Test audio system on startup (Windows compatibility)
        self.test_audio_system_startup()

    def closeEvent(self, event):
        """Handle GUI close event - cleanup audio playback"""
        # Stop any playing voice sample
        if hasattr(self, '_voice_playing') and self._voice_playing:
            self.stop_voice_sample()

        # Accept the close event
        event.accept()

    def test_audio_system_startup(self):
        """Test audio system on startup and show status"""
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.quit()
            # Audio system working - no message needed
        except ImportError:
            # pygame not available
            self.statusBar().showMessage("Ready - Voice preview unavailable (missing audio components)")
        except Exception as e:
            # pygame failed - show in status bar
            import platform
            if platform.system() == "Windows":
                self.statusBar().showMessage("Ready - Voice preview may not work (check audio drivers)")
            else:
                self.statusBar().showMessage("Ready - Voice preview unavailable")

    def create_convert_book_tab(self):
        """Tab 1: Convert a book (GenTTS) - Main functionality"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "1. Convert Book")

        # Create scroll area for Tab 1
        self.tab1_scroll_area = QScrollArea(tab)
        self.tab1_scroll_area.setWidgetResizable(True)
        self.tab1_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tab1_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create scrollable content widget
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)

        # Add scroll area to tab
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(self.tab1_scroll_area)
        self.tab1_scroll_area.setWidget(scroll_content)

        # Book Selection
        book_layout = QFormLayout()

        self.book_path_edit = QLineEdit()
        self.book_path_edit.setPlaceholderText("Select book folder...")
        book_browse_btn = QPushButton("Browse...")
        book_browse_btn.clicked.connect(self.browse_book_folder)

        book_row = QHBoxLayout()
        book_row.addWidget(self.book_path_edit)
        book_row.addWidget(book_browse_btn)
        book_layout.addRow("Book Folder:", book_row)
        self.book_path_edit.textChanged.connect(self.on_book_selection_changed)

        # Text file selection (populated after book selection)
        self.text_file_combo = QComboBox()
        book_layout.addRow("Text File:", self.text_file_combo)

        layout.addLayout(book_layout)

        # Voice Selection
        voice_layout = QFormLayout()

        self.voice_path_edit = QLineEdit()
        self.voice_path_edit.setPlaceholderText("Select voice sample...")

        voice_browse_btn = QPushButton("Browse...")
        voice_browse_btn.clicked.connect(self.browse_voice_file)
        voice_browse_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Add play button for voice sample
        self.voice_play_btn = QPushButton("▶️ Play")
        self.voice_play_btn.clicked.connect(self.play_voice_sample)
        self.voice_play_btn.setToolTip("Play selected voice sample")
        self.voice_play_btn.setEnabled(False)  # Disabled until file selected
        self.voice_play_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Add stop button for voice sample
        self.voice_stop_btn = QPushButton("⏹️ Stop")
        self.voice_stop_btn.clicked.connect(self.stop_voice_sample)
        self.voice_stop_btn.setToolTip("Stop voice sample playback")
        self.voice_stop_btn.setEnabled(False)  # Disabled until playing
        self.voice_stop_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice_path_edit, 1)  # Stretch factor 1 - expands to fill space
        voice_row.addWidget(voice_browse_btn, 0)      # Stretch factor 0 - stays compact
        voice_row.addWidget(self.voice_play_btn, 0)   # Stretch factor 0 - stays compact
        voice_row.addWidget(self.voice_stop_btn, 0)   # Stretch factor 0 - stays compact
        voice_layout.addRow("Voice Sample:", voice_row)

        layout.addLayout(voice_layout)



        # VADER and ASR Settings
        vader_layout = QVBoxLayout()

        # VADER and ASR Settings
        vader_layout = QVBoxLayout()

        # Horizontal layout for VADER and ASR checkboxes
        checkbox_layout = QHBoxLayout()

        self.asr_checkbox = QCheckBox("🎤 Enable ASR validation")
        self.asr_checkbox.setChecked(False)
        self.asr_checkbox.setToolTip("Smart quality control with automatic model selection")
        checkbox_layout.addWidget(self.asr_checkbox)

        self.validate_btn = QPushButton("🔍 Validate")
        self.validate_btn.setEnabled(False)
        self.validate_btn.setToolTip("Run ASR validation on completed conversion")
        self.validate_btn.clicked.connect(self.run_validation)
        checkbox_layout.addWidget(self.validate_btn)

        vader_layout.addLayout(checkbox_layout)

        # ASR threshold control (shows when ASR enabled)
        self.asr_threshold_container = QWidget()
        threshold_layout = QHBoxLayout(self.asr_threshold_container)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_label = QLabel("Similarity Threshold:")
        threshold_label.setStyleSheet("font-weight: bold;")
        self.asr_threshold_spinner = NoScrollDoubleSpinBox()
        self.asr_threshold_spinner.setRange(0.5, 1.0)
        self.asr_threshold_spinner.setSingleStep(0.01)
        self.asr_threshold_spinner.setDecimals(2)
        self.asr_threshold_spinner.setValue(DEFAULT_ASR_THRESHOLD)
        self.asr_threshold_spinner.setToolTip("Minimum similarity score required for ASR validation to pass")
        self.asr_threshold_spinner.setMaximumWidth(80)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.asr_threshold_spinner)
        threshold_layout.addStretch()
        self.asr_threshold_container.setVisible(False)
        vader_layout.addWidget(self.asr_threshold_container)

#        vader_info = QLabel("VADER dynamically adjusts TTS parameters based on emotional content of each chunk")
#        vader_info.setStyleSheet("color: #666; font-style: italic;")
#        vader_layout.addWidget(vader_info)

        layout.addLayout(vader_layout)

        # Middle section: TTS Parameters (left) + Quality Enhancement (right)
        middle_container = QHBoxLayout()

        # LEFT COLUMN: TTS Parameters - All in vertical list
        tts_params_layout = QFormLayout()

        # Temperature - Tab 1 uses config limits
        self.temperature_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.temperature_spin, 'DEFAULT_TEMPERATURE')
        self.temperature_spin.setRange(TTS_PARAM_MIN_TEMPERATURE, TTS_PARAM_MAX_TEMPERATURE)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(DEFAULT_TEMPERATURE)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setMaximumWidth(100)
        tts_params_layout.addRow("Temperature:", self.temperature_spin)

        # Min-P
        self.min_p_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.min_p_spin, 'DEFAULT_MIN_P')
        self.min_p_spin.setRange(0.0, 0.5)
        self.min_p_spin.setSingleStep(0.01)
        self.min_p_spin.setValue(0.05)  # DEFAULT_MIN_P
        self.min_p_spin.setDecimals(3)
        self.min_p_spin.setMaximumWidth(100)
        tts_params_layout.addRow("Min-P:", self.min_p_spin)

        # Top-P
        self.top_p_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.top_p_spin, 'DEFAULT_TOP_P')
        self.top_p_spin.setRange(0.5, 1.0)
        self.top_p_spin.setSingleStep(0.1)
        self.top_p_spin.setValue(1.0)  # DEFAULT_TOP_P
        self.top_p_spin.setDecimals(2)
        self.top_p_spin.setMaximumWidth(100)
        tts_params_layout.addRow("Top-P:", self.top_p_spin)

        # Repetition Penalty
        self.repetition_penalty_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.repetition_penalty_spin, 'DEFAULT_REPETITION_PENALTY')
        self.repetition_penalty_spin.setRange(1.0, 3.0)
        self.repetition_penalty_spin.setSingleStep(0.1)
        self.repetition_penalty_spin.setValue(1.2)  # DEFAULT_REPETITION_PENALTY
        self.repetition_penalty_spin.setDecimals(1)
        self.repetition_penalty_spin.setMaximumWidth(100)
        tts_params_layout.addRow("Rep. Penalty:", self.repetition_penalty_spin)

        tts_params_widget = QWidget()
        tts_params_widget.setLayout(tts_params_layout)

        # RIGHT COLUMN: Quality Enhancement Settings
        quality_layout = QHBoxLayout()  # Horizontal layout for regen + detection side-by-side

        # Regeneration Loop Section
        regen_layout = QFormLayout()

        self.regeneration_enabled_checkbox = QCheckBox("Enable automatic chunk regeneration on quality failure")
        self.regeneration_enabled_checkbox.setChecked(ENABLE_REGENERATION_LOOP)
        regen_layout.addRow(self.regeneration_enabled_checkbox)

        # Max attempts
        self.max_attempts_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.max_attempts_spin, 'MAX_REGENERATION_ATTEMPTS')
        self.max_attempts_spin.setRange(1, 10)
        self.max_attempts_spin.setValue(MAX_REGENERATION_ATTEMPTS)
        self.max_attempts_spin.setMaximumWidth(60)  # Reduced width
        regen_layout.addRow("Max Attempts:", self.max_attempts_spin)

        # Quality threshold
        self.quality_threshold_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.quality_threshold_spin, 'QUALITY_THRESHOLD')
        self.quality_threshold_spin.setRange(0.1, 1.0)
        self.quality_threshold_spin.setSingleStep(0.05)
        self.quality_threshold_spin.setValue(QUALITY_THRESHOLD)
        self.quality_threshold_spin.setDecimals(2)
        self.quality_threshold_spin.setMaximumWidth(80)  # Reduced width
        regen_layout.addRow("Quality Threshold:", self.quality_threshold_spin)

        # Advanced Detection Section
        detection_layout = QFormLayout()

        self.mfcc_validation_checkbox = QCheckBox("Enable MFCC-based spectral analysis")
        self.mfcc_validation_checkbox.setChecked(ENABLE_MFCC_VALIDATION)
        detection_layout.addRow(self.mfcc_validation_checkbox)

        self.output_validation_checkbox = QCheckBox("Enable output validation")
        self.output_validation_checkbox.setChecked(ENABLE_OUTPUT_VALIDATION)
        detection_layout.addRow(self.output_validation_checkbox)

        # Note about ASR consolidation
        asr_note = QLabel("ℹ️ Reuses existing ASR model if available")
        asr_note.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        detection_layout.addRow(asr_note)

        self.spectral_threshold_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.spectral_threshold_spin, 'SPECTRAL_ANOMALY_THRESHOLD')
        self.spectral_threshold_spin.setRange(0.1, 1.0)
        self.spectral_threshold_spin.setSingleStep(0.05)
        self.spectral_threshold_spin.setValue(SPECTRAL_ANOMALY_THRESHOLD)
        self.spectral_threshold_spin.setDecimals(2)
        self.spectral_threshold_spin.setMaximumWidth(80)  # Reduced width
        detection_layout.addRow("Spectral Threshold:", self.spectral_threshold_spin)

        self.output_threshold_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.output_threshold_spin, 'OUTPUT_VALIDATION_THRESHOLD')
        self.output_threshold_spin.setRange(0.1, 1.0)
        self.output_threshold_spin.setSingleStep(0.05)
        self.output_threshold_spin.setValue(OUTPUT_VALIDATION_THRESHOLD)
        self.output_threshold_spin.setDecimals(2)
        self.output_threshold_spin.setMaximumWidth(80)
        detection_layout.addRow("Output Threshold:", self.output_threshold_spin)

        # Add all sections to horizontal layout
        # Create simple containers for the sections
        regen_widget = QWidget()
        regen_widget.setLayout(regen_layout)

        detection_widget = QWidget()
        detection_widget.setLayout(detection_layout)

        quality_layout.addWidget(regen_widget)
        quality_layout.addWidget(detection_widget)

        # Add overall info label below the sections
        main_quality_layout = QVBoxLayout()
        main_quality_layout.addLayout(quality_layout)

#        quality_info = QLabel("💡 These settings control the Phase 1 quality enhancement features")
#        quality_info.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
#        main_quality_layout.addWidget(quality_info)

        # Add the whole quality section to main layout
        quality_container = QWidget()
        quality_container.setLayout(main_quality_layout)

        # Add both columns to middle container
        middle_container.addWidget(tts_params_widget)
        middle_container.addWidget(quality_container)

        layout.addLayout(middle_container)

        # Action Buttons with Batch Checkbox
        button_layout = QHBoxLayout()

        self.convert_btn = QPushButton("🚀 Start Conversion")
        self.convert_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        self.convert_btn.clicked.connect(self.start_conversion)
        button_layout.addWidget(self.convert_btn)

        # Tab 1: Reload this tab from saved config
        self.tab1_reload_btn = QPushButton("Reload This Tab")
        self.tab1_reload_btn.setToolTip("Reload Tab 1 values from saved config")
        self.tab1_reload_btn.clicked.connect(self.reload_tab1_from_config)
        button_layout.addWidget(self.tab1_reload_btn)

        # ASR Event Handlers
        self.asr_checkbox.stateChanged.connect(self.handle_asr_toggle)

        # Batch processing checkbox moved to button row
        self.add_to_batch_checkbox = QCheckBox("📦 Add to batch queue")
        button_layout.addWidget(self.add_to_batch_checkbox)

        # Add spacing
        button_layout.addSpacing(40)

        # M4B Playback Speed Control
        # M4B Sample Rate (left of speed spinner)
        sample_rate_label = QLabel("M4B Sample Rate:")
        button_layout.addWidget(sample_rate_label)

        self.main_m4b_sample_rate_combo = QComboBox()
        self.main_m4b_sample_rate_combo.addItems(["22050", "24000", "44100", "48000", "96000"])
        self.main_m4b_sample_rate_combo.setCurrentText(str(M4B_SAMPLE_RATE))
        self.main_m4b_sample_rate_combo.setMaximumWidth(80)
        self.main_m4b_sample_rate_combo.setToolTip("Output is 24KHz. This matches input samples. Use the drop down to change output. Note: increasing output only really increases size not quality.")
        button_layout.addWidget(self.main_m4b_sample_rate_combo)

        button_layout.addWidget(QLabel("Hz"))

        speed_label = QLabel("M4B Speed:")
        button_layout.addWidget(speed_label)

        self.main_playback_speed_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.main_playback_speed_spin, 'ATEMPO_SPEED')
        self.main_playback_speed_spin.setRange(0.5, 2.0)
        self.main_playback_speed_spin.setSingleStep(0.05)
        self.main_playback_speed_spin.setDecimals(2)
        self.main_playback_speed_spin.setValue(ATEMPO_SPEED)
        self.main_playback_speed_spin.setMaximumWidth(80)
        self.main_playback_speed_spin.setToolTip("Playback speed multiplier for M4B output (<1 slower, >1 faster)")
        button_layout.addWidget(self.main_playback_speed_spin)

        # Regenerate button
        self.regenerate_m4b_btn = QPushButton("REGENERATE")
        self.regenerate_m4b_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        self.regenerate_m4b_btn.clicked.connect(self.regenerate_m4b)
        self.regenerate_m4b_btn.setToolTip("Regenerate selected audio file with new speed setting")
        self.regenerate_m4b_btn.setEnabled(False)  # Disabled until file selected
        button_layout.addWidget(self.regenerate_m4b_btn)

        # Browse button (standard style like other browse buttons on Tab 1)
        self.browse_m4b_btn = QPushButton("Browse")
        self.browse_m4b_btn.clicked.connect(self.browse_m4b_file)
        self.browse_m4b_btn.setToolTip("Select WAV or M4B file to regenerate or play")
        button_layout.addWidget(self.browse_m4b_btn)

        # Audio Controls (replaces green ETA display)
        self.play_btn = QPushButton("PLAY")
        self.play_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_m4b_file)
        self.play_btn.setToolTip("Click Play after conversion to hear m4b. Browse and select a wav file. Click play to preview the file. Click regenerate to change the playback speed. Then click Play to hear the new file.")
        button_layout.addWidget(self.play_btn)

        # Track audio files for priority system
        self.selected_wav_file = None      # WAV file selected via browse button
        self.selected_m4b_file = None      # M4B file selected via browse button
        self.regenerated_m4b_file = None   # File created by regeneration
        self.converted_m4b_file = None     # File created by conversion
        self.current_m4b_file = None       # Legacy - most recent file for play button

# Redundant status display removed - information now in TTS generation status

        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Add structured status panel for Tab 1
        self.tab1_status_panel = StructuredStatusPanel("🚀 TTS Generation Status")
        layout.addWidget(self.tab1_status_panel)

        # Initialize real-time status manager for Tab 1
        self.tab1_status_manager = RealTimeStatusManager(self.tab1_status_panel)
        set_status_manager(self.tab1_status_manager)

        # Detect and update device status
        self.detect_and_update_device_status()

        # Connect audio control buttons
# Audio button connections moved to main button area creation
# Audio button connections moved to main button area creation

        # Initialize audio player state
        self.audio_player = None
        self.is_playing = False
        self.current_m4b_file = None

        layout.addStretch()

    def handle_vader_toggle(self, state):
        """When VADER is turned off, disable VADER in config flags."""
        is_checked = bool(state)
        try:
            from config import config as _cfg
        except Exception:
            _cfg = None

        if not is_checked:
            try:
                if _cfg is not None:
                    _cfg.ENABLE_VADER_MICRO_BATCHING = False
            except Exception:
                pass

    def _attach_spin_reset(self, spin, config_key: str):
        """Attaches a configuration key to a widget and sets up context menu handling.
        Args:
        - spin (QWidget): The widget to configure.
        - config_key (str): The configuration key to set.
        - widget (QWidget): The widget to configure.
        Returns:
        None
        """
        try:
            spin.setProperty('config_key', config_key)
            spin.setContextMenuPolicy(Qt.CustomContextMenu)
            spin.customContextMenuRequested.connect(lambda pos, s=spin: self._show_spin_context_menu(s, pos))
        except Exception:
            pass

    def _attach_config_key(self, widget, config_key: str):
        """Attaches a configuration key to a widget.
        Args:
        widget (QWidget): The widget to which the config key will be attached.
        config_key (str): The key to attach to the widget.
        Reads and returns the value from a widget.
        Args:
        w (QWidget): The widget from which to read the value.
        Returns:
        The value of the widget, converted to an appropriate type.
        """
        try:
            widget.setProperty('config_key', config_key)
        except Exception:
            pass

    def _read_widget_value(self, w):
        """Reads and returns the value of a widget.
        Args:
        w (QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox): The widget to read from.
        Returns:
        float, int, bool, or str: The value of the widget. Returns an integer for numeric spin boxes, a float for double spin boxes, a boolean for check boxes, and a string for combo boxes with numeric values.
        """
        try:
            from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox
            if isinstance(w, QDoubleSpinBox):
                return float(w.value())
            if isinstance(w, QSpinBox):
                return int(w.value())
            if isinstance(w, QCheckBox):
                return bool(w.isChecked())
            if isinstance(w, QComboBox):
                text = w.currentText()
                # Best-effort numeric cast for numeric combos
                try:
                    if text.strip().isdigit():
                        return int(text.strip())
                    return float(text)
                except Exception:
                    return text
        except Exception:
            pass
        return None

    def _build_effective_settings(self):
        """Normalizes GUI keys to runtime keys used by engine/modules.
        Args:
        None
        Returns:
        dict: A dictionary mapping normalized GUI keys to their corresponding runtime keys.
        """
        # Normalize GUI keys (CONFIG_VAR -> runtime key used by engine/modules)
        normalize = {
            'MAX_WORKERS': 'max_workers',
            'BATCH_SIZE': 'batch_size',
            'TTS_BATCH_SIZE': 'tts_batch_size',
            'MIN_CHUNK_WORDS': 'min_chunk_words',
            'MAX_CHUNK_WORDS': 'max_chunk_words',
            'CHUNKING_QUALITY': 'chunking_quality',
            'ENABLE_MID_DROP_CHECK': 'enable_mid_drop_check',
            'ENABLE_HUM_DETECTION': 'enable_hum_detection',
            'ENABLE_NORMALIZATION': 'enable_normalization',
            'NORMALIZATION_TYPE': 'normalization_type',
            'TARGET_LUFS': 'target_lufs',
            'TARGET_PEAK_DB': 'target_peak_db',
            'M4B_SAMPLE_RATE': 'm4b_sample_rate',
            'ENABLE_AUDIO_TRIMMING': 'enable_audio_trimming',
            'SPEECH_ENDPOINT_THRESHOLD': 'speech_threshold',
            'TRIMMING_BUFFER_MS': 'trimming_buffer',
            'ATEMPO_SPEED': 'playback_speed',
            'SILENCE_CHAPTER_START': 'silence_chapter_start',
            'SILENCE_CHAPTER_END': 'silence_chapter_end',
            'SILENCE_SECTION_BREAK': 'silence_section',
            'SILENCE_PARAGRAPH_END': 'silence_paragraph',
            'SILENCE_COMMA': 'silence_comma',
            'SILENCE_PERIOD': 'silence_period',
            'SILENCE_QUESTION_MARK': 'silence_question',
            'SILENCE_EXCLAMATION': 'silence_exclamation',
            'ENABLE_CHUNK_END_SILENCE': 'enable_chunk_silence',
            'CHUNK_END_SILENCE_MS': 'chunk_silence_duration',
            'VADER_TEMPERATURE_SENSITIVITY': 'vader_temp_sensitivity',
            'ENABLE_PUNCTUATION_PAUSES': 'enable_punctuation_pauses',
        }

        values = {}
        for w in self.findChildren(QWidget):
            key = w.property('config_key') if hasattr(w, 'property') else None
            if not key:
                continue
            val = self._read_widget_value(w)
            if val is None:
                continue
            runtime_key = normalize.get(key, None)
            if runtime_key:
                values[runtime_key] = val

        # Add inline pause values (not using config_key mechanism)
        if hasattr(self, 'enable_inline_pauses_checkbox'):
            values["enable_punctuation_pauses"] = self.enable_inline_pauses_checkbox.isChecked()
            values["inline_comma_ms"] = self.inline_comma_spin.value()
            values["inline_period_ms"] = self.inline_period_spin.value()
            values["inline_question_ms"] = self.inline_question_spin.value()
            values["inline_exclamation_ms"] = self.inline_exclamation_spin.value()

        return values

    def _show_spin_context_menu(self, spin, pos):
        """Displays a context menu for a spin widget.
        Args:
        spin: The QSpinbox widget to show the context menu for.
        pos: The position at which to show the context menu.
        Returns:
        None
        """
        try:
            menu = QMenu(spin)
            action = menu.addAction("Reset to saved config")
            action.triggered.connect(lambda _: self._reset_spin_from_saved(spin))
            menu.exec_(spin.mapToGlobal(pos))
        except Exception:
            pass

    def _reset_spin_from_saved(self, spin):
        """Resets the spin box value from a saved configuration key.
        Args:
        spin (QAbstractSpinBox): The spin box to reset.
        Returns:
        None
        """
        try:
            key = spin.property('config_key')
            if not key:
                return
            importlib.reload(config_mod)
            if not hasattr(config_mod, key):
                return
            value = getattr(config_mod, key)
            # Clamp to spin range
            try:
                if isinstance(spin, QDoubleSpinBox):
                    value = float(value)
                else:
                    value = int(value)
            except Exception:
                return
            value = max(spin.minimum(), min(spin.maximum(), value))
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
            self.statusBar().showMessage(f"Reset {key} from saved config", 2000)
        except Exception:
            pass

    def reload_tab1_from_config(self):
        """Reloads Tab 1 from configuration and reapply spinners.
        Args:
        None
        Returns:
        None
        """
        try:
            importlib.reload(config_mod)
            # Reapply Tab 1 spinners from saved config
            mapping = [
                (self.max_attempts_spin, 'MAX_REGENERATION_ATTEMPTS'),
                (self.quality_threshold_spin, 'QUALITY_THRESHOLD'),
                (self.spectral_threshold_spin, 'SPECTRAL_ANOMALY_THRESHOLD'),
                (self.output_threshold_spin, 'OUTPUT_VALIDATION_THRESHOLD'),
                (self.temperature_spin, 'DEFAULT_TEMPERATURE'),
                (self.min_p_spin, 'DEFAULT_MIN_P'),
                (self.top_p_spin, 'DEFAULT_TOP_P'),
                (self.repetition_penalty_spin, 'DEFAULT_REPETITION_PENALTY'),
                (self.main_playback_speed_spin, 'ATEMPO_SPEED'),
            ]
            for spin, key in mapping:
                if hasattr(config_mod, key):
                    val = getattr(config_mod, key)
                    if isinstance(spin, QDoubleSpinBox):
                        try:
                            val = float(val)
                        except Exception:
                            continue
                    else:
                        try:
                            val = int(val)
                        except Exception:
                            continue
                    val = max(spin.minimum(), min(spin.maximum(), val))
                    spin.blockSignals(True)
                    spin.setValue(val)
                    spin.blockSignals(False)
            self.statusBar().showMessage("Reloaded Tab 1 from saved config", 2500)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to reload Tab 1: {e}", 3000)

    def create_config_tab(self):
        """Tab 2: Configuration Settings"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "2. Config")

        # Create scrollable area for all settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Top Row - Core Settings with Group Boxes
        top_row_layout = QHBoxLayout()

        # Workers/Batch Settings Group
        workers_group = QGroupBox()
        workers_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        workers_layout = QFormLayout(workers_group)

        # Add descriptive text
        workers_desc = QLabel("Set # workers for parallel processing. (too many workers\nwill use up VRAM. Only increase if VRAM and GPU %\nare below 60% utilized.)\nBatch size: set to determine when model is reloaded to\nflush VRAM and avoid recursive problems and slowdowns.")
        workers_desc.setStyleSheet("font-size: 10px; color: #666; margin: 5px;")
        workers_desc.setWordWrap(True)
        workers_layout.addRow(workers_desc)

        self.batch_size_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.batch_size_spin, 'BATCH_SIZE')
        self.batch_size_spin.setRange(50, 999999)
        self.batch_size_spin.setSingleStep(50)
        self.batch_size_spin.setValue(BATCH_SIZE)
        self.batch_size_spin.setMaximumWidth(60)
        workers_layout.addRow("Reload Model Batch Size:", self.batch_size_spin)

        # Word Count & Chunking Quality Group
        words_group = QGroupBox("Word Count & Chunking Quality")
        words_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        words_layout = QFormLayout(words_group)

        # Add descriptive text
        words_desc = QLabel("Set Min/Max for words in a text chunk. Too many\nwords can lead to poor TTS.")
        words_desc.setStyleSheet("font-size: 10px; color: #666; margin: 5px;")
        words_desc.setWordWrap(True)
        words_layout.addRow(words_desc)

        self.min_chunk_words_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.min_chunk_words_spin, 'MIN_CHUNK_WORDS')
        self.min_chunk_words_spin.setRange(1, 50)
        self.min_chunk_words_spin.setValue(MIN_CHUNK_WORDS)
        self.min_chunk_words_spin.setMaximumWidth(60)
        words_layout.addRow("Min Words:", self.min_chunk_words_spin)

        self.max_chunk_words_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.max_chunk_words_spin, 'MAX_CHUNK_WORDS')
        self.max_chunk_words_spin.setRange(10, 100)
        self.max_chunk_words_spin.setValue(MAX_CHUNK_WORDS)
        self.max_chunk_words_spin.setMaximumWidth(60)
        words_layout.addRow("Max Words:", self.max_chunk_words_spin)

        # Chunking Quality Selection
        quality_desc = QLabel("Low: Original sentence-based chunking\nHigh: New paragraph-based chunking")
        quality_desc.setStyleSheet("font-size: 10px; color: #666; margin: 5px;")
        quality_desc.setWordWrap(True)
        words_layout.addRow(quality_desc)

        self.chunking_quality_combo = QComboBox()
        self.chunking_quality_combo.addItems(["Low", "High"])
        self.chunking_quality_combo.setCurrentText(CHUNKING_QUALITY)
        self.chunking_quality_combo.setMaximumWidth(100)
        self._attach_config_key(self.chunking_quality_combo, 'CHUNKING_QUALITY')
        words_layout.addRow("Chunking Quality:", self.chunking_quality_combo)

        # Audio Detection Group
        detection_group = QGroupBox()
        detection_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        detection_layout = QFormLayout(detection_group)

        # Add descriptive text
        detection_desc = QLabel("Detects when audio speech stops and just noise\nor silence follow at end of audio chunk.\nUse the threshold to change detection range of\nvoice.\nBuffer adds a ms silence to end of audio chunk.")
        detection_desc.setStyleSheet("font-size: 10px; color: #666; margin: 5px;")
        detection_desc.setWordWrap(True)
        detection_layout.addRow(detection_desc)

        self.normalization_check = QCheckBox("Audio normalization")
        self.normalization_check.setChecked(ENABLE_NORMALIZATION)
        self._attach_config_key(self.normalization_check, 'ENABLE_NORMALIZATION')

        self.target_lufs_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.target_lufs_spin, 'TARGET_LUFS')
        self.target_lufs_spin.setRange(-30, -6)
        self.target_lufs_spin.setValue(TARGET_LUFS)
        self.target_lufs_spin.setMaximumWidth(60)

        # Compact LUFS row
        lufs_layout = QHBoxLayout()
        lufs_layout.addWidget(self.normalization_check)
        lufs_layout.addWidget(QLabel("DB:"))
        lufs_layout.addWidget(self.target_lufs_spin)
        lufs_layout.addStretch()
        detection_layout.addRow(lufs_layout)

        # M4B Sample Rate setting
        self.m4b_sample_rate_combo = QComboBox()
        self.m4b_sample_rate_combo.addItems(["22050", "24000", "44100", "48000", "96000"])
        self.m4b_sample_rate_combo.setCurrentText(str(M4B_SAMPLE_RATE))
        self.m4b_sample_rate_combo.setMaximumWidth(80)
        self._attach_config_key(self.m4b_sample_rate_combo, 'M4B_SAMPLE_RATE')

        m4b_layout = QHBoxLayout()
        m4b_layout.addWidget(QLabel("M4B Sample Rate:"))
        m4b_layout.addWidget(self.m4b_sample_rate_combo)
        m4b_layout.addWidget(QLabel("Hz"))
        m4b_layout.addStretch()
        detection_layout.addRow(m4b_layout)

        self.audio_trimming_check = QCheckBox("Automatic audio trimming")
        self.audio_trimming_check.setChecked(ENABLE_AUDIO_TRIMMING)
        self._attach_config_key(self.audio_trimming_check, 'ENABLE_AUDIO_TRIMMING')
        detection_layout.addRow(self.audio_trimming_check)

        # Trimming details
        self.speech_threshold_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.speech_threshold_spin, 'SPEECH_ENDPOINT_THRESHOLD')
        self.speech_threshold_spin.setRange(0.001, 0.1)
        self.speech_threshold_spin.setSingleStep(0.001)
        self.speech_threshold_spin.setDecimals(3)
        self.speech_threshold_spin.setValue(SPEECH_ENDPOINT_THRESHOLD)
        self.speech_threshold_spin.setMaximumWidth(80)

        self.trimming_buffer_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.trimming_buffer_spin, 'TRIMMING_BUFFER_MS')
        self.trimming_buffer_spin.setRange(0, 500)
        self.trimming_buffer_spin.setValue(TRIMMING_BUFFER_MS)
        self.trimming_buffer_spin.setMaximumWidth(60)

        trim_details_layout = QHBoxLayout()
        trim_details_layout.addWidget(QLabel("Threshold:"))
        trim_details_layout.addWidget(self.speech_threshold_spin)
        trim_details_layout.addWidget(QLabel("Buffer:"))
        trim_details_layout.addWidget(self.trimming_buffer_spin)
        trim_details_layout.addStretch()
        detection_layout.addRow(trim_details_layout)

        # Add columns to top row
        top_row_layout.addWidget(workers_group)
        top_row_layout.addWidget(words_group)
        top_row_layout.addWidget(detection_group)
        top_row_layout.addStretch()

        scroll_layout.addLayout(top_row_layout)

        # Hidden advanced settings (keep for compatibility but don't show)
        self.normalization_type_combo = QComboBox()
        self.normalization_type_combo.addItems(["loudness", "peak", "simple", "none"])
        self.normalization_type_combo.setCurrentText(NORMALIZATION_TYPE)
        self._attach_config_key(self.normalization_type_combo, 'NORMALIZATION_TYPE')
        self.normalization_type_combo.setVisible(False)  # Hidden

        # Additional audio settings (Hidden but needed for compatibility)
        self.target_peak_db_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.target_peak_db_spin, 'TARGET_PEAK_DB')
        self.target_peak_db_spin.setRange(-6.0, 0.0)
        self.target_peak_db_spin.setSingleStep(0.1)
        self.target_peak_db_spin.setDecimals(1)
        self.target_peak_db_spin.setValue(TARGET_PEAK_DB)
        self.target_peak_db_spin.setMaximumWidth(60)
        self.target_peak_db_spin.setVisible(False)  # Hidden

        # Missing controls that need to be kept for compatibility
        self.mid_drop_check = QCheckBox("Mid-chunk energy drop detection")
        self.mid_drop_check.setChecked(ENABLE_MID_DROP_CHECK)
        self._attach_config_key(self.mid_drop_check, 'ENABLE_MID_DROP_CHECK')
        self.mid_drop_check.setVisible(False)  # Hidden

        self.hum_detection_check = QCheckBox("TTS hum artifact detection")
        self.hum_detection_check.setChecked(ENABLE_HUM_DETECTION)
        self._attach_config_key(self.hum_detection_check, 'ENABLE_HUM_DETECTION')
        self.hum_detection_check.setVisible(False)  # Hidden

        # TTS Parameter Limits Group
        limits_group = QGroupBox("🔒 TTS Parameter Limits")
        limits_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        limits_layout = QFormLayout(limits_group)

        # Add descriptive text
        limits_desc = QLabel("Set the upper and lower limits that\nTTS Params can be automatically\nadjusted to by Vader and other\nfunctions.")
        limits_desc.setStyleSheet("font-size: 10px; color: #666; margin: 5px;")
        limits_desc.setWordWrap(True)
        limits_layout.addRow(limits_desc)

        # CFG Weight limits - Config tab uses software max (1.0)
        self.cfg_min_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.cfg_min_spin, 'TTS_PARAM_MIN_CFG_WEIGHT')
        self.cfg_min_spin.setRange(0.0, 1.0)  # Software max = 1.0
        self.cfg_min_spin.setSingleStep(0.05)
        self.cfg_min_spin.setDecimals(2)
        self.cfg_min_spin.setValue(TTS_PARAM_MIN_CFG_WEIGHT)
        self.cfg_min_spin.setMaximumWidth(60)

        self.cfg_max_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.cfg_max_spin, 'TTS_PARAM_MAX_CFG_WEIGHT')
        self.cfg_max_spin.setRange(0.0, 1.0)  # Software max = 1.0
        self.cfg_max_spin.setSingleStep(0.05)
        self.cfg_max_spin.setDecimals(2)
        self.cfg_max_spin.setValue(TTS_PARAM_MAX_CFG_WEIGHT)
        self.cfg_max_spin.setMaximumWidth(60)

        # Temperature limits - Config tab uses software max (5.0)
        self.temp_min_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.temp_min_spin, 'TTS_PARAM_MIN_TEMPERATURE')
        self.temp_min_spin.setRange(0.0, 5.0)  # Software max = 5.0
        self.temp_min_spin.setSingleStep(0.05)
        self.temp_min_spin.setDecimals(2)
        self.temp_min_spin.setValue(TTS_PARAM_MIN_TEMPERATURE)
        self.temp_min_spin.setMaximumWidth(60)

        self.temp_max_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.temp_max_spin, 'TTS_PARAM_MAX_TEMPERATURE')
        self.temp_max_spin.setRange(0.0, 5.0)  # Software max = 5.0
        self.temp_max_spin.setSingleStep(0.05)
        self.temp_max_spin.setDecimals(2)
        self.temp_max_spin.setValue(TTS_PARAM_MAX_TEMPERATURE)
        self.temp_max_spin.setMaximumWidth(60)

        # Compact rows for min/max limits
        cfg_limits_row = QHBoxLayout()
        cfg_limits_row.addWidget(QLabel("CFG Limits:"))
        cfg_limits_row.addWidget(QLabel("Min:"))
        cfg_limits_row.addWidget(self.cfg_min_spin)
        cfg_limits_row.addWidget(QLabel("Max:"))
        cfg_limits_row.addWidget(self.cfg_max_spin)
        cfg_limits_row.addStretch()

        temp_limits_row = QHBoxLayout()
        temp_limits_row.addWidget(QLabel("Temp Limits:"))
        temp_limits_row.addWidget(QLabel("Min:"))
        temp_limits_row.addWidget(self.temp_min_spin)
        temp_limits_row.addWidget(QLabel("Max:"))
        temp_limits_row.addWidget(self.temp_max_spin)
        temp_limits_row.addStretch()

        limits_layout.addRow(cfg_limits_row)
        limits_layout.addRow(temp_limits_row)

        scroll_layout.addWidget(limits_group)

        # TTS Defaults and VADER Sensitivity Section
        middle_row_layout = QHBoxLayout()

        # TTS Defaults Group
        tts_defaults_group = QGroupBox("TTS Defaults")
        tts_defaults_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        tts_defaults_layout = QFormLayout(tts_defaults_group)

        # Add descriptive text
        tts_defaults_desc = QLabel("Exaq: 0.50  CFG: 0.50  Temp: 0.80")
        tts_defaults_desc.setStyleSheet("font-size: 10px; color: #666; margin: 5px;")
        tts_defaults_layout.addRow(tts_defaults_desc)

        # Base TTS values - Config tab uses software max
        self.default_exag_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.default_exag_spin, 'DEFAULT_EXAGGERATION')
        self.default_exag_spin.setRange(0.0, 2.0)  # Software max = 2.0
        self.default_exag_spin.setSingleStep(0.05)
        self.default_exag_spin.setDecimals(2)
        self.default_exag_spin.setValue(DEFAULT_EXAGGERATION)
        self.default_exag_spin.setMaximumWidth(60)

        self.default_cfg_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.default_cfg_spin, 'DEFAULT_CFG_WEIGHT')
        self.default_cfg_spin.setRange(0.0, 1.0)  # Software max = 1.0
        self.default_cfg_spin.setSingleStep(0.05)
        self.default_cfg_spin.setDecimals(2)
        self.default_cfg_spin.setValue(DEFAULT_CFG_WEIGHT)
        self.default_cfg_spin.setMaximumWidth(60)

        self.default_temp_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.default_temp_spin, 'DEFAULT_TEMPERATURE')
        self.default_temp_spin.setRange(0.0, 5.0)  # Software max = 5.0
        self.default_temp_spin.setSingleStep(0.05)
        self.default_temp_spin.setDecimals(2)
        self.default_temp_spin.setValue(DEFAULT_TEMPERATURE)
        self.default_temp_spin.setMaximumWidth(60)

        # Compact TTS defaults row
        tts_defaults_row = QHBoxLayout()
        tts_defaults_row.addWidget(QLabel("Exaq:"))
        tts_defaults_row.addWidget(self.default_exag_spin)
        tts_defaults_row.addWidget(QLabel("CFG:"))
        tts_defaults_row.addWidget(self.default_cfg_spin)
        tts_defaults_row.addWidget(QLabel("Temp:"))
        tts_defaults_row.addWidget(self.default_temp_spin)
        tts_defaults_row.addStretch()
        tts_defaults_layout.addRow(tts_defaults_row)

        # VADER Sensitivity Group
        vader_group = QGroupBox("VADER Sensitivity")
        vader_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        vader_layout = QFormLayout(vader_group)

        # Add descriptive text
        vader_desc = QLabel("Exaq Sens: 0.30  CFG Sens: 0.30  Temp Sens: 0.30\n\nSet defaults for TTS params Exaggeration, CFG, Temperature. Use Tab #1 to adjust for\na single use.\nVader Sensitivity: Sets how much vader adjusts the above params based on Vader weight.")
        vader_desc.setStyleSheet("font-size: 10px; color: #666; margin: 5px;")
        vader_desc.setWordWrap(True)
        vader_layout.addRow(vader_desc)



        # VADER Sensitivity
        self.vader_temp_sens_spin = NoScrollDoubleSpinBox()
        self._attach_spin_reset(self.vader_temp_sens_spin, 'VADER_TEMPERATURE_SENSITIVITY')
        self.vader_temp_sens_spin.setRange(0.0, 1.0)
        self.vader_temp_sens_spin.setSingleStep(0.01)
        self.vader_temp_sens_spin.setDecimals(2)
        self.vader_temp_sens_spin.setValue(VADER_TEMPERATURE_SENSITIVITY)
        self.vader_temp_sens_spin.setMaximumWidth(60)

        # Compact VADER sensitivity row
        vader_sens_row = QHBoxLayout()
        vader_sens_row.addWidget(QLabel("Temp Sens:"))
        vader_sens_row.addWidget(self.vader_temp_sens_spin)
        vader_sens_row.addStretch()
        vader_layout.addRow(vader_sens_row)

        # Add both groups to middle row
        middle_row_layout.addWidget(tts_defaults_group)
        middle_row_layout.addWidget(vader_group)
        middle_row_layout.addStretch()

        scroll_layout.addLayout(middle_row_layout)

        # Silence Settings Group
        silence_group = QGroupBox()
        silence_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        silence_layout = QFormLayout(silence_group)

        # Add descriptive text
        silence_desc = QLabel("Set the silence added to audio chunks for each type of chunk. ie chapter start/end, period,\nparagraph. For each setting silence is added for pacing.")
        silence_desc.setStyleSheet("font-size: 10px; color: #666; margin: 5px;")
        silence_desc.setWordWrap(True)
        silence_layout.addRow(silence_desc)

        # Chapter/Section silence
        self.silence_chapter_start_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.silence_chapter_start_spin, 'SILENCE_CHAPTER_START')
        self.silence_chapter_start_spin.setRange(0, 9999)
        self.silence_chapter_start_spin.setValue(SILENCE_CHAPTER_START)
        self.silence_chapter_start_spin.setMaximumWidth(60)

        self.silence_chapter_end_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.silence_chapter_end_spin, 'SILENCE_CHAPTER_END')
        self.silence_chapter_end_spin.setRange(0, 9999)
        self.silence_chapter_end_spin.setValue(SILENCE_CHAPTER_END)
        self.silence_chapter_end_spin.setMaximumWidth(60)

        self.silence_section_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.silence_section_spin, 'SILENCE_SECTION_BREAK')
        self.silence_section_spin.setRange(0, 9999)
        self.silence_section_spin.setValue(SILENCE_SECTION_BREAK)
        self.silence_section_spin.setMaximumWidth(60)

        self.silence_paragraph_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.silence_paragraph_spin, 'SILENCE_PARAGRAPH_END')
        self.silence_paragraph_spin.setRange(0, 9999)
        self.silence_paragraph_spin.setValue(SILENCE_PARAGRAPH_END)
        self.silence_paragraph_spin.setMaximumWidth(60)

        # Compact chapter/section row
        chapter_layout = QHBoxLayout()
        chapter_layout.addWidget(QLabel("Ch Start:"))
        chapter_layout.addWidget(self.silence_chapter_start_spin)
        chapter_layout.addWidget(QLabel("Ch End:"))
        chapter_layout.addWidget(self.silence_chapter_end_spin)
        chapter_layout.addWidget(QLabel("Section:"))
        chapter_layout.addWidget(self.silence_section_spin)
        chapter_layout.addWidget(QLabel("Para:"))
        chapter_layout.addWidget(self.silence_paragraph_spin)
        chapter_layout.addStretch()

        # Punctuation silence
        self.silence_comma_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.silence_comma_spin, 'SILENCE_COMMA')
        self.silence_comma_spin.setRange(0, 9999)
        self.silence_comma_spin.setValue(SILENCE_COMMA)
        self.silence_comma_spin.setMaximumWidth(60)

        self.silence_period_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.silence_period_spin, 'SILENCE_PERIOD')
        self.silence_period_spin.setRange(0, 9999)
        self.silence_period_spin.setValue(SILENCE_PERIOD)
        self.silence_period_spin.setMaximumWidth(60)

        self.silence_question_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.silence_question_spin, 'SILENCE_QUESTION_MARK')
        self.silence_question_spin.setRange(0, 9999)
        self.silence_question_spin.setValue(SILENCE_QUESTION_MARK)
        self.silence_question_spin.setMaximumWidth(60)

        self.silence_exclamation_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.silence_exclamation_spin, 'SILENCE_EXCLAMATION')
        self.silence_exclamation_spin.setRange(0, 9999)
        self.silence_exclamation_spin.setValue(SILENCE_EXCLAMATION)
        self.silence_exclamation_spin.setMaximumWidth(60)

        # Compact punctuation row
        punct_layout = QHBoxLayout()
        punct_layout.addWidget(QLabel("Comma:"))
        punct_layout.addWidget(self.silence_comma_spin)
        punct_layout.addWidget(QLabel("Period:"))
        punct_layout.addWidget(self.silence_period_spin)
        punct_layout.addWidget(QLabel("?:"))
        punct_layout.addWidget(self.silence_question_spin)
        punct_layout.addWidget(QLabel("!:"))
        punct_layout.addWidget(self.silence_exclamation_spin)
        punct_layout.addStretch()

        # Chunk silence settings
        self.chunk_end_silence_check = QCheckBox("Chunk End Silence")
        self.chunk_end_silence_spin = NoScrollSpinBox()
        self._attach_spin_reset(self.chunk_end_silence_spin, 'CHUNK_END_SILENCE_MS')
        self.chunk_end_silence_spin.setRange(0, 9999)
        self.chunk_end_silence_spin.setValue(CHUNK_END_SILENCE_MS)
        self.chunk_end_silence_spin.setMaximumWidth(60)

        chunk_silence_layout = QHBoxLayout()
        chunk_silence_layout.addWidget(self.chunk_end_silence_check)
        chunk_silence_layout.addWidget(self.chunk_end_silence_spin)
        chunk_silence_layout.addWidget(QLabel("ms"))
        chunk_silence_layout.addStretch()

        silence_layout.addRow("Structure (ms):", chapter_layout)
        silence_layout.addRow("Boundary Punct (ms):", punct_layout)
        silence_layout.addRow(chunk_silence_layout)

        # Inline pause settings with punctuation mapping
        inline_group = QGroupBox()
        inline_group.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }"
        )
        inline_main_layout = QVBoxLayout(inline_group)

        # Checkbox row
        checkbox_layout = QHBoxLayout()
        try:
            from config.config import ENABLE_PUNCTUATION_PAUSES
        except Exception:
            ENABLE_PUNCTUATION_PAUSES = False

        self.enable_inline_pauses_checkbox = QCheckBox("Enable Inline Pauses")
        self.enable_inline_pauses_checkbox.setChecked(bool(ENABLE_PUNCTUATION_PAUSES))
        self._attach_config_key(self.enable_inline_pauses_checkbox, "ENABLE_PUNCTUATION_PAUSES")
        checkbox_layout.addWidget(self.enable_inline_pauses_checkbox)
        checkbox_layout.addStretch()
        inline_main_layout.addLayout(checkbox_layout)

        # Punctuation pause spinners row (initially hidden)
        self.inline_pauses_widget = QWidget()
        inline_spinners_layout = QHBoxLayout(self.inline_pauses_widget)
        inline_spinners_layout.setContentsMargins(0, 0, 0, 0)

        # Load pause values from PUNCTUATION_PAUSE_MAPPING
        try:
            from config.config import PUNCTUATION_PAUSE_MAPPING
            comma_val = getattr(config, PUNCTUATION_PAUSE_MAPPING.get(',', 'SILENCE_COMMA'), 150)
            period_val = getattr(config, PUNCTUATION_PAUSE_MAPPING.get('.', 'SILENCE_PERIOD'), 550)
            question_val = getattr(config, PUNCTUATION_PAUSE_MAPPING.get('?', 'SILENCE_QUESTION_MARK'), 650)
            exclamation_val = getattr(config, PUNCTUATION_PAUSE_MAPPING.get('!', 'SILENCE_EXCLAMATION'), 200)
        except Exception:
            comma_val, period_val, question_val, exclamation_val = 150, 550, 650, 200

        self.inline_comma_spin = NoScrollSpinBox()
        self.inline_comma_spin.setRange(0, 5000)
        self.inline_comma_spin.setValue(comma_val)
        self.inline_comma_spin.setMaximumWidth(60)

        self.inline_period_spin = NoScrollSpinBox()
        self.inline_period_spin.setRange(0, 5000)
        self.inline_period_spin.setValue(period_val)
        self.inline_period_spin.setMaximumWidth(60)

        self.inline_question_spin = NoScrollSpinBox()
        self.inline_question_spin.setRange(0, 5000)
        self.inline_question_spin.setValue(question_val)
        self.inline_question_spin.setMaximumWidth(60)

        self.inline_exclamation_spin = NoScrollSpinBox()
        self.inline_exclamation_spin.setRange(0, 5000)
        self.inline_exclamation_spin.setValue(exclamation_val)
        self.inline_exclamation_spin.setMaximumWidth(60)

        inline_spinners_layout.addWidget(QLabel("Comma:"))
        inline_spinners_layout.addWidget(self.inline_comma_spin)
        inline_spinners_layout.addSpacing(10)
        inline_spinners_layout.addWidget(QLabel("Period:"))
        inline_spinners_layout.addWidget(self.inline_period_spin)
        inline_spinners_layout.addSpacing(10)
        inline_spinners_layout.addWidget(QLabel("?:"))
        inline_spinners_layout.addWidget(self.inline_question_spin)
        inline_spinners_layout.addSpacing(10)
        inline_spinners_layout.addWidget(QLabel("!:"))
        inline_spinners_layout.addWidget(self.inline_exclamation_spin)
        inline_spinners_layout.addStretch()

        inline_main_layout.addWidget(self.inline_pauses_widget)

        # Show/hide logic based on checkbox state
        self.inline_pauses_widget.setVisible(self.enable_inline_pauses_checkbox.isChecked())
        self.enable_inline_pauses_checkbox.stateChanged.connect(
            lambda state: self.inline_pauses_widget.setVisible(bool(state))
        )

        silence_layout.addRow(inline_group)

        scroll_layout.addWidget(silence_group)

        scroll_layout.addStretch()

        # Config action buttons
        config_buttons_layout = QHBoxLayout()

        save_btn = QPushButton("💾 Save Config")
        save_btn.clicked.connect(self.save_config_to_file)
        save_btn.setToolTip("Save current GUI settings to config file")
        save_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        config_buttons_layout.addWidget(save_btn)

        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.clicked.connect(self.reset_config_defaults)
        reset_btn.setToolTip("Reset all configuration values to defaults")
        config_buttons_layout.addWidget(reset_btn)

        config_buttons_layout.addStretch()
        scroll_layout.addLayout(config_buttons_layout)

        scroll.setWidget(scroll_widget)

        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)

        # Save original config values and set up change tracking
        self.save_original_config_values()
        self.setup_config_change_tracking()

    def create_resume_tab(self):
        """Tab 3: Resume from specific chunk"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "3. Resume Processing")

        layout = QVBoxLayout(tab)

        info_label = QLabel("🔄 Resume interrupted book processing from a specific chunk")
        info_label.setStyleSheet("font-weight: bold; color: #2196F3; padding: 10px;")
        layout.addWidget(info_label)

        # Incomplete books list
        incomplete_group = QGroupBox("📋 Incomplete Books")
        incomplete_layout = QVBoxLayout(incomplete_group)

        self.incomplete_books_list = QListWidget()
        incomplete_layout.addWidget(self.incomplete_books_list)

        refresh_btn = QPushButton("🔄 Refresh List")
        refresh_btn.clicked.connect(self.refresh_incomplete_books)
        incomplete_layout.addWidget(refresh_btn)

        layout.addWidget(incomplete_group)

        # Resume button
        resume_btn = QPushButton("▶️ Resume Selected Book")
        resume_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 10px; }")
        resume_btn.clicked.connect(self.resume_processing)
        layout.addWidget(resume_btn)

        # Add simple status bar for Tab 2 (Resume)
        self.tab2_status_bar = QProgressBar()
        self.tab2_status_bar.setVisible(False)
        layout.addWidget(self.tab2_status_bar)

        layout.addStretch()

    def create_combine_tab(self):
        """Tab 4: Combine audio chunks"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "4. Combine Audio")

        layout = QVBoxLayout(tab)

        info_label = QLabel("🎵 Combine processed audio chunks into final audiobook")
        info_label.setStyleSheet("font-weight: bold; color: #9C27B0; padding: 10px;")
        layout.addWidget(info_label)

        # Add folder selection note
        note_label = QLabel("📁 <b>Important:</b> Select the main book folder (e.g., 'Audiobook/BookName'), NOT the TTS or audio_chunks subfolder")
        note_label.setStyleSheet("color: #E65100; padding: 5px 10px; background-color: #FFF3E0; border: 1px solid #FFB74D; border-radius: 4px; margin: 5px;")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)

        # Book selection for combining
        combine_group = QGroupBox("📚 Select Book to Combine")
        combine_layout = QFormLayout(combine_group)

        self.combine_book_edit = QLineEdit()
        self.combine_book_edit.setPlaceholderText("Select book folder with audio chunks...")
        combine_browse_btn = QPushButton("Browse...")
        combine_browse_btn.clicked.connect(self.browse_combine_book)

        combine_row = QHBoxLayout()
        combine_row.addWidget(self.combine_book_edit)
        combine_row.addWidget(combine_browse_btn)
        combine_layout.addRow("Book Folder:", combine_row)

        layout.addWidget(combine_group)

        # Combine button
        combine_btn = QPushButton("🔗 Combine Audio Chunks")
        combine_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 10px; }")
        combine_btn.clicked.connect(self.combine_audio)
        layout.addWidget(combine_btn)

        # Add simple status bar for Tab 3 (Combine)
        self.tab3_status_bar = QProgressBar()
        self.tab3_status_bar.setVisible(False)
        layout.addWidget(self.tab3_status_bar)

        layout.addStretch()

    def create_prepare_text_tab(self):
        """Tab 5: Prepare text file for chunking"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "5. Prepare Text")

        layout = QVBoxLayout(tab)

        info_label = QLabel("📝 Prepare and chunk text files for processing")
        info_label.setStyleSheet("font-weight: bold; color: #607D8B; padding: 10px;")
        layout.addWidget(info_label)

        # Guidance note
        guidance_label = QLabel(
            "💡 <b>Important:</b> Configure TTS parameters and VADER settings on the <b>Main Tab</b> before preparing text.\n"
            "This process will use your current Main Tab settings for:\n"
            "• Base TTS parameters (Exaggeration, CFG Weight, Temperature)\n"
            "• VADER sentiment analysis (enabled/disabled)\n"
            "• Sentiment smoothing settings\n"
            "• VADER sensitivity values from Config Tab"
        )
        guidance_label.setStyleSheet(
            "background-color: #E3F2FD; "
            "border: 1px solid #2196F3; "
            "border-radius: 5px; "
            "padding: 10px; "
            "margin: 5px; "
            "color: #1565C0;"
        )
        guidance_label.setWordWrap(True)
        layout.addWidget(guidance_label)

        # Text file selection
        prepare_group = QGroupBox("📄 Text File Selection")
        prepare_layout = QFormLayout(prepare_group)

        self.prepare_text_edit = QLineEdit()
        self.prepare_text_edit.setPlaceholderText("Select text file to prepare...")
        prepare_browse_btn = QPushButton("Browse...")
        prepare_browse_btn.clicked.connect(self.browse_prepare_text)

        prepare_row = QHBoxLayout()
        prepare_row.addWidget(self.prepare_text_edit)
        prepare_row.addWidget(prepare_browse_btn)
        prepare_layout.addRow("Text File:", prepare_row)

        layout.addWidget(prepare_group)

        # Prepare button
        prepare_btn = QPushButton("📝 Prepare Text for Chunking")
        prepare_btn.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 10px; }")
        prepare_btn.clicked.connect(self.prepare_text)
        layout.addWidget(prepare_btn)

        # Add simple status bar for Tab 4 (Prepare Text)
        self.tab4_status_bar = QProgressBar()
        self.tab4_status_bar.setVisible(False)
        layout.addWidget(self.tab4_status_bar)

        layout.addStretch()

    def create_test_chunking_tab(self):
        """Tab 6: Test chunking logic"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "6. Test Chunking")

        layout = QVBoxLayout(tab)

        info_label = QLabel("🧪 Test and verify text chunking logic")
        info_label.setStyleSheet("font-weight: bold; color: #FF5722; padding: 10px;")
        layout.addWidget(info_label)

        # Test parameters
        test_group = QGroupBox("⚙️ Chunking Parameters")
        test_layout = QFormLayout(test_group)

        self.test_text_edit = QTextEdit()
        self.test_text_edit.setPlaceholderText("Enter test text with pause tags: Hello[pause:0.5s]world...")
        self.test_text_edit.setMaximumHeight(100)
        test_layout.addRow("Test Text:", self.test_text_edit)

        # Pause tag help label
        pause_help = QLabel("💡 Pause tag examples: [pause:0.5s] [pause:1.2s] [pause:500ms]")
        pause_help.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        test_layout.addRow("", pause_help)

        self.max_words_spin = NoScrollSpinBox()
        self.max_words_spin.setRange(1, 200)
        self.max_words_spin.setValue(30)
        test_layout.addRow("Max Words per Chunk:", self.max_words_spin)

        self.min_words_spin = NoScrollSpinBox()
        self.min_words_spin.setRange(1, 50)
        self.min_words_spin.setValue(4)
        test_layout.addRow("Min Words per Chunk:", self.min_words_spin)

        # Audio processing controls
        audio_group = QGroupBox("🔧 Audio Processing")
        audio_layout = QFormLayout(audio_group)

        # Artifact cleaning controls
        self.enable_artifact_cleaning = QCheckBox("Enable artifact cleaning")
        self.enable_artifact_cleaning.setChecked(False)
        self.enable_artifact_cleaning.setToolTip("Remove audio artifacts using auto-editor (slower processing)")
        audio_layout.addRow("", self.enable_artifact_cleaning)

        self.artifact_threshold_spin = NoScrollDoubleSpinBox()
        self.artifact_threshold_spin.setRange(0.01, 0.20)
        self.artifact_threshold_spin.setValue(0.06)
        self.artifact_threshold_spin.setSingleStep(0.01)
        self.artifact_threshold_spin.setDecimals(2)
        self.artifact_threshold_spin.setEnabled(False)
        self.artifact_threshold_spin.setToolTip("Detection threshold for artifacts (lower = more sensitive)")
        audio_layout.addRow("Artifact Threshold:", self.artifact_threshold_spin)

        self.artifact_margin_spin = NoScrollDoubleSpinBox()
        self.artifact_margin_spin.setRange(0.05, 0.50)
        self.artifact_margin_spin.setValue(0.20)
        self.artifact_margin_spin.setSingleStep(0.05)
        self.artifact_margin_spin.setDecimals(2)
        self.artifact_margin_spin.setEnabled(False)
        self.artifact_margin_spin.setToolTip("Safety margin around detected artifacts")
        audio_layout.addRow("Artifact Margin:", self.artifact_margin_spin)

        # Connect checkbox to enable/disable controls
        self.enable_artifact_cleaning.toggled.connect(self.artifact_threshold_spin.setEnabled)
        self.enable_artifact_cleaning.toggled.connect(self.artifact_margin_spin.setEnabled)

        layout.addWidget(test_group)
        layout.addWidget(audio_group)

        # Test button
        test_btn = QPushButton("🧪 Run Chunking Test")
        test_btn.setStyleSheet("QPushButton { background-color: #FF5722; color: white; font-weight: bold; padding: 10px; }")
        test_btn.clicked.connect(self.test_chunking)
        layout.addWidget(test_btn)

        layout.addStretch()

    def create_repair_tool_tab(self):
        """Tab 7: Chunk repair tool"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "7. Repair Tool")

        layout = QVBoxLayout(tab)

        info_label = QLabel("🔧 Chunk Repair and Revision Tool")
        info_label.setStyleSheet("font-weight: bold; color: #795548; padding: 10px;")
        layout.addWidget(info_label)

        # Book selection section
        book_group = QGroupBox("📚 Book Selection")
        book_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        book_layout = QVBoxLayout(book_group)

        self.repair_book_combo = QComboBox()
        self.repair_book_combo.currentTextChanged.connect(self.load_chunks_for_repair)
        book_layout.addWidget(QLabel("Select book to repair:"))
        book_layout.addWidget(self.repair_book_combo)

        refresh_books_btn = QPushButton("🔄 Refresh Book List")
        refresh_books_btn.clicked.connect(self.refresh_repair_books)
        book_layout.addWidget(refresh_books_btn)

        layout.addWidget(book_group)

        # Search section
        search_group = QGroupBox("🔍 Search and Select Chunks")
        search_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        search_layout = QVBoxLayout(search_group)

        search_layout.addWidget(QLabel("Search for text fragment:"))
        self.repair_search_edit = QLineEdit()
        self.repair_search_edit.setPlaceholderText("Enter text to search for...")
        self.repair_search_edit.returnPressed.connect(self.search_chunks_for_repair)
        search_layout.addWidget(self.repair_search_edit)

        search_btn = QPushButton("🔍 Search Chunks")
        search_btn.clicked.connect(self.search_chunks_for_repair)
        search_layout.addWidget(search_btn)

        # Results list
        search_layout.addWidget(QLabel("Search Results:"))
        self.repair_results_list = QListWidget()
        self.repair_results_list.itemClicked.connect(self.select_chunk_for_repair)
        search_layout.addWidget(self.repair_results_list)

        layout.addWidget(search_group)

        # Chunk editor section
        edit_group = QGroupBox("✏️ Edit Selected Chunk")
        edit_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        edit_layout = QVBoxLayout(edit_group)

        # Chunk info display
        self.repair_chunk_info = QLabel("No chunk selected")
        self.repair_chunk_info.setStyleSheet("background-color: #f5f5f5; padding: 10px; border: 1px solid #ddd;")
        edit_layout.addWidget(self.repair_chunk_info)

        # Text editing
        edit_layout.addWidget(QLabel("Chunk Text:"))
        self.repair_text_edit = QTextEdit()
        self.repair_text_edit.setMaximumHeight(100)
        edit_layout.addWidget(self.repair_text_edit)

        # Metadata editing
        metadata_layout = QHBoxLayout()

        # Boundary type
        metadata_layout.addWidget(QLabel("Boundary:"))
        self.repair_boundary_combo = QComboBox()
        # Add all possible boundary types from the text processor
        boundary_types = [
            "none", "paragraph_end", "chapter_start", "chapter_end", "section_break",
            "period", "comma", "semicolon", "colon", "question_mark", "exclamation",
            "dash", "ellipsis", "quote_end"
        ]
        self.repair_boundary_combo.addItems(boundary_types)
        metadata_layout.addWidget(self.repair_boundary_combo)

        # Sentiment score removed - it's just a stored value that doesn't affect processing

        edit_layout.addLayout(metadata_layout)

        # TTS parameters - group labels with their spinners
        tts_layout = QHBoxLayout()
        tts_layout.setSpacing(10)  # Reduce overall spacing

        # Exag parameter
        exag_label = QLabel("TTS Params - Exag:")
        self.repair_exag_spin = NoScrollDoubleSpinBox()
        self.repair_exag_spin.setRange(0.0, 3.0)
        self.repair_exag_spin.setSingleStep(0.1)
        self.repair_exag_spin.setDecimals(1)
        self.repair_exag_spin.setMaximumWidth(80)
        tts_layout.addWidget(exag_label)
        tts_layout.addWidget(self.repair_exag_spin)

        # CFG parameter
        cfg_label = QLabel("CFG:")
        self.repair_cfg_spin = NoScrollDoubleSpinBox()
        self.repair_cfg_spin.setRange(0.0, 2.0)
        self.repair_cfg_spin.setSingleStep(0.1)
        self.repair_cfg_spin.setDecimals(1)
        self.repair_cfg_spin.setMaximumWidth(80)
        tts_layout.addWidget(cfg_label)
        tts_layout.addWidget(self.repair_cfg_spin)

        # Temp parameter
        temp_label = QLabel("Temp:")
        self.repair_temp_spin = NoScrollDoubleSpinBox()
        self.repair_temp_spin.setRange(0.0, 2.0)
        self.repair_temp_spin.setSingleStep(0.1)
        self.repair_temp_spin.setDecimals(1)
        self.repair_temp_spin.setMaximumWidth(80)
        tts_layout.addWidget(temp_label)
        tts_layout.addWidget(self.repair_temp_spin)

        # Add stretch to push everything to the left
        tts_layout.addStretch()

        edit_layout.addLayout(tts_layout)

        # Action buttons
        actions_layout = QHBoxLayout()

        play_btn = QPushButton("🔊 Play Original")
        play_btn.clicked.connect(self.play_original_chunk)
        actions_layout.addWidget(play_btn)

        save_btn = QPushButton("💾 Save Changes")
        save_btn.clicked.connect(self.save_chunk_changes)
        actions_layout.addWidget(save_btn)

        resynth_btn = QPushButton("🎤 Resynthesize")
        resynth_btn.clicked.connect(self.resynthesize_chunk)
        actions_layout.addWidget(resynth_btn)

        play_rev_btn = QPushButton("🔊 Play Revised")
        play_rev_btn.clicked.connect(self.play_revised_chunk)
        actions_layout.addWidget(play_rev_btn)

        accept_btn = QPushButton("✅ Accept Revision")
        accept_btn.clicked.connect(self.accept_chunk_revision)
        actions_layout.addWidget(accept_btn)

        edit_layout.addLayout(actions_layout)
        layout.addWidget(edit_group)

        # Voice selection section
        voice_group = QGroupBox("🎤 Voice Selection")
        voice_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        voice_layout = QVBoxLayout(voice_group)

        # Voice detection info
        self.repair_voice_info = QLabel("No voice detected")
        self.repair_voice_info.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        voice_layout.addWidget(self.repair_voice_info)

        # Voice selection dropdown
        voice_layout.addWidget(QLabel("Select voice for resynthesis:"))
        self.repair_voice_combo = QComboBox()

        # Style the first item (placeholder) in red
        self.repair_voice_combo.setStyleSheet("""
            QComboBox QAbstractItemView::item:first-child {
                color: red;
                font-style: italic;
            }
        """)

        voice_layout.addWidget(self.repair_voice_combo)

        refresh_voices_btn = QPushButton("🔄 Re-detect Voice Candidates")
        refresh_voices_btn.clicked.connect(self.refresh_available_voices)
        voice_layout.addWidget(refresh_voices_btn)

        layout.addWidget(voice_group)

        # Initialize repair tool state
        self.current_repair_chunks = None
        self.current_repair_chunk = None
        self.current_repair_book_path = None
        self.current_repair_audio_dir = None
        self.current_repair_voice_name = None
        self.current_repair_voice_path = None

        # Initialize voice combo with placeholder
        self.repair_voice_combo.addItem("-- Please Select Voice --", None)

        # Populate books on tab creation
        self.refresh_repair_books()

        layout.addStretch()

    def create_json_generate_tab(self):
        """Tab 8: Generate from JSON with voice selection and playback controls"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "8. Generate from JSON")

        layout = QVBoxLayout(tab)

        info_label = QLabel("📄 Generate Audiobook from JSON Files")
        info_label.setStyleSheet("font-weight: bold; color: #E91E63; padding: 10px;")
        layout.addWidget(info_label)

        # JSON file selection
        json_group = QGroupBox("📄 JSON File Selection")
        json_layout = QFormLayout(json_group)

        self.json_file_edit = QLineEdit()
        self.json_file_edit.setPlaceholderText("Select JSON chunks file...")
        json_browse_btn = QPushButton("Browse...")
        json_browse_btn.clicked.connect(self.browse_json_file)

        json_row = QHBoxLayout()
        json_row.addWidget(self.json_file_edit)
        json_row.addWidget(json_browse_btn)
        json_layout.addRow("JSON File:", json_row)

        layout.addWidget(json_group)

        # Voice selection
        voice_group = QGroupBox("🎤 Voice Selection")
        voice_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        voice_layout = QFormLayout(voice_group)

        voice_layout.addRow(QLabel("Select voice for audiobook generation:"))
        self.json_voice_combo = QComboBox()
        voice_layout.addRow("Voice:", self.json_voice_combo)

        refresh_json_voices_btn = QPushButton("🔄 Refresh Voice List")
        refresh_json_voices_btn.clicked.connect(self.refresh_json_voices)
        voice_layout.addRow(refresh_json_voices_btn)

        layout.addWidget(voice_group)


        # Generation controls
        generate_group = QGroupBox("🎵 Audio Generation")
        generate_layout = QVBoxLayout(generate_group)

        # Generate button
        self.json_generate_btn = QPushButton("🎵 Generate Audiobook from JSON")
        self.json_generate_btn.setStyleSheet("QPushButton { background-color: #E91E63; color: white; font-weight: bold; padding: 12px; }")
        self.json_generate_btn.clicked.connect(self.generate_from_json)
        generate_layout.addWidget(self.json_generate_btn)

        # Progress bar
        self.json_progress = QProgressBar()
        self.json_progress.setVisible(False)
        generate_layout.addWidget(self.json_progress)

        layout.addWidget(generate_group)

        # Audio playback controls
        playback_group = QGroupBox("🔊 Audio Playback Controls")
        playback_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin: 5px; padding-top: 10px; }")
        playback_layout = QVBoxLayout(playback_group)

        # Current file display
        self.json_current_file = QLabel("No audiobook generated")
        self.json_current_file.setStyleSheet("background-color: #f8f9fa; padding: 8px; border: 1px solid #dee2e6; border-radius: 4px;")
        playback_layout.addWidget(self.json_current_file)

        # Playback buttons
        controls_layout = QHBoxLayout()

        self.json_play_btn = QPushButton("▶️ Play")
        self.json_play_btn.clicked.connect(self.play_json_audio)
        self.json_play_btn.setEnabled(False)
        controls_layout.addWidget(self.json_play_btn)

        self.json_pause_btn = QPushButton("⏸️ Pause")
        self.json_pause_btn.clicked.connect(self.pause_json_audio)
        self.json_pause_btn.setEnabled(False)
        controls_layout.addWidget(self.json_pause_btn)

        self.json_stop_btn = QPushButton("⏹️ Stop")
        self.json_stop_btn.clicked.connect(self.stop_json_audio)
        self.json_stop_btn.setEnabled(False)
        controls_layout.addWidget(self.json_stop_btn)

        self.json_rewind_btn = QPushButton("⏪ -10s")
        self.json_rewind_btn.clicked.connect(self.rewind_json_audio)
        self.json_rewind_btn.setEnabled(False)
        controls_layout.addWidget(self.json_rewind_btn)

        self.json_ff_btn = QPushButton("⏩ +10s")
        self.json_ff_btn.clicked.connect(self.ff_json_audio)
        self.json_ff_btn.setEnabled(False)
        controls_layout.addWidget(self.json_ff_btn)

        playback_layout.addLayout(controls_layout)

        # Position slider
        self.json_position_slider = QSlider(Qt.Horizontal)
        self.json_position_slider.setEnabled(False)
        self.json_position_slider.sliderPressed.connect(self.json_slider_pressed)
        self.json_position_slider.sliderReleased.connect(self.json_slider_released)
        playback_layout.addWidget(self.json_position_slider)

        # Time display
        self.json_time_label = QLabel("00:00 / 00:00")
        self.json_time_label.setAlignment(Qt.AlignCenter)
        playback_layout.addWidget(self.json_time_label)

        layout.addWidget(playback_group)

        # Initialize state
        self.json_audio_file = None
        self.json_audio_process = None
        self.json_audio_position = 0
        self.json_audio_duration = 0
        self.json_slider_dragging = False

        # Populate voices on tab creation
        self.refresh_json_voices()

        # Add structured status panel for Tab 8
        self.tab8_status_panel = StructuredStatusPanel("🎵 JSON Generation Status")
        layout.addWidget(self.tab8_status_panel)

        # Initialize real-time status manager for Tab 8
        self.tab8_status_manager = RealTimeStatusManager(self.tab8_status_panel)

        layout.addStretch()

    def create_output_area_widget(self, widget):
        """Create output/log area as a widget for splitter"""
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        output_group = QGroupBox("📝 Output Log")
        output_layout = QVBoxLayout(output_group)

        self.output_text = QTextEdit()
        self.output_text.setMinimumHeight(150)  # Minimum height instead of maximum
        self.output_text.setStyleSheet("font-family: monospace; background-color: #f5f5f5;")
        output_layout.addWidget(self.output_text)

        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.output_text.clear)
        output_layout.addWidget(clear_btn)

        layout.addWidget(output_group)

    def create_output_area(self, layout):
        """Legacy method for backwards compatibility"""
        self.create_output_area_widget(QWidget())

    # Browse Methods
    def browse_book_folder(self):
        """Browse book folder and populate text file combo box.
        Args:
        folder_path (str): Path of the selected book folder.
        Returns:
        None
        """
        last_book_folder = self.settings.value("last_book_folder", "")
        folder = QFileDialog.getExistingDirectory(self, "Select Book Folder", last_book_folder)
        if folder:
            self.book_path_edit.setText(folder)
            self.settings.setValue("last_book_folder", folder)
            self.populate_text_files(folder)

    def populate_text_files(self, folder_path):
        """Populate text file combo box when book folder is selected"""
        self.text_file_combo.clear()
        folder = Path(folder_path)
        txt_files = list(folder.glob("*.txt"))
        for txt_file in txt_files:
            self.text_file_combo.addItem(txt_file.name, str(txt_file))

    def browse_voice_file(self):
        """Opens a file dialog to select a voice file and updates the settings and UI accordingly. Returns None.
        Args:
        None
        Returns:
        None
        """
        last_voice_folder = self.settings.value("last_voice_folder", "")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Sample", last_voice_folder,
            "Audio Files (*.wav *.mp3 *.flac *.m4a);;All Files (*)"
        )
        if file_path:
            self.voice_path_edit.setText(file_path)
            self.settings.setValue("last_voice_folder", str(Path(file_path).parent))
            # Enable play button when file is selected
            if hasattr(self, 'voice_play_btn'):
                self.voice_play_btn.setEnabled(True)

    def play_voice_sample(self):
        """Play the selected voice sample"""
        voice_path = self.voice_path_edit.text()
        if not voice_path or not Path(voice_path).exists():
            QMessageBox.warning(self, "No Voice File", "Please select a valid voice sample file first.")
            return

        # Auto-stop any currently playing audio
        if hasattr(self, '_voice_playing') and self._voice_playing:
            self.stop_voice_sample()

        try:
            # Import audio playback library
            import pygame

            # Initialize pygame mixer if not already done
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

            # Start playing
            pygame.mixer.music.load(voice_path)
            pygame.mixer.music.play()

            # Update button states
            self.voice_play_btn.setEnabled(False)
            self.voice_stop_btn.setEnabled(True)
            self._voice_playing = True

            # Set timer to reset buttons when playback finishes
            self._voice_timer = QTimer()
            self._voice_timer.timeout.connect(self._check_voice_playback)
            self._voice_timer.start(200)  # Check every 200ms

        except ImportError:
            # pygame not available - show helpful message and try system fallback
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Audio Preview Unavailable")
            msg.setText("Voice preview is not available due to missing audio components.")
            msg.setInformativeText("This doesn't affect TTS generation - only GUI audio preview.\n\n"
                                 "To fix: Update audio drivers or restart your computer.")
            msg.exec_()
            return

        except Exception as e:
            # pygame failed - try Windows fallback methods
            import platform
            if platform.system() == "Windows":
                try:
                    # Try winsound first (most reliable on Windows)
                    import winsound
                    winsound.PlaySound(voice_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

                    # Update button states
                    self.voice_play_btn.setEnabled(False)
                    self.voice_stop_btn.setEnabled(False)  # winsound can't be stopped easily
                    self._voice_playing = True

                    # Auto-reset after estimated duration
                    QTimer.singleShot(5000, self._reset_voice_buttons)
                    return

                except ImportError:
                    # Try PowerShell as last resort
                    try:
                        import subprocess
                        subprocess.Popen(['powershell', '-c', f'(New-Object Media.SoundPlayer "{voice_path}").PlaySync()'],
                                       creationflags=subprocess.CREATE_NO_WINDOW)

                        self.voice_play_btn.setEnabled(False)
                        self.voice_stop_btn.setEnabled(False)
                        self._voice_playing = True
                        QTimer.singleShot(5000, self._reset_voice_buttons)
                        return

                    except Exception:
                        pass

            # All methods failed - show user-friendly error
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Audio Preview Error")
            msg.setText("Unable to play voice sample.")
            msg.setInformativeText(f"Technical details: {str(e)}\n\n"
                                 "This doesn't affect TTS generation - only GUI preview.\n\n"
                                 "Common fixes:\n"
                                 "• Update audio drivers\n"
                                 "• Check Windows audio settings\n"
                                 "• Restart your computer")
            msg.exec_()

    def stop_voice_sample(self):
        """Stop voice sample playback"""
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except ImportError:
            # Try to stop Windows audio if possible
            import platform
            if platform.system() == "Windows":
                try:
                    import winsound
                    winsound.PlaySound(None, winsound.SND_PURGE)  # Stop all winsound playback
                except ImportError:
                    # For PowerShell and other system commands, we can't easily stop them
                    # They will continue playing until finished
                    pass
        except Exception as e:
            print(f"Error stopping voice sample: {e}")

        # Reset button states
        self._reset_voice_buttons()

    def _reset_voice_buttons(self):
        """Reset voice playback button states"""
        self.voice_play_btn.setEnabled(True)
        self.voice_stop_btn.setEnabled(False)
        self._voice_playing = False
        if hasattr(self, '_voice_timer'):
            self._voice_timer.stop()

    def _check_voice_playback(self):
        """Check if voice sample is still playing"""
        try:
            import pygame
            if not pygame.mixer.music.get_busy() and hasattr(self, '_voice_playing') and self._voice_playing:
                # Playback finished naturally
                self._reset_voice_buttons()
        except ImportError:
            # For system commands, assume it's done after a timeout
            self._reset_voice_buttons()
        except Exception as e:
            print(f"Error checking voice playback: {e}")
            self._reset_voice_buttons()

    def handle_asr_toggle(self, state):
        """Show/hide ASR configuration when ASR is toggled"""
        if hasattr(self, 'asr_threshold_container'):
            self.asr_threshold_container.setVisible(state == 2)  # 2 = Qt.Checked

    def _auto_select_asr_config(self):
        """Build an ASR configuration using adaptive system recommendations."""
        try:
            from modules.system_detector import get_system_profile, recommend_asr_models

            profile = get_system_profile()
            recommendations = recommend_asr_models(profile)

            for level in ('moderate', 'safe', 'insane'):
                if level in recommendations:
                    selected = recommendations[level]
                    primary = selected['primary']
                    fallback = selected['fallback']
                    return {
                        'enabled': True,
                        'level': level,
                        'primary_model': primary['model'],
                        'primary_device': primary['device'],
                        'fallback_model': fallback['model'],
                        'fallback_device': fallback['device']
                    }

            print("⚠️ No ASR recommendations matched expected levels; using default config.")
        except Exception as e:
            print(f"⚠️ Error configuring ASR automatically: {e}")

        return self._default_asr_config()

    def _default_asr_config(self):
        """Fallback ASR configuration when recommendations are unavailable."""
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except Exception:
            has_gpu = False

        primary_model = 'small' if has_gpu else 'tiny'
        primary_device = 'gpu' if has_gpu else 'cpu'

        return {
            'enabled': True,
            'level': 'auto',
            'primary_model': primary_model,
            'primary_device': primary_device,
            'fallback_model': 'tiny',
            'fallback_device': 'cpu'
        }

    def on_book_selection_changed(self):
        """Enable or disable validation button based on book selection"""
        if hasattr(self, 'validate_btn'):
            has_book = bool(self.book_path_edit.text().strip())
            self.validate_btn.setEnabled(has_book)

    def browse_combine_book(self):
        """Opens a dialog to select an existing directory for book files and updates the combine book edit field.
        Args:
        folder (str): The selected folder path.
        Returns:
        None
        ---
        Opens a dialog to select a text file for preparation and updates the prepare text edit field.
        Args:
        file_path (str): The selected file path.
        Returns:
        None
        ---
        Opens a dialog to select a JSON file.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Book Folder to Combine")
        if folder:
            self.combine_book_edit.setText(folder)

    def browse_prepare_text(self):
        """Opens a file dialog to select a text file and sets its path in the prepare_text_edit widget.
        Args:
        None
        Returns:
        None
        ---
        Opens a file dialog to select a JSON file and sets its path in the json_file_edit widget.
        Args:
        None
        Returns:
        None
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Text File to Prepare", "",
            "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.prepare_text_edit.setText(file_path)

    def browse_json_file(self):
        """Selects a JSON file and updates the UI to display the selected file path.
        Args:
        None
        Returns:
        None
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON File", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.json_file_edit.setText(file_path)

    # Action Methods

    def update_status_display(self, status_text):
        """Update the status display - now uses TTS generation status panel"""
        # Status information is now handled by the TTS generation status panel
        # This method is kept for compatibility but no longer displays anything
        pass

    def start_conversion(self):
        """Button click handler - validates inputs and starts conversion"""
        # Validate inputs
        if not self.book_path_edit.text():
            QMessageBox.warning(self, "Error", "Please select a book folder")
            return

        if self.text_file_combo.currentData() is None:
            QMessageBox.warning(self, "Error", "Please select a text file")
            return

        # Collect parameters
        book_path = Path(self.book_path_edit.text())
        voice_path = Path(self.voice_path_edit.text()) if self.voice_path_edit.text() else None
        text_file_path = Path(self.text_file_combo.currentData())
        use_vader = False  # VADER disabled (widget removed)
        enable_asr = self.asr_checkbox.isChecked()

        # Build ASR configuration
        asr_config = {'enabled': False}
        if enable_asr:
            asr_config = self._auto_select_asr_config()

        asr_threshold = self.asr_threshold_spinner.value()

        # Collect TTS parameters
        tts_params = {
            'exaggeration': DEFAULT_EXAGGERATION,  # Default (widget removed)
            'cfg_weight': DEFAULT_CFG_WEIGHT,  # Default (widget removed)
            'temperature': self.temperature_spin.value(),
            'min_p': self.min_p_spin.value(),
            'top_p': self.top_p_spin.value(),
            'repetition_penalty': self.repetition_penalty_spin.value(),
            'use_vader': use_vader,
            'enable_asr': asr_config.get('enabled', False),
            'asr_threshold': asr_threshold,
            'seed': DEFAULT_SEED  # Default (widget removed)
        }

        # Collect quality enhancement parameters
        quality_params = {
            'regeneration_enabled': self.regeneration_enabled_checkbox.isChecked(),
            'max_attempts': self.max_attempts_spin.value(),
            'quality_threshold': self.quality_threshold_spin.value(),
            'sentiment_smoothing': False,  # Disabled (widget removed)
            'smoothing_window': SENTIMENT_SMOOTHING_WINDOW,  # Default (widget removed)
            'smoothing_method': SENTIMENT_SMOOTHING_METHOD,  # Default (widget removed)
            'mfcc_validation': self.mfcc_validation_checkbox.isChecked(),
            'spectral_threshold': self.spectral_threshold_spin.value(),
            'output_validation': self.output_validation_checkbox.isChecked(),
            'output_threshold': self.output_threshold_spin.value()
        }

        # Collect effective config parameters from tabs (real-time overrides)
        config_params = self._build_effective_settings()
        # Include ASR config
        config_params['asr_config'] = asr_config

        self.log_output(f"Starting conversion for: {book_path.name}")
        self.update_status_display(f"🔄 Processing {book_path.name}")
        
        if voice_path:
            self.log_output(f"Voice: {voice_path.name}")
        else:
            self.log_output(f"Voice: Default Turbo Model Voice")
        
        self.log_output(f"Text file: {text_file_path.name}")
        self.log_output(f"VADER enabled: {use_vader}")

        # Display ASR configuration details
        if asr_config.get('enabled'):
            level = asr_config.get('level', 'auto')
            primary_model = asr_config.get('primary_model', 'unknown')
            primary_device = asr_config.get('primary_device', 'unknown')
            self.log_output(f"🎤 ASR enabled: {level.upper()} level ({primary_model} on {primary_device.upper()})")
            self.log_output(f"🎚 ASR threshold: {asr_threshold:.2f}")
        else:
            self.log_output(f"🎤 ASR disabled")

        self.log_output(f"TTS params: {tts_params}")
        self.log_output(f"Quality enhancements: {quality_params}")
        self.log_output(f"🔧 Config params: {config_params}")

        # Reset and prepare status panel
        if hasattr(self, 'tab1_status_panel'):
            self.tab1_status_panel.reset()
            self.tab1_status_panel.update_status(operation="🚀 Starting conversion...")

        # Reset audio controls
        if hasattr(self, 'play_btn'):
            self.play_btn.setEnabled(False)

            # Auto-scroll main tab container to show status panel at bottom
            if hasattr(self, 'main_scroll_area'):
                # Use QTimer to ensure layout is updated, then scroll to bottom
                def debug_scroll():
                    """Scrolls the main scroll area to its maximum value and prints a debug message.
                    Args:
                    self: The current instance of the class.
                    Returns:
                    None
                    """
                    max_val = self.main_scroll_area.verticalScrollBar().maximum()
                    print(f"DEBUG: Main scroll maximum value: {max_val}")
                    self.main_scroll_area.verticalScrollBar().setValue(max_val)
                QTimer.singleShot(50, debug_scroll)

        # Disable the button during processing
        self.convert_btn.setEnabled(False)
        self.convert_btn.setText("🔄 Processing...")
        if hasattr(self, 'validate_btn'):
            self.validate_btn.setEnabled(False)

        # Disable audio controls during new conversion
        # Audio controls reset moved to above
        # Audio controls reset moved to above
        self.current_m4b_file = None

        # Start processing in background thread
        self.process_thread = ProcessThread(
            self.run_book_conversion,
            book_path, text_file_path, voice_path, tts_params, quality_params, config_params
        )
        self.process_thread.output_signal.connect(self.log_output)
        self.process_thread.finished_signal.connect(self.on_conversion_finished)
        self.process_thread.status_signal.connect(self.update_status_display)
        self.process_thread.structured_status_signal.connect(self.update_tab1_status_panel)
        self.process_thread.start()

    def run_book_conversion(self, book_path, text_file_path, voice_path, tts_params, quality_params, config_params):
        """Execute the actual book conversion with all GUI parameters"""
        try:
            print(f"🚀 Starting book conversion with GUI parameters")
            print(f"📖 Book: {book_path}")
            print(f"🎤 Voice: {voice_path if voice_path else 'Default Turbo Model Voice'}")
            print(f"🎛️ TTS Params: {tts_params}")
            print(f"🔬 Quality Params: {quality_params}")

            # Extract enable_asr from tts_params
            enable_asr = tts_params.get('enable_asr', False)

            # Detect best available device
            from modules.tts_engine import get_best_available_device
            device = get_best_available_device()
            print(f"🖥️ Using device: {device.upper()}")

            # Call the TTS engine with all parameters
            result = process_book_folder(
                book_dir=book_path,
                voice_path=voice_path,
                tts_params=tts_params,
                device=device,
                skip_cleanup=False,
                enable_asr=enable_asr,
                quality_params=quality_params,
                config_params=config_params,
                specific_text_file=text_file_path
            )

            print(f"✅ Conversion completed successfully")
            
            # Unload TTS model to free GPU memory
            try:
                print("🧹 Unloading TTS model from GPU memory...")
                _release_global_tts_model()
                print("✅ TTS model unloaded successfully")
            except Exception as cleanup_error:
                print(f"⚠️ Model cleanup warning: {cleanup_error}")
            
            return True, "Conversion completed successfully"

        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Unload TTS model even on failure to prevent memory accumulation
            try:
                print("🧹 Unloading TTS model after error...")
                _release_global_tts_model()
            except Exception:
                pass  # Silently ignore cleanup errors after conversion failure
            
            return False, f"Conversion failed: {e}"

    def run_validation(self):
        """Run ASR validation on completed conversion"""
        from modules.asr_validator import resolve_tts_dir, discover_chunks
        from pathlib import Path

        book_path_text = self.book_path_edit.text().strip()
        if not book_path_text:
            QMessageBox.warning(self, "Error", "Please select a book folder first")
            return

        book_path = Path(book_path_text)
        if not book_path.exists():
            QMessageBox.warning(self, "Error", f"Book path not found:\n{book_path}")
            return

        tts_dir = resolve_tts_dir(book_path)
        if not tts_dir:
            QMessageBox.warning(
                self,
                "No TTS Output",
                f"No TTS output found for this book.\n\n"
                f"Expected: Audiobook/{book_path.name}/TTS/\n\n"
                f"Run conversion first, or select a different book.",
            )
            return

        chunks = discover_chunks(tts_dir)
        if not chunks:
            QMessageBox.warning(
                self,
                "No Files",
                f"TTS directory exists but no chunk files found.\n\n"
                f"Path: {tts_dir}\n\n"
                f"Make sure conversion completed successfully.",
            )
            return

        threshold = self.asr_threshold_spinner.value()

        reply = QMessageBox.question(
            self,
            "ASR Validation Options",
            "Choose validation mode:\n\n"
            "• Yes: Run full validation with retry/regeneration of failed chunks\n"
            "• No: Report only (run ASR on existing files without retries)\n\n"
            "Note: Retry mode may regenerate audio chunks that fail validation.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        resynthesize_failed = reply == QMessageBox.Yes

        if resynthesize_failed:
            self.log_output("🎤 Starting ASR validation with retry/regeneration...")

            try:
                from modules.asr_manager import get_real_time_vram_status

                vram = get_real_time_vram_status()
                if vram["available_mb"] < 2000:
                    QMessageBox.warning(
                        self,
                        "Insufficient VRAM",
                        f"Not enough GPU memory for ASR validation.\n\n"
                        f"Available: {vram['available_mb']:,}MB\n"
                        f"Required: 2,000MB minimum\n\n"
                        f"Close other GPU applications and try again.",
                    )
                    return
            except Exception as e:
                logging.warning(f"VRAM check failed: {e}")

            progress = QProgressDialog("Running ASR validation...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setCancelButton(None)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            try:
                # Use new concurrent ASR system with regeneration
                from modules.asr_client import ASRClient
                from modules.regeneration_engine import regenerate_with_best_selection
                
                # Start ASR client for initial validation
                asr_client = ASRClient(tts_dir, self.asr_threshold_spinner.value())
                if not asr_client.start_daemon():
                    raise RuntimeError("ASR daemon failed to start - cannot proceed with validation")

                # Collect ASR results with concurrent submission
                print("🏳 Collecting ASR validation results...")
                total_chunks = discover_chunks(tts_dir)
                _, failed_chunks = asr_client.collect_all_results()
                
                if failed_chunks:
                    print(f"⚠️ {len(failed_chunks)} chunks failed initial validation")
                    
                    # Multi-attempt regeneration
                    print(f"🔄 Starting multi-attempt regeneration for {len(failed_chunks)} failed chunks...")
                    regeneration_report, still_failed = regenerate_with_best_selection(
                        failed_chunks, tts_dir, self.asr_threshold_spinner.value(),
                        model=None, voice_path=None, tts_params={}, asr_client=asr_client
                    )
                    
                    # Write regeneration report
                    report_file = tts_dir / "asr_regeneration_report.json"
                    report_file.write_text(json.dumps(regeneration_report, indent=2))
                    print(f"📄 Wrote regeneration report to {report_file.name}")
                
                    # Final validation of regenerated chunks
                    if still_failed:
                        final_failures_file = tts_dir / "asr_remaining_failures.json"
                        final_failures_file.write_text(json.dumps(still_failed, indent=2))
                        print(f"📄 Wrote remaining failures to {final_failures_file.name}")
                else:
                    print("✅ All chunks passed initial validation")
                
                # Shutdown ASR client
                asr_client.shutdown_daemon()
                
                progress.close()
                return {"total": total_chunks, "passed": total_chunks - len(failed_chunks), "regenerated": len(failed_chunks)}

                dialog = QDialog(self)
                dialog.setWindowTitle("ASR Validation with Retries Complete")
                dialog.setMinimumSize(600, 400)
                layout = QVBoxLayout()

                summary_text = (
                    f"Validation with Retries Complete\n\n"
                    f"Total Chunks: {result['total']}\n"
                    f"Passed: {result['passed']}\n"
                    f"Failed after retries: {result['failed_after_retries']}\n\n"
                    f"Check validations.log and 2ndfail.log for full results."
                )
                summary_label = QLabel(summary_text)
                summary_label.setStyleSheet("font-weight: bold; padding: 10px;")
                layout.addWidget(summary_label)

                buttons = QDialogButtonBox(QDialogButtonBox.Close)
                buttons.rejected.connect(dialog.reject)
                layout.addWidget(buttons)

                dialog.setLayout(layout)
                dialog.exec_()

            except Exception as e:
                progress.close()
                QMessageBox.critical(self, "Validation Error", f"ASR validation failed:\n{str(e)}")

        else:
            self.log_output("🎤 Running ASR validation on existing files (no regeneration)...")

            progress = QProgressDialog("Running ASR validation...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setCancelButton(None)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            try:
                from modules.asr_validator import validate_batch
                from modules.file_manager import write_validation_log

                result = validate_batch(tts_dir, threshold, progress_callback=None)
                progress.close()

                write_validation_log(tts_dir, result["results"])
                self.log_output(f"✅ Validation complete: {result['passed']}/{result['total']} passed")

                self.show_validation_results(result, tts_dir)

            except Exception as e:
                progress.close()
                self.log_output(f"❌ Validation failed: {e}")
                QMessageBox.critical(self, "Validation Error", f"ASR validation failed:\n{str(e)}")

    def show_validation_results(self, result, tts_dir):
        """Show validation results in resizable dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("ASR Validation Results")
        dialog.setMinimumSize(900, 700)
        dialog.resize(1000, 800)

        layout = QVBoxLayout()

        summary_text = (
            f"Validation Complete\n\n"
            f"Total Chunks: {result['total']}\n"
            f"Passed: {result['passed']} ({result['passed']/result['total']*100:.1f}%)\n"
            f"Failed: {result['failed']} ({result['failed']/result['total']*100:.1f}%)\n\n"
            f"Validation logs saved to: {tts_dir}/validations.log"
        )
        summary_label = QLabel(summary_text)
        summary_label.setStyleSheet("font-weight: bold; padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(summary_label)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Monospace", 9))

        details = ["=" * 80, "DETAILED VALIDATION RESULTS", "=" * 80, ""]

        failed_chunks = [r for r in result["results"] if not r["passed"]]
        if failed_chunks:
            details.append("FAILED CHUNKS:")
            details.append("-" * 80)
            for r in failed_chunks:
                details.append(f"Chunk {r['chunk_num']}: Score {r['score']:.3f}")
                details.append(f"  Expected: {r['ref_normalized'][:100]}...")
                details.append(f"  Got:      {r['hyp_normalized'][:100]}...")
                details.append("")
            details.append("")

        passed_chunks = [r["chunk_num"] for r in result["results"] if r["passed"]]
        if passed_chunks:
            details.append(f"PASSED CHUNKS: {len(passed_chunks)}")
            details.append("-" * 80)
            details.append(", ".join(passed_chunks))
            details.append("")

        text_edit.setPlainText("\n".join(details))
        layout.addWidget(text_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)
        dialog.exec_()

    def refresh_incomplete_books(self):
        """Refresh list of incomplete books"""
        self.incomplete_books_list.clear()
        try:
            incomplete_books = find_incomplete_books()
            for book in incomplete_books:
                self.incomplete_books_list.addItem(str(book))
            self.log_output(f"Found {len(incomplete_books)} incomplete books")
        except Exception as e:
            self.log_output(f"Error finding incomplete books: {e}")

    def resume_processing(self):
        """Resume processing selected book"""
        current_item = self.incomplete_books_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a book to resume")
            return

        book_path = current_item.text()
        self.log_output(f"Resuming processing for: {book_path}")

        try:
            from modules.resume_handler import resume_book_from_chunk
            # Start resume in background
            self.process_thread = ProcessThread(resume_book_from_chunk, 0)
            self.process_thread.output_signal.connect(self.log_output)
            self.process_thread.finished_signal.connect(lambda s, m: self.log_output("Resume completed" if s else f"Resume failed: {m}"))
            self.process_thread.status_signal.connect(self.update_status_display)
            self.process_thread.start()
        except Exception as e:
            self.log_output(f"Error starting resume: {e}")

    def combine_audio(self):
        """Combine audio chunks"""
        if not self.combine_book_edit.text():
            QMessageBox.warning(self, "Error", "Please select a book folder")
            return

        book_path = self.combine_book_edit.text()
        self.log_output(f"Combining audio for: {Path(book_path).name}")
        self.update_status_display(f"🔗 Combining audio chunks for {Path(book_path).name}")

        try:
            # Import the new GUI-friendly combine function
            from tools.combine_only import combine_audio_for_book

            # Start combine in background with the selected book path
            self.process_thread = ProcessThread(combine_audio_for_book, book_path)
            self.process_thread.output_signal.connect(self.log_output)
            self.process_thread.finished_signal.connect(lambda s, m: self.on_combine_finished(s, m))
            self.process_thread.status_signal.connect(self.update_status_display)
            self.process_thread.start()
        except Exception as e:
            self.log_output(f"Error starting combine: {e}")
            self.update_status_display(f"❌ Error starting combine: {e}")

    def on_combine_finished(self, success, message):
        """Handle combine completion"""
        if success:
            self.log_output("✅ Audio combining completed successfully!")
            self.update_status_display("✅ Audio combining completed successfully!")
            QMessageBox.information(self, "Success", "Audio chunks combined successfully!")
        else:
            self.log_output(f"❌ Combine failed: {message}")
            self.update_status_display(f"❌ Combine failed: {message}")
            QMessageBox.critical(self, "Error", f"Combine failed:\n{message}")

    def prepare_text(self):
        """Prepare text for chunking"""
        if not self.prepare_text_edit.text():
            QMessageBox.warning(self, "Error", "Please select a text file")
            return

        text_path = Path(self.prepare_text_edit.text())
        self.log_output(f"Preparing text: {text_path.name}")
        self.update_status_display(f"📝 Preparing text: {text_path.name}")

        try:
            # Start text preparation in background thread
            self.process_thread = ProcessThread(self._process_text_file, text_path)
            self.process_thread.output_signal.connect(self.log_output)
            self.process_thread.finished_signal.connect(lambda s, m: self.on_text_prep_finished(s, m))
            self.process_thread.status_signal.connect(self.update_status_display)
            self.process_thread.start()
        except Exception as e:
            self.log_output(f"Error starting text preparation: {e}")
            self.update_status_display(f"❌ Error starting text preparation: {e}")

    def _process_text_file(self, text_path):
        """Process text file with VADER sentiment analysis and JSON generation (matches CLI Option 4)"""
        from modules.tts_engine import generate_enriched_chunks
        from config.config import AUDIOBOOK_ROOT, TEXT_INPUT_ROOT

        print(f"📝 Processing text file: {text_path.name}")
        print("🎭 Generating enriched chunks with VADER sentiment analysis...")

        # Determine book name and setup output directory structure
        # Check if file is in Text_Input directory structure
        if TEXT_INPUT_ROOT in text_path.parents:
            # File is in Text_Input/BookName/file.txt structure
            book_name = text_path.parent.name
        else:
            # Standalone file - use filename as book name
            book_name = text_path.stem

        # Create TTS processing directory structure (matches CLI)
        book_output_dir = AUDIOBOOK_ROOT / book_name / "TTS" / "text_chunks"
        book_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Book: {book_name}")
        print(f"📂 Output directory: {book_output_dir}")

        # Get TTS parameters from Main Tab (same as used in convert_audiobook)
        use_vader = False  # VADER disabled (widget removed)
        user_tts_params = {
            'exaggeration': DEFAULT_EXAGGERATION,  # Default (widget removed)
            'cfg_weight': DEFAULT_CFG_WEIGHT,  # Default (widget removed)
            'temperature': self.temperature_spin.value(),
            'min_p': self.min_p_spin.value(),
            'top_p': self.top_p_spin.value(),
            'repetition_penalty': self.repetition_penalty_spin.value(),
            'use_vader': use_vader
        }
        print(f"🔧 Using Main Tab TTS parameters: {user_tts_params}")

        # Get quality parameters from Main Tab (sentiment smoothing)
        quality_params = {
            'sentiment_smoothing': False,  # Disabled (widget removed)
            'smoothing_window': SENTIMENT_SMOOTHING_WINDOW,  # Default (widget removed)
            'smoothing_method': SENTIMENT_SMOOTHING_METHOD  # Default (widget removed)
        }
        print(f"🔧 Using Main Tab quality parameters: {quality_params}")

        # Build effective config from tabs (includes VADER sensitivity)
        config_params = self._build_effective_settings()
        print(f"🔧 Using Config Tab VADER sensitivity parameters: {config_params}")

        # Generate enriched chunks with VADER analysis (same as CLI Option 4)
        enriched_chunks = generate_enriched_chunks(
            text_path,
            book_output_dir,
            user_tts_params,
            quality_params,
            config_params
        )

        json_path = book_output_dir / "chunks_info.json"

        print(f"✅ Text preparation completed!")
        print(f"   📋 Generated {len(enriched_chunks)} enriched chunks")
        print(f"   🎭 VADER sentiment analysis applied to each chunk")
        print(f"   📄 JSON metadata saved: {json_path}")
        print(f"   📁 Ready for TTS conversion with Option 1!")

        return True

    def on_text_prep_finished(self, success, message):
        """Handle text preparation completion"""
        if success:
            self.log_output(f"✅ Text preparation completed!")
            self.update_status_display("✅ Text preparation completed!")
            QMessageBox.information(self, "Success", "Text preparation completed successfully!\n\nCheck the output log for details.")
        else:
            self.log_output(f"❌ Text preparation failed: {message}")
            self.update_status_display(f"❌ Text preparation failed")
            QMessageBox.critical(self, "Error", f"Text preparation failed:\n{message}")

    def test_chunking(self):
        """Test chunking logic and show results in popup window"""
        test_text = self.test_text_edit.toPlainText().strip()
        max_words = self.max_words_spin.value()
        min_words = self.min_words_spin.value()

        self.log_output(f"Running chunking test - Max: {max_words}, Min: {min_words}")
        if test_text:
            self.log_output(f"Custom text: {test_text[:50]}...")
        else:
            self.log_output("Using default test text")

        try:
            from modules.text_processor import test_chunking
            import io
            import contextlib

            # Capture the test_chunking output
            captured_output = io.StringIO()

            with contextlib.redirect_stdout(captured_output):
                test_chunking(
                    test_text if test_text else None,
                    max_words,
                    min_words
                )

            # Get the captured results
            results_text = captured_output.getvalue()

            # Create and show popup window with results
            chunking_window = ChunkingTestWindow(self)
            chunking_window.set_chunking_results(results_text)
            chunking_window.show()

            self.log_output("✅ Chunking test completed - results shown in popup window")

        except Exception as e:
            self.log_output(f"❌ Error running chunking test: {e}")
            QMessageBox.critical(self, "Chunking Test Error", f"Failed to run chunking test:\n{e}")

    def refresh_repair_books(self):
        """Refresh the list of available books for repair"""
        from pathlib import Path

        self.repair_book_combo.clear()
        available_books = []

        # Check TTS processing directories
        audiobook_root = Path(AUDIOBOOK_ROOT)
        if audiobook_root.exists():
            for book_dir in audiobook_root.iterdir():
                if book_dir.is_dir():
                    tts_chunks_dir = book_dir / "TTS" / "text_chunks"
                    json_path = tts_chunks_dir / "chunks_info.json"
                    if json_path.exists():
                        available_books.append((book_dir.name, json_path, "TTS"))

        # Check Text_Input directory for fallback
        text_input_dir = Path("Text_Input")
        if text_input_dir.exists():
            for chunk_file in text_input_dir.glob("*_chunks.json"):
                book_name = chunk_file.stem.replace("_chunks", "")
                # Only add if not already found in TTS directories
                if not any(book[0] == book_name for book in available_books):
                    available_books.append((book_name, chunk_file, "Text_Input"))

        # Add placeholder as first item
        self.repair_book_combo.addItem("-- Select a Book --", None)

        if available_books:
            for book_name, json_path, source in available_books:
                self.repair_book_combo.addItem(f"{book_name} ({source})", (book_name, json_path, source))
            self.log_output(f"Found {len(available_books)} books available for repair")
        else:
            self.log_output("No chunk files found for repair")

    def load_chunks_for_repair(self):
        """Load chunks for the selected book"""
        current_data = self.repair_book_combo.currentData()
        if not current_data:
            # Clear display when placeholder is selected
            self.repair_results_list.clear()
            self.current_repair_chunk = None
            self.update_repair_chunk_display()
            return

        book_name, json_path, source = current_data

        try:
            from wrapper.chunk_loader import load_chunks
            self.current_repair_chunks = load_chunks(str(json_path))
            self.current_repair_book_path = json_path

            # Ensure chunks have index fields (0-based indexing)
            for i, chunk in enumerate(self.current_repair_chunks):
                if 'index' not in chunk:
                    chunk['index'] = i
                    self.log_output(f"🔧 Added missing index {i} to chunk")

            # Determine audio directory
            from pathlib import Path
            audiobook_root = Path(AUDIOBOOK_ROOT)
            self.current_repair_audio_dir = audiobook_root / book_name / "TTS" / "audio_chunks"

            self.log_output(f"Loaded {len(self.current_repair_chunks)} chunks from {json_path.name}")

            # Clear search results
            self.repair_results_list.clear()
            self.repair_search_edit.clear()
            self.current_repair_chunk = None
            self.update_repair_chunk_display()

            # Detect voice for this book
            self.detect_and_update_voice_info()

        except Exception as e:
            self.log_output(f"Error loading chunks: {e}")

    def detect_and_update_voice_info(self):
        """Detect and display voice information for the current book"""
        if not self.current_repair_book_path:
            return

        try:
            current_data = self.repair_book_combo.currentData()
            if not current_data:
                return
            book_name, json_path, source = current_data

            from modules.voice_detector import get_likely_voices_for_book
            likely_voices = get_likely_voices_for_book(book_name, json_path)

            # Clear and populate voice combo with ONLY likely candidates
            self.repair_voice_combo.clear()

            # Always start with placeholder - no auto-selection
            self.repair_voice_combo.addItem("-- Please Select Voice --", None)

            if likely_voices:
                # Add detected voice candidates (but don't auto-select any)
                for voice_name, voice_path, detection_method in likely_voices:
                    display_text = f"{voice_name} ({detection_method})"
                    self.repair_voice_combo.addItem(display_text, (voice_name, voice_path, detection_method))

                # Clear current selection - force user choice
                self.current_repair_voice_name = None
                self.current_repair_voice_path = None

                info_text = f"✅ Found {len(likely_voices)} candidate(s). Please select voice before resynthesizing."
                self.repair_voice_info.setText(info_text)
                self.repair_voice_info.setStyleSheet("background-color: #d4edda; padding: 5px; border: 1px solid #c3e6cb; color: #155724;")

                self.log_output(f"🎤 Found {len(likely_voices)} likely voice candidates for {book_name}")
            else:
                self.current_repair_voice_name = None
                self.current_repair_voice_path = None

                info_text = "❌ No voice candidates detected. Check JSON metadata, run.log, or filename patterns."
                self.repair_voice_info.setText(info_text)
                self.repair_voice_info.setStyleSheet("background-color: #f8d7da; padding: 5px; border: 1px solid #f5c6cb; color: #721c24;")

                self.log_output(f"❌ No voice candidates found for {book_name}")

        except Exception as e:
            self.log_output(f"Error detecting voice: {e}")

    def refresh_available_voices(self):
        """Refresh voice candidates for current book (not all voices)"""
        if self.current_repair_book_path:
            self.detect_and_update_voice_info()
        else:
            self.log_output("No book selected - cannot refresh voice candidates")

    def search_chunks_for_repair(self):
        """Search for chunks containing the specified text"""
        if not self.current_repair_chunks:
            QMessageBox.warning(self, "No Chunks", "Please select a book first")
            return

        query = self.repair_search_edit.text().strip()
        if not query:
            QMessageBox.warning(self, "No Query", "Please enter text to search for")
            return

        try:
            from wrapper.chunk_search import search_chunks
            results = search_chunks(self.current_repair_chunks, query)

            self.repair_results_list.clear()

            if results:
                for chunk in results:
                    text_preview = chunk['text'][:60] + "..." if len(chunk['text']) > 60 else chunk['text']
                    item_text = f"[{chunk['index']}] {text_preview}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, chunk)
                    self.repair_results_list.addItem(item)

                self.log_output(f"Found {len(results)} matching chunks")
            else:
                self.log_output("No matching chunks found")

        except Exception as e:
            self.log_output(f"Error searching chunks: {e}")

    def select_chunk_for_repair(self, item):
        """Select a chunk for editing"""
        chunk = item.data(Qt.UserRole)
        if chunk:
            self.current_repair_chunk = chunk
            self.update_repair_chunk_display()
            self.log_output(f"Selected chunk {chunk['index']} for editing")

    def update_repair_chunk_display(self):
        """Update the chunk editor display with current chunk data"""
        if not self.current_repair_chunk:
            self.repair_chunk_info.setText("No chunk selected")
            self.repair_text_edit.clear()
            return

        chunk = self.current_repair_chunk

        # Update info display
        sentiment_compound = chunk.get('sentiment_compound', chunk.get('sentiment_score', 'N/A'))
        tts_params = chunk.get('tts_params', {})

        info_text = f"""Index: {chunk['index']} | Boundary: {chunk['boundary_type']} | Words: {chunk.get('word_count', 'N/A')}
Sentiment: {sentiment_compound} | TTS: exag={tts_params.get('exaggeration', 'N/A')}, cfg={tts_params.get('cfg_weight', 'N/A')}, temp={tts_params.get('temperature', 'N/A')}
Audio: chunk_{chunk['index']+1:05d}.wav"""

        self.repair_chunk_info.setText(info_text)

        # Update text editor
        self.repair_text_edit.setPlainText(chunk['text'])

        # Update metadata controls
        self.repair_boundary_combo.setCurrentText(chunk.get('boundary_type', 'none'))
        # Sentiment spinner removed - value is preserved from original chunk

        # Update TTS parameters
        self.repair_exag_spin.setValue(tts_params.get('exaggeration', 1.0))
        self.repair_cfg_spin.setValue(tts_params.get('cfg_weight', 0.7))
        self.repair_temp_spin.setValue(tts_params.get('temperature', 0.7))

    def save_chunk_changes(self):
        """Save changes to the current chunk"""
        if not self.current_repair_chunk or not self.current_repair_chunks:
            QMessageBox.warning(self, "No Chunk", "No chunk selected for editing")
            return

        try:
            # Debug: Check chunk data before update
            chunk = self.current_repair_chunk
            chunk_index = chunk['index']
            self.log_output(f"🔍 DEBUG: Chunk keys before update: {list(chunk.keys())}")
            self.log_output(f"🔍 DEBUG: Chunk index: {chunk_index}")

            # Create updated chunk with standard field order
            from collections import OrderedDict
            import copy

            # Define standard field order template (matches most common chunk structure)
            standard_field_order = [
                'index', 'text', 'word_count', 'boundary_type',
                'sentiment_compound', 'sentiment_raw', 'tts_params'
            ]

            # Create ordered chunk following standard structure
            updated_chunk = OrderedDict()

            # Add fields in standard order
            for field in standard_field_order:
                if field in chunk:
                    if field == 'text':
                        updated_chunk[field] = self.repair_text_edit.toPlainText()
                    elif field == 'word_count':
                        updated_chunk[field] = len(self.repair_text_edit.toPlainText().split())
                    elif field == 'boundary_type':
                        updated_chunk[field] = self.repair_boundary_combo.currentText()
                    elif field == 'tts_params':
                        # Standard TTS params order
                        tts_order = ['exaggeration', 'cfg_weight', 'temperature', 'min_p', 'top_p', 'repetition_penalty']
                        updated_tts_params = OrderedDict()
                        original_tts = chunk.get('tts_params', {})

                        # Add TTS params in standard order
                        for tts_field in tts_order:
                            if tts_field in original_tts:
                                if tts_field == 'exaggeration':
                                    updated_tts_params[tts_field] = round(self.repair_exag_spin.value(), 2)
                                elif tts_field == 'cfg_weight':
                                    updated_tts_params[tts_field] = round(self.repair_cfg_spin.value(), 2)
                                elif tts_field == 'temperature':
                                    updated_tts_params[tts_field] = round(self.repair_temp_spin.value(), 2)
                                else:
                                    updated_tts_params[tts_field] = original_tts[tts_field]

                        updated_chunk[field] = updated_tts_params
                    else:
                        # Copy field unchanged (index, sentiment_compound, sentiment_raw)
                        updated_chunk[field] = copy.deepcopy(chunk[field])

            # Add any extra fields that weren't in the standard order (shouldn't happen but just in case)
            for field in chunk:
                if field not in updated_chunk:
                    updated_chunk[field] = copy.deepcopy(chunk[field])

            # Update the chunk in the list and current reference
            self.current_repair_chunks[chunk_index] = updated_chunk
            self.current_repair_chunk = updated_chunk
            self.log_output(f"🔧 Updated chunk at index {chunk_index} in list (preserving structure)")

            # Save to file
            from wrapper.chunk_loader import save_chunks
            json_path = str(self.current_repair_book_path)

            # Debug logging
            self.log_output(f"💾 Saving changes to: {json_path}")
            self.log_output(f"💾 Updated chunk {updated_chunk['index'] + 1:05d}: text='{updated_chunk['text'][:50]}...'")
            self.log_output(f"💾 Boundary: {updated_chunk['boundary_type']}, Sentiment: {updated_chunk['sentiment_compound']}")
            self.log_output(f"💾 TTS Params: exag={updated_chunk['tts_params']['exaggeration']}, cfg={updated_chunk['tts_params']['cfg_weight']}, temp={updated_chunk['tts_params']['temperature']}")

            # Debug: Show what we're actually saving
            actual_chunk_in_list = self.current_repair_chunks[chunk_index]
            self.log_output(f"🔍 DEBUG: Chunk in list at index {chunk_index}: text='{actual_chunk_in_list['text'][:50]}...'")
            self.log_output(f"🔍 DEBUG: Are they the same object? {updated_chunk is actual_chunk_in_list}")

            save_chunks(json_path, self.current_repair_chunks)

            # Verify the save by checking file modification time and re-reading the file
            import os
            from datetime import datetime
            if os.path.exists(json_path):
                mod_time = datetime.fromtimestamp(os.path.getmtime(json_path))
                self.log_output(f"💾 File modified at: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Re-read the file to verify what was actually saved
                try:
                    from wrapper.chunk_loader import load_chunks
                    saved_chunks = load_chunks(json_path)
                    if chunk_index < len(saved_chunks):
                        saved_chunk = saved_chunks[chunk_index]
                        self.log_output(f"🔍 VERIFY: Saved chunk text: '{saved_chunk['text'][:50]}...'")
                        self.log_output(f"🔍 VERIFY: Saved boundary: {saved_chunk.get('boundary_type', 'MISSING')}")
                        self.log_output(f"🔍 VERIFY: Saved sentiment: {saved_chunk.get('sentiment_compound', 'MISSING')}")
                except Exception as e:
                    self.log_output(f"❌ Error verifying save: {e}")

            self.log_output(f"✅ Saved changes to chunk {updated_chunk['index'] + 1:05d}")
            QMessageBox.information(self, "Saved", f"Chunk changes saved to:\n{json_path}")

        except Exception as e:
            self.log_output(f"Error saving chunk: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save chunk: {e}")

    def play_original_chunk(self):
        """Play the original audio for the current chunk"""
        if not self.current_repair_chunk or not self.current_repair_audio_dir:
            QMessageBox.warning(self, "No Chunk", "No chunk selected or audio directory not found")
            return

        try:
            from pathlib import Path
            chunk_index = self.current_repair_chunk['index']
            audio_path = self.current_repair_audio_dir / f"chunk_{chunk_index+1:05d}.wav"

            if audio_path.exists():
                from wrapper.chunk_player import play_chunk_audio
                play_chunk_audio(str(audio_path))
                self.log_output(f"🔊 Playing original audio: {audio_path.name}")
            else:
                QMessageBox.warning(self, "Audio Not Found", f"Audio file not found: {audio_path.name}")

        except Exception as e:
            self.log_output(f"Error playing audio: {e}")

    def resynthesize_chunk(self):
        """Resynthesize the current chunk with updated parameters"""
        if not self.current_repair_chunk or not self.current_repair_audio_dir:
            QMessageBox.warning(self, "No Chunk", "No chunk selected or audio directory not found")
            return

        try:
            chunk = self.current_repair_chunk
            chunk_index = chunk['index']

            # Get book name and JSON path from combo box
            current_data = self.repair_book_combo.currentData()
            if not current_data:
                QMessageBox.warning(self, "No Book", "No book selected")
                return
            book_name, json_path, source = current_data

            # Get selected voice from combo box
            combo_data = self.repair_voice_combo.currentData()
            selected_voice_display = self.repair_voice_combo.currentText()

            # Check if user selected the placeholder or no valid voice
            if not combo_data or combo_data is None:
                if selected_voice_display == "-- Please Select Voice --":
                    QMessageBox.warning(self, "Voice Selection Required",
                                      "Please select a voice from the dropdown before resynthesizing.\n\n"
                                      "The dropdown contains only voice candidates detected for this book.")
                else:
                    QMessageBox.warning(self, "No Voice", "No voice candidates available. Please check book's voice detection.")
                return

            voice_name, voice_path, detection_method = combo_data

            # Create updated chunk with current GUI values
            updated_chunk = chunk.copy()

            # Update text from GUI
            updated_chunk['text'] = self.repair_text_edit.toPlainText()
            updated_chunk['word_count'] = len(updated_chunk['text'].split())

            # Update metadata from GUI
            updated_chunk['boundary_type'] = self.repair_boundary_combo.currentText()
            # Preserve original sentiment values - not editable by user

            # Update TTS parameters from GUI
            if 'tts_params' not in updated_chunk:
                updated_chunk['tts_params'] = {}

            updated_chunk['tts_params']['exaggeration'] = self.repair_exag_spin.value()
            updated_chunk['tts_params']['cfg_weight'] = self.repair_cfg_spin.value()
            updated_chunk['tts_params']['temperature'] = self.repair_temp_spin.value()

            self.log_output(f"🎤 Resynthesizing chunk {chunk_index+1:05d} with voice: {selected_voice_display}")
            self.log_output(f"📊 Using TTS params: exag={updated_chunk['tts_params']['exaggeration']}, cfg={updated_chunk['tts_params']['cfg_weight']}, temp={updated_chunk['tts_params']['temperature']}")
            self.log_output(f"📝 Text: {updated_chunk['text'][:50]}...")

            from wrapper.chunk_synthesizer import synthesize_chunk
            revised_path = synthesize_chunk(updated_chunk, chunk_index, book_name, self.current_repair_audio_dir,
                                          revision=True, chunks_json_path=json_path,
                                          override_voice_name=voice_name, override_voice_path=str(voice_path))

            if revised_path:
                self.log_output(f"✅ Chunk resynthesized: {revised_path}")
                QMessageBox.information(self, "Success", f"Chunk resynthesized successfully:\n{revised_path}")
            else:
                self.log_output("❌ Failed to resynthesize chunk")
                QMessageBox.warning(self, "Failed", "Failed to resynthesize chunk")

        except Exception as e:
            self.log_output(f"Error resynthesizing chunk: {e}")
            QMessageBox.critical(self, "Error", f"Failed to resynthesize chunk: {e}")

    def play_revised_chunk(self):
        """Play the revised audio for the current chunk"""
        if not self.current_repair_chunk or not self.current_repair_audio_dir:
            QMessageBox.warning(self, "No Chunk", "No chunk selected or audio directory not found")
            return

        try:
            from pathlib import Path
            chunk_index = self.current_repair_chunk['index']
            revised_path = self.current_repair_audio_dir / f"chunk_{chunk_index+1:05d}_rev.wav"

            if revised_path.exists():
                from wrapper.chunk_player import play_chunk_audio
                play_chunk_audio(str(revised_path))
                self.log_output(f"🔊 Playing revised audio: {revised_path.name}")
            else:
                QMessageBox.warning(self, "Audio Not Found", f"Revised audio file not found: {revised_path.name}")

        except Exception as e:
            self.log_output(f"Error playing revised audio: {e}")

    def accept_chunk_revision(self):
        """Accept the revision by replacing original with revised audio"""
        if not self.current_repair_chunk or not self.current_repair_audio_dir:
            QMessageBox.warning(self, "No Chunk", "No chunk selected or audio directory not found")
            return

        try:
            chunk_index = self.current_repair_chunk['index']

            # Check if revised file exists before attempting to accept
            revised_path = self.current_repair_audio_dir / f"chunk_{chunk_index+1:05d}_rev.wav"
            if not revised_path.exists():
                QMessageBox.warning(self, "No Revision", "No revised audio file found. Please resynthesize first.")
                return

            from wrapper.chunk_revisions import accept_revision
            accept_revision(chunk_index, self.current_repair_audio_dir)

            self.log_output(f"✅ Revision accepted for chunk {chunk_index+1:05d}")
            self.log_output(f"📦 Original archived to Audio_Revisions/chunk_{chunk_index+1:05d}_orig.wav")

            # Update UI state after successful acceptance
            # Clear the revised audio reference since it's now the main file
            if hasattr(self, 'revised_audio_file'):
                self.revised_audio_file = None

            # Update chunk display to show the change has been accepted
            self.update_repair_chunk_display()

            QMessageBox.information(self, "Success",
                f"Revision accepted for chunk {chunk_index+1:05d}\n\n" +
                "Original has been archived and revised version is now active.")

        except Exception as e:
            self.log_output(f"❌ Error accepting revision: {e}")
            QMessageBox.critical(self, "Error", f"Failed to accept revision:\n{e}")

    def generate_from_json(self):
        """Generate audio from JSON"""
        if not self.json_file_edit.text():
            QMessageBox.warning(self, "Error", "Please select a JSON file")
            return

        json_path = self.json_file_edit.text()
        self.log_output(f"Generating audio from: {Path(json_path).name}")
        # TODO: Integrate with JSON generation functionality
        QMessageBox.information(self, "Started", "JSON generation started - check output log")

    def browse_m4b_file(self):
        """Browse and select a WAV or M4B file for regeneration or playback"""
        # Set default directory to Audiobook folder
        default_dir = str(Path("Audiobook").resolve()) if Path("Audiobook").exists() else str(Path.cwd())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            default_dir,
            "Audio Files (*.wav *.m4b);;WAV Files (*.wav);;M4B Files (*.m4b);;All Files (*)"
        )

        if file_path:
            selected_file = Path(file_path)
            file_extension = selected_file.suffix.lower()

            if file_extension == '.wav':
                # TRIGGER: Browse WAV - Clear all higher priority results
                self.converted_m4b_file = None    # Clear conversion results
                self.regenerated_m4b_file = None  # Clear regeneration results
                self.selected_m4b_file = None     # Clear M4B selection

                # Store WAV file for regeneration
                self.selected_wav_file = selected_file
                self.log_output(f"📁 Selected WAV file: {selected_file.name}")

                # Enable regenerate button for WAV file
                self.regenerate_m4b_btn.setEnabled(True)
                self.regenerate_m4b_btn.setToolTip(f"Create M4B from: {selected_file.name}")

                # Enable play button for immediate WAV playback
                self.play_btn.setEnabled(True)
                self.play_btn.setToolTip(f"Play selected WAV file: {selected_file.name}")

            elif file_extension == '.m4b':
                # TRIGGER: Browse M4B - Clear all higher priority results
                self.converted_m4b_file = None    # Clear conversion results
                self.regenerated_m4b_file = None  # Clear regeneration results
                self.selected_wav_file = None     # Clear WAV selection

                # Store M4B file for regeneration/playback
                self.selected_m4b_file = selected_file
                self.log_output(f"📁 Selected M4B file: {selected_file.name}")

                # Enable regenerate button for M4B file
                self.regenerate_m4b_btn.setEnabled(True)
                self.regenerate_m4b_btn.setToolTip(f"Regenerate: {selected_file.name}")

                # Enable play button for immediate playback
                self.play_btn.setEnabled(True)
                self.play_btn.setToolTip(f"Play selected file: {selected_file.name}")

                # Update current file for priority system
                self.current_m4b_file = selected_file

            else:
                QMessageBox.warning(self, "Unsupported File",
                                  f"Unsupported file type: {file_extension}\n\n"
                                  "Please select a WAV or M4B file.")

    def regenerate_m4b(self):
        """Regenerate M4B file with new speed setting from WAV or M4B file"""
        try:
            # Check if user has selected a file via browse button
            has_wav = hasattr(self, 'selected_wav_file') and self.selected_wav_file
            has_m4b = hasattr(self, 'selected_m4b_file') and self.selected_m4b_file

            if not has_wav and not has_m4b:
                QMessageBox.warning(self, "No File Selected",
                                  "Please use the Browse button to select a WAV or M4B file to regenerate.")
                return

            # Handle WAV file selection (direct path)
            if has_wav:
                if not self.selected_wav_file.exists():
                    QMessageBox.warning(self, "File Not Found",
                                      f"Selected WAV file no longer exists:\n{self.selected_wav_file}")
                    return

                # Use WAV file directly
                source_wav = self.selected_wav_file
                book_dir = source_wav.parent
                self.log_output(f"🔄 Creating M4B from WAV: {source_wav.name}")

            # Handle M4B file selection (find corresponding WAV)
            elif has_m4b:
                if not self.selected_m4b_file.exists():
                    QMessageBox.warning(self, "File Not Found",
                                      f"Selected M4B file no longer exists:\n{self.selected_m4b_file}")
                    return

                # Find corresponding WAV file (existing logic)
                book_dir = self.selected_m4b_file.parent
                m4b_base_name = self.selected_m4b_file.stem

                # Look for matching WAV file
                wav_files = []
                for wav_file in book_dir.glob("*.wav"):
                    if "TTS" not in str(wav_file) and "chunk_" not in wav_file.name:
                        wav_files.append(wav_file)

                if not wav_files:
                    QMessageBox.warning(self, "Error",
                                      f"No WAV file found in the same directory as the selected M4B file:\n{book_dir}")
                    return

                # Try to find matching WAV file
                source_wav = None
                for wav_file in wav_files:
                    wav_base = wav_file.stem
                    m4b_clean = m4b_base_name.split('[')[0].strip()
                    wav_clean = wav_base.split('[')[0].strip()
                    if m4b_clean == wav_clean:
                        source_wav = wav_file
                        break

                if not source_wav:
                    source_wav = max(wav_files, key=lambda p: p.stat().st_mtime)
                    QMessageBox.information(self, "File Match",
                                          f"Using most recent WAV file in directory:\n{source_wav.name}")

                self.log_output(f"🔄 Regenerating M4B from WAV: {source_wav.name}")

            # Get current speed and sample rate settings from GUI
            new_speed = self.main_playback_speed_spin.value()
            custom_sample_rate = int(self.main_m4b_sample_rate_combo.currentText())

            # Import the conversion function
            from modules.file_manager import convert_to_m4b

            # Determine M4B output path based on input type
            if has_wav:
                # For WAV input: create new M4B with matching name
                wav_base_name = source_wav.stem
                m4b_name = f"{wav_base_name}.m4b"
                final_m4b_path = book_dir / m4b_name
            else:
                # For M4B input: use the selected M4B file as the target for regeneration
                final_m4b_path = self.selected_m4b_file
                m4b_name = self.selected_m4b_file.name

            # Create temp file for conversion (Option C implementation)
            temp_m4b_path = book_dir / f"temp_speed_conversion.m4b"

            # Perform conversion with custom speed and sample rate parameters
            try:
                convert_to_m4b(source_wav, temp_m4b_path, custom_speed=new_speed, custom_sample_rate=custom_sample_rate)

                # Check if the temp file was created successfully
                if temp_m4b_path.exists() and temp_m4b_path.stat().st_size > 0:
                    # Atomic rename: replace original with temp file
                    if final_m4b_path.exists():
                        final_m4b_path.unlink()  # Remove original
                    temp_m4b_path.rename(final_m4b_path)  # Rename temp to final

                    # TRIGGER: Regenerate - Clear conversion results, set regenerated result
                    self.converted_m4b_file = None    # Clear conversion results (regenerate takes priority)
                    self.regenerated_m4b_file = final_m4b_path  # Set regenerated result

                    # Enable play button now that M4B is created/updated
                    self.play_btn.setEnabled(True)
                    self.play_btn.setToolTip(f"Play regenerated M4B: {m4b_name}")

                    # If this was a WAV file, update the selection to the new M4B
                    if has_wav:
                        self.selected_m4b_file = final_m4b_path
                        self.selected_wav_file = None  # Clear WAV selection
                        self.log_output(f"✅ M4B created successfully from WAV: {m4b_name}")
                        QMessageBox.information(self, "Success", f"M4B file created with {new_speed}x speed:\n{m4b_name}")
                    else:
                        self.log_output(f"✅ M4B regenerated successfully: {m4b_name}")
                        QMessageBox.information(self, "Success", f"M4B file regenerated with {new_speed}x speed:\n{m4b_name}")
                else:
                    self.log_output(f"❌ M4B regeneration failed: Temp file not created or empty")
                    QMessageBox.critical(self, "Error", "M4B regeneration failed: Temp file not created or empty")
                    # Clean up failed temp file
                    if temp_m4b_path.exists():
                        temp_m4b_path.unlink()
            except FileNotFoundError as ffmpeg_error:
                # FFmpeg not available - show user-friendly message
                self.log_output(f"❌ FFmpeg not available: {ffmpeg_error}")
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("FFmpeg Required")
                msg.setText("Cannot create M4B audiobook - FFmpeg is not installed.")
                msg.setInformativeText("FFmpeg is required for M4B audiobook creation.\n\n"
                                     "To install FFmpeg:\n"
                                     "• Download from https://ffmpeg.org/download.html\n"
                                     "• Or re-run the installation script\n\n"
                                     "WAV audio files are still available in the Audiobook folder.")
                msg.exec_()
            except Exception as conv_error:
                self.log_output(f"❌ M4B conversion error: {conv_error}")
                QMessageBox.critical(self, "Error", f"M4B conversion error:\n{conv_error}")

        except Exception as e:
            self.log_output(f"❌ Error during M4B regeneration: {e}")
            QMessageBox.critical(self, "Error", f"Error during M4B regeneration:\n{e}")

    def on_conversion_finished(self, success, message):
        """Handle conversion completion"""
        # Re-enable the button
        self.convert_btn.setEnabled(True)
        self.convert_btn.setText("🚀 Start Conversion")

        if success:
            self.log_output("✅ Conversion completed successfully!")
            self.update_status_display("✅ Conversion completed successfully!")
            if hasattr(self, 'validate_btn'):
                self.validate_btn.setEnabled(True)

            # Find and enable audio controls for the newly generated M4B file
            from pathlib import Path
            from modules.file_manager import sanitize_filename
            book_path = Path(self.book_path_edit.text())
            voice_path_text = self.voice_path_edit.text()

            if voice_path_text:
                voice_stem = Path(voice_path_text).stem
            else:
                voice_stem = "default-turbo"

            # M4B is created in Audiobook/BookName/ directory with pattern: BookName[VoiceName].m4b
            audiobook_dir = Path("Audiobook") / sanitize_filename(book_path.name)
            expected_m4b_name = f"{book_path.name}[{voice_stem}].m4b"
            generated_m4b = audiobook_dir / expected_m4b_name

            # Also check for files with space in the pattern: "BookName [VoiceName].m4b"
            alt_m4b_name = f"{book_path.name} [{voice_stem}].m4b"
            alt_generated_m4b = audiobook_dir / alt_m4b_name

            if generated_m4b.exists():
                # TRIGGER: Convert - Clear all other results, set converted result (highest priority)
                self.regenerated_m4b_file = None   # Clear regeneration results
                self.selected_m4b_file = None      # Clear browse selections
                self.selected_wav_file = None      # Clear browse selections
                self.converted_m4b_file = generated_m4b  # Set conversion result
                self.current_m4b_file = generated_m4b  # Legacy support
                self.play_btn.setEnabled(True)
                self.log_output(f"🎵 Audio controls enabled for: {generated_m4b.name}")
            elif alt_generated_m4b.exists():
                # TRIGGER: Convert - Clear all other results, set converted result (highest priority)
                self.regenerated_m4b_file = None   # Clear regeneration results
                self.selected_m4b_file = None      # Clear browse selections
                self.selected_wav_file = None      # Clear browse selections
                self.converted_m4b_file = alt_generated_m4b  # Set conversion result
                self.current_m4b_file = alt_generated_m4b  # Legacy support
                self.play_btn.setEnabled(True)
                self.log_output(f"🎵 Audio controls enabled for: {alt_generated_m4b.name}")
            else:
                self.log_output(f"⚠️ Could not find generated M4B at: {generated_m4b} or {alt_generated_m4b}")

            QMessageBox.information(self, "Success", "Book conversion completed successfully!")
        else:
            self.log_output(f"❌ Conversion failed: {message}")
            self.update_status_display(f"❌ Conversion failed: {message}")
            QMessageBox.critical(self, "Error", f"Conversion failed:\n{message}")

    def reset_config_defaults(self):
        """Reload Tab 2 values from the saved config (defaults = current saved config)."""
        try:
            importlib.reload(config_mod)

            # Performance settings
            self.workers_spin.setValue(getattr(config_mod, 'MAX_WORKERS', self.workers_spin.value()))
            self.batch_size_spin.setValue(getattr(config_mod, 'BATCH_SIZE', self.batch_size_spin.value()))
            self.tts_batch_size_spin.setValue(getattr(config_mod, 'TTS_BATCH_SIZE', self.tts_batch_size_spin.value()))
            self.min_chunk_words_spin.setValue(getattr(config_mod, 'MIN_CHUNK_WORDS', self.min_chunk_words_spin.value()))
            self.max_chunk_words_spin.setValue(getattr(config_mod, 'MAX_CHUNK_WORDS', self.max_chunk_words_spin.value()))

            # Detection settings
            self.mid_drop_check.setChecked(getattr(config_mod, 'ENABLE_MID_DROP_CHECK', self.mid_drop_check.isChecked()))
            self.hum_detection_check.setChecked(getattr(config_mod, 'ENABLE_HUM_DETECTION', self.hum_detection_check.isChecked()))

            # Audio processing
            self.normalization_check.setChecked(getattr(config_mod, 'ENABLE_NORMALIZATION', self.normalization_check.isChecked()))
            self.normalization_type_combo.setCurrentText(getattr(config_mod, 'NORMALIZATION_TYPE', self.normalization_type_combo.currentText()))
            self.target_lufs_spin.setValue(getattr(config_mod, 'TARGET_LUFS', self.target_lufs_spin.value()))
            self.target_peak_db_spin.setValue(getattr(config_mod, 'TARGET_PEAK_DB', self.target_peak_db_spin.value()))
            m4b_sr = str(getattr(config_mod, 'M4B_SAMPLE_RATE', int(self.m4b_sample_rate_combo.currentText())))
            self.m4b_sample_rate_combo.setCurrentText(m4b_sr)
            if hasattr(self, 'main_m4b_sample_rate_combo'):
                self.main_m4b_sample_rate_combo.setCurrentText(m4b_sr)
            self.audio_trimming_check.setChecked(getattr(config_mod, 'ENABLE_AUDIO_TRIMMING', self.audio_trimming_check.isChecked()))
            self.speech_threshold_spin.setValue(getattr(config_mod, 'SPEECH_ENDPOINT_THRESHOLD', self.speech_threshold_spin.value()))
            self.trimming_buffer_spin.setValue(getattr(config_mod, 'TRIMMING_BUFFER_MS', self.trimming_buffer_spin.value()))
            if hasattr(self, 'main_playback_speed_spin'):
                self.main_playback_speed_spin.setValue(getattr(config_mod, 'ATEMPO_SPEED', self.main_playback_speed_spin.value()))

            # Silence settings
            self.silence_chapter_start_spin.setValue(getattr(config_mod, 'SILENCE_CHAPTER_START', self.silence_chapter_start_spin.value()))
            self.silence_chapter_end_spin.setValue(getattr(config_mod, 'SILENCE_CHAPTER_END', self.silence_chapter_end_spin.value()))
            self.silence_section_spin.setValue(getattr(config_mod, 'SILENCE_SECTION_BREAK', self.silence_section_spin.value()))
            self.silence_paragraph_spin.setValue(getattr(config_mod, 'SILENCE_PARAGRAPH_END', self.silence_paragraph_spin.value()))
            self.silence_comma_spin.setValue(getattr(config_mod, 'SILENCE_COMMA', self.silence_comma_spin.value()))
            self.silence_period_spin.setValue(getattr(config_mod, 'SILENCE_PERIOD', self.silence_period_spin.value()))
            self.silence_question_spin.setValue(getattr(config_mod, 'SILENCE_QUESTION_MARK', self.silence_question_spin.value()))
            self.silence_exclamation_spin.setValue(getattr(config_mod, 'SILENCE_EXCLAMATION', self.silence_exclamation_spin.value()))
            self.chunk_end_silence_check.setChecked(getattr(config_mod, 'ENABLE_CHUNK_END_SILENCE', self.chunk_end_silence_check.isChecked()))
            self.chunk_end_silence_spin.setValue(getattr(config_mod, 'CHUNK_END_SILENCE_MS', self.chunk_end_silence_spin.value()))

            # TTS parameter defaults
            self.default_exag_spin.setValue(getattr(config_mod, 'DEFAULT_EXAGGERATION', self.default_exag_spin.value()))
            self.default_cfg_spin.setValue(getattr(config_mod, 'DEFAULT_CFG_WEIGHT', self.default_cfg_spin.value()))
            self.default_temp_spin.setValue(getattr(config_mod, 'DEFAULT_TEMPERATURE', self.default_temp_spin.value()))

            # VADER sensitivity
            self.vader_temp_sens_spin.setValue(getattr(config_mod, 'VADER_TEMPERATURE_SENSITIVITY', self.vader_temp_sens_spin.value()))

            # Parameter limits
            self.cfg_min_spin.setValue(getattr(config_mod, 'TTS_PARAM_MIN_CFG_WEIGHT', self.cfg_min_spin.value()))
            self.cfg_max_spin.setValue(getattr(config_mod, 'TTS_PARAM_MAX_CFG_WEIGHT', self.cfg_max_spin.value()))
            self.temp_min_spin.setValue(getattr(config_mod, 'TTS_PARAM_MIN_TEMPERATURE', self.temp_min_spin.value()))
            self.temp_max_spin.setValue(getattr(config_mod, 'TTS_PARAM_MAX_TEMPERATURE', self.temp_max_spin.value()))

            self.log_output("🔄 Reloaded configuration from saved config (Tab 2)")
            self.statusBar().showMessage("Reloaded Tab 2 from saved config", 2500)
        except Exception as e:
            self.log_output(f"❌ Error reloading config: {e}")
            QMessageBox.critical(self, "Error", f"Failed to reload configuration:\n{e}")

    def save_original_config_values(self):
        """Save original config values to track changes"""
        self.original_config_values = {
            'batch_size': self.batch_size_spin.value(),
            'min_chunk_words': self.min_chunk_words_spin.value(),
            'max_chunk_words': self.max_chunk_words_spin.value(),
            'normalization': self.normalization_check.isChecked(),
            'target_lufs': self.target_lufs_spin.value(),
            'audio_trimming': self.audio_trimming_check.isChecked(),
            'speech_threshold': self.speech_threshold_spin.value(),
            'trimming_buffer': self.trimming_buffer_spin.value(),
            'default_exag': self.default_exag_spin.value(),
            'default_cfg': self.default_cfg_spin.value(),
            'default_temp': self.default_temp_spin.value(),
            'vader_temp_sens': self.vader_temp_sens_spin.value(),
            'cfg_min': self.cfg_min_spin.value(),
            'cfg_max': self.cfg_max_spin.value(),
            'temp_min': self.temp_min_spin.value(),
            'temp_max': self.temp_max_spin.value(),
        }
        self.config_has_unsaved_changes = False

    def setup_config_change_tracking(self):
        """Connect all config widgets to change tracking"""
        # Connect spinboxes to change tracking
        config_widgets = [
            self.batch_size_spin, self.min_chunk_words_spin,
            self.max_chunk_words_spin, self.target_lufs_spin, self.speech_threshold_spin,
            self.trimming_buffer_spin, self.default_exag_spin, self.default_cfg_spin,
            self.default_temp_spin, self.vader_temp_sens_spin,
            self.cfg_min_spin, self.cfg_max_spin, self.temp_min_spin, self.temp_max_spin
        ]

        for widget in config_widgets:
            widget.valueChanged.connect(self.mark_config_changed)

        # Connect checkboxes to change tracking
        config_checkboxes = [self.normalization_check, self.audio_trimming_check]
        for checkbox in config_checkboxes:
            checkbox.toggled.connect(self.mark_config_changed)

        # Connect combo boxes to change tracking
        config_combos = [self.m4b_sample_rate_combo]
        for combo in config_combos:
            combo.currentTextChanged.connect(self.mark_config_changed)

    def mark_config_changed(self):
        """Mark config as having unsaved changes"""
        self.config_has_unsaved_changes = True

    def check_unsaved_config_changes(self):
        """Check if there are unsaved config changes and prompt user"""
        if self.config_has_unsaved_changes:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes in the configuration.\n\nDo you want to save them before continuing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self.save_config_to_file()
                return True
            elif reply == QMessageBox.No:
                return True
            else:  # Cancel
                return False
        return True

    def on_tab_changed(self, index):
        """Handle tab change - check for unsaved config changes when leaving config tab"""
        # Config tab is index 1 (0-based: 0=Convert Book, 1=Config)
        previous_index = getattr(self, '_previous_tab_index', 0)

        # If leaving config tab (previous was 1), check for unsaved changes
        if previous_index == 1 and index != 1:  # Leaving config tab
            if not self.check_unsaved_config_changes():
                # User cancelled, switch back to config tab
                self.tab_widget.setCurrentIndex(1)
                return

        # Update previous tab index
        self._previous_tab_index = index

    def save_config_to_file(self):
        """Save current GUI settings to config file"""
        try:
            import os
            config_path = "config/config.py"

            # Read current config file
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()

            # Create backup
            backup_path = config_path + ".backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(config_content)

            # Update specific values from GUI
            gui_values = {
                'MAX_WORKERS': self.workers_spin.value(),
                'BATCH_SIZE': self.batch_size_spin.value(),
                'TTS_BATCH_SIZE': self.tts_batch_size_spin.value(),
                'MIN_CHUNK_WORDS': self.min_chunk_words_spin.value(),
                'MAX_CHUNK_WORDS': self.max_chunk_words_spin.value(),
                'CHUNKING_QUALITY': self.chunking_quality_combo.currentText(),
                'ENABLE_MID_DROP_CHECK': self.mid_drop_check.isChecked(),
                'ENABLE_HUM_DETECTION': self.hum_detection_check.isChecked(),
                'ENABLE_NORMALIZATION': self.normalization_check.isChecked(),
                'NORMALIZATION_TYPE': self.normalization_type_combo.currentText(),
                'TARGET_LUFS': self.target_lufs_spin.value(),
                'TARGET_PEAK_DB': self.target_peak_db_spin.value(),
                'M4B_SAMPLE_RATE': int(self.m4b_sample_rate_combo.currentText()),
                'ENABLE_AUDIO_TRIMMING': self.audio_trimming_check.isChecked(),
                'SPEECH_ENDPOINT_THRESHOLD': self.speech_threshold_spin.value(),
                'TRIMMING_BUFFER_MS': self.trimming_buffer_spin.value(),
                'DEFAULT_EXAGGERATION': self.default_exag_spin.value(),
                'DEFAULT_CFG_WEIGHT': self.default_cfg_spin.value(),
                'DEFAULT_TEMPERATURE': self.default_temp_spin.value(),
                'VADER_TEMPERATURE_SENSITIVITY': self.vader_temp_sens_spin.value(),
                'TTS_PARAM_MIN_CFG_WEIGHT': self.cfg_min_spin.value(),
                'TTS_PARAM_MAX_CFG_WEIGHT': self.cfg_max_spin.value(),
                'TTS_PARAM_MIN_TEMPERATURE': self.temp_min_spin.value(),
                'TTS_PARAM_MAX_TEMPERATURE': self.temp_max_spin.value(),
                # Silence settings
                'SILENCE_CHAPTER_START': self.silence_chapter_start_spin.value(),
                'SILENCE_CHAPTER_END': self.silence_chapter_end_spin.value(),
                'SILENCE_SECTION_BREAK': self.silence_section_spin.value(),
                'SILENCE_PARAGRAPH_END': self.silence_paragraph_spin.value(),
                'SILENCE_COMMA': self.silence_comma_spin.value(),
                'SILENCE_PERIOD': self.silence_period_spin.value(),
                'SILENCE_QUESTION_MARK': self.silence_question_spin.value(),
                'SILENCE_EXCLAMATION': self.silence_exclamation_spin.value(),
                'ENABLE_CHUNK_END_SILENCE': self.chunk_end_silence_check.isChecked(),
                'CHUNK_END_SILENCE_MS': self.chunk_end_silence_spin.value(),
            }

            # Update config content
            import re
            for key, value in gui_values.items():
                # Match the config line pattern
                pattern = rf'^{key}\s*=\s*.*$'

                # Format value based on type
                if isinstance(value, str):
                    # Quote string values
                    formatted_value = f'"{value}"'
                else:
                    # Keep numeric values as-is
                    formatted_value = str(value)

                replacement = f'{key} = {formatted_value}'
                config_content = re.sub(pattern, replacement, config_content, flags=re.MULTILINE)

            # Write updated config
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)

            self.log_output("💾 Configuration saved to config/config.py")
            self.log_output(f"📁 Backup created: {backup_path}")

            # Release cached model and clear VRAM BEFORE reloading config to prevent OOM on save
            try:
                from modules.tts_engine import _release_global_tts_model
                _release_global_tts_model()
                # Best-effort host cleanup
                try:
                    import torch, gc
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    gc.collect()
                except Exception:
                    pass
                self.log_output("🧹 Released cached TTS model prior to config reload (VRAM cleared)")
            except Exception as rel_e:
                self.log_output(f"⚠️  Warning: Failed to release cached TTS model prior to reload: {rel_e}")

            # Reload config module and apply values app-wide
            try:
                importlib.reload(config_mod)
                # Optionally sync mirrored UI fields that depend on config
                try:
                    # Keep Tab 1 M4B sample rate and speed in sync with saved config
                    m4b_sr = str(getattr(config_mod, 'M4B_SAMPLE_RATE', int(self.m4b_sample_rate_combo.currentText())))
                    if hasattr(self, 'main_m4b_sample_rate_combo'):
                        self.main_m4b_sample_rate_combo.setCurrentText(m4b_sr)
                    if hasattr(self, 'main_playback_speed_spin') and hasattr(config_mod, 'ATEMPO_SPEED'):
                        self.main_playback_speed_spin.blockSignals(True)
                        self.main_playback_speed_spin.setValue(getattr(config_mod, 'ATEMPO_SPEED'))
                        self.main_playback_speed_spin.blockSignals(False)
                except Exception:
                    pass

                self.statusBar().showMessage("Saved and applied config from disk", 3000)
            except Exception as reload_e:
                self.log_output(f"⚠️  Warning: Config reload after save failed: {reload_e}")


        except Exception as e:
            self.log_output(f"❌ Error saving config: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{e}")


    def play_m4b_file(self):
        """Open audio file in system default player with priority system"""
        self.log_output("🔍 Play button clicked")
        try:
            # Priority system: conversion > regeneration > browse selection (WAV or M4B)
            file_to_play = None
            source_description = ""

            # 1. Highest priority: converted M4B file
            if hasattr(self, 'converted_m4b_file') and self.converted_m4b_file and self.converted_m4b_file.exists():
                file_to_play = self.converted_m4b_file
                source_description = "converted M4B"
            # 2. Medium priority: regenerated M4B file
            elif hasattr(self, 'regenerated_m4b_file') and self.regenerated_m4b_file and self.regenerated_m4b_file.exists():
                file_to_play = self.regenerated_m4b_file
                source_description = "regenerated M4B"
            # 3. Browse-selected files: M4B takes priority over WAV
            elif hasattr(self, 'selected_m4b_file') and self.selected_m4b_file and self.selected_m4b_file.exists():
                file_to_play = self.selected_m4b_file
                source_description = "selected M4B"
            # 4. Browse-selected WAV file
            elif hasattr(self, 'selected_wav_file') and self.selected_wav_file and self.selected_wav_file.exists():
                file_to_play = self.selected_wav_file
                source_description = "selected WAV"
            # 5. Fallback: legacy current_m4b_file
            elif hasattr(self, 'current_m4b_file') and self.current_m4b_file and self.current_m4b_file.exists():
                file_to_play = self.current_m4b_file
                source_description = "current M4B"

            if not file_to_play:
                self.log_output("❌ No audio file available to play")
                QMessageBox.information(self, "No File",
                                      "No audio file available to play.\n\n"
                                      "Use the Browse button to select a WAV or M4B file, "
                                      "or convert/regenerate an M4B file first.")
                return

            # Use system default player to open audio file
            import subprocess
            import sys

            if sys.platform == "win32":
                subprocess.Popen(["start", str(file_to_play)], shell=True)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(file_to_play)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(file_to_play)])

            self.log_output(f"🎵 Playing {source_description}: {file_to_play.name}")

        except Exception as e:
            self.log_output(f"❌ Error opening audio file: {e}")
            QMessageBox.critical(self, "Playback Error", f"Error opening audio file:\n{e}")


    def detect_and_update_device_status(self):
        """Detect and update device status in the GUI using comprehensive CUDA checking"""
        try:
            import torch
            import subprocess
            import sys

            # Use the same comprehensive CUDA checking logic as our launcher scripts
            cuda_status = "UNKNOWN"
            device_text = "❓ Checking device..."

            try:
                # Check if PyTorch has CUDA support
                if not hasattr(torch.version, 'cuda') or torch.version.cuda is None:
                    cuda_status = "CPU_ONLY"
                else:
                    pytorch_cuda = torch.version.cuda

                    # Try to detect system CUDA
                    try:
                        nvcc_result = subprocess.run(['nvcc', '--version'],
                                                   capture_output=True, text=True, timeout=5)
                        if nvcc_result.returncode == 0:
                            import re
                            match = re.search(r'release (\d+\.\d+)', nvcc_result.stdout)
                            if match:
                                system_cuda = match.group(1)

                                # CUDA compatibility check with special handling for CUDA 12.0
                                pytorch_version = float(pytorch_cuda)
                                system_version = float(system_cuda)

                                # CUDA compatibility check - be more permissive
                                compatible = False

                                # CUDA 12.x family compatibility (12.0, 12.1, 12.2, etc.)
                                if (system_version >= 12.0 and pytorch_version >= 12.0 and
                                    int(system_version) == 12 and int(pytorch_version) == 12):
                                    compatible = True
                                # CUDA 11.x family compatibility
                                elif (system_version >= 11.0 and pytorch_version >= 11.0 and
                                      int(system_version) == 11 and int(pytorch_version) == 11):
                                    compatible = True
                                # General rule: PyTorch CUDA should be <= System CUDA + tolerance
                                elif pytorch_version <= system_version + 0.5:
                                    compatible = True

                                if compatible:
                                    # Additional check: can we actually use CUDA?
                                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                                        cuda_status = "COMPATIBLE"
                                    else:
                                        cuda_status = "NO_DEVICES"
                                else:
                                    cuda_status = f"MISMATCH:{pytorch_cuda}:{system_cuda}"
                            else:
                                cuda_status = "UNKNOWN"
                        else:
                            cuda_status = "NO_NVCC"
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        cuda_status = "NO_NVCC"

            except Exception as e:
                cuda_status = f"ERROR:{str(e)}"

            # DETECTION REPORT ONLY - Don't override GPU usage, just report status
            if cuda_status == "COMPATIBLE":
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    device_text = f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB) [DETECTION: Compatible]"
                    self.tab1_status_panel.device_label.setStyleSheet("color: #4CAF50; font-weight: bold;")  # Green
                except:
                    device_text = "🚀 GPU: CUDA Available [DETECTION: Compatible]"
                    self.tab1_status_panel.device_label.setStyleSheet("color: #4CAF50; font-weight: bold;")  # Green
            elif cuda_status == "CPU_ONLY":
                device_text = "🚀 GPU: Forcing GPU Usage [DETECTION: PyTorch CPU-only]"
                self.tab1_status_panel.device_label.setStyleSheet("color: #FF9800; font-weight: bold;")  # Orange
            elif cuda_status == "NO_NVCC":
                device_text = "🚀 GPU: Forcing GPU Usage [DETECTION: No CUDA toolkit]"
                self.tab1_status_panel.device_label.setStyleSheet("color: #FF9800; font-weight: bold;")  # Orange
            elif cuda_status == "NO_DEVICES":
                device_text = "🚀 GPU: Forcing GPU Usage [DETECTION: No CUDA devices]"
                self.tab1_status_panel.device_label.setStyleSheet("color: #FF9800; font-weight: bold;")  # Orange
            elif cuda_status.startswith("MISMATCH"):
                parts = cuda_status.split(":")
                if len(parts) == 3:
                    pytorch_cuda, system_cuda = parts[1], parts[2]
                    device_text = f"🚀 GPU: Forcing GPU Usage [DETECTION: Mismatch {pytorch_cuda}/{system_cuda}]"
                else:
                    device_text = "🚀 GPU: Forcing GPU Usage [DETECTION: Version mismatch]"
                self.tab1_status_panel.device_label.setStyleSheet("color: #FF9800; font-weight: bold;")  # Orange
            else:
                device_text = f"🚀 GPU: Forcing GPU Usage [DETECTION: {cuda_status}]"
                self.tab1_status_panel.device_label.setStyleSheet("color: #FF9800; font-weight: bold;")  # Orange

            self.tab1_status_panel.device_label.setText(device_text)
            self.log_output(f"Device detected: {device_text}")

        except Exception as e:
            device_text = "❌ Device Detection Failed"
            self.tab1_status_panel.device_label.setText(device_text)
            self.tab1_status_panel.device_label.setStyleSheet("color: #F44336; font-weight: bold;")  # Red
            self.log_output(f"Device detection error: {e}")

    def log_output(self, message):
        """Add message to output log with ANSI code filtering"""
        import re
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Filter out ANSI color codes
        clean_message = re.sub(r'\x1b\[[0-9;]*m', '', message)

        # Safety check in case output_text hasn't been initialized yet
        if hasattr(self, 'output_text') and self.output_text is not None:
            self.output_text.append(f"[{timestamp}] {clean_message}")
            # Force scroll to bottom
            scrollbar = self.output_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        else:
            # Fallback to print if GUI not ready
            print(f"[{timestamp}] {clean_message}")

    def update_tab1_status_panel(self, status_data):
        """Update Tab 1 structured status panel with parsed data"""
        if hasattr(self, 'tab1_status_panel'):
            self.tab1_status_panel.update_status(
                operation=status_data.get('operation'),
                progress=status_data.get('progress'),
                elapsed=status_data.get('elapsed'),
                eta=status_data.get('eta'),
                remaining=status_data.get('remaining'),
                realtime=status_data.get('realtime'),
                vram=status_data.get('vram'),
                chunk_info=status_data.get('chunk_info')
            )

    def update_tab8_status_panel(self, status_data):
        """Update Tab 8 structured status panel with parsed data"""
        if hasattr(self, 'tab8_status_panel'):
            self.tab8_status_panel.update_status(
                operation=status_data.get('operation'),
                progress=status_data.get('progress'),
                elapsed=status_data.get('elapsed'),
                eta=status_data.get('eta'),
                remaining=status_data.get('remaining'),
                realtime=status_data.get('realtime'),
                vram=status_data.get('vram'),
                chunk_info=status_data.get('chunk_info')
            )

    # JSON Generate Tab Methods
    def refresh_json_voices(self):
        """Refresh the list of all available voices for JSON generation"""
        try:
            from modules.file_manager import list_voice_samples
            voice_files = list_voice_samples()

            self.json_voice_combo.clear()
            for voice_file in voice_files:
                voice_name = voice_file.stem
                self.json_voice_combo.addItem(voice_name, str(voice_file))

            self.log_output(f"Loaded {len(voice_files)} voices for JSON generation")

        except Exception as e:
            self.log_output(f"Error loading voices: {e}")

    def browse_json_file(self):
        """Browse for JSON chunks file"""
        last_json_folder = self.settings.value("last_json_folder", "")

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select JSON Chunks File",
            last_json_folder,
            "JSON files (*.json);;All files (*.*)"
        )

        if file_path:
            self.json_file_edit.setText(file_path)
            self.settings.setValue("last_json_folder", os.path.dirname(file_path))
            self.log_output(f"Selected JSON file: {os.path.basename(file_path)}")

    def generate_from_json(self):
        """Generate audiobook from JSON file with selected voice"""
        json_path = self.json_file_edit.text().strip()
        if not json_path:
            QMessageBox.warning(self, "No JSON File", "Please select a JSON chunks file first.")
            return

        if not os.path.exists(json_path):
            QMessageBox.warning(self, "File Not Found", f"JSON file not found: {json_path}")
            return

        selected_voice = self.json_voice_combo.currentText() or None

        try:
            # Get speed/temp setting from config tab
            temp_setting = self.default_temp_spin.value()

            self.log_output(f"🎵 Starting audiobook generation from JSON...")
            self.log_output(f"📄 JSON file: {os.path.basename(json_path)}")
            self.log_output(f"🎤 Voice: {selected_voice if selected_voice else 'Default Turbo Model Voice'}")
            self.log_output(f"⚡ Temperature: {temp_setting}")

            # Reset and prepare status panel
            if hasattr(self, 'tab8_status_panel'):
                self.tab8_status_panel.reset()
                self.tab8_status_panel.update_status(operation="🎵 Starting JSON generation...")

            # Disable generate button during processing
            self.json_generate_btn.setEnabled(False)
            self.json_progress.setVisible(True)
            self.json_progress.setValue(0)

            # Start generation in background thread
            self.json_generation_thread = ProcessThread(
                self._generate_audiobook_from_json,
                json_path,
                selected_voice,
                temp_setting
            )
            self.json_generation_thread.output_signal.connect(self.log_output)
            self.json_generation_thread.finished_signal.connect(self.json_generation_finished)
            self.json_generation_thread.structured_status_signal.connect(self.update_tab1_status_panel)
            # Skip progress signal connection to avoid Qt threading issues
            self.json_generation_thread.start()

        except Exception as e:
            self.log_output(f"❌ Error starting JSON generation: {e}")
            QMessageBox.critical(self, "Generation Error", f"Failed to start generation:\n{e}")

    def _generate_audiobook_from_json(self, json_path, voice_name, temp_setting):
        """Generate audiobook from JSON using dedicated GUI module"""
        try:
            from modules.gui_json_generator import generate_audiobook_from_json

            print(f"🎵 Starting GUI JSON generation...")
            print(f"📄 JSON: {json_path}")
            print(f"🎤 Voice: {voice_name}")
            print(f"🌡️ Temperature: {temp_setting}")

            # Call the dedicated GUI JSON generator with status callback
            # Note: self is not directly accessible here, need to get it from thread
            success, message, audiobook_path = generate_audiobook_from_json(
                json_path=json_path,
                voice_name=voice_name,
                temp_setting=temp_setting,
                status_callback=getattr(self, 'parse_and_emit_status', None) if hasattr(self, 'parse_and_emit_status') else None
            )

            if success and audiobook_path:
                self.json_audio_file = audiobook_path
                print(f"✅ {message}")
                print(f"📁 Audiobook: {audiobook_path}")
                return True
            else:
                print(f"❌ {message}")
                return False

        except Exception as e:
            print(f"❌ GUI JSON generation error: {e}")
            return False

    def json_generation_finished(self, success, message):
        """Handle completion of JSON generation"""
        self.json_generate_btn.setEnabled(True)
        self.json_progress.setVisible(False)

        if success:
            self.log_output("✅ Audiobook generation completed successfully!")

            # Update UI with generated file
            if self.json_audio_file:
                filename = os.path.basename(self.json_audio_file)
                self.json_current_file.setText(f"📁 Generated: {filename}")
                self.json_current_file.setStyleSheet("background-color: #d4edda; padding: 8px; border: 1px solid #c3e6cb; color: #155724; border-radius: 4px;")

                # Enable playback controls
                self.json_play_btn.setEnabled(True)
                self.json_stop_btn.setEnabled(True)
                self.json_rewind_btn.setEnabled(True)
                self.json_ff_btn.setEnabled(True)
                self.json_position_slider.setEnabled(True)

                QMessageBox.information(self, "Generation Complete", f"Audiobook generated successfully!\n\nFile: {filename}")

        else:
            self.log_output(f"❌ Audiobook generation failed: {message}")
            QMessageBox.critical(self, "Generation Failed", f"Audiobook generation failed:\n{message}")

    # Audio Playback Control Methods
    def play_json_audio(self):
        """Play the generated audiobook"""
        if not self.json_audio_file or not os.path.exists(self.json_audio_file):
            QMessageBox.warning(self, "No Audio", "No audiobook file available for playback")
            return

        try:
            import subprocess
            import platform

            # Use system default media player
            if platform.system() == "Linux":
                self.json_audio_process = subprocess.Popen(['xdg-open', self.json_audio_file])
            elif platform.system() == "Darwin":  # macOS
                self.json_audio_process = subprocess.Popen(['open', self.json_audio_file])
            elif platform.system() == "Windows":
                self.json_audio_process = subprocess.Popen(['start', self.json_audio_file], shell=True)

            self.json_play_btn.setEnabled(False)
            self.json_pause_btn.setEnabled(True)
            self.log_output(f"▶️ Playing: {os.path.basename(self.json_audio_file)}")

        except Exception as e:
            self.log_output(f"❌ Error playing audio: {e}")
            QMessageBox.critical(self, "Playback Error", f"Failed to play audio:\n{e}")

    def pause_json_audio(self):
        """Pause/Resume audio playback"""
        # Note: This is a simplified implementation
        # Full media control would require a dedicated audio library like pygame or VLC bindings
        if self.json_pause_btn.text() == "⏸️ Pause":
            self.json_pause_btn.setText("▶️ Resume")
            self.log_output("⏸️ Audio paused (system player)")
        else:
            self.json_pause_btn.setText("⏸️ Pause")
            self.log_output("▶️ Audio resumed (system player)")

    def stop_json_audio(self):
        """Stop audio playback.
        Args:
        - None
        Returns:
        - None
        """
        """Stop audio playback"""
        if self.json_audio_process:
            try:
                self.json_audio_process.terminate()
                self.json_audio_process = None
            except:
                pass

        self.json_play_btn.setEnabled(True)
        self.json_pause_btn.setEnabled(False)
        self.json_pause_btn.setText("⏸️ Pause")
        self.log_output("⏹️ Audio playback stopped")

    def rewind_json_audio(self):
        """Rewind 10 seconds (simplified implementation)"""
        self.log_output("⏪ Rewind 10 seconds (system player - manual control)")

    def ff_json_audio(self):
        """Fast forward 10 seconds (simplified implementation)"""
        self.log_output("⏩ Fast forward 10 seconds (system player - manual control)")

    def json_slider_pressed(self):
        """Handle slider press for seeking"""
        self.json_slider_dragging = True

    def json_slider_released(self):
        """Handle slider release for seeking"""
        self.json_slider_dragging = False
        # In a full implementation, this would seek to the slider position

    def create_voice_analyzer_tab(self):
        """Tab 9: Voice Sample Analyzer for TTS Suitability"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "9. Voice Analyzer")

        layout = QVBoxLayout(tab)

        # Title and description
        info_label = QLabel("🎤 Voice Sample Analyzer for TTS")
        info_label.setStyleSheet("font-weight: bold; color: #9C27B0; padding: 10px;")
        layout.addWidget(info_label)

        desc_label = QLabel("Analyze voice samples for TTS suitability, get quality scores, and apply automated fixes")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(desc_label)

        if VOICE_ANALYZER_AVAILABLE:
            # Build the voice analyzer GUI directly in the tab
            try:
                self.build_voice_analyzer_gui(layout)

            except Exception as e:
                error_layout = QVBoxLayout()
                error_label = QLabel(f"❌ Error loading Voice Analyzer: {str(e)}")
                error_label.setStyleSheet("color: red; padding: 10px;")
                error_layout.addWidget(error_label)

                # Add troubleshooting info
                help_text = QLabel("""
Troubleshooting:
1. Make sure all dependencies are installed:
   pip install -r voice_analyzer/requirements.txt

2. Required packages:
   - praat-parselmouth
   - librosa
   - matplotlib
   - scipy
   - soundfile

3. Try running the analyzer standalone first:
   python voice_analyzer/main.py
                """)
                help_text.setStyleSheet("font-family: monospace; color: #666; padding: 10px;")
                error_layout.addWidget(help_text)

                layout.addLayout(error_layout)
        else:
            # Voice analyzer not available - show installation instructions
            not_available_layout = QVBoxLayout()

            not_available_label = QLabel("❌ Voice Analyzer Not Available")
            not_available_label.setStyleSheet("font-weight: bold; color: red; padding: 10px;")
            not_available_layout.addWidget(not_available_label)

            install_label = QLabel("The Voice Analyzer requires additional dependencies.")
            install_label.setStyleSheet("padding: 5px;")
            not_available_layout.addWidget(install_label)

            # Installation instructions
            instructions = QLabel("""
To enable the Voice Analyzer:

1. Install required dependencies:
   pip install praat-parselmouth librosa matplotlib scipy soundfile

2. Or install from requirements file:
   pip install -r voice_analyzer/requirements.txt

3. Restart ChatterboxTTS

The Voice Analyzer provides:
• TTS suitability scoring
• Voice quality analysis using Praat
• Automated audio fixes (clipping, noise, tempo, etc.)
• Comparison tools for multiple voice samples
• Export capabilities for reports and plots
            """)
            instructions.setStyleSheet("font-family: monospace; background-color: #f5f5f5; padding: 15px; border: 1px solid #ddd; border-radius: 5px;")
            instructions.setWordWrap(True)
            not_available_layout.addWidget(instructions)

            # Quick install button
            install_btn = QPushButton("🔧 Try Auto-Install Dependencies")
            install_btn.clicked.connect(self.try_install_voice_analyzer_deps)
            install_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px; font-weight: bold; }")
            not_available_layout.addWidget(install_btn)

            layout.addLayout(not_available_layout)

        layout.addStretch()

    def try_install_voice_analyzer_deps(self):
        """Try to auto-install voice analyzer dependencies"""
        reply = QMessageBox.question(
            self, "Install Dependencies",
            "This will try to install the Voice Analyzer dependencies using pip.\n\n"
            "Dependencies to install:\n"
            "• praat-parselmouth\n"
            "• librosa\n"
            "• matplotlib\n"
            "• scipy\n"
            "• soundfile\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self.log_output("📦 Installing Voice Analyzer dependencies...")

                # Try to install the packages
                import subprocess
                import sys

                packages = [
                    "praat-parselmouth",
                    "librosa",
                    "matplotlib",
                    "scipy",
                    "soundfile"
                ]

                for package in packages:
                    self.log_output(f"Installing {package}...")
                    result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        self.log_output(f"✅ {package} installed successfully")
                    else:
                        self.log_output(f"❌ Failed to install {package}: {result.stderr}")

                self.log_output("🔄 Installation complete. Please restart ChatterboxTTS to use the Voice Analyzer.")

                QMessageBox.information(
                    self, "Installation Complete",
                    "Dependency installation finished.\n\n"
                    "Please restart ChatterboxTTS to enable the Voice Analyzer tab."
                )

            except Exception as e:
                error_msg = f"Installation failed: {str(e)}"
                self.log_output(f"❌ {error_msg}")
                QMessageBox.critical(self, "Installation Error", error_msg)

    def build_voice_analyzer_gui(self, parent_layout):
        """Build the complete voice analyzer GUI directly in the tab"""
        # Import the analysis functions
        from voice_analyzer.analyzer import analyze_voice_sample
        from voice_analyzer.audio_processor import process_voice_sample

        # Initialize analyzer variables
        self.analyzer_results = []
        self.analyzer_current_result = None
        self.analyzer_processing_thread = None

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        parent_layout.addWidget(main_splitter)

        # Left panel - File selection and controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)

        # File selection group
        file_group = QGroupBox("Voice Sample Selection")
        file_layout = QVBoxLayout(file_group)

        self.analyzer_file_list = QListWidget()
        self.analyzer_file_list.itemClicked.connect(self.on_analyzer_file_selected)
        file_layout.addWidget(self.analyzer_file_list)

        # File buttons
        file_buttons = QHBoxLayout()
        self.analyzer_add_file_btn = QPushButton("Add Files")
        self.analyzer_add_file_btn.clicked.connect(self.add_analyzer_files)
        self.analyzer_remove_file_btn = QPushButton("Remove")
        self.analyzer_remove_file_btn.clicked.connect(self.remove_analyzer_file)
        self.analyzer_clear_files_btn = QPushButton("Clear All")
        self.analyzer_clear_files_btn.clicked.connect(self.clear_analyzer_files)

        file_buttons.addWidget(self.analyzer_add_file_btn)
        file_buttons.addWidget(self.analyzer_remove_file_btn)
        file_buttons.addWidget(self.analyzer_clear_files_btn)
        file_layout.addLayout(file_buttons)

        left_layout.addWidget(file_group)

        # Analysis options
        options_group = QGroupBox("Analysis Options")
        options_layout = QVBoxLayout(options_group)

        self.analyzer_detailed_cb = QCheckBox("Detailed Praat Analysis")
        self.analyzer_detailed_cb.setChecked(True)
        self.analyzer_detailed_cb.setToolTip("Enable advanced voice quality analysis using Praat")
        options_layout.addWidget(self.analyzer_detailed_cb)

        left_layout.addWidget(options_group)

        # Analysis controls
        controls_group = QGroupBox("Analysis Controls")
        controls_layout = QVBoxLayout(controls_group)

        self.analyzer_analyze_btn = QPushButton("🔍 Analyze Selected")
        self.analyzer_analyze_btn.clicked.connect(self.analyze_selected_voice)
        self.analyzer_analyze_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")

        self.analyzer_analyze_all_btn = QPushButton("🔍 Analyze All")
        self.analyzer_analyze_all_btn.clicked.connect(self.analyze_all_voices)

        self.analyzer_progress_bar = QProgressBar()
        self.analyzer_progress_bar.setVisible(False)

        self.analyzer_status_label = QLabel("Ready - Add voice samples to begin")
        self.analyzer_status_label.setStyleSheet("padding: 5px;")

        controls_layout.addWidget(self.analyzer_analyze_btn)
        controls_layout.addWidget(self.analyzer_analyze_all_btn)
        controls_layout.addWidget(self.analyzer_progress_bar)
        controls_layout.addWidget(self.analyzer_status_label)

        left_layout.addWidget(controls_group)

        # Export controls
        export_group = QGroupBox("Export Results")
        export_layout = QVBoxLayout(export_group)

        self.analyzer_export_plot_btn = QPushButton("Save Plot")
        self.analyzer_export_plot_btn.clicked.connect(self.export_analyzer_plot)
        self.analyzer_export_plot_btn.setEnabled(False)

        self.analyzer_export_report_btn = QPushButton("Save Report")
        self.analyzer_export_report_btn.clicked.connect(self.export_analyzer_report)
        self.analyzer_export_report_btn.setEnabled(False)

        export_layout.addWidget(self.analyzer_export_plot_btn)
        export_layout.addWidget(self.analyzer_export_report_btn)

        left_layout.addWidget(export_group)
        left_layout.addStretch()

        # Right panel - Results display with full tabbed interface
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Full results tab widget
        self.analyzer_results_tabs = QTabWidget()

        # 1. Scores tab
        self.setup_analyzer_scores_tab()

        # 2. Analysis Plots tab
        self.setup_analyzer_plots_tab()

        # 3. Recommendations tab
        self.setup_analyzer_recommendations_tab()

        # 4. Compare Samples tab
        self.setup_analyzer_comparison_tab()

        # 5. Auto-Fix tab
        self.setup_analyzer_autofix_tab()

        right_layout.addWidget(self.analyzer_results_tabs)

        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([300, 1100])

        # Initialize UI state
        self.update_analyzer_ui_state()

    def setup_analyzer_scores_tab(self):
        """Setup the scores display tab"""
        scores_tab = QWidget()
        layout = QVBoxLayout(scores_tab)

        # Overall score display
        self.analyzer_overall_score_label = QLabel("No analysis results")
        self.analyzer_overall_score_label.setAlignment(Qt.AlignCenter)
        self.analyzer_overall_score_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.analyzer_overall_score_label.setStyleSheet("padding: 20px; border: 2px solid #ccc; border-radius: 10px; margin: 10px;")
        layout.addWidget(self.analyzer_overall_score_label)

        # Detailed scores grid
        self.analyzer_scores_scroll = QScrollArea()
        self.analyzer_scores_widget = QWidget()
        self.analyzer_scores_layout = QGridLayout(self.analyzer_scores_widget)
        self.analyzer_scores_scroll.setWidget(self.analyzer_scores_widget)
        self.analyzer_scores_scroll.setWidgetResizable(True)
        layout.addWidget(self.analyzer_scores_scroll)

        self.analyzer_results_tabs.addTab(scores_tab, "Scores")

    def setup_analyzer_plots_tab(self):
        """Setup the visualization tab with matplotlib"""
        plots_tab = QWidget()
        layout = QVBoxLayout(plots_tab)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            # Matplotlib figure
            self.analyzer_figure = Figure(figsize=(12, 8))
            self.analyzer_canvas = FigureCanvas(self.analyzer_figure)
            layout.addWidget(self.analyzer_canvas)

            # Initially show placeholder
            ax = self.analyzer_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Select a voice sample and run analysis\nto see detailed plots',
                    ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title('Voice Analysis Plots')
            self.analyzer_canvas.draw()

        except ImportError:
            # Fallback if matplotlib not available
            placeholder_label = QLabel("📊 Analysis plots require matplotlib\nInstall with: pip install matplotlib")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_label.setStyleSheet("color: #666; font-size: 14px; padding: 50px;")
            layout.addWidget(placeholder_label)
            self.analyzer_canvas = None

        self.analyzer_results_tabs.addTab(plots_tab, "Analysis Plots")

    def setup_analyzer_recommendations_tab(self):
        """Setup the recommendations tab"""
        recommendations_tab = QWidget()
        layout = QVBoxLayout(recommendations_tab)

        self.analyzer_recommendations_text = QTextEdit()
        self.analyzer_recommendations_text.setReadOnly(True)
        self.analyzer_recommendations_text.setFont(QFont("Consolas", 11))
        self.analyzer_recommendations_text.setText("No recommendations available - run analysis first")

        layout.addWidget(self.analyzer_recommendations_text)

        self.analyzer_results_tabs.addTab(recommendations_tab, "Recommendations")

    def setup_analyzer_comparison_tab(self):
        """Setup the comparison tab"""
        comparison_tab = QWidget()
        layout = QVBoxLayout(comparison_tab)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            # Comparison matplotlib figure
            self.analyzer_comparison_figure = Figure(figsize=(12, 8))
            self.analyzer_comparison_canvas = FigureCanvas(self.analyzer_comparison_figure)
            layout.addWidget(self.analyzer_comparison_canvas)

            # Initially show placeholder
            ax = self.analyzer_comparison_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Analyze multiple voice samples\nto see comparison plots',
                    ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title('Sample Comparison')
            self.analyzer_comparison_canvas.draw()

        except ImportError:
            # Fallback if matplotlib not available
            placeholder_label = QLabel("📊 Comparison plots require matplotlib\nInstall with: pip install matplotlib")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_label.setStyleSheet("color: #666; font-size: 14px; padding: 50px;")
            layout.addWidget(placeholder_label)
            self.analyzer_comparison_canvas = None

        self.analyzer_results_tabs.addTab(comparison_tab, "Compare Samples")

    def setup_analyzer_autofix_tab(self):
        """Setup the auto-fix tab with comprehensive audio processing fixes"""
        autofix_tab = QWidget()
        layout = QVBoxLayout(autofix_tab)

        # Title and instructions
        title_label = QLabel("Automated Audio Processing Fixes")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        instructions = QLabel("Select the fixes to apply to your voice sample. Each fix will only run when checked.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(instructions)

        # Scroll area for fix options
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Audio Quality Fixes Group
        audio_group = QGroupBox("Audio Quality Fixes")
        audio_layout = QVBoxLayout(audio_group)

        self.analyzer_fix_clipping_cb = QCheckBox("Fix Clipping (0.22% detected)")
        self.analyzer_fix_clipping_cb.setToolTip("Repair digital clipping using interpolation and soft limiting")
        audio_layout.addWidget(self.analyzer_fix_clipping_cb)

        self.analyzer_normalize_volume_cb = QCheckBox("Normalize Volume Consistency")
        self.analyzer_normalize_volume_cb.setToolTip("Apply dynamic range compression to even out volume levels")
        audio_layout.addWidget(self.analyzer_normalize_volume_cb)

        self.analyzer_reduce_noise_cb = QCheckBox("Reduce Background Noise")
        self.analyzer_reduce_noise_cb.setToolTip("Apply spectral noise reduction to improve SNR")
        audio_layout.addWidget(self.analyzer_reduce_noise_cb)

        self.analyzer_optimize_dynamic_range_cb = QCheckBox("Optimize Dynamic Range")
        self.analyzer_optimize_dynamic_range_cb.setToolTip("Enhance dynamic range for better TTS compatibility")
        audio_layout.addWidget(self.analyzer_optimize_dynamic_range_cb)

        scroll_layout.addWidget(audio_group)

        # Voice Enhancement Group
        voice_group = QGroupBox("Voice Enhancement")
        voice_layout = QVBoxLayout(voice_group)

        self.analyzer_apply_tts_eq_cb = QCheckBox("Apply TTS-Optimized EQ")
        self.analyzer_apply_tts_eq_cb.setToolTip("Apply frequency shaping optimized for TTS training")
        voice_layout.addWidget(self.analyzer_apply_tts_eq_cb)

        self.analyzer_normalize_lufs_cb = QCheckBox("Normalize to -16 LUFS")
        self.analyzer_normalize_lufs_cb.setToolTip("Normalize loudness to broadcast standard (-16 LUFS)")
        voice_layout.addWidget(self.analyzer_normalize_lufs_cb)

        self.analyzer_enhance_clarity_cb = QCheckBox("Enhance Voice Clarity")
        self.analyzer_enhance_clarity_cb.setToolTip("Apply subtle enhancement to improve voice definition")
        voice_layout.addWidget(self.analyzer_enhance_clarity_cb)

        self.analyzer_reduce_sibilance_cb = QCheckBox("Reduce Harsh Sibilants (De-essing)")
        self.analyzer_reduce_sibilance_cb.setToolTip("Split-band de-esser to reduce harsh 's' and 'sh' sounds for cleaner TTS training")
        voice_layout.addWidget(self.analyzer_reduce_sibilance_cb)

        self.analyzer_slow_speaking_rate_cb = QCheckBox("Slow Down Speaking Rate")
        self.analyzer_slow_speaking_rate_cb.setToolTip("Reduce speaking tempo for better TTS compatibility and clarity")
        voice_layout.addWidget(self.analyzer_slow_speaking_rate_cb)

        scroll_layout.addWidget(voice_group)

        # Advanced Processing Group
        advanced_group = QGroupBox("Advanced Processing")
        advanced_layout = QVBoxLayout(advanced_group)

        self.analyzer_remove_dc_offset_cb = QCheckBox("Remove DC Offset")
        self.analyzer_remove_dc_offset_cb.setToolTip("Remove any DC bias from the audio signal")
        advanced_layout.addWidget(self.analyzer_remove_dc_offset_cb)

        self.analyzer_normalize_sample_rate_cb = QCheckBox("Normalize Sample Rate to 24kHz")
        self.analyzer_normalize_sample_rate_cb.setToolTip("Resample audio to optimal rate for TTS processing")
        advanced_layout.addWidget(self.analyzer_normalize_sample_rate_cb)

        self.analyzer_trim_silence_cb = QCheckBox("Trim Start/End Silence")
        self.analyzer_trim_silence_cb.setToolTip("Remove excessive silence from beginning and end")
        advanced_layout.addWidget(self.analyzer_trim_silence_cb)

        scroll_layout.addWidget(advanced_group)

        # Processing Controls
        controls_group = QGroupBox("Processing Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Output quality setting
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Output Quality:"))
        self.analyzer_quality_spin = NoScrollSpinBox()
        self.analyzer_quality_spin.setRange(1, 10)
        self.analyzer_quality_spin.setValue(8)
        self.analyzer_quality_spin.setToolTip("Processing quality (1=fast, 10=best quality)")
        quality_layout.addWidget(self.analyzer_quality_spin)
        quality_layout.addStretch()
        controls_layout.addLayout(quality_layout)

        # Preserve characteristics checkbox
        self.analyzer_preserve_characteristics_cb = QCheckBox("Preserve Natural Voice Characteristics")
        self.analyzer_preserve_characteristics_cb.setChecked(True)
        self.analyzer_preserve_characteristics_cb.setToolTip("Maintain pitch variation and breathing that makes TTS sound human")
        controls_layout.addWidget(self.analyzer_preserve_characteristics_cb)

        scroll_layout.addWidget(controls_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self.analyzer_select_all_fixes_btn = QPushButton("Select All")
        self.analyzer_select_all_fixes_btn.clicked.connect(self.select_all_analyzer_fixes)
        button_layout.addWidget(self.analyzer_select_all_fixes_btn)

        self.analyzer_select_recommended_fixes_btn = QPushButton("Select Recommended")
        self.analyzer_select_recommended_fixes_btn.clicked.connect(self.select_recommended_analyzer_fixes)
        button_layout.addWidget(self.analyzer_select_recommended_fixes_btn)

        self.analyzer_clear_all_fixes_btn = QPushButton("Clear All")
        self.analyzer_clear_all_fixes_btn.clicked.connect(self.clear_all_analyzer_fixes)
        button_layout.addWidget(self.analyzer_clear_all_fixes_btn)

        button_layout.addStretch()

        self.analyzer_apply_fixes_btn = QPushButton("🔧 Apply Selected Fixes")
        self.analyzer_apply_fixes_btn.clicked.connect(self.apply_analyzer_fixes)
        self.analyzer_apply_fixes_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; background-color: #4CAF50; color: white; }")
        self.analyzer_apply_fixes_btn.setEnabled(False)
        button_layout.addWidget(self.analyzer_apply_fixes_btn)

        scroll_layout.addLayout(button_layout)

        # Progress and status
        self.analyzer_fix_progress_bar = QProgressBar()
        self.analyzer_fix_progress_bar.setVisible(False)
        scroll_layout.addWidget(self.analyzer_fix_progress_bar)

        self.analyzer_fix_status_label = QLabel("Select a voice sample and analysis result to enable fixes")
        self.analyzer_fix_status_label.setStyleSheet("padding: 5px; color: #666;")
        scroll_layout.addWidget(self.analyzer_fix_status_label)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Connect all checkboxes to update function
        checkboxes = [
            self.analyzer_fix_clipping_cb, self.analyzer_normalize_volume_cb, self.analyzer_reduce_noise_cb,
            self.analyzer_optimize_dynamic_range_cb, self.analyzer_apply_tts_eq_cb, self.analyzer_normalize_lufs_cb,
            self.analyzer_enhance_clarity_cb, self.analyzer_reduce_sibilance_cb, self.analyzer_slow_speaking_rate_cb,
            self.analyzer_remove_dc_offset_cb, self.analyzer_normalize_sample_rate_cb, self.analyzer_trim_silence_cb
        ]

        for cb in checkboxes:
            cb.stateChanged.connect(self.update_analyzer_fix_ui_state)

        self.analyzer_results_tabs.addTab(autofix_tab, "Auto-Fix")

    def add_analyzer_files(self):
        """Add voice sample files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Voice Samples", "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*)"
        )

        for file_path in file_paths:
            if file_path:
                item = QListWidgetItem(Path(file_path).name)
                item.setData(Qt.UserRole, file_path)
                item.setToolTip(file_path)
                self.analyzer_file_list.addItem(item)

        self.update_analyzer_ui_state()
        self.log_output(f"📁 Added {len(file_paths)} voice samples for analysis")

    def remove_analyzer_file(self):
        """Remove selected file"""
        current_item = self.analyzer_file_list.currentItem()
        current_row = self.analyzer_file_list.currentRow()

        if current_row >= 0 and current_item:
            file_path = current_item.data(Qt.UserRole)
            file_name = Path(file_path).name

            # Remove from file list
            self.analyzer_file_list.takeItem(current_row)

            # Remove corresponding result
            self.analyzer_results = [r for r in self.analyzer_results if r.filename != file_name]

            # Clear current result if it was the removed file
            if self.analyzer_current_result and self.analyzer_current_result.filename == file_name:
                self.analyzer_current_result = None
                self.clear_analyzer_displays()

        self.update_analyzer_ui_state()

    def clear_analyzer_files(self):
        """Clears all files and results in the analyzer.
        Args:
        None
        Returns:
        None
        """
        """Clear all files"""
        self.analyzer_file_list.clear()
        self.analyzer_results.clear()
        self.analyzer_current_result = None
        self.clear_analyzer_displays()
        self.update_analyzer_ui_state()

    def on_analyzer_file_selected(self, item):
        """Handle file selection"""
        if item:
            file_path = item.data(Qt.UserRole)
            file_name = Path(file_path).name

            # Find corresponding result
            for result in self.analyzer_results:
                if result.filename == file_name:
                    self.analyzer_current_result = result
                    self.update_analyzer_result_display(result)
                    break
        self.update_analyzer_ui_state()

    def analyze_selected_voice(self):
        """Analyze selected voice file"""
        current_item = self.analyzer_file_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a voice sample to analyze.")
            return

        file_path = current_item.data(Qt.UserRole)
        self.start_voice_analysis([file_path])

    def analyze_all_voices(self):
        """Analyze all voice files"""
        if self.analyzer_file_list.count() == 0:
            QMessageBox.warning(self, "No Files", "Please add voice samples to analyze.")
            return

        file_paths = []
        for i in range(self.analyzer_file_list.count()):
            item = self.analyzer_file_list.item(i)
            file_paths.append(item.data(Qt.UserRole))

        self.start_voice_analysis(file_paths)

    def start_voice_analysis(self, file_paths):
        """Start voice analysis"""
        from voice_analyzer.analyzer import analyze_voice_sample

        self.analyzer_progress_bar.setVisible(True)
        self.analyzer_progress_bar.setRange(0, len(file_paths))
        self.analyzer_status_label.setText("Analyzing voice samples...")

        self.analyzer_analyze_btn.setEnabled(False)
        self.analyzer_analyze_all_btn.setEnabled(False)

        # Process files synchronously for simplicity
        for i, file_path in enumerate(file_paths):
            try:
                self.analyzer_progress_bar.setValue(i)
                self.analyzer_status_label.setText(f"Analyzing {Path(file_path).name}...")

                # Perform analysis
                result = analyze_voice_sample(file_path, self.analyzer_detailed_cb.isChecked())

                # Update or add result
                file_name = Path(file_path).name
                existing_index = -1
                for j, existing_result in enumerate(self.analyzer_results):
                    if existing_result.filename == file_name:
                        existing_index = j
                        break

                if existing_index >= 0:
                    self.analyzer_results[existing_index] = result
                else:
                    self.analyzer_results.append(result)

                # Update display if this is the current selection
                current_item = self.analyzer_file_list.currentItem()
                if current_item and Path(current_item.data(Qt.UserRole)).name == result.filename:
                    self.analyzer_current_result = result
                    self.update_analyzer_result_display(result)

                self.log_output(f"✅ {file_name}: Score {result.overall_score:.1f}/100 ({result.suitability_rating})")

            except Exception as e:
                self.log_output(f"❌ Analysis failed for {Path(file_path).name}: {str(e)}")

        self.analyzer_progress_bar.setValue(len(file_paths))
        self.analyzer_progress_bar.setVisible(False)
        self.analyzer_status_label.setText("Analysis complete")

        self.analyzer_analyze_btn.setEnabled(True)
        self.analyzer_analyze_all_btn.setEnabled(True)
        self.update_analyzer_ui_state()

    def update_analyzer_result_display(self, result):
        """Update the display with analysis result"""
        if not result.success:
            self.analyzer_overall_score_label.setText(f"Analysis Failed: {result.error_message}")
            return

        # Update overall score
        color = "green" if result.overall_score >= 75 else "orange" if result.overall_score >= 50 else "red"
        self.analyzer_overall_score_label.setText(f"{result.filename}\nOverall Score: {result.overall_score:.1f}/100\n{result.suitability_rating}")
        self.analyzer_overall_score_label.setStyleSheet(f"padding: 20px; border: 2px solid {color}; border-radius: 10px; margin: 10px; color: {color};")

        # Update detailed scores
        self.clear_analyzer_scores_grid()

        scores_data = [
            ("Audio Quality", result.audio_quality_score),
            ("Noise Level", result.noise_score),
            ("Dynamic Range", result.dynamic_range_score),
            ("Clipping", result.clipping_score),
            ("Pitch Stability", result.pitch_stability_score),
            ("Voice Quality", result.voice_quality_score),
            ("Speaking Rate", result.speaking_rate_score),
            ("Sibilance", result.sibilance_score),
            ("Consistency", result.consistency_score)
        ]

        for i, (label, score) in enumerate(scores_data):
            row = i // 3
            col = i % 3
            score_widget = self.create_score_widget(label, score)
            self.analyzer_scores_layout.addWidget(score_widget, row, col)

        # Update recommendations
        recommendations_text = f"Analysis Results for: {result.filename}\n"
        recommendations_text += "=" * 60 + "\n\n"
        recommendations_text += f"Duration: {result.duration:.2f} seconds\n"
        recommendations_text += f"Sample Rate: {result.sample_rate} Hz\n"
        recommendations_text += f"Channels: {result.channels}\n\n"
        recommendations_text += "Recommendations:\n"
        recommendations_text += "-" * 20 + "\n"

        for i, rec in enumerate(result.recommendations, 1):
            recommendations_text += f"{i}. {rec}\n"

        self.analyzer_recommendations_text.setText(recommendations_text)

        # Update visualization
        self.update_analyzer_visualization(result)

        # Update comparison plot if multiple results
        if len(self.analyzer_results) > 1:
            self.update_analyzer_comparison_plot()

    def create_score_widget(self, label, score):
        """Create a score display widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Score label
        score_label = QLabel(f"{score:.1f}")
        score_label.setAlignment(Qt.AlignCenter)
        score_label.setFont(QFont("Arial", 14, QFont.Bold))

        # Color coding
        if score >= 75:
            color = "green"
        elif score >= 50:
            color = "orange"
        else:
            color = "red"

        score_label.setStyleSheet(f"color: {color}; padding: 5px;")

        # Category label
        category_label = QLabel(label)
        category_label.setAlignment(Qt.AlignCenter)
        category_label.setFont(QFont("Arial", 9))
        category_label.setWordWrap(True)

        layout.addWidget(score_label)
        layout.addWidget(category_label)

        # Add border
        widget.setStyleSheet("QWidget { border: 1px solid #ccc; border-radius: 5px; margin: 2px; }")

        return widget

    def clear_analyzer_scores_grid(self):
        """Clear the scores grid"""
        while self.analyzer_scores_layout.count():
            child = self.analyzer_scores_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def clear_analyzer_displays(self):
        """Clear all result displays"""
        self.analyzer_overall_score_label.setText("No analysis results")
        self.analyzer_overall_score_label.setStyleSheet("padding: 20px; border: 2px solid #ccc; border-radius: 10px; margin: 10px;")

        self.clear_analyzer_scores_grid()
        self.analyzer_recommendations_text.setText("No recommendations available - run analysis first")


    def update_analyzer_ui_state(self):
        """Update UI element states"""
        has_files = self.analyzer_file_list.count() > 0
        has_selection = self.analyzer_file_list.currentItem() is not None
        has_results = len(self.analyzer_results) > 0

        self.analyzer_analyze_btn.setEnabled(has_files and has_selection)
        self.analyzer_analyze_all_btn.setEnabled(has_files)
        self.analyzer_remove_file_btn.setEnabled(has_selection)
        self.analyzer_clear_files_btn.setEnabled(has_files)
        self.analyzer_export_plot_btn.setEnabled(has_results and self.analyzer_current_result is not None)
        self.analyzer_export_report_btn.setEnabled(has_results and self.analyzer_current_result is not None)

    def select_all_analyzer_fixes(self):
        """Select all fix checkboxes"""
        checkboxes = [
            self.analyzer_fix_clipping_cb, self.analyzer_normalize_volume_cb, self.analyzer_reduce_noise_cb,
            self.analyzer_optimize_dynamic_range_cb, self.analyzer_apply_tts_eq_cb, self.analyzer_normalize_lufs_cb,
            self.analyzer_enhance_clarity_cb, self.analyzer_reduce_sibilance_cb, self.analyzer_slow_speaking_rate_cb,
            self.analyzer_remove_dc_offset_cb, self.analyzer_normalize_sample_rate_cb, self.analyzer_trim_silence_cb
        ]
        for cb in checkboxes:
            cb.setChecked(True)

    def select_recommended_analyzer_fixes(self):
        """Select recommended fixes based on current analysis"""
        if not self.analyzer_current_result:
            QMessageBox.information(self, "No Analysis", "Please analyze a voice sample first.")
            return

        # Clear all first
        self.clear_all_analyzer_fixes()

        # Select based on scores
        result = self.analyzer_current_result
        if result.clipping_score < 90:
            self.analyzer_fix_clipping_cb.setChecked(True)
        if result.noise_score < 75:
            self.analyzer_reduce_noise_cb.setChecked(True)
        if result.dynamic_range_score < 70:
            self.analyzer_optimize_dynamic_range_cb.setChecked(True)
        if result.consistency_score < 75:
            self.analyzer_normalize_volume_cb.setChecked(True)
        if result.audio_quality_score < 80:
            self.analyzer_apply_tts_eq_cb.setChecked(True)
        if result.speaking_rate_score < 75:
            self.analyzer_slow_speaking_rate_cb.setChecked(True)
        if result.sibilance_score < 70:
            self.analyzer_reduce_sibilance_cb.setChecked(True)

        # Always recommend these basic fixes
        self.analyzer_remove_dc_offset_cb.setChecked(True)
        self.analyzer_normalize_lufs_cb.setChecked(True)

        self.log_output("🎯 Recommended fixes selected based on analysis results")

    def clear_all_analyzer_fixes(self):
        """Clear all fix checkboxes"""
        checkboxes = [
            self.analyzer_fix_clipping_cb, self.analyzer_normalize_volume_cb, self.analyzer_reduce_noise_cb,
            self.analyzer_optimize_dynamic_range_cb, self.analyzer_apply_tts_eq_cb, self.analyzer_normalize_lufs_cb,
            self.analyzer_enhance_clarity_cb, self.analyzer_reduce_sibilance_cb, self.analyzer_slow_speaking_rate_cb,
            self.analyzer_remove_dc_offset_cb, self.analyzer_normalize_sample_rate_cb, self.analyzer_trim_silence_cb
        ]
        for cb in checkboxes:
            cb.setChecked(False)

    def update_analyzer_fix_ui_state(self):
        """Update the fix UI state based on selections"""
        checkboxes = [
            self.analyzer_fix_clipping_cb, self.analyzer_normalize_volume_cb, self.analyzer_reduce_noise_cb,
            self.analyzer_optimize_dynamic_range_cb, self.analyzer_apply_tts_eq_cb, self.analyzer_normalize_lufs_cb,
            self.analyzer_enhance_clarity_cb, self.analyzer_reduce_sibilance_cb, self.analyzer_slow_speaking_rate_cb,
            self.analyzer_remove_dc_offset_cb, self.analyzer_normalize_sample_rate_cb, self.analyzer_trim_silence_cb
        ]

        any_selected = any(cb.isChecked() for cb in checkboxes)
        has_current_file = self.analyzer_current_result is not None
        has_selection = self.analyzer_file_list.currentItem() is not None

        self.analyzer_apply_fixes_btn.setEnabled(any_selected and has_current_file and has_selection)

        if any_selected and has_current_file:
            selected_count = sum(1 for cb in checkboxes if cb.isChecked())
            self.analyzer_fix_status_label.setText(f"Ready to apply {selected_count} fixes to {self.analyzer_current_result.filename}")
        elif not has_current_file:
            self.analyzer_fix_status_label.setText("Select and analyze a voice sample to enable fixes")
        else:
            self.analyzer_fix_status_label.setText("Select fixes to apply")

    def apply_analyzer_fixes(self):
        """Apply selected audio fixes using the comprehensive Auto-Fix system"""
        current_item = self.analyzer_file_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a voice sample.")
            return

        # Collect selected fixes
        selected_fixes = []
        if self.analyzer_fix_clipping_cb.isChecked():
            selected_fixes.append("fix_clipping")
        if self.analyzer_normalize_volume_cb.isChecked():
            selected_fixes.append("normalize_volume")
        if self.analyzer_reduce_noise_cb.isChecked():
            selected_fixes.append("reduce_noise")
        if self.analyzer_optimize_dynamic_range_cb.isChecked():
            selected_fixes.append("optimize_dynamic_range")
        if self.analyzer_apply_tts_eq_cb.isChecked():
            selected_fixes.append("apply_tts_eq")
        if self.analyzer_normalize_lufs_cb.isChecked():
            selected_fixes.append("normalize_lufs")
        if self.analyzer_enhance_clarity_cb.isChecked():
            selected_fixes.append("enhance_clarity")
        if self.analyzer_reduce_sibilance_cb.isChecked():
            selected_fixes.append("reduce_sibilance")
        if self.analyzer_slow_speaking_rate_cb.isChecked():
            selected_fixes.append("slow_speaking_rate")
        if self.analyzer_remove_dc_offset_cb.isChecked():
            selected_fixes.append("remove_dc_offset")
        if self.analyzer_normalize_sample_rate_cb.isChecked():
            selected_fixes.append("normalize_sample_rate")
        if self.analyzer_trim_silence_cb.isChecked():
            selected_fixes.append("trim_silence")

        if not selected_fixes:
            QMessageBox.information(self, "No Fixes Selected", "Please select at least one fix to apply.")
            return

        input_file_path = current_item.data(Qt.UserRole)
        input_path = Path(input_file_path)
        default_output = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"

        output_file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Fixed Audio File",
            str(default_output),
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )

        if not output_file_path:
            return

        # Show confirmation dialog
        fix_count = len(selected_fixes)
        quality_level = self.analyzer_quality_spin.value()
        preserve_characteristics = self.analyzer_preserve_characteristics_cb.isChecked()

        msg = f"Apply {fix_count} audio fixes to:\n{Path(input_file_path).name}\n\n"
        msg += f"Quality Level: {quality_level}/10\n"
        msg += f"Preserve Natural Characteristics: {'Yes' if preserve_characteristics else 'No'}\n\n"
        msg += f"Output: {Path(output_file_path).name}"

        reply = QMessageBox.question(self, "Confirm Audio Processing", msg,
                                   QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                from voice_analyzer.audio_processor import process_voice_sample

                self.analyzer_fix_status_label.setText("Processing audio fixes...")
                self.analyzer_fix_progress_bar.setVisible(True)
                self.analyzer_fix_progress_bar.setRange(0, 0)  # Indeterminate

                def progress_callback(message):
                    """Updates the status label with a message during the processing of a voice sample.
                    Args:
                    message (str): The message to be displayed on the status label.
                    Returns:
                    None
                    """
                    self.analyzer_fix_status_label.setText(message)

                results = process_voice_sample(
                    input_file_path,
                    output_file_path,
                    selected_fixes,
                    preserve_characteristics,
                    quality_level,
                    progress_callback
                )

                self.analyzer_fix_progress_bar.setVisible(False)

                if results['success']:
                    fixes_count = len(results['fixes_applied'])
                    duration_change = results['final_duration'] - results['original_duration']

                    # Show detailed results
                    message = f"✅ Audio processing complete!\n\n"
                    message += f"Applied {fixes_count} fixes successfully\n"
                    message += f"Original duration: {results['original_duration']:.2f}s\n"
                    message += f"Final duration: {results['final_duration']:.2f}s\n"
                    message += f"Duration change: {duration_change:+.2f}s\n"
                    message += f"Sample rate: {results['original_sample_rate']}Hz → {results['final_sample_rate']}Hz\n\n"
                    message += f"Output saved to:\n{Path(results['output_path']).name}"

                    # Add processing statistics
                    if 'statistics' in results and results['statistics']:
                        message += "\n\nProcessing Statistics:\n"
                        for fix_name, stats in results['statistics'].items():
                            if isinstance(stats, dict):
                                for key, value in stats.items():
                                    if isinstance(value, float) and abs(value) > 0.001:
                                        message += f"• {fix_name}: {key} = {value:.3f}\n"

                    QMessageBox.information(self, "Processing Complete", message)
                    self.analyzer_fix_status_label.setText(f"Processing complete! {fixes_count} fixes applied successfully.")

                    self.log_output(f"✅ Applied {fixes_count} audio fixes to {Path(input_file_path).name}")
                    self.log_output(f"📁 Output saved: {Path(output_file_path).name}")

                    # Ask if user wants to analyze the processed file
                    reply = QMessageBox.question(
                        self, "Analyze Processed File",
                        "Would you like to add and analyze the processed file to compare results?",
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if reply == QMessageBox.Yes:
                        # Add processed file to the list
                        processed_path = results['output_path']
                        item = QListWidgetItem(Path(processed_path).name)
                        item.setData(Qt.UserRole, processed_path)
                        item.setToolTip(processed_path)
                        self.analyzer_file_list.addItem(item)

                        # Select and analyze it
                        self.analyzer_file_list.setCurrentItem(item)
                        self.analyze_selected_voice()
                else:
                    error_msg = results.get('error', 'Unknown error')
                    self.log_output(f"❌ Audio processing failed: {error_msg}")
                    QMessageBox.critical(self, "Processing Failed", f"Audio processing failed:\n{error_msg}")
                    self.analyzer_fix_status_label.setText("Processing failed")

            except Exception as e:
                self.analyzer_fix_progress_bar.setVisible(False)
                error_msg = str(e)
                self.log_output(f"❌ Audio processing error: {error_msg}")
                QMessageBox.critical(self, "Processing Error", f"Audio processing error:\n{error_msg}")
                self.analyzer_fix_status_label.setText("Processing failed")

    def export_analyzer_plot(self):
        """Export current analysis plot"""
        if not self.analyzer_current_result:
            QMessageBox.warning(self, "No Results", "Please analyze a voice sample first.")
            return

        if not hasattr(self, 'analyzer_canvas') or not self.analyzer_canvas:
            QMessageBox.warning(self, "No Plot", "Matplotlib not available for plot export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Plot",
            f"{self.analyzer_current_result.filename}_analysis.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )

        if file_path:
            try:
                self.analyzer_figure.savefig(file_path, dpi=150, bbox_inches='tight')
                QMessageBox.information(self, "Export Success", f"Plot saved to:\n{file_path}")
                self.log_output(f"📊 Analysis plot exported: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save plot:\n{str(e)}")

    def export_analyzer_report(self):
        """Export current analysis report"""
        if not self.analyzer_current_result:
            QMessageBox.warning(self, "No Results", "Please analyze a voice sample first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Report",
            f"{self.analyzer_current_result.filename}_report.txt",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                from voice_analyzer.visualizer import create_summary_report
                report = create_summary_report(self.analyzer_current_result)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                QMessageBox.information(self, "Export Success", f"Report saved to:\n{file_path}")
                self.log_output(f"📝 Analysis report exported: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save report:\n{str(e)}")

    def update_analyzer_visualization(self, result):
        """Update the analysis plots"""
        if not hasattr(self, 'analyzer_canvas') or not self.analyzer_canvas:
            return

        if not result.success:
            return

        try:
            self.analyzer_figure.clear()

            # Create a simple summary plot like in the standalone version
            ax = self.analyzer_figure.add_subplot(111)

            # Create summary bar chart
            categories = ['Audio\nQuality', 'Noise', 'Dynamic\nRange', 'Clipping',
                         'Pitch\nStability', 'Voice\nQuality', 'Speaking\nRate', 'Consistency']
            scores = [result.audio_quality_score, result.noise_score, result.dynamic_range_score,
                     result.clipping_score, result.pitch_stability_score, result.voice_quality_score,
                     result.speaking_rate_score, result.consistency_score]

            colors = ['red' if s < 50 else 'orange' if s < 75 else 'green' for s in scores]
            bars = ax.bar(categories, scores, color=colors, alpha=0.7)

            ax.set_title(f'Quality Scores - {result.filename}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 100)
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Poor/Fair')
            ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='Fair/Good')

            # Rotate x-axis labels for better readability
            import matplotlib.pyplot as plt
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Add score values on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{score:.0f}', ha='center', va='bottom', fontsize=10)

            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')

            # Add legend
            ax.legend(loc='upper right')

            self.analyzer_figure.tight_layout()
            self.analyzer_canvas.draw()

        except Exception as e:
            # Fallback to error message
            self.analyzer_figure.clear()
            ax = self.analyzer_figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Visualization error:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Visualization Error')
            self.analyzer_canvas.draw()

    def update_analyzer_comparison_plot(self):
        """Update comparison plot for multiple samples"""
        if not hasattr(self, 'analyzer_comparison_canvas') or not self.analyzer_comparison_canvas:
            return

        if len(self.analyzer_results) < 2:
            return

        try:
            self.analyzer_comparison_figure.clear()

            # Create comparison plot
            ax1 = self.analyzer_comparison_figure.add_subplot(2, 1, 1)
            ax2 = self.analyzer_comparison_figure.add_subplot(2, 1, 2)

            # Overall scores comparison
            filenames = [r.filename[:15] + '...' if len(r.filename) > 15 else r.filename for r in self.analyzer_results]
            overall_scores = [r.overall_score for r in self.analyzer_results]

            colors = ['red' if s < 50 else 'orange' if s < 75 else 'green' for s in overall_scores]
            bars1 = ax1.bar(range(len(filenames)), overall_scores, color=colors, alpha=0.7)
            ax1.set_title('Overall TTS Suitability Scores', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 100)
            ax1.set_xticks(range(len(filenames)))
            ax1.set_xticklabels(filenames, rotation=45, ha='right')
            ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
            ax1.axhline(y=75, color='orange', linestyle='--', alpha=0.5)
            ax1.grid(True, alpha=0.3, axis='y')

            # Add score values
            for bar, score in zip(bars1, overall_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{score:.0f}', ha='center', va='bottom', fontsize=9)

            # Detailed comparison (key metrics only)
            import numpy as np
            categories = ['Audio Qual.', 'Noise', 'Pitch Stab.', 'Voice Qual.']
            x = np.arange(len(categories))
            width = 0.8 / len(self.analyzer_results) if len(self.analyzer_results) > 0 else 0.8

            for i, result in enumerate(self.analyzer_results):
                scores = [result.audio_quality_score, result.noise_score,
                         result.pitch_stability_score, result.voice_quality_score]
                ax2.bar(x + i * width, scores, width, label=filenames[i], alpha=0.7)

            ax2.set_title('Key Quality Metrics Comparison', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 100)
            ax2.set_xticks(x + width * (len(self.analyzer_results) - 1) / 2)
            ax2.set_xticklabels(categories)
            if len(self.analyzer_results) <= 4:  # Only show legend if not too crowded
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3, axis='y')

            self.analyzer_comparison_figure.tight_layout()
            self.analyzer_comparison_canvas.draw()

        except Exception as e:
            # Fallback to error message
            self.analyzer_comparison_figure.clear()
            ax = self.analyzer_comparison_figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Comparison error:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Comparison Error')
            self.analyzer_comparison_canvas.draw()

    def create_audio_output_analyzer_tab(self):
        """Tab 10: Audio Output Analyzer for finished audiobooks"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "10. Audio Output Analyzer")

        layout = QVBoxLayout(tab)

        # Title and description
        info_label = QLabel("🎧 Audio Output Analyzer for Finished Audiobooks")
        info_label.setStyleSheet("font-weight: bold; color: #FF5722; padding: 10px;")
        layout.addWidget(info_label)

        desc_label = QLabel("Analyze completed audiobook files for production quality, consistency, and technical standards")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(desc_label)

        # Initialize output analyzer variables
        self.output_analyzer_results = []
        self.output_analyzer_current_result = None

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)

        # Left panel - File selection and controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(380)
        left_layout = QVBoxLayout(left_panel)

        # File selection group
        file_group = QGroupBox("Audiobook File Selection")
        file_layout = QVBoxLayout(file_group)

        self.output_file_list = QListWidget()
        self.output_file_list.itemClicked.connect(self.on_output_file_selected)
        file_layout.addWidget(self.output_file_list)

        # File buttons
        file_buttons = QHBoxLayout()
        self.output_add_file_btn = QPushButton("Add Audiobook Files")
        self.output_add_file_btn.clicked.connect(self.add_output_files)
        self.output_remove_file_btn = QPushButton("Remove")
        self.output_remove_file_btn.clicked.connect(self.remove_output_file)
        self.output_clear_files_btn = QPushButton("Clear All")
        self.output_clear_files_btn.clicked.connect(self.clear_output_files)

        file_buttons.addWidget(self.output_add_file_btn)
        file_buttons.addWidget(self.output_remove_file_btn)
        file_buttons.addWidget(self.output_clear_files_btn)
        file_layout.addLayout(file_buttons)

        left_layout.addWidget(file_group)

        # Analysis options
        options_group = QGroupBox("Analysis Options")
        options_layout = QVBoxLayout(options_group)

        self.output_detailed_analysis_cb = QCheckBox("Detailed Technical Analysis")
        self.output_detailed_analysis_cb.setChecked(True)
        self.output_detailed_analysis_cb.setToolTip("Enable comprehensive production quality analysis")
        options_layout.addWidget(self.output_detailed_analysis_cb)

        self.output_chapter_analysis_cb = QCheckBox("Chapter-by-Chapter Analysis")
        self.output_chapter_analysis_cb.setChecked(False)
        self.output_chapter_analysis_cb.setToolTip("Analyze consistency between chapters (requires chapter markers)")
        options_layout.addWidget(self.output_chapter_analysis_cb)

        self.output_commercial_standards_cb = QCheckBox("Commercial Production Standards")
        self.output_commercial_standards_cb.setChecked(True)
        self.output_commercial_standards_cb.setToolTip("Apply commercial audiobook production standards")
        options_layout.addWidget(self.output_commercial_standards_cb)

        left_layout.addWidget(options_group)

        # Analysis controls
        controls_group = QGroupBox("Analysis Controls")
        controls_layout = QVBoxLayout(controls_group)

        self.output_analyze_btn = QPushButton("🔍 Analyze Selected")
        self.output_analyze_btn.clicked.connect(self.analyze_selected_output)
        self.output_analyze_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")

        self.output_analyze_all_btn = QPushButton("🔍 Analyze All Files")
        self.output_analyze_all_btn.clicked.connect(self.analyze_all_outputs)

        self.output_progress_bar = QProgressBar()
        self.output_progress_bar.setVisible(False)

        self.output_status_label = QLabel("Ready - Add audiobook files to begin")
        self.output_status_label.setStyleSheet("padding: 5px;")

        controls_layout.addWidget(self.output_analyze_btn)
        controls_layout.addWidget(self.output_analyze_all_btn)
        controls_layout.addWidget(self.output_progress_bar)
        controls_layout.addWidget(self.output_status_label)

        left_layout.addWidget(controls_group)

        # Quality assessment
        quality_group = QGroupBox("Quality Assessment")
        quality_layout = QVBoxLayout(quality_group)

        # Overall quality display
        self.output_overall_quality_label = QLabel("No analysis results")
        self.output_overall_quality_label.setAlignment(Qt.AlignCenter)
        self.output_overall_quality_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.output_overall_quality_label.setStyleSheet("padding: 15px; border: 2px solid #ccc; border-radius: 5px; margin: 5px;")
        quality_layout.addWidget(self.output_overall_quality_label)

        # Key metrics
        metrics_layout = QGridLayout()

        # Create metric labels
        self.output_duration_label = QLabel("Duration: --")
        self.output_size_label = QLabel("Size: --")
        self.output_bitrate_label = QLabel("Bitrate: --")
        self.output_sample_rate_label = QLabel("Sample Rate: --")
        self.output_channels_label = QLabel("Channels: --")
        self.output_format_label = QLabel("Format: --")

        metrics_layout.addWidget(QLabel("📏"), 0, 0)
        metrics_layout.addWidget(self.output_duration_label, 0, 1)
        metrics_layout.addWidget(QLabel("💾"), 1, 0)
        metrics_layout.addWidget(self.output_size_label, 1, 1)
        metrics_layout.addWidget(QLabel("🎵"), 2, 0)
        metrics_layout.addWidget(self.output_bitrate_label, 2, 1)
        metrics_layout.addWidget(QLabel("⚡"), 0, 2)
        metrics_layout.addWidget(self.output_sample_rate_label, 0, 3)
        metrics_layout.addWidget(QLabel("🔊"), 1, 2)
        metrics_layout.addWidget(self.output_channels_label, 1, 3)
        metrics_layout.addWidget(QLabel("📂"), 2, 2)
        metrics_layout.addWidget(self.output_format_label, 2, 3)

        quality_layout.addLayout(metrics_layout)
        left_layout.addWidget(quality_group)

        # Export controls
        export_group = QGroupBox("Export Results")
        export_layout = QVBoxLayout(export_group)

        self.output_export_report_btn = QPushButton("Save Analysis Report")
        self.output_export_report_btn.clicked.connect(self.export_output_report)
        self.output_export_report_btn.setEnabled(False)

        self.output_export_plot_btn = QPushButton("Save Quality Plots")
        self.output_export_plot_btn.clicked.connect(self.export_output_plot)
        self.output_export_plot_btn.setEnabled(False)

        export_layout.addWidget(self.output_export_report_btn)
        export_layout.addWidget(self.output_export_plot_btn)

        left_layout.addWidget(export_group)
        left_layout.addStretch()

        # Right panel - Results display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Results tabs
        self.output_results_tabs = QTabWidget()

        # 1. Quality Scores tab
        self.setup_output_quality_tab()

        # 2. Technical Analysis tab
        self.setup_output_technical_tab()

        # 3. Production Standards tab
        self.setup_output_standards_tab()

        # 4. Chapter Analysis tab (if enabled)
        self.setup_output_chapter_tab()

        # 5. Comparison tab
        self.setup_output_comparison_tab()

        right_layout.addWidget(self.output_results_tabs)

        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([380, 920])

        # Initialize UI state
        self.update_output_analyzer_ui_state()

    def setup_output_quality_tab(self):
        """Setup the quality scores tab for output analysis"""
        quality_tab = QWidget()
        layout = QVBoxLayout(quality_tab)

        # Overall quality score
        self.output_quality_score_label = QLabel("No analysis results")
        self.output_quality_score_label.setAlignment(Qt.AlignCenter)
        self.output_quality_score_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.output_quality_score_label.setStyleSheet("padding: 20px; border: 2px solid #ccc; border-radius: 10px; margin: 10px;")
        layout.addWidget(self.output_quality_score_label)

        # Quality metrics grid
        self.output_quality_scroll = QScrollArea()
        self.output_quality_widget = QWidget()
        self.output_quality_layout = QGridLayout(self.output_quality_widget)
        self.output_quality_scroll.setWidget(self.output_quality_widget)
        self.output_quality_scroll.setWidgetResizable(True)
        layout.addWidget(self.output_quality_scroll)

        self.output_results_tabs.addTab(quality_tab, "Quality Scores")

    def setup_output_technical_tab(self):
        """Setup the technical analysis tab"""
        technical_tab = QWidget()
        layout = QVBoxLayout(technical_tab)

        self.output_technical_text = QTextEdit()
        self.output_technical_text.setReadOnly(True)
        self.output_technical_text.setFont(QFont("Consolas", 10))
        self.output_technical_text.setText("No technical analysis available - run analysis first")

        layout.addWidget(self.output_technical_text)

        self.output_results_tabs.addTab(technical_tab, "Technical Analysis")

    def setup_output_standards_tab(self):
        """Setup the production standards compliance tab"""
        standards_tab = QWidget()
        layout = QVBoxLayout(standards_tab)

        self.output_standards_text = QTextEdit()
        self.output_standards_text.setReadOnly(True)
        self.output_standards_text.setFont(QFont("Consolas", 10))
        self.output_standards_text.setText("No standards analysis available - run analysis first")

        layout.addWidget(self.output_standards_text)

        self.output_results_tabs.addTab(standards_tab, "Production Standards")

    def setup_output_chapter_tab(self):
        """Setup the chapter analysis tab"""
        chapter_tab = QWidget()
        layout = QVBoxLayout(chapter_tab)

        self.output_chapter_text = QTextEdit()
        self.output_chapter_text.setReadOnly(True)
        self.output_chapter_text.setFont(QFont("Consolas", 10))
        self.output_chapter_text.setText("No chapter analysis available - enable option and run analysis")

        layout.addWidget(self.output_chapter_text)

        self.output_results_tabs.addTab(chapter_tab, "Chapter Analysis")

    def setup_output_comparison_tab(self):
        """Setup the output comparison tab"""
        comparison_tab = QWidget()
        layout = QVBoxLayout(comparison_tab)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            # Comparison matplotlib figure
            self.output_comparison_figure = Figure(figsize=(12, 8))
            self.output_comparison_canvas = FigureCanvas(self.output_comparison_figure)
            layout.addWidget(self.output_comparison_canvas)

            # Initially show placeholder
            ax = self.output_comparison_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Analyze multiple audiobook files\nto see quality comparison plots',
                    ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title('Audiobook Quality Comparison')
            self.output_comparison_canvas.draw()

        except ImportError:
            placeholder_label = QLabel("📊 Comparison plots require matplotlib\nInstall with: pip install matplotlib")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_label.setStyleSheet("color: #666; font-size: 14px; padding: 50px;")
            layout.addWidget(placeholder_label)
            self.output_comparison_canvas = None

        self.output_results_tabs.addTab(comparison_tab, "File Comparison")

    def add_output_files(self):
        """Add audiobook files for analysis"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audiobook Files", "",
            "Audio Files (*.mp3 *.m4a *.wav *.flac *.ogg *.aac);;All Files (*)"
        )

        for file_path in file_paths:
            if file_path:
                item = QListWidgetItem(Path(file_path).name)
                item.setData(Qt.UserRole, file_path)
                item.setToolTip(file_path)
                self.output_file_list.addItem(item)

        self.update_output_analyzer_ui_state()
        self.log_output(f"📁 Added {len(file_paths)} audiobook files for analysis")

    def remove_output_file(self):
        """Remove selected audiobook file"""
        current_item = self.output_file_list.currentItem()
        current_row = self.output_file_list.currentRow()

        if current_row >= 0 and current_item:
            file_path = current_item.data(Qt.UserRole)
            file_name = Path(file_path).name

            # Remove from file list
            self.output_file_list.takeItem(current_row)

            # Remove corresponding result
            self.output_analyzer_results = [r for r in self.output_analyzer_results if r['filename'] != file_name]

            # Clear current result if it was the removed file
            if self.output_analyzer_current_result and self.output_analyzer_current_result['filename'] == file_name:
                self.output_analyzer_current_result = None
                self.clear_output_displays()

        self.update_output_analyzer_ui_state()

    def clear_output_files(self):
        """Clear all audiobook files"""
        self.output_file_list.clear()
        self.output_analyzer_results.clear()
        self.output_analyzer_current_result = None
        self.clear_output_displays()
        self.update_output_analyzer_ui_state()

    def on_output_file_selected(self, item):
        """Handle audiobook file selection"""
        if item:
            file_path = item.data(Qt.UserRole)
            file_name = Path(file_path).name

            # Find corresponding result
            for result in self.output_analyzer_results:
                if result['filename'] == file_name:
                    self.output_analyzer_current_result = result
                    self.update_output_result_display(result)
                    break
        self.update_output_analyzer_ui_state()

    def analyze_selected_output(self):
        """Analyze selected audiobook file"""
        current_item = self.output_file_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select an audiobook file to analyze.")
            return

        file_path = current_item.data(Qt.UserRole)
        self.start_output_analysis([file_path])

    def analyze_all_outputs(self):
        """Analyze all audiobook files"""
        if self.output_file_list.count() == 0:
            QMessageBox.warning(self, "No Files", "Please add audiobook files to analyze.")
            return

        file_paths = []
        for i in range(self.output_file_list.count()):
            item = self.output_file_list.item(i)
            file_paths.append(item.data(Qt.UserRole))

        self.start_output_analysis(file_paths)

    def start_output_analysis(self, file_paths):
        """Start audiobook output analysis"""
        self.output_progress_bar.setVisible(True)
        self.output_progress_bar.setRange(0, len(file_paths))
        self.output_status_label.setText("Analyzing audiobook files...")

        self.output_analyze_btn.setEnabled(False)
        self.output_analyze_all_btn.setEnabled(False)

        # Process files synchronously for simplicity
        for i, file_path in enumerate(file_paths):
            try:
                self.output_progress_bar.setValue(i)
                file_name = Path(file_path).name
                self.output_status_label.setText(f"Analyzing {file_name}...")

                # Perform comprehensive audiobook analysis
                result = self.analyze_audiobook_file(file_path)

                # Update or add result
                existing_index = -1
                for j, existing_result in enumerate(self.output_analyzer_results):
                    if existing_result['filename'] == file_name:
                        existing_index = j
                        break

                if existing_index >= 0:
                    self.output_analyzer_results[existing_index] = result
                else:
                    self.output_analyzer_results.append(result)

                # Update display if this is the current selection
                current_item = self.output_file_list.currentItem()
                if current_item and Path(current_item.data(Qt.UserRole)).name == result['filename']:
                    self.output_analyzer_current_result = result
                    self.update_output_result_display(result)

                # Calculate quality grade
                overall_score = result['overall_score']
                if overall_score >= 90:
                    grade = "Excellent"
                elif overall_score >= 75:
                    grade = "Good"
                elif overall_score >= 60:
                    grade = "Fair"
                else:
                    grade = "Poor"

                self.log_output(f"✅ {file_name}: Quality Score {overall_score:.1f}/100 ({grade})")

            except Exception as e:
                self.log_output(f"❌ Analysis failed for {Path(file_path).name}: {str(e)}")

        self.output_progress_bar.setValue(len(file_paths))
        self.output_progress_bar.setVisible(False)
        self.output_status_label.setText("Analysis complete")

        self.output_analyze_btn.setEnabled(True)
        self.output_analyze_all_btn.setEnabled(True)
        self.update_output_analyzer_ui_state()

        # Update comparison plot if multiple results
        if len(self.output_analyzer_results) > 1 and hasattr(self, 'output_comparison_canvas'):
            self.update_output_comparison_plot()

    def analyze_audiobook_file(self, file_path):
        """Comprehensive analysis of audiobook file"""
        import os
        import librosa
        import numpy as np
        from pathlib import Path

        file_name = Path(file_path).name
        file_size = os.path.getsize(file_path)

        # Load audio for analysis
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            duration = len(audio_data) / sample_rate
        except Exception as e:
            return {
                'filename': file_name,
                'success': False,
                'error': str(e),
                'overall_score': 0
            }

        # Basic file information
        file_info = {
            'filename': file_name,
            'success': True,
            'file_size': file_size,
            'duration': duration,
            'sample_rate': sample_rate,
            'channels': 1 if len(audio_data.shape) == 1 else audio_data.shape[0],
            'format': Path(file_path).suffix.lower()
        }

        # Audio quality metrics
        rms_level = np.sqrt(np.mean(audio_data**2))
        peak_level = np.max(np.abs(audio_data))

        # Dynamic range analysis
        rms_db = 20 * np.log10(rms_level + 1e-10)
        peak_db = 20 * np.log10(peak_level + 1e-10)

        # Clipping detection
        clipping_samples = np.sum(np.abs(audio_data) >= 0.95)
        clipping_percentage = (clipping_samples / len(audio_data)) * 100

        # Noise floor estimation
        sorted_samples = np.sort(np.abs(audio_data))
        noise_floor = np.mean(sorted_samples[:int(len(sorted_samples) * 0.1)])
        noise_floor_db = 20 * np.log10(noise_floor + 1e-10)

        # Frequency analysis
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(fft), 1/sample_rate)

        # Find dominant frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]

        # Spectral centroid
        spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)

        # Calculate quality scores
        scores = {}

        # Audio fidelity score (sample rate, bit depth estimation)
        if sample_rate >= 44100:
            scores['audio_fidelity'] = 100
        elif sample_rate >= 22050:
            scores['audio_fidelity'] = 80
        else:
            scores['audio_fidelity'] = 60

        # Dynamic range score
        dynamic_range = peak_db - noise_floor_db
        if dynamic_range >= 60:
            scores['dynamic_range'] = 100
        elif dynamic_range >= 40:
            scores['dynamic_range'] = 80
        elif dynamic_range >= 20:
            scores['dynamic_range'] = 60
        else:
            scores['dynamic_range'] = 40

        # Clipping score
        if clipping_percentage < 0.01:
            scores['clipping'] = 100
        elif clipping_percentage < 0.1:
            scores['clipping'] = 80
        elif clipping_percentage < 1.0:
            scores['clipping'] = 60
        else:
            scores['clipping'] = 30

        # Level consistency score (based on RMS variations)
        chunk_size = int(sample_rate * 5)  # 5-second chunks
        rms_chunks = []
        for i in range(0, len(audio_data) - chunk_size, chunk_size):
            chunk_rms = np.sqrt(np.mean(audio_data[i:i+chunk_size]**2))
            rms_chunks.append(chunk_rms)

        if len(rms_chunks) > 1:
            rms_std = np.std(rms_chunks) / np.mean(rms_chunks)
            if rms_std < 0.1:
                scores['consistency'] = 100
            elif rms_std < 0.2:
                scores['consistency'] = 80
            elif rms_std < 0.3:
                scores['consistency'] = 60
            else:
                scores['consistency'] = 40
        else:
            scores['consistency'] = 90  # Short files get benefit of doubt

        # Production standards score
        standards_score = 100

        # Check file format
        if file_info['format'] not in ['.mp3', '.m4a', '.aac']:
            standards_score -= 10

        # Check sample rate
        if sample_rate not in [22050, 44100, 48000]:
            standards_score -= 10

        # Check duration (audiobooks should be substantial)
        if duration < 300:  # Less than 5 minutes
            standards_score -= 20

        scores['production_standards'] = max(standards_score, 0)

        # Overall score (weighted average)
        weights = {
            'audio_fidelity': 0.25,
            'dynamic_range': 0.25,
            'clipping': 0.25,
            'consistency': 0.15,
            'production_standards': 0.10
        }

        overall_score = sum(scores[metric] * weights[metric] for metric in scores)

        # Compile full result
        result = {
            **file_info,
            'overall_score': overall_score,
            'scores': scores,
            'metrics': {
                'rms_level_db': rms_db,
                'peak_level_db': peak_db,
                'dynamic_range_db': dynamic_range,
                'noise_floor_db': noise_floor_db,
                'clipping_percentage': clipping_percentage,
                'spectral_centroid_hz': spectral_centroid,
                'estimated_bitrate': self.estimate_bitrate(file_path, duration)
            }
        }

        return result

    def estimate_bitrate(self, file_path, duration):
        """Estimate bitrate from file size and duration"""
        try:
            file_size = os.path.getsize(file_path)
            bitrate_bps = (file_size * 8) / duration
            return int(bitrate_bps / 1000)  # Convert to kbps
        except:
            return 0

    def update_output_result_display(self, result):
        """Update the display with audiobook analysis result"""
        if not result['success']:
            self.output_overall_quality_label.setText(f"Analysis Failed: {result['error']}")
            return

        # Update overall quality score
        overall_score = result['overall_score']
        if overall_score >= 90:
            color = "green"
            grade = "Excellent"
        elif overall_score >= 75:
            color = "orange"
            grade = "Good"
        elif overall_score >= 60:
            color = "gold"
            grade = "Fair"
        else:
            color = "red"
            grade = "Poor"

        self.output_overall_quality_label.setText(f"{result['filename']}\nQuality Score: {overall_score:.1f}/100\n{grade}")
        self.output_overall_quality_label.setStyleSheet(f"padding: 15px; border: 2px solid {color}; border-radius: 5px; margin: 5px; color: {color};")

        # Update file metrics
        duration_str = f"{int(result['duration'] // 3600):02d}:{int((result['duration'] % 3600) // 60):02d}:{int(result['duration'] % 60):02d}"
        size_str = f"{result['file_size'] / (1024*1024):.1f} MB"
        bitrate_str = f"{result['metrics']['estimated_bitrate']} kbps"

        self.output_duration_label.setText(f"Duration: {duration_str}")
        self.output_size_label.setText(f"Size: {size_str}")
        self.output_bitrate_label.setText(f"Bitrate: {bitrate_str}")
        self.output_sample_rate_label.setText(f"Sample Rate: {result['sample_rate']} Hz")
        self.output_channels_label.setText(f"Channels: {result['channels']}")
        self.output_format_label.setText(f"Format: {result['format'].upper()}")

        # Update quality scores grid
        self.clear_output_quality_grid()

        scores_data = [
            ("Audio Fidelity", result['scores']['audio_fidelity']),
            ("Dynamic Range", result['scores']['dynamic_range']),
            ("Clipping", result['scores']['clipping']),
            ("Consistency", result['scores']['consistency']),
            ("Production Standards", result['scores']['production_standards'])
        ]

        for i, (label, score) in enumerate(scores_data):
            row = i // 3
            col = i % 3
            score_widget = self.create_output_score_widget(label, score)
            self.output_quality_layout.addWidget(score_widget, row, col)

        # Update technical analysis
        technical_text = f"Technical Analysis for: {result['filename']}\n"
        technical_text += "=" * 60 + "\n\n"
        technical_text += f"File Information:\n"
        technical_text += f"  Duration: {duration_str}\n"
        technical_text += f"  File Size: {size_str}\n"
        technical_text += f"  Sample Rate: {result['sample_rate']} Hz\n"
        technical_text += f"  Channels: {result['channels']}\n"
        technical_text += f"  Format: {result['format'].upper()}\n"
        technical_text += f"  Estimated Bitrate: {bitrate_str}\n\n"

        technical_text += f"Audio Metrics:\n"
        technical_text += f"  RMS Level: {result['metrics']['rms_level_db']:.2f} dB\n"
        technical_text += f"  Peak Level: {result['metrics']['peak_level_db']:.2f} dB\n"
        technical_text += f"  Dynamic Range: {result['metrics']['dynamic_range_db']:.2f} dB\n"
        technical_text += f"  Noise Floor: {result['metrics']['noise_floor_db']:.2f} dB\n"
        technical_text += f"  Clipping: {result['metrics']['clipping_percentage']:.3f}%\n"
        technical_text += f"  Spectral Centroid: {result['metrics']['spectral_centroid_hz']:.1f} Hz\n"

        self.output_technical_text.setText(technical_text)

        # Update production standards
        standards_text = f"Production Standards Analysis for: {result['filename']}\n"
        standards_text += "=" * 60 + "\n\n"

        standards_text += "Format Compliance:\n"
        if result['format'] in ['.mp3', '.m4a', '.aac']:
            standards_text += "  ✅ Compatible audiobook format\n"
        else:
            standards_text += f"  ⚠️ Format {result['format']} may not be optimal for audiobooks\n"

        standards_text += "\nSample Rate Compliance:\n"
        if result['sample_rate'] in [22050, 44100, 48000]:
            standards_text += f"  ✅ Standard sample rate: {result['sample_rate']} Hz\n"
        else:
            standards_text += f"  ⚠️ Non-standard sample rate: {result['sample_rate']} Hz\n"

        standards_text += "\nAudio Quality Assessment:\n"
        if result['metrics']['clipping_percentage'] < 0.01:
            standards_text += "  ✅ No significant clipping detected\n"
        else:
            standards_text += f"  ❌ Clipping detected: {result['metrics']['clipping_percentage']:.3f}%\n"

        if result['metrics']['rms_level_db'] > -30:
            standards_text += "  ✅ Adequate audio levels\n"
        else:
            standards_text += "  ⚠️ Audio levels may be too low\n"

        if result['duration'] >= 300:
            standards_text += f"  ✅ Appropriate duration: {duration_str}\n"
        else:
            standards_text += f"  ⚠️ Short duration: {duration_str}\n"

        standards_text += "\nRecommendations:\n"
        recommendations = []

        if result['scores']['clipping'] < 90:
            recommendations.append("• Consider re-mastering to reduce clipping")
        if result['scores']['dynamic_range'] < 80:
            recommendations.append("• Improve dynamic range through better mastering")
        if result['scores']['consistency'] < 80:
            recommendations.append("• Apply level normalization for consistency")
        if result['format'] not in ['.mp3', '.m4a']:
            recommendations.append("• Consider converting to MP3 or M4A format")

        if not recommendations:
            recommendations.append("• Audio meets production standards")

        for rec in recommendations:
            standards_text += rec + "\n"

        self.output_standards_text.setText(standards_text)

    def create_output_score_widget(self, label, score):
        """Create a score display widget for output analysis"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Score label
        score_label = QLabel(f"{score:.0f}")
        score_label.setAlignment(Qt.AlignCenter)
        score_label.setFont(QFont("Arial", 14, QFont.Bold))

        # Color coding
        if score >= 90:
            color = "green"
        elif score >= 75:
            color = "orange"
        elif score >= 60:
            color = "gold"
        else:
            color = "red"

        score_label.setStyleSheet(f"color: {color}; padding: 5px;")

        # Category label
        category_label = QLabel(label)
        category_label.setAlignment(Qt.AlignCenter)
        category_label.setFont(QFont("Arial", 9))
        category_label.setWordWrap(True)

        layout.addWidget(score_label)
        layout.addWidget(category_label)

        # Add border
        widget.setStyleSheet("QWidget { border: 1px solid #ccc; border-radius: 5px; margin: 2px; }")

        return widget

    def clear_output_quality_grid(self):
        """Clear the quality scores grid"""
        while self.output_quality_layout.count():
            child = self.output_quality_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def clear_output_displays(self):
        """Clear all output analysis displays"""
        self.output_overall_quality_label.setText("No analysis results")
        self.output_overall_quality_label.setStyleSheet("padding: 15px; border: 2px solid #ccc; border-radius: 5px; margin: 5px;")

        # Reset metrics
        self.output_duration_label.setText("Duration: --")
        self.output_size_label.setText("Size: --")
        self.output_bitrate_label.setText("Bitrate: --")
        self.output_sample_rate_label.setText("Sample Rate: --")
        self.output_channels_label.setText("Channels: --")
        self.output_format_label.setText("Format: --")

        self.clear_output_quality_grid()
        self.output_technical_text.setText("No technical analysis available - run analysis first")
        self.output_standards_text.setText("No standards analysis available - run analysis first")

    def update_output_comparison_plot(self):
        """Update comparison plot for multiple audiobook files"""
        if not hasattr(self, 'output_comparison_canvas') or not self.output_comparison_canvas:
            return

        if len(self.output_analyzer_results) < 2:
            return

        try:
            self.output_comparison_figure.clear()

            # Create comparison plot
            ax1 = self.output_comparison_figure.add_subplot(2, 1, 1)
            ax2 = self.output_comparison_figure.add_subplot(2, 1, 2)

            # Overall quality scores comparison
            filenames = [r['filename'][:20] + '...' if len(r['filename']) > 20 else r['filename'] for r in self.output_analyzer_results]
            overall_scores = [r['overall_score'] for r in self.output_analyzer_results]

            colors = ['red' if s < 60 else 'gold' if s < 75 else 'orange' if s < 90 else 'green' for s in overall_scores]
            bars1 = ax1.bar(range(len(filenames)), overall_scores, color=colors, alpha=0.7)
            ax1.set_title('Overall Audiobook Quality Scores', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 100)
            ax1.set_xticks(range(len(filenames)))
            ax1.set_xticklabels(filenames, rotation=45, ha='right')
            ax1.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Fair')
            ax1.axhline(y=75, color='gold', linestyle='--', alpha=0.5, label='Good')
            ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.legend(loc='upper right')

            # Add score values
            for bar, score in zip(bars1, overall_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{score:.0f}', ha='center', va='bottom', fontsize=9)

            # Detailed metrics comparison
            import numpy as np
            categories = ['Fidelity', 'Dynamic Range', 'Clipping', 'Consistency']
            x = np.arange(len(categories))
            width = 0.8 / len(self.output_analyzer_results) if len(self.output_analyzer_results) > 0 else 0.8

            for i, result in enumerate(self.output_analyzer_results):
                scores = [result['scores']['audio_fidelity'], result['scores']['dynamic_range'],
                         result['scores']['clipping'], result['scores']['consistency']]
                ax2.bar(x + i * width, scores, width, label=filenames[i], alpha=0.7)

            ax2.set_title('Detailed Quality Metrics Comparison', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 100)
            ax2.set_xticks(x + width * (len(self.output_analyzer_results) - 1) / 2)
            ax2.set_xticklabels(categories)
            if len(self.output_analyzer_results) <= 4:
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3, axis='y')

            self.output_comparison_figure.tight_layout()
            self.output_comparison_canvas.draw()

        except Exception as e:
            # Fallback to error message
            self.output_comparison_figure.clear()
            ax = self.output_comparison_figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Comparison error:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Comparison Error')
            self.output_comparison_canvas.draw()

    def update_output_analyzer_ui_state(self):
        """Update UI element states for output analyzer"""
        has_files = self.output_file_list.count() > 0
        has_selection = self.output_file_list.currentItem() is not None
        has_results = len(self.output_analyzer_results) > 0

        self.output_analyze_btn.setEnabled(has_files and has_selection)
        self.output_analyze_all_btn.setEnabled(has_files)
        self.output_remove_file_btn.setEnabled(has_selection)
        self.output_clear_files_btn.setEnabled(has_files)
        self.output_export_report_btn.setEnabled(has_results and self.output_analyzer_current_result is not None)
        self.output_export_plot_btn.setEnabled(has_results and self.output_analyzer_current_result is not None)

    def export_output_report(self):
        """Export audiobook analysis report"""
        if not self.output_analyzer_current_result:
            QMessageBox.warning(self, "No Results", "Please analyze an audiobook file first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Audiobook Analysis Report",
            f"{self.output_analyzer_current_result['filename']}_audiobook_report.txt",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                result = self.output_analyzer_current_result

                report = f"Audiobook Quality Analysis Report\n"
                report += "=" * 50 + "\n\n"
                report += f"File: {result['filename']}\n"
                report += f"Overall Quality Score: {result['overall_score']:.1f}/100\n\n"

                # Add all the detailed information
                duration_str = f"{int(result['duration'] // 3600):02d}:{int((result['duration'] % 3600) // 60):02d}:{int(result['duration'] % 60):02d}"
                report += f"File Information:\n"
                report += f"  Duration: {duration_str}\n"
                report += f"  File Size: {result['file_size'] / (1024*1024):.1f} MB\n"
                report += f"  Sample Rate: {result['sample_rate']} Hz\n"
                report += f"  Channels: {result['channels']}\n"
                report += f"  Format: {result['format'].upper()}\n"
                report += f"  Estimated Bitrate: {result['metrics']['estimated_bitrate']} kbps\n\n"

                report += f"Quality Scores:\n"
                for metric, score in result['scores'].items():
                    report += f"  {metric.replace('_', ' ').title()}: {score:.1f}/100\n"

                report += f"\nTechnical Metrics:\n"
                for metric, value in result['metrics'].items():
                    if isinstance(value, float):
                        report += f"  {metric.replace('_', ' ').title()}: {value:.3f}\n"
                    else:
                        report += f"  {metric.replace('_', ' ').title()}: {value}\n"

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)

                QMessageBox.information(self, "Export Success", f"Report saved to:\n{file_path}")
                self.log_output(f"📝 Audiobook analysis report exported: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save report:\n{str(e)}")

    def export_output_plot(self):
        """Export audiobook quality plot"""
        if not self.output_analyzer_current_result:
            QMessageBox.warning(self, "No Results", "Please analyze an audiobook file first.")
            return

        if not hasattr(self, 'output_comparison_canvas') or not self.output_comparison_canvas:
            QMessageBox.warning(self, "No Plot", "Matplotlib not available for plot export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Audiobook Quality Plot",
            f"{self.output_analyzer_current_result['filename']}_quality_plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )

        if file_path:
            try:
                self.output_comparison_figure.savefig(file_path, dpi=150, bbox_inches='tight')
                QMessageBox.information(self, "Export Success", f"Plot saved to:\n{file_path}")
                self.log_output(f"📊 Audiobook quality plot exported: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save plot:\n{str(e)}")


def main():
    """Runs the application's main loop.
    Args:
    - None
    Returns:
    - None
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    window = ChatterboxMainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GUI Walker: enumerates and clicks GUI actions under SPIDER_DRY_RUN=1.

Best-effort: launches chatterbox_gui, finds the main window, iterates tabs and buttons,
clicks each once with timeouts. Requires PyQt5 and supports offscreen mode.

Outputs a simple log at reports/spider/gui_walk.log
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('SPIDER_DRY_RUN', '1')

log_lines = []

def log(msg: str):
    """Main entry point for the script. Starts a PyQt5 application and initializes logging.
    Args: None
    Returns: None
    """
    print(msg)
    log_lines.append(msg)


def save_log():
    """Saves a log file named 'gui_walk.log' in the reports/spider directory of the repository.
    Args:
    None
    Returns:
    None
    """
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / 'reports' / 'spider'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'gui_walk.log').write_text('\n'.join(log_lines), encoding='utf-8')


def main():
    """Initializes the PyQt5 library and attempts to load necessary modules dynamically. Args: None Returns: None"""
    try:
        from PyQt5 import QtWidgets, QtCore
        from PyQt5.QtTest import QTest
    except Exception as e:
        log(f"PyQt5 not available: {e}")
        save_log()
        return

    # Try to import the GUI module (with lightweight stubs to avoid heavy deps during discovery)
    try:
        import importlib, types
        from importlib.machinery import ModuleSpec

        # Dynamic stub finder for librosa/* and soundfile to avoid heavy deps during import
        class _PkgStubLoader:
            """A private class for loading package stubs, handling module creation and execution, particularly for no-op functions in specific packages like librosa."""
            def create_module(self, spec):
                """Creates a module from a specification.
                Args:
                spec (ModuleSpec): The module specification.
                Returns:
                ModuleType: The created module.
                """
                mod = types.ModuleType(spec.name)
                mod.__spec__ = spec
                if getattr(spec, 'submodule_search_locations', None) is not None:
                    # Mark as package
                    mod.__path__ = []  # type: ignore[attr-defined]
                return mod
            def exec_module(self, module):
                """Executes a module and replaces certain functions with no-ops based on module name.
                Args:
                module: The module to execute.
                Returns:
                None
                """
                name = module.__name__
                # Add a couple of common no-ops
                def _noop(*a, **k):
                    """No-op for specific module attributes.
                    Args:
                    *a: Variable length argument list.
                    **k: Arbitrary keyword arguments.
                    Returns:
                    None.
                    """
                    return None
                if name.startswith('librosa'):
                    setattr(module, 'load', _noop)
                    setattr(module, 'resample', _noop)
                if name == 'librosa.filters':
                    setattr(module, 'mel', _noop)
                if name == 'soundfile':
                    setattr(module, 'write', _noop)
                if name == 'vaderSentiment.vaderSentiment' or name == 'vaderSentiment':
                    class _SIA:
                        """Provides a mock implementation of sentiment analysis and package stub finder for specific libraries."""
                        def polarity_scores(self, text):
                            """```python
                            Creates a mock sentiment analyzer returning neutral scores for polarity analysis.
                            Args:
                            text (str): The input text to analyze.
                            Returns:
                            dict: A dictionary containing negative, neutral, positive, and compound scores all set to 0.0.
                            ```
                            """
                            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
                    setattr(module, 'SentimentIntensityAnalyzer', _SIA)

        class _PkgStubFinder:
            """Class for dynamically finding and loading stub packages, specifically targeting 'librosa', 'soundfile', and 'vaderSentiment'. Integrates with sys.meta_path to intercept package imports."""
            def find_spec(self, fullname, path, target=None):
                """Finds a module specification for certain specified modules. Args: fullname (str): The full name of the module. path (list of str): The search path for modules. target (any): Optional target object. Returns: ModuleSpec or None: A module spec if found, otherwise None."""
                if fullname.startswith('librosa') or fullname == 'soundfile' or fullname.startswith('vaderSentiment'):
                    is_pkg = True if fullname in ('librosa', 'vaderSentiment') else False
                    spec = ModuleSpec(fullname, _PkgStubLoader(), is_package=is_pkg)
                    if is_pkg:
                        spec.submodule_search_locations = []  # type: ignore[attr-defined]
                    return spec
                return None

        sys.meta_path.insert(0, _PkgStubFinder())
        gui_mod = importlib.import_module('chatterbox_gui')
    except Exception as e:
        log(f"Failed to import chatterbox_gui: {e}")
        save_log()
        return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Find a QMainWindow subclass to instantiate
    main_cls = None
    for name in dir(gui_mod):
        obj = getattr(gui_mod, name)
        try:
            from PyQt5.QtWidgets import QMainWindow
            if isinstance(obj, type) and issubclass(obj, QMainWindow):
                main_cls = obj
                break
        except Exception:
            pass

    if main_cls is None:
        log("No QMainWindow subclass found; cannot walk GUI.")
        save_log()
        return

    win = main_cls()
    win.show()
    app.processEvents()
    QTest.qWait(200)

    # Enumerate tabs and buttons
    from PyQt5.QtWidgets import QTabWidget, QPushButton

    tab_widgets = win.findChildren(QTabWidget)
    log(f"Found {len(tab_widgets)} QTabWidget(s)")
    for tw in tab_widgets:
        count = tw.count()
        for idx in range(count):
            tw.setCurrentIndex(idx)
            app.processEvents()
            QTest.qWait(100)
            log(f"Selected tab '{tw.tabText(idx)}'")
            # Click buttons in this tab
            buttons = tw.currentWidget().findChildren(QPushButton)
            log(f"  Found {len(buttons)} buttons in tab")
            for b in buttons:
                try:
                    log(f"  Clicking '{b.text()}'")
                    QTest.mouseClick(b, QtCore.Qt.LeftButton)
                    app.processEvents()
                    QTest.qWait(100)
                except Exception as e:
                    log(f"  Failed click on '{b.text()}': {e}")

    # Done
    save_log()


if __name__ == '__main__':
    main()

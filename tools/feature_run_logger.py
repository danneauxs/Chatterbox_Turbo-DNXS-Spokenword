#!/usr/bin/env python3
"""
Feature Run Logger: instrument GUI action slots to log actual UI values at press time.

Usage:
  PYTHONPATH=. SPIDER_TRACE=1 python tools/feature_run_logger.py

Outputs:
  reports/spider/feature_runs.ndjson   (one line per slot invocation)

Notes:
  - Non-invasive: wraps slot methods on the window instance at runtime.
  - Uses tools/gui_static_map.py output to know which slots and inputs to log.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import types
import sys
import threading


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'reports' / 'spider'
FMAP_PATH = OUT_DIR / 'feature_map.json'


def parse_input_expr(expr: str):
    """Parse 'self.widget.sub.value()' into (attr_chain:list[str], getter:str)."""
    if not expr.startswith('self.') or not expr.endswith('()'):
        return None, None
    body = expr[len('self.'): -2]  # strip self. and ()
    parts = body.split('.')
    if len(parts) < 2:
        return None, None
    getter = parts[-1]
    chain = parts[:-1]
    return chain, getter


def get_input_value(inst, chain, getter):
    obj = inst
    for name in chain:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    meth = getattr(obj, getter, None)
    if meth is None:
        return None
    try:
        return meth()
    except Exception:
        return None


def _parse_self_qualname(qname: str):
    # 'self.foo.bar' -> ['foo','bar']
    if not qname.startswith('self.'):
        return []
    return qname[len('self.'):].split('.')


def _resolve_attr_chain(inst, chain):
    obj = inst
    for name in chain:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def instrument_window(window_cls, feature_map):
    # Build mapping: slot_name -> list[input_expr]
    slot_inputs = {}
    slot_buttons = {}
    for btn in feature_map.get('buttons', []):
        slot = btn.get('slot', '')
        if not slot:
            continue
        slot_name = slot.split('.')[-1]
        inputs = btn.get('ui_inputs', [])
        slot_inputs.setdefault(slot_name, set()).update(inputs)
        slot_buttons.setdefault(slot_name, []).append({
            'button': btn.get('button'),
            'button_text': btn.get('button_text')
        })

    original_init = window_cls.__init__

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # After UI is built, wrap slots
        wrapped_count = 0
        # Connect lightweight listeners on known buttons to capture which one was pressed
        try:
            from PyQt5 import QtCore
            for btns in slot_buttons.values():
                for b in btns:
                    bname = b.get('button') or ''
                    chain = _parse_self_qualname(bname)
                    if not chain:
                        continue
                    obj = _resolve_attr_chain(self, chain)
                    if obj is None or not hasattr(obj, 'clicked'):
                        continue
                    # Connect once to record last pressed button
                    try:
                        obj.clicked.connect(lambda checked=False, desc=b: setattr(self, '_feature_pressed_button', desc))
                    except Exception:
                        pass
        except Exception:
            pass
        for slot_name, input_exprs in slot_inputs.items():
            if not hasattr(self, slot_name):
                continue
            orig = getattr(self, slot_name)
            if not callable(orig):
                continue
            # Avoid double-wrapping
            if getattr(orig, '__feature_wrapped__', False):
                continue

            def make_wrapper(_orig, _slot_name, _exprs):
                def _wrapper(*a, **k):
                    t0 = time.time()
                    # Collect inputs
                    values = {}
                    for expr in _exprs:
                        chain, getter = parse_input_expr(expr)
                        if chain and getter:
                            val = get_input_value(self, chain, getter)
                            values[expr] = val
                    # Capture context: pressed button and tab(s)
                    btn = getattr(self, '_feature_pressed_button', None)
                    # Tab info
                    tabs = []
                    try:
                        from PyQt5.QtWidgets import QTabWidget
                        for tw in self.findChildren(QTabWidget):
                            try:
                                idx = tw.currentIndex()
                                tabs.append({'object': getattr(tw, 'objectName', lambda: '')(), 'index': idx, 'text': tw.tabText(idx)})
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Per-slot lightweight call tracer (only active during this slot)
                    repo_root = REPO.resolve()
                    called_files = set()
                    def _should_keep(path_str: str) -> bool:
                        try:
                            p = Path(path_str).resolve()
                            return repo_root in p.parents or p == repo_root
                        except Exception:
                            return False
                    def _prof(frame, event, arg):
                        if event != 'call':
                            return
                        fn = frame.f_code.co_filename
                        if fn and _should_keep(fn):
                            called_files.add(str(Path(fn).resolve().relative_to(repo_root)))
                    # Execute original and log outcome
                    old_prof = sys.getprofile()
                    old_thr_prof = threading.getprofile()
                    sys.setprofile(_prof)
                    threading.setprofile(_prof)
                    OUT_DIR.mkdir(parents=True, exist_ok=True)
                    path = OUT_DIR / 'feature_runs.ndjson'
                    rec = {
                        't_start': t0,
                        'slot': _slot_name,
                        'button': btn,
                        'tabs': tabs,
                        'values': values
                    }
                    err = None
                    try:
                        result = _orig(*a, **k)
                        return result
                    except Exception as e:
                        err = repr(e)
                        raise
                    finally:
                        # Disable tracer
                        sys.setprofile(old_prof)
                        threading.setprofile(old_thr_prof)
                        rec['t_end'] = time.time()
                        rec['duration_s'] = rec['t_end'] - rec['t_start']
                        if err:
                            rec['error'] = err
                        try:
                            with open(path, 'a', encoding='utf-8') as fh:
                                fh.write(json.dumps(rec) + '\n')
                                fh.flush()
                            # Persist per-slot called files
                            calls_dir = OUT_DIR / 'feature_slot_calls'
                            calls_dir.mkdir(parents=True, exist_ok=True)
                            files_path = calls_dir / f"{_slot_name}.files.json"
                            # Merge with any existing set for this slot
                            try:
                                import json as _json
                                if files_path.exists():
                                    prev = set(_json.loads(files_path.read_text(encoding='utf-8')))
                                else:
                                    prev = set()
                            except Exception:
                                prev = set()
                            merged = sorted(prev.union(called_files))
                            files_path.write_text(json.dumps(merged, indent=2), encoding='utf-8')
                            print(f"[FeatureRun] {_slot_name} logged {len(values)} inputs; traced {len(called_files)} files → {path}")
                        except Exception:
                            pass
                _wrapper.__feature_wrapped__ = True  # type: ignore[attr-defined]
                _wrapper.__name__ = getattr(_orig, '__name__', _slot_name)
                return _wrapper

            setattr(self, slot_name, types.MethodType(make_wrapper(orig, slot_name, list(input_exprs)), self))
            wrapped_count += 1

        # Emit an initialization status file for validation
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / 'feature_run_status.json').write_text(
                json.dumps({
                    'wrapped_slots': wrapped_count,
                    'slots_available': sorted(list(slot_inputs.keys())),
                    'log_path': str((OUT_DIR / 'feature_runs.ndjson').resolve()),
                }, indent=2), encoding='utf-8')
            print(f"[FeatureRun] Initialized: wrapped {wrapped_count} slots; logging to {(OUT_DIR/'feature_runs.ndjson').resolve()}")
        except Exception:
            pass

    window_cls.__init__ = wrapped_init


def main():
    if not FMAP_PATH.exists():
        raise SystemExit(f"Feature map not found: {FMAP_PATH}. Run tools/gui_static_map.py first.")
    fmap = json.loads(FMAP_PATH.read_text(encoding='utf-8'))

    # Import GUI and instrument class
    import importlib
    gui = importlib.import_module('chatterbox_gui')

    # Find QMainWindow subclass
    from PyQt5.QtWidgets import QMainWindow
    target_cls = None
    for name in dir(gui):
        obj = getattr(gui, name)
        if isinstance(obj, type) and issubclass(obj, QMainWindow):
            target_cls = obj
            break
    if target_cls is None:
        raise SystemExit('No QMainWindow subclass found in chatterbox_gui')

    instrument_window(target_cls, fmap)

    # Additionally, wrap run_book_conversion to capture final params and per-call files
    if hasattr(target_cls, 'run_book_conversion'):
        orig_rbc = target_cls.run_book_conversion
        def _wrap_rbc(self, book_path, text_file_path, voice_path, tts_params, quality_params, config_params):
            # Per-call tracer for this long-running operation
            import sys as _sys, threading as _thr, time as _time, json as _json
            repo_root = REPO.resolve()
            files = set()
            def _keep(pstr: str) -> bool:
                try:
                    p = Path(pstr).resolve()
                    return repo_root in p.parents or p == repo_root
                except Exception:
                    return False
            def _prof(frame, event, arg):
                if event != 'call':
                    return
                fn = frame.f_code.co_filename
                if fn and _keep(fn):
                    try:
                        files.add(str(Path(fn).resolve().relative_to(repo_root)))
                    except Exception:
                        pass
            oldp = _sys.getprofile(); oldtp = _thr.getprofile()
            _sys.setprofile(_prof); _thr.setprofile(_prof)
            t0 = _time.time(); err = None
            try:
                return orig_rbc(self, book_path, text_file_path, voice_path, tts_params, quality_params, config_params)
            except Exception as e:
                err = repr(e)
                raise
            finally:
                _sys.setprofile(oldp); _thr.setprofile(oldtp)
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                evt = {
                    't_start': t0,
                    't_end': _time.time(),
                    'duration_s': _time.time() - t0,
                    'slot': 'run_book_conversion',
                    'book_path': str(book_path),
                    'text_file_path': str(text_file_path),
                    'voice_path': str(voice_path),
                    'tts_params': tts_params,
                    'quality_params': quality_params,
                    'config_params': config_params,
                }
                with open(OUT_DIR / 'feature_runs.ndjson', 'a', encoding='utf-8') as fh:
                    fh.write(_json.dumps(evt) + '\n')
                calls_dir = OUT_DIR / 'feature_slot_calls'; calls_dir.mkdir(parents=True, exist_ok=True)
                (calls_dir / 'run_book_conversion.files.json').write_text(_json.dumps(sorted(files), indent=2), encoding='utf-8')
                print(f"[FeatureRun] run_book_conversion captured params; traced {len(files)} files")
        target_cls.run_book_conversion = _wrap_rbc

    # Launch app via module's main()
    if hasattr(gui, 'main') and callable(gui.main):
        gui.main()
    else:
        raise SystemExit('chatterbox_gui.main() not found')


if __name__ == '__main__':
    main()

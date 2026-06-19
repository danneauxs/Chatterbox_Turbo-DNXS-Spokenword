#!/usr/bin/env python3
"""
GUI Static Map: Analyze chatterbox_gui.py to map buttons → slots → inputs → external calls.

Outputs reports/spider/feature_map.json containing, for each QPushButton:
  - button_name, button_text (if statically available)
  - slot (method) it connects to
  - slot location (line number)
  - ui_inputs read inside the slot (self.<widget>.<getter>())
  - file_dialog_targets (variables assigned from QFileDialog calls)
  - external_calls (function/attr calls referenced in the slot)

Static only; does not execute GUI code.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

REPO = Path(__file__).resolve().parents[1]
GUI_FILE = REPO / 'chatterbox_gui.py'
OUT_DIR = REPO / 'reports' / 'spider'


UI_GETTERS = {
    'text', 'toPlainText', 'value', 'isChecked', 'currentText', 'currentIndex',
    'toPlainText', 'date', 'time', 'isEnabled', 'isVisible'
}

FILE_DIALOG_FUNCS = {
    'QFileDialog.getOpenFileName',
    'QFileDialog.getOpenFileNames',
    'QFileDialog.getSaveFileName',
    'QFileDialog.getExistingDirectory',
}


def qualname_from_attr(node: ast.AST) -> str:
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    elif isinstance(cur, ast.Call):
        # e.g., module.function() as callee; take function name only
        if isinstance(cur.func, ast.Attribute):
            parts.append(cur.func.attr)
        elif isinstance(cur.func, ast.Name):
            parts.append(cur.func.id)
    return '.'.join(reversed(parts))


class GUISpy(ast.NodeVisitor):
    def __init__(self):
        self.buttons: Dict[str, Dict] = {}  # var -> {text}
        self.connections: List[Tuple[str, str, int]] = []  # (button_var, slot_name, lineno)
        self.method_defs: Dict[str, ast.FunctionDef] = {}

    def visit_Assign(self, node: ast.Assign):
        # Look for: self.foo = QtWidgets.QPushButton("Text") or QPushButton("Text")
        try:
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr == 'QPushButton':
                    text = None
                    if node.value.args and isinstance(node.value.args[0], ast.Constant) and isinstance(node.value.args[0].value, str):
                        text = node.value.args[0].value
                    for t in node.targets:
                        if isinstance(t, ast.Attribute):  # self.foo
                            var = f"{qualname_from_attr(t)}"
                        elif isinstance(t, ast.Name):
                            var = t.id
                        else:
                            continue
                        self.buttons[var] = {'text': text}
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Look for .clicked.connect(self.some_slot)
        try:
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'connect':
                owner = node.func.value  # e.g., self.button.clicked
                if isinstance(owner, ast.Attribute) and owner.attr == 'clicked':
                    # extract button var name (self.button)
                    btn_expr = owner.value
                    btn_name = qualname_from_attr(btn_expr)
                    # slot name from arg like self.some_slot
                    if node.args:
                        arg0 = node.args[0]
                        slot_name = qualname_from_attr(arg0)
                        self.connections.append((btn_name, slot_name, node.lineno))
        except Exception:
            pass
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Collect methods (for slots)
        self.method_defs[node.name] = node
        self.generic_visit(node)


class SlotAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.ui_inputs: Set[str] = set()
        self.file_targets: Set[str] = set()
        self.external_calls: Set[str] = set()

    def visit_Call(self, node: ast.Call):
        # UI getter: self.<widget>.<getter>()
        try:
            if isinstance(node.func, ast.Attribute):
                getter = node.func.attr
                if getter in UI_GETTERS:
                    qual = qualname_from_attr(node.func.value)
                    self.ui_inputs.add(f"{qual}.{getter}()")
                # File dialog
                qn = qualname_from_attr(node.func)
                if qn in FILE_DIALOG_FUNCS:
                    # record assignment target name if parent is Assign
                    parent = getattr(node, 'parent', None)
                    if isinstance(parent, ast.Assign) and parent.targets:
                        target = parent.targets[0]
                        if isinstance(target, ast.Name):
                            self.file_targets.add(target.id)
                        elif isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    self.file_targets.add(elt.id)
                # External-ish calls: modules.* or bare names commonly used
                base = qualname_from_attr(node.func)
                if base and not base.startswith('self.'):
                    self.external_calls.add(base)
        except Exception:
            pass
        self.generic_visit(node)

    def generic_visit(self, node):
        # Wire parents for Assign capture above
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
            self.visit(child)


def build_feature_map(gui_path: Path) -> Dict:
    src = gui_path.read_text(encoding='utf-8')
    tree = ast.parse(src, filename=str(gui_path))
    spy = GUISpy()
    spy.visit(tree)

    features = []
    for btn_name, slot_name, lineno in spy.connections:
        slot_short = slot_name.split('.')[-1]
        slot_def = spy.method_defs.get(slot_short)
        inputs: List[str] = []
        files: List[str] = []
        calls: List[str] = []
        if slot_def is not None:
            sa = SlotAnalyzer()
            sa.visit(slot_def)
            inputs = sorted(sa.ui_inputs)
            files = sorted(sa.file_targets)
            calls = sorted(sa.external_calls)
        btn_meta = spy.buttons.get(btn_name, {})
        features.append({
            'button': btn_name,
            'button_text': btn_meta.get('text'),
            'slot': slot_name,
            'slot_lineno': slot_def.lineno if slot_def is not None else lineno,
            'ui_inputs': inputs,
            'file_dialog_targets': files,
            'external_calls': calls,
        })
    return {
        'source': str(gui_path),
        'buttons': features,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not GUI_FILE.exists():
        print(f"GUI file not found: {GUI_FILE}")
        return
    fmap = build_feature_map(GUI_FILE)
    out = OUT_DIR / 'feature_map.json'
    out.write_text(json.dumps(fmap, indent=2), encoding='utf-8')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()


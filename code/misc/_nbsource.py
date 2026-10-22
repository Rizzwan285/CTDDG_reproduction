#!/usr/bin/env python3
"""Load definitions out of a notebook source file, by name.

Every evidence script needs the same thing: run the REAL function from the
shipped notebook rather than a copy. A copy can pass its own test while the
shipped code differs, which is the whole failure mode 99_consistency_audit.py
exists to catch — so the evidence scripts must not reintroduce it.

Paths resolve relative to THIS file, so the scripts work from any directory.
"""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def source_path(name):
    """A notebook source file in this project, as an absolute path."""
    p = os.path.join(HERE, name)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"{name} not found next to {os.path.basename(__file__)} (looked in "
            f"{HERE}). Evidence scripts read the shipped notebook sources, so they "
            f"must sit in the same directory as them.")
    return p


def load_defs(name, wanted, namespace=None):
    """Exec the named top-level definitions from `name` into `namespace`.

    Returns the namespace. Raises if any requested name is absent, so a renamed
    or deleted definition fails loudly instead of leaving a stale copy in play.
    """
    path = source_path(name)
    src = open(path).read()
    lines = src.split("\n")
    ns = {} if namespace is None else namespace

    spans = {}
    for node in ast.parse(src).body:
        names = ([node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 else [t.id for t in node.targets if isinstance(t, ast.Name)]
                 if isinstance(node, ast.Assign) else [])
        for nm in names:
            spans.setdefault(nm, (node.lineno, node.end_lineno))

    missing = [w for w in wanted if w not in spans]
    if missing:
        raise KeyError(f"{name} has no top-level definition of {missing}. "
                       f"The evidence script and the notebook have drifted.")

    for w in wanted:
        a, b = spans[w]
        exec(compile("\n".join(lines[a - 1:b]), path, "exec"), ns)
    return ns

#!/usr/bin/env python3
"""Percent-format .py -> .ipynb, with no jupytext dependency.

    python3 _to_ipynb.py src.py [src2.py ...]
"""
import json
import os
import sys


def parse(path):
    cells, kind, buf = [], "code", []

    def flush():
        if not buf:
            return
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            return
        if kind == "markdown":
            src = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#")
                            for l in text.split("\n"))
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": src.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [],
                          "source": text.splitlines(keepends=True)})

    with open(path) as f:
        for line in f.read().split("\n"):
            if line.startswith("# %%"):
                flush()
                kind = "markdown" if "[markdown]" in line else "code"
                buf = []
                continue
            buf.append(line)
    flush()
    return cells


def verify(path, cells):
    """Every substantive line of the source must survive into some cell.

    A missing `# %%` between a markdown block and the code after it makes the
    markdown cell absorb that code: the notebook still looks fine and whole
    functions are simply gone. This refuses to write such a notebook.
    """
    src_lines = [l for l in open(path).read().split("\n")
                 if l.strip() and not l.startswith("# %%")]
    code_lines, md_lines = [], []
    for c in cells:
        body = "".join(c["source"]).split("\n")
        (md_lines if c["cell_type"] == "markdown" else code_lines).extend(
            l for l in body if l.strip())

    got_code = set(code_lines)
    got_md = {("# " + l) for l in md_lines} | {("#" + l) for l in md_lines} \
        | {l for l in md_lines} | {"#"}
    lost = [l for l in src_lines if l not in got_code and l not in got_md]
    if lost:
        raise SystemExit(
            f"{path}: {len(lost)} source line(s) reached no cell. The usual cause "
            f"is a markdown block with no `# %%` after it, which absorbs the code "
            f"below.\n  first lost line: {lost[0][:70]!r}")
    n_code = sum(1 for l in src_lines if not l.lstrip().startswith("#"))
    if len(got_code) and n_code:
        missing = [l for l in src_lines
                   if not l.lstrip().startswith("#") and l not in got_code]
        if missing:
            raise SystemExit(f"{path}: {len(missing)} CODE line(s) landed in a "
                             f"markdown cell, e.g. {missing[0][:70]!r}")


def convert(path):
    cells = parse(path)
    verify(path, cells)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.splitext(path)[0] + ".ipynb"
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"{out}  {len(nb['cells'])} cells ({n_md} markdown)")
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)

#!/usr/bin/env python3
"""Insert a `# %%` wherever a markdown block runs straight into code.

Without the marker the markdown cell absorbs that code and the converted
notebook loses it silently. Only marker lines are added.
"""
import ast, sys, pathlib


def repair(path):
    lines = pathlib.Path(path).read_text().split("\n")
    inserts, i = [], 0
    while i < len(lines):
        if lines[i].startswith("# %% [markdown]"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("# %%"):
                j += 1
            # First substantive non-comment line inside the block is code.
            for k in range(i + 1, j):
                if lines[k].strip() and not lines[k].lstrip().startswith("#"):
                    inserts.append(k)
                    break
            i = j
        else:
            i += 1

    if not inserts:
        return 0

    out, pos = [], set(inserts)
    for idx, ln in enumerate(lines):
        if idx in pos:
            while out and not out[-1].strip():
                out.pop()
            out.append("")
            out.append("# %%")
        out.append(ln)
    text = "\n".join(out)
    ast.parse(text)
    before = [l for l in "\n".join(lines).split("\n")
              if l.strip() and not l.startswith("# %%")]
    after = [l for l in text.split("\n") if l.strip() and not l.startswith("# %%")]
    assert before == after, f"{path}: repair altered content"
    pathlib.Path(path).write_text(text)
    return len(inserts)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p:34s} {repair(p):3d} missing marker(s) restored")

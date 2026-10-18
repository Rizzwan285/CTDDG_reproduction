#!/usr/bin/env python3
"""Normalise percent-format cell structure. Deterministic and idempotent.

Only `# %%` marker lines move; no code line is added, removed or reordered, and
`# %% [markdown]` cells are preserved verbatim. Boundaries go:
  * before every top-level `def` / `class` (with its decorators and lead comment),
  * between runs of simple statements, chunked to at most TARGET lines, broken at
    blank-line boundaries so a logical group stays together.
"""
import ast, sys, pathlib

TARGET = 34          # soft maximum for a statement chunk
HARD = 46            # a single unit longer than this is atomic; left alone


def recell(path, target=TARGET):
    src = pathlib.Path(path).read_text()
    lines = src.split("\n")

    # Markdown cells are copied through untouched.
    md = set()
    i = 0
    while i < len(lines):
        if lines[i].startswith("# %% [markdown]"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("# %%"):
                j += 1
            md.update(range(i, j))
            i = j
        else:
            i += 1

    tree = ast.parse(src)
    defs, simple = [], []
    for node in tree.body:
        start = min([node.lineno] + [d.lineno for d in
                                     getattr(node, "decorator_list", [])])
        (defs if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)) else simple).append(
            (start, node.end_lineno))

    def lead(start, floor):
        """Pull a definition's lead comment block up with it.

        Stops at a `# %%` marker: walking past one puts the break on the marker
        line, which the emit loop drops, and the definition then merges into the
        cell above.
        """
        k = start - 1
        while k - 1 >= floor and lines[k - 1].strip().startswith("#") \
                and not lines[k - 1].startswith("# %%") and (k - 1) not in md:
            k -= 1
        return k

    def place(i):
        """Shift a break out of any markdown block it landed in.

        A break emitted on a markdown line is swallowed by the markdown branch
        below, and the original marker on that line was already dropped - so the
        markdown cell absorbs the code that follows. This is the guard against
        that; `assert_markers_intact` re-checks the result.
        """
        while i - 1 < len(lines) and (i - 1) in md:
            i += 1
        return i

    breaks = set()
    for s, _e in defs:
        breaks.add(place(lead(s, 1)))

    # Chunk consecutive simple statements.
    run = []
    for s, e in sorted(simple):
        if run and s <= run[-1][1] + 3 and not any(
                d_s <= s <= d_e for d_s, d_e in defs):
            run.append((s, e))
        else:
            if run:
                breaks.update(place(b) for b in _chunk(run, target))
            run = [(s, e)]
    if run:
        breaks.update(place(b) for b in _chunk(run, target))

    out, prev_blank = [], True
    for i, ln in enumerate(lines, 1):
        idx = i - 1
        if idx in md:
            out.append(ln)
            prev_blank = not ln.strip()
            continue
        # The break is emitted BEFORE an old marker on the same line is dropped;
        # the other order silently swallows the break.
        if i in breaks:
            while out and not out[-1].strip():
                out.pop()
            if out:
                out.append("")
            out.append("# %%")
        if ln.startswith("# %%"):
            continue                                   # drop old code markers
        out.append(ln)
        prev_blank = not ln.strip()

    text = "\n".join(out)
    ast.parse(text)
    assert_markers_intact(path, text)

    # No substantive line may have changed: only cell markers and the blank lines
    # around them are allowed to move.
    def code(t):
        return [l for l in t.split("\n")
                if l.strip() and not l.startswith("# %%")]
    before, after = code(src), code(text)
    if before != after:
        import difflib
        d = list(difflib.unified_diff(before, after, "before", "after",
                                      lineterm="", n=1))[:20]
        raise SystemExit(f"{path}: recell altered code lines\n" + "\n".join(d))

    pathlib.Path(path).write_text(text)
    return text


def assert_markers_intact(path, text):
    """Every markdown block must be closed by a `# %%` marker or end the file.

    Without this, a markdown cell silently absorbs the code after it and the
    notebook loses whole functions while still parsing as valid Python.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("# %% [markdown]"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("# %%"):
                j += 1
            tail = [l for l in lines[i + 1:j] if l.strip()]
            bad = [l for l in tail if not l.lstrip().startswith("#")]
            if bad:
                raise SystemExit(
                    f"{path}: markdown cell at line {i + 1} absorbs code - first "
                    f"offending line: {bad[0][:70]!r}")
            i = j
        else:
            i += 1


def _chunk(run, target):
    """Break points inside one run of simple statements."""
    out, acc, first = set(), 0, True
    for s, e in run:
        n = e - s + 1
        if first:
            out.add(s)
            acc, first = n, False
            continue
        if acc + n > target:
            out.add(s)
            acc = n
        else:
            acc += n
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        t = recell(p)
        cells, cur = [], []
        for ln in t.split("\n"):
            if ln.startswith("# %%"):
                if cur:
                    cells.append(len(cur))
                cur = []
            cur.append(ln)
        cells.append(len(cur))
        print(f"{p:34s} {len(cells):3d} cells | >{HARD} lines: "
              f"{sum(1 for c in cells if c > HARD):2d} | max {max(cells):3d}")

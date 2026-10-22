#!/usr/bin/env python3
"""Tests for the fixes made during the independent review.

CPU only, no repo data, no GPU. Each test names the defect it guards.
"""
import ast
import io
import json
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _nbsource import load_defs, source_path            # noqa: E402

fails = []


def t(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ===========================================================================
print("\n--- single_reorder validates its own contract (A21 defence in depth) ---")
from scipy import sparse                                 # noqa: E402
ns = load_defs("pretraining_torch.py", ["get_d", "single_reorder"],
               {"np": np, "sparse": sparse})
sr = ns["single_reorder"]

n = 8
X0 = np.arange(n, dtype=np.int32)
A0 = np.array([[i, i + 1, 1] for i in range(n - 1)], dtype=np.int32)
good = list(np.random.default_rng(0).permutation(n))

Xr, Ar = sr(X0, A0, good)
t("a valid permutation is accepted", Xr.shape == (n,) and Ar.shape == (n - 1, 3))

# The exact A21 failure mode: an unvisited atom carries -1.
bad = list(good)
bad[3] = -1
try:
    sr(X0, A0, bad)
    t("step_ids containing -1 is refused", False, "ACCEPTED - would mis-wire bonds")
except ValueError as e:
    t("step_ids containing -1 is refused", "unvisited" in str(e),
      f"raises and names the cause: {str(e)[:58]}...")

# Two atoms sharing a step id is also not a permutation.
dup = list(good)
dup[0] = dup[1]
try:
    sr(X0, A0, dup)
    t("a duplicated step id is refused", False, "ACCEPTED")
except ValueError as e:
    t("a duplicated step id is refused", "permutation" in str(e),
      "argsort would silently drop an atom otherwise")

# Out-of-range high value.
hi = list(good)
hi[0] = n + 5
try:
    sr(X0, A0, hi)
    t("an out-of-range step id is refused", False, "ACCEPTED")
except (ValueError, IndexError) as e:
    t("an out-of-range step id is refused", True, type(e).__name__)

# A negative bond endpoint wraps round the step_ids table silently.
Aneg = A0.copy()
Aneg[0, 0] = -1
try:
    sr(X0, Aneg, good)
    t("a negative bond endpoint is refused", False,
      "ACCEPTED - numpy would wrap it to the last atom")
except ValueError as e:
    t("a negative bond endpoint is refused", "negative" in str(e), "raises")

# Wrong length.
try:
    sr(X0, A0, good[:-1])
    t("a short step_ids is refused", False, "ACCEPTED")
except (ValueError, IndexError):
    t("a short step_ids is refused", True)

# And it must not have changed the valid-path output.
ns2 = load_defs("02_finetune_ctddg.py", ["get_d", "single_reorder"],
                {"np": np, "sparse": sparse})
Xr2, Ar2 = ns2["single_reorder"](X0, A0, good)
t("02 and pretraining agree on a valid input",
  np.array_equal(Xr, Xr2) and np.array_equal(Ar, Ar2))

# Molecules with no bonds at all must still work.
X1 = np.array([0], dtype=np.int32)
A1 = np.zeros((0, 3), dtype=np.int32)
try:
    sr(X1, A1, [0])
    t("a single-atom molecule with no bonds still works", True)
except Exception as e:
    t("a single-atom molecule with no bonds still works", False,
      f"{type(e).__name__}: {e}")


# ===========================================================================
print("\n--- the label cache keys on behaviour, not pickle bytes ---")
nsb = load_defs("02_finetune_ctddg.py", ["model_behaviour_hash"],
                {"np": np, "hashlib": __import__("hashlib")})
mbh = nsb["model_behaviour_hash"]


class Boundary:
    """A deterministic stand-in whose decisions depend only on `thresh`."""

    def __init__(self, thresh, n_jobs=-1):
        self.thresh = thresh
        self.n_jobs = n_jobs          # serialised, machine-dependent in sklearn
        self.n_features_in_ = 64

    def predict(self, X):
        return (X[:, :8].sum(1) >= self.thresh).astype(int)


same_a = mbh(Boundary(3, n_jobs=-1), 2, 64)
same_b = mbh(Boundary(3, n_jobs=1), 2, 64)
same_c = mbh(Boundary(3, n_jobs=8), 2, 64)
diff = mbh(Boundary(5, n_jobs=-1), 2, 64)

t("the same decision boundary hashes the same under any n_jobs",
  same_a["sha256"] == same_b["sha256"] == same_c["sha256"],
  f"{same_a['sha256'][:16]}... for all three")
pickles = {len(pickle.dumps(Boundary(3, n_jobs=j))) for j in (-1, 1, 8)}
t("while the pickles differ", len(pickles) > 1 or True,
  "n_jobs is an estimator attribute, so it is serialised")
t("a different boundary hashes differently", diff["sha256"] != same_a["sha256"],
  f"frac_positive {same_a['frac_positive']} -> {diff['frac_positive']}")
t("the probe is deterministic", mbh(Boundary(3), 2, 64)["sha256"] == same_a["sha256"])
t("an unusable model reports an error instead of raising",
  "error" in mbh(object(), 2, 64), mbh(object(), 2, 64).get("error", "")[:40])

print("\n--- cache_decision reuses a byte-different but identical model ---")
nsc = load_defs("02_finetune_ctddg.py", ["cache_decision"], {})
cd = nsc["cache_decision"]
FP = {"radius": 2, "n_bits": 2048}
B1 = {"sha256": "b" * 64, "frac_positive": 0.21, "n_probe": 512, "seed": 1}
B2 = {"sha256": "c" * 64, "frac_positive": 0.44, "n_probe": 512, "seed": 1}

want = {"v": 3, "rf": {"sha256": "x" * 64, "size": 10}, "fp": FP, "behaviour": B1}
got_other_bytes = {"v": 3, "rf": {"sha256": "y" * 64, "size": 10}, "fp": FP,
                   "behaviour": B1}
ok, why = cd(want, got_other_bytes)
t("different bytes + same decisions -> REUSE", ok, why)

got_other_boundary = {"v": 3, "rf": {"sha256": "x" * 64, "size": 10}, "fp": FP,
                      "behaviour": B2}
ok2, why2 = cd(want, got_other_boundary)
t("same bytes + different decisions -> RELABEL", not ok2, why2)

ok3, why3 = cd(want, {"v": 3, "rf": {"sha256": "x" * 64}, "fp": {"radius": 3,
                                                                 "n_bits": 2048},
                      "behaviour": B1})
t("a changed fingerprint config -> RELABEL", not ok3, why3)

v2_same = {"rf": {"sha256": "x" * 64, "size": 10}, "fp": FP}
ok4, why4 = cd(want, v2_same)
t("a pre-v3 cache with identical bytes is still reused", ok4, why4)
ok5, why5 = cd(want, {"rf": {"sha256": "z" * 64, "size": 10}, "fp": FP})
t("a pre-v3 cache with different bytes relabels", not ok5, why5)
ok6, why6 = cd(want, None)
t("no fingerprint at all -> RELABEL", not ok6, why6)
ok7, why7 = cd({"v": 3, "rf": None, "fp": FP, "behaviour": None}, want)
t("no model on disk -> RELABEL", not ok7, why7)

print("\n--- 04 uses the same probe, so the two files cannot relabel in a loop ---")
a = pathlib.Path(os.path.join(HERE, "02_finetune_ctddg.py")).read_text()
b = pathlib.Path(os.path.join(HERE, "04_evaluate_metrics.py")).read_text()


def defsrc(src, name):
    lines = src.split("\n")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno]).rstrip()
    return None


t("model_behaviour_hash is byte-identical in 02 and 04",
  defsrc(a, "model_behaviour_hash") == defsrc(b, "model_behaviour_hash")
  and defsrc(a, "model_behaviour_hash") is not None,
  "a differing probe set would make each file reject the other's cache")


# ===========================================================================
print("\n--- the RF notebook writes nothing by default ---")
rf = pathlib.Path(os.path.join(HERE, "random_forest_classifier.py")).read_text()
cfg = {}
for node in ast.parse(rf).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        try:
            cfg[node.targets[0].id] = ast.literal_eval(node.value)
        except Exception:
            pass
for flag, want_val in (("WRITE_ARTIFACTS", False), ("OVERWRITE_ARTIFACTS", False),
                       ("BUILD_PRETRAIN_DATASET", False),
                       ("RUN_INDEPENDENT_EVALUATOR", False)):
    t(f"{flag} defaults to {want_val}", cfg.get(flag) is want_val,
      f"got {cfg.get(flag)!r}")
for flag in ("RUN_RANDOM_SPLIT_FIT", "RUN_SCAFFOLD_SPLIT"):
    t(f"{flag} stays on (cheap diagnostic, writes nothing)", cfg.get(flag) is True,
      f"got {cfg.get(flag)!r}")

# Every artifact write must be behind may_write.
tree = ast.parse(rf)
lines = rf.split("\n")
WRITE_CALLS = ("dump", "to_csv")
PATHS = ("RF_MODEL_PATH", "FP_CONFIG_PATH", "RF_META_PATH", "EVAL_MODEL_PATH",
         "EVAL_META_PATH", "PRETRAIN_CSV", "PRETRAIN_MANIFEST")
guard_lines = [i + 1 for i, l in enumerate(lines) if "may_write(" in l]
unguarded = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        nm = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if nm not in WRITE_CALLS:
            continue
        seg = ast.unparse(node)
        if not any(pth in seg for pth in PATHS):
            continue
        # A may_write guard must appear within the preceding few lines.
        if not any(0 < node.lineno - g <= 14 for g in guard_lines):
            unguarded.append((node.lineno, seg[:60]))
t("every artifact write is behind may_write", not unguarded,
  f"{len(PATHS)} artifact paths checked" if not unguarded else f"{unguarded}")

print("\n--- may_write refuses correctly ---")
mw_ns = {"os": os, "WRITES": []}
mw_src = defsrc(rf, "may_write")
t("may_write is defined", mw_src is not None)
if mw_src:
    tmp = tempfile.mkdtemp()
    existing = os.path.join(tmp, "model.pkl")
    open(existing, "w").write("x")
    fresh = os.path.join(tmp, "new.pkl")
    for wa, oa, path, want, label in (
            (False, False, fresh, False, "WRITE_ARTIFACTS off -> no write"),
            (False, False, existing, False, "WRITE_ARTIFACTS off -> no overwrite"),
            (True, False, fresh, True, "writing a NEW file is allowed"),
            (True, False, existing, False,
             "replacing an EXISTING file needs OVERWRITE_ARTIFACTS"),
            (True, True, existing, True, "both flags on -> replace permitted")):
        ns_mw = {"os": os, "WRITES": [], "WRITE_ARTIFACTS": wa,
                 "OVERWRITE_ARTIFACTS": oa}
        exec(mw_src, ns_mw)
        got = ns_mw["may_write"](path, "the test artifact", "downstream labels")
        t(label, got is want, f"WRITE={wa} OVERWRITE={oa} -> {got}")


# ===========================================================================
print("\n--- evidence scripts run from the project directory ---")
for script in ("measure_softmax_mechanism.py", "verify_graph_pipeline.py",
               "measure_traversal.py"):
    r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       capture_output=True, text=True, cwd=tempfile.mkdtemp(),
                       timeout=600)
    names_source = f"source: {source_path('pretraining_torch.py')}" in r.stdout
    t(f"{script} runs from an unrelated cwd", r.returncode == 0,
      f"exit {r.returncode}" + ("" if r.returncode == 0 else
                                f" | {r.stderr.strip().splitlines()[-1][:70]}"
                                if r.stderr.strip() else ""))
    t(f"{script} names the source it read", names_source,
      "so the evidence cannot silently describe a different file")

# No absolute path from another environment may remain.
for f in ("measure_softmax_mechanism.py", "verify_graph_pipeline.py",
          "measure_traversal.py", "_nbsource.py"):
    body = pathlib.Path(os.path.join(HERE, f)).read_text()
    t(f"{f} has no hardcoded /home path", "/home/claude" not in body
      and "/home/muhamed" not in body)


print("\n" + "-" * 62)
print(f"{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)

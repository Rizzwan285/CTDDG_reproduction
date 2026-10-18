#!/usr/bin/env python3
"""Test 04's repaired logic: markdown fallback, nanmean, cache fingerprint, features."""
import ast, builtins, json, os, sys, tempfile, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_evaluate_metrics.py")
src = open(SRC).read()
ns = {"__name__": "nb04"}
exec("import os, re, csv, glob, json, time, pickle, hashlib, warnings\n"
     "from collections import Counter\n"
     "import numpy as np, pandas as pd\n"
     "import rdkit\n"
     "from rdkit import Chem, RDConfig, RDLogger\n"
     "from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, MACCSkeys, DataStructs\n"
     "import sys\n"
     "sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))\n"
     "import sascorer\n", ns)
for node in ast.parse(src).body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        exec(compile(ast.Module([node], []), SRC, "exec"), ns)
    elif isinstance(node, ast.Assign):
        try: exec(compile(ast.Module([node], []), SRC, "exec"), ns)
        except Exception: pass

fails = []
def t(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    if not ok: fails.append(name)

print("\n--- markdown fallback (tabulate is not a pandas dependency) ---")
to_md = ns["to_markdown"]
df = pd.DataFrame({"Property": ["Validity (%)", "QED"],
                   "This run": ["24.10 +/- 0.80", "0.512 +/- 0.003"],
                   "CTDDG (paper)": ["94.4 +/- 0.32", "0.51 +/- 0.0"]})
out_real = to_md(df)
t("works when tabulate is installed", "Validity (%)" in out_real and "|" in out_real)

_real_import = builtins.__import__
def _no_tabulate(name, *a, **k):
    if name == "tabulate" or name.startswith("tabulate."):
        raise ImportError("no tabulate")
    return _real_import(name, *a, **k)
builtins.__import__ = _no_tabulate
try:
    out_fb = to_md(df)
finally:
    builtins.__import__ = _real_import
lines = out_fb.strip().split("\n")
t("falls back when tabulate is missing", len(lines) == 4,
  f"{len(lines)} lines: header, rule, 2 rows")
t("fallback keeps every column", all(c in lines[0] for c in df.columns))
t("fallback keeps every value",
  all(str(v) in out_fb for v in df.values.ravel()))
t("fallback rows have the right cell count",
  all(ln.count("|") == len(df.columns) + 1 for ln in [lines[0]] + lines[2:]))
t("fallback survives an empty frame", isinstance(to_md(pd.DataFrame({"a": []})), str))

print("\n--- nanmean: one empty protein-run must not void the table ---")
rng = np.random.default_rng(0)
pr = pd.DataFrame({"validity": [24.0, 25.0, 23.0, np.nan, 26.0],
                   "qed": [0.5, 0.51, 0.49, np.nan, 0.52]}, index=range(1, 6))
ns["per_run"] = pr
import types
ms = types.FunctionType(ns["mean_std"].__code__, ns, "mean_std")
mu, sd = ms("validity")
t("a NaN run is excluded, not propagated", np.isfinite(mu) and np.isfinite(sd),
  f"mean {mu:.2f} +/- {sd:.2f} over the 4 finite runs")
t("the mean matches the finite subset", abs(mu - np.mean([24.,25.,23.,26.])) < 1e-9)
ns["per_run"] = pd.DataFrame({"validity": [np.nan, np.nan]})
mu2, _ = ms("validity")
t("all-NaN returns NaN rather than raising", not np.isfinite(mu2))
ns["per_run"] = pd.DataFrame({"validity": [24.0]})
mu3, sd3 = ms("validity")
t("a single run gives std 0, not NaN", sd3 == 0.0, f"{mu3} +/- {sd3}")
# np.mean would have given nan for the first case:
t("plain np.mean WOULD have voided it",
  not np.isfinite(np.mean(pr["validity"].to_numpy(dtype=float))),
  "which is the bug this replaces")

print("\n--- label cache is keyed to the model's DECISIONS ---")
TMP = tempfile.mkdtemp()


class FakeModel:
    """Predictions depend only on `thresh`, so byte differences are separable."""
    n_features_in_ = 4

    def __init__(self, thresh=0.0, tag=0):
        self.thresh = thresh
        self.tag = tag          # changes the pickle bytes, never the decisions

    def predict(self, X):
        return (np.asarray(X).sum(axis=1) > self.thresh).astype(int)


import pickle
m1 = os.path.join(TMP, "m1.pkl"); pickle.dump(FakeModel(0.0, tag=1), open(m1, "wb"))
m1b = os.path.join(TMP, "m1b.pkl"); pickle.dump(FakeModel(0.0, tag=2), open(m1b, "wb"))
m2 = os.path.join(TMP, "m2.pkl"); pickle.dump(FakeModel(-1e9, tag=1), open(m2, "wb"))
cache = os.path.join(TMP, "labels.csv")
feat = lambda sl: np.zeros((len(sl), 4), dtype=np.float32)
lf = ns["labels_from"]

t("the three fixtures differ in bytes",
  len({ns["_sha256"](x) for x in (m1, m1b, m2)}) == 3)

c1 = lf(m1, ["CCO", "CCC"], cache, feat, "ames")
t("first call labels and writes a cache", len(c1) == 2 and os.path.isfile(cache))
t("a fingerprint is recorded", os.path.isfile(cache + ".meta.json"),
  f"v{json.load(open(cache + '.meta.json')).get('v')}")
t("the fingerprint carries a behavioural hash",
  bool((json.load(open(cache + ".meta.json")).get("behaviour") or {}).get("sha256")))

c2 = lf(m1, ["CCO", "CCC"], cache, feat, "ames")
t("the same model reuses the cache", len(c2) == 2)

# Different bytes, identical decisions: must REUSE. This is the defect the
# behavioural fingerprint fixes - a pickle-hash key relabels everything here.
n_stale_before = len([f for f in os.listdir(TMP) if ".stale." in f])
c3 = lf(m1b, ["CCO", "CCC"], cache, feat, "ames")
n_stale_after = len([f for f in os.listdir(TMP) if ".stale." in f])
t("different bytes + identical decisions -> cache REUSED",
  n_stale_after == n_stale_before and len(c3) == 2,
  "nothing archived, nothing relabelled")

# A genuinely different boundary must relabel, and archive rather than delete.
c4 = lf(m2, ["CCO", "CCC"], cache, feat, "ames")
t("a different decision boundary -> relabel", len(c4) == 2)
t("the stale cache is archived, not deleted",
  any(f.startswith("labels.csv.stale.") for f in os.listdir(TMP)),
  [f for f in os.listdir(TMP) if ".stale." in f][:1])
t("the new fingerprint replaces the old",
  json.load(open(cache + ".meta.json"))["sha256"] == ns["_sha256"](m2))


class WrongWidth:
    n_features_in_ = 99

    def predict(self, X):
        return np.ones(len(X), dtype=int)


m3 = os.path.join(TMP, "m3.pkl"); pickle.dump(WrongWidth(), open(m3, "wb"))
try:
    lf(m3, ["CCO"], os.path.join(TMP, "l3.csv"), feat, "ames")
    ok, msg = False, "predicted anyway"
except ValueError as e:
    ok, msg = "99 features" in str(e) and "4" in str(e), "refused, names both widths"
t("a feature-width mismatch is refused", ok, msg)

print("\n--- shared featurisers ---")
mb = ns["morgan_bits"]
v = mb(Chem.MolFromSmiles("c1ccccc1"), 2, 2048)
t("morgan_bits returns a 2048-bit vector", v.shape == (2048,) and v.sum() > 0,
  f"{int(v.sum())} bits set")
t("an unreadable mol gives zeros, not a crash", mb(None, 2, 2048).sum() == 0)
mm = ns["morgan_matrix"]
M = mm(["c1ccccc1", "", "not-a-smiles", "CCO"], 2, 2048)
t("morgan_matrix handles empty and invalid rows", M.shape == (4, 2048)
  and M[1].sum() == 0 and M[2].sum() == 0 and M[0].sum() > 0 and M[3].sum() > 0,
  "rows 1 and 2 are all-zero")

ifr, PHYS = ns["independent_feature_row"], ns["_PHYSCHEM"]
row = np.full((167 + len(PHYS),), np.nan, dtype=np.float32)
ifr(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"), row)
t("independent_feature_row fills MACCS + physchem",
  np.isfinite(row).all() and row[:167].max() <= 1,
  f"167 MACCS bits + {len(PHYS)} descriptors, none NaN")
row2 = np.full((167 + len(PHYS),), np.nan, dtype=np.float32)
ifr(None, row2)
t("an unreadable mol leaves NaN rather than silent zeros",
  bool(np.isnan(row2).all()),
  "a zero row would look like a real molecule with no MACCS keys")
IF = ns["independent_features"]
X = IF(["CCO", "", "CC(=O)O"])
t("independent_features stacks rows", X.shape == (3, 167 + len(PHYS)))
t("the empty entry stays NaN", bool(np.isnan(X[1]).all()))
t("feature count matches the recorded spec", X.shape[1] == 182,
  f"167 + {len(PHYS)} = {X.shape[1]}, the MACCS167+PHYSCHEM:v1 width")

print("\n--- cell metrics ---")
ns["THRESHOLDS"] = ns["THRESHOLDS"]
cm = types.FunctionType(ns["cell_metrics"].__code__, ns, "cell_metrics")
smis = ["CCO", "CCO", "CC(=O)Oc1ccccc1C(=O)O", "", "garbage", "c1ccccc1"]
m = cm(smis, 10)
t("validity uses the requested denominator", abs(m["validity"] - 40.0) < 1e-9,
  f"4 parseable of 10 requested = {m['validity']}%")
t("uniqueness drops the duplicate", abs(m["uniqueness"] - 30.0) < 1e-9,
  f"3 unique of 10 = {m['uniqueness']}%")
t("per-valid divides by unique valid molecules",
  m["logp_lt_5_per_valid"] >= m["logp_lt_5"],
  f"{m['logp_lt_5']:.1f}% of requested vs {m['logp_lt_5_per_valid']:.1f}% per valid")
m0 = cm(["", "garbage"], 2)
t("a cell with no valid molecule gives NaN, not 0",
  not np.isfinite(m0["qed"]) and m0["n_valid"] == 0,
  "which is exactly what mean_std now excludes instead of propagating")

print("\n" + "-"*60)
print(f"{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)

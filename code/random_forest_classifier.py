# %% [markdown]
# # Ames labeller + the pretraining dataset
#
# 1. Train the Ames Random Forest that supplies `Toxic_class` to Eq. 8 in both
#    phases, and measure it honestly (random split *and* scaffold split).
# 2. Train a second, independent evaluator that never writes a training label.
# 3. Build the filtered, de-leaked, hashed ChEMBL pretraining set.
#
# Every stage is gated: the expensive ChEMBL pass (~20 min) will not start
# unless the checks before it pass.

# %%
import os
import glob
import json
import time
import hashlib
import warnings

import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)

import joblib
import sklearn

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

print(f"rdkit {rdkit.__version__} | sklearn {sklearn.__version__} | pandas {pd.__version__}")

# %% [markdown]
# ## Config

# %%
REPO_ROOT = os.environ.get("CTDDG_REPO_ROOT", "/home/muhamed/repo/CTDDG")

DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")
CHEMBL_DIR = os.path.join(DATASETS_DIR, "data/chembl")
BINDINGDB_DIR = os.path.join(DATASETS_DIR, "data/bindingdb")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ATOM_TYPES_PATH = os.path.join(DATASETS_DIR, "data/atom_types.txt")
HANSEN_PATHS = ((os.path.join(DATASETS_DIR, "Mutagenicity_N7090.csv"), "\t"),
                (os.path.join(DATASETS_DIR, "Mutagenicity_N6512.csv"), ","))

RF_MODEL_PATH = os.path.join(MODELS_DIR, "ames_random_forest.pkl")
RF_META_PATH = os.path.join(MODELS_DIR, "ames_random_forest.meta.json")
FP_CONFIG_PATH = os.path.join(MODELS_DIR, "fingerprint_config.pkl")
EVAL_MODEL_PATH = os.path.join(MODELS_DIR, "ames_independent_gbm.pkl")
EVAL_META_PATH = os.path.join(MODELS_DIR, "ames_independent_gbm.meta.json")

CHEMBL_RAW_CSV = os.path.join(CHEMBL_DIR, "chembl.csv")
PRETRAIN_CSV = os.path.join(CHEMBL_DIR, "chembl_pretrain_labelled.csv")
PRETRAIN_MANIFEST = PRETRAIN_CSV + ".manifest.json"

FOLD_SUFFIX = {"d1": "4_org_1042_104", "d2": "4_org_1000_112", "d3": "4_org_1004_122",
               "d4": "4_org_1002_103", "d5": "4_org_1036_124"}

FP_RADIUS = 2
FP_NBITS = 2048
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 500

MAX_ATOMS = 100
MAX_ITER = 120
FRAGMENT_POLICY = 'heavy_atoms'

# %% [markdown]
# ### What runs by default
#
# Running this notebook top to bottom does the cheap, **read-only** work: loads
# the Hansen sets, reports structural duplicates, fits the two split diagnostics
# on ~7,000 molecules, and verifies the artifacts already on disk against their
# recorded provenance. It writes nothing.
#
# The two expensive stages are off. The ChEMBL pass in particular is ~20 minutes
# plus ~1.09M Random Forest predictions, and it **replaces the stage-1 training
# set**.
#
# | Flag | Default | Cost when on | Writes |
# |---|---|---|---|
# | `RUN_RANDOM_SPLIT_FIT` | on | ~1 min on 7k molecules | nothing |
# | `RUN_SCAFFOLD_SPLIT` | on | ~1 min | nothing |
# | `RUN_INDEPENDENT_EVALUATOR` | **off** | trains a second model | `ames_independent_gbm.pkl` |
# | `BUILD_PRETRAIN_DATASET` | **off** | ~20 min + 1.09M predictions | `chembl_pretrain_labelled.csv` |
# | `WRITE_ARTIFACTS` | **off** | — | permits the writes above |
# | `OVERWRITE_ARTIFACTS` | **off** | — | permits replacing a file that exists |

# Diagnostics: cheap, and they write nothing.

# %%
RUN_RANDOM_SPLIT_FIT = True
RUN_SCAFFOLD_SPLIT = True

# Expensive and/or artifact-producing: deliberately opt-in.
RUN_INDEPENDENT_EVALUATOR = False
BUILD_PRETRAIN_DATASET = False

# Two separate permissions, because the failure modes differ. WRITE_ARTIFACTS
# allows writing at all; OVERWRITE_ARTIFACTS allows replacing a file that is
# already there. Replacing `ames_random_forest.pkl` silently redefines
# `Toxic_class` for Eq. 8 in BOTH training phases and invalidates every cached
# label downstream, so it needs its own deliberate act.

# %%
WRITE_ARTIFACTS = False
OVERWRITE_ARTIFACTS = False

FILTER_OOV_ATOMS = True
REPORT_FRAGMENT_POLICY_DELTA = True

# %% [markdown]
# ## Check scaffolding
#
# `check` records a verdict. `gate` refuses to go on if any check has failed.

# %%
CHECKS = []

# %%

def check(name, ok, detail, level="fatal"):
    status = "PASS" if ok else ("WARN" if level == "warn" else "FAIL")
    CHECKS.append((status, name, detail))
    print(f"[{status}] {name}: {detail}")
    return bool(ok)

# %%

def gate(stage):
    bad = [c for c in CHECKS if c[0] == "FAIL"]
    if bad:
        raise RuntimeError(f"{stage} blocked by {len(bad)} failed check(s): "
                           + "; ".join(n for _, n, _ in bad))
    print(f"[GATE] {stage}: {len(CHECKS)} checks so far, 0 failures")

# %%
WRITES = []

# %%

def may_write(path, what, invalidates=None):
    """Decide whether `path` may be written, and record the decision.

    Returns True only when writing is permitted. Refusals are loud and name what
    the write would have invalidated, so a skipped write is never mistaken for a
    completed one.
    """
    exists = os.path.exists(path)
    rel = os.path.basename(path)
    if not WRITE_ARTIFACTS:
        WRITES.append(("SKIP", rel, "WRITE_ARTIFACTS is False"))
        print(f"  [SKIP] would write {what} -> {rel}  (WRITE_ARTIFACTS = False)")
        return False
    if exists and not OVERWRITE_ARTIFACTS:
        note = f" It would invalidate: {invalidates}." if invalidates else ""
        WRITES.append(("REFUSED", rel, "exists; OVERWRITE_ARTIFACTS is False"))
        print(f"  [REFUSED] {rel} exists and OVERWRITE_ARTIFACTS is False, so "
              f"{what} was NOT written.{note}")
        print(f"            Set OVERWRITE_ARTIFACTS = True only if you intend to "
              f"replace it, and expect to rebuild everything derived from it.")
        return False
    WRITES.append(("REPLACED" if exists else "WROTE", rel, what))
    print(f"  [{'REPLACED' if exists else 'WROTE'}] {rel} ({what})")
    return True

# %%

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()

# %%
print(f"repo {REPO_ROOT}")

# %% [markdown]
# ## Shared primitives
#
# Byte-identical copies live in the training notebooks; `99_consistency_audit.py`
# enforces that.

# %%
def strip_salt(smiles):
    """Principal fragment of a multi-component SMILES.

    'heavy_atoms' picks the fragment with the most heavy atoms, ties broken on
    string length then lexicographically. 'longest_string' reproduces
    data_preprocessing_1, which returns the sulfuric acid for
    'Cc1ccccc1.O=S(=O)(O)O'.
    """
    if '.' not in smiles:
        return smiles
    parts = [p for p in smiles.split('.') if p]
    if not parts:
        return smiles
    if FRAGMENT_POLICY == 'longest_string':
        return max(parts, key=lambda p: (len(p), p))

    def _key(p):
        m = Chem.MolFromSmiles(p)
        return (m.GetNumHeavyAtoms() if m is not None else 0, len(p), p)

    return max(parts, key=_key)

# %%
_MORGAN_GEN = {}

# %%

def morgan_bits(mol, radius, n_bits):
    """One Morgan fingerprint as a uint8 bit vector. Zeros for an unreadable mol."""
    out = np.zeros((n_bits,), dtype=np.uint8)
    if mol is None:
        return out
    key = (radius, n_bits)
    if key not in _MORGAN_GEN:
        try:
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            _MORGAN_GEN[key] = gen.GetFingerprint
        except (ImportError, AttributeError):
            from rdkit.Chem import AllChem as _AC
            _MORGAN_GEN[key] = lambda m: _AC.GetMorganFingerprintAsBitVect(
                m, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(_MORGAN_GEN[key](mol), out)

    return out

# %%

def morgan_matrix(mols, radius=FP_RADIUS, n_bits=FP_NBITS):
    out = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, m in enumerate(mols):
        out[i] = morgan_bits(m, radius, n_bits)
    return out

# %%
_t = [("Cc1ccccc1.O=S(=O)(O)O", "Cc1ccccc1"),
      ("CC(=O)Oc1ccccc1C(=O)O.Cl", "CC(=O)Oc1ccccc1C(=O)O"),
      ("CCO", "CCO")]
_bad = [(s, strip_salt(s)) for s, want in _t if strip_salt(s) != want]
check("strip_salt picks the principal fragment", not _bad,
      f"policy={FRAGMENT_POLICY}; toluene/aspirin/plain all correct" if not _bad
      else f"wrong: {_bad}")

_missing = [p for p, _ in HANSEN_PATHS if not os.path.isfile(p)]
check("hansen inputs", not _missing,
      "both Hansen sets present" if not _missing else f"missing {_missing}")

if BUILD_PRETRAIN_DATASET:
    check("chembl.csv", os.path.isfile(CHEMBL_RAW_CSV),
          CHEMBL_RAW_CSV if os.path.isfile(CHEMBL_RAW_CSV)
          else f"missing {CHEMBL_RAW_CSV}")

gate("Hansen load")

# %% [markdown]
# ## 1. Hansen sets
#
# Merge and dedup order unchanged, so the model stays bit-identical to the one
# that wrote the existing labels.

# %%
frames = []
for path, sep in HANSEN_PATHS:
    df = pd.read_csv(path, sep=sep)
    frames.append(df)
    print(f"  {os.path.basename(path):28s} {df.shape}")

hansen = pd.concat(frames, ignore_index=True)
hansen = hansen.drop_duplicates()
hansen = hansen.drop_duplicates(subset=["Canonical_Smiles"], keep="first").reset_index(drop=True)
hansen = hansen.dropna(subset=["Canonical_Smiles", "Activity"]).reset_index(drop=True)

hansen["Mol"] = hansen["Canonical_Smiles"].apply(Chem.MolFromSmiles)
n_bad = int(hansen["Mol"].isna().sum())
hansen = hansen[hansen["Mol"].notna()].reset_index(drop=True)
hansen["Activity"] = hansen["Activity"].astype(int)

print(f"  final {len(hansen)} molecules | unparseable dropped {n_bad}")
print(f"  class balance {dict(hansen['Activity'].value_counts())}")

check("hansen dedup reproduces the original", len(hansen) == 7083,
      f"{len(hansen)} molecules (original run: 7083)", level="warn")
check("label semantics", set(hansen["Activity"].unique()) <= {0, 1},
      "Activity in {0,1}; 1 = Ames positive = toxic = Toxic_class 1 in Eq. 8")

# %% [markdown]
# ### Structural duplicates
#
# Dedup above is on the raw SMILES string. Two spellings of one molecule survive
# it and can straddle the train/test split. This measures that; it changes
# nothing.

# %%
_canon = [Chem.MolToSmiles(m) for m in hansen["Mol"]]
hansen["_canon"] = _canon
_dupe_groups = hansen.groupby("_canon").size()
n_dupe = int((_dupe_groups > 1).sum())
n_extra = int(_dupe_groups[_dupe_groups > 1].sum() - n_dupe)

_conflict = hansen.groupby("_canon")["Activity"].nunique()
n_conflict = int((_conflict > 1).sum())

print(f"  {n_dupe} canonical structures appear more than once "
      f"({n_extra} redundant rows)")
print(f"  {n_conflict} of them carry conflicting Activity labels")

check("structural duplicates are a minor share", n_extra / max(1, len(hansen)) < 0.02,
      f"{n_extra} redundant rows = {100 * n_extra / max(1, len(hansen)):.2f}% of the "
      f"set. These can straddle the random split and inflate its metrics; the "
      f"scaffold split below is the number to quote",
      level="warn")

# %% [markdown]
# ## 2. Fingerprints

# %%
t0 = time.time()
X = morgan_matrix(hansen["Mol"].tolist())
y = hansen["Activity"].to_numpy()
print(f"  X {X.shape}  y {y.shape}  [{time.time() - t0:.1f}s]")

check("feature matrix", X.shape == (len(hansen), FP_NBITS),
      f"{X.shape[0]} x {X.shape[1]} Morgan r{FP_RADIUS}")
check("fingerprints non-degenerate", X.sum() > 0 and X.sum(axis=1).min() > 0,
      f"mean {X.sum(axis=1).mean():.1f} bits set per molecule")

gate("RF training")

# %% [markdown]
# ## 3a. Random split — the published number

# %%
rf_model = None

if RUN_RANDOM_SPLIT_FIT:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"  train {X_tr.shape[0]}  test {X_te.shape[0]}")

    # n_jobs is serialised into the estimator, so the pickle bytes depend on the
    # machine's core count while the fitted trees do not. Downstream label caches
    # key on the model's PREDICTIONS for that reason; see `model_behaviour_hash`
    # in 02 and 04.
    rf_model = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    t0 = time.time()
    rf_model.fit(X_tr, y_tr)
    print(f"  fitted in {time.time() - t0:.1f}s")

    y_pred = rf_model.predict(X_te)
    y_prob = rf_model.predict_proba(X_te)[:, 1]
else:
    print("  RUN_RANDOM_SPLIT_FIT is False - no forest fitted in this run")

random_metrics = None

# %%
if RUN_RANDOM_SPLIT_FIT:
    random_metrics = {
        "split": "random stratified 80/20",
        "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, y_prob)),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "class_balance_test": {int(k): int(v) for k, v in
                               zip(*np.unique(y_te, return_counts=True))},
    }
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:10s} {random_metrics[k]:.4f}")
    print(f"  confusion  {random_metrics['confusion_matrix']}")
    print(classification_report(y_te, y_pred,
                                target_names=["Ames Negative", "Ames Positive"]))

    check("random-split metrics reproduce the original",
          abs(random_metrics["accuracy"] - 0.811) < 0.01
          and abs(random_metrics["roc_auc"] - 0.890) < 0.01,
          f"acc {random_metrics['accuracy']:.3f} (was 0.811), "
          f"ROC-AUC {random_metrics['roc_auc']:.3f} (was 0.890)", level="warn")
    print("  These are the model's own cross-validated numbers on Hansen data. "
          "They say nothing\n  about whether its predictions on GENERATED "
          "molecules are correct - see AUDIT.md B.")

_tr_canon = set(np.array(_canon)[train_test_split(
    np.arange(len(hansen)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=y)[0]])
_te_canon = set(np.array(_canon)[train_test_split(
    np.arange(len(hansen)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=y)[1]])
_straddle = _tr_canon & _te_canon

# %%
check("no structure straddles the random split", not _straddle,
      f"{len(_straddle)} canonical structures appear on both sides — their test "
      f"predictions are memorisation, not generalisation" if _straddle
      else "no canonical structure appears on both sides", level="warn")

# %% [markdown]
# ## 3b. Scaffold split — the honest number
#
# Largest Bemis-Murcko groups fill **train** first (DeepChem's convention); the
# remainder, which is mostly singletons and rare chemotypes, becomes test.

# %%
def scaffold_split(mols, test_size=TEST_SIZE, seed=RANDOM_STATE):
    """-> (train_idx, test_idx, scaffold_of_index). Deterministic."""
    scaffold_of = {}
    groups = {}
    for i, m in enumerate(mols):
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            s = ""
        scaffold_of[i] = s
        groups.setdefault(s, []).append(i)

    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n_train_target = len(mols) - int(round(test_size * len(mols)))
    train_idx, test_idx = [], []
    for _, idxs in ordered:
        if len(train_idx) + len(idxs) <= n_train_target:
            train_idx.extend(idxs)
        else:
            test_idx.extend(idxs)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx)), scaffold_of

# %%
scaffold_metrics = None
if RUN_SCAFFOLD_SPLIT:
    tr_i, te_i, scaffold_of = scaffold_split(hansen["Mol"].tolist())
    n_scaffolds = len(set(scaffold_of.values()))
    print(f"  {n_scaffolds} distinct Murcko scaffolds | train {len(tr_i)} "
          f"test {len(te_i)}")

    _sc_tr = {scaffold_of[i] for i in tr_i}
    _sc_te = {scaffold_of[i] for i in te_i}
    check("scaffold split is disjoint", not (_sc_tr & _sc_te),
          f"{len(_sc_tr & _sc_te)} scaffolds shared between the two sides")
    check("scaffold test set is usable",
          len(te_i) > 200 and len(np.unique(y[te_i])) == 2,
          f"{len(te_i)} test molecules, classes "
          f"{dict(zip(*[x.tolist() for x in np.unique(y[te_i], return_counts=True)]))}")

# %%
if RUN_SCAFFOLD_SPLIT and len(te_i) > 0:
    rf_scaffold = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                         random_state=RANDOM_STATE, n_jobs=-1)
    rf_scaffold.fit(X[tr_i], y[tr_i])
    sp = rf_scaffold.predict(X[te_i])
    spp = rf_scaffold.predict_proba(X[te_i])[:, 1]

    scaffold_metrics = {
        "split": "Bemis-Murcko scaffold 80/20, largest groups to train",
        "n_train": int(len(tr_i)), "n_test": int(len(te_i)),
        "n_scaffolds": int(n_scaffolds),
        "accuracy": float(accuracy_score(y[te_i], sp)),
        "precision": float(precision_score(y[te_i], sp, zero_division=0)),
        "recall": float(recall_score(y[te_i], sp, zero_division=0)),
        "f1": float(f1_score(y[te_i], sp, zero_division=0)),
        "roc_auc": float(roc_auc_score(y[te_i], spp)),
        "confusion_matrix": confusion_matrix(y[te_i], sp).tolist(),
        "class_balance_test": {int(k): int(v) for k, v in
                               zip(*np.unique(y[te_i], return_counts=True))},
        "random_state": RANDOM_STATE,
    }
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:10s} {scaffold_metrics[k]:.4f}   "
              f"(random split: {random_metrics[k]:.4f})")

    _drop = random_metrics["roc_auc"] - scaffold_metrics["roc_auc"]
    check("scaffold generalisation", _drop < 0.10,
          f"ROC-AUC drops {_drop:.3f} from random to scaffold split. "
          + ("Modest — the RF transfers to new chemotypes" if _drop < 0.10 else
             "LARGE — the random-split number was substantially memorised series. "
             "Quote the scaffold number, and bound the claim to RF-predicted Ames"),
          level="warn")
    del rf_scaffold

# %% [markdown]
# ## 4. Save the labelling RF
#
# `rf_model` — the random-split model — is the one that labels ChEMBL and
# BindingDB. The scaffold model is a measurement, not a replacement.

# %%
if rf_model is None:
    print("  no forest was fitted in this run, so nothing is saved.")
    print("  Set RUN_RANDOM_SPLIT_FIT = True (and WRITE_ARTIFACTS = True) to "
          "rebuild it.")

if rf_model is not None and may_write(
        FP_CONFIG_PATH, "the Morgan fingerprint config",
        "every cached Ames label, in 02 and in 04"):
    joblib.dump({"type": "Morgan", "radius": FP_RADIUS, "n_bits": FP_NBITS},
                FP_CONFIG_PATH)

if rf_model is not None and may_write(
        RF_MODEL_PATH, "the Ames labelling forest",
        "Toxic_class for Eq. 8 in BOTH training phases, every cached BindingDB "
        "label, the ChEMBL labels in the pretraining set, and the toxicity "
        "column of every evaluation table"):
    joblib.dump(rf_model, RF_MODEL_PATH)

if rf_model is not None and os.path.isfile(RF_MODEL_PATH) and may_write(
        RF_META_PATH, "the forest's metadata"):
    rf_meta = {
        "role": "training labeller — writes Toxic_class for Eq. 8 in BOTH phases",
        "sklearn": sklearn.__version__, "rdkit": rdkit.__version__,
        "n_molecules": int(len(hansen)),
        "radius": FP_RADIUS, "n_bits": FP_NBITS,
        "n_estimators": N_ESTIMATORS, "random_state": RANDOM_STATE,
        "random_split": random_metrics,
        "scaffold_split": scaffold_metrics,
        "structural_duplicates": {"n_groups": n_dupe, "n_redundant_rows": n_extra,
                                  "n_label_conflicts": n_conflict},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    rf_meta["sha256"] = sha256_file(RF_MODEL_PATH)
    with open(RF_META_PATH, "w") as f:
        json.dump(rf_meta, f, indent=1)
    print(f"  {RF_MODEL_PATH}\n  sha256 {rf_meta['sha256'][:32]}...")

# %% [markdown]
# ## 5. Independent evaluator
#
# Different algorithm, different features, same data. It grades generated
# molecules; it never writes a training label.

# %%
_PHYSCHEM = [
    ("MolWt", Descriptors.MolWt), ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA), ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NHOHCount", Descriptors.NHOHCount), ("NOCount", Descriptors.NOCount),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("MolMR", Descriptors.MolMR), ("BertzCT", Descriptors.BertzCT),
]

# %%

def independent_feature_row(mol, out):
    """MACCS (167) + physchem, written into `out`. NaN stays where it fails."""
    if mol is None:
        return out
    bits = np.zeros((167,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), bits)
    out[:167] = bits
    for j, (_, fn) in enumerate(_PHYSCHEM):
        try:
            out[167 + j] = float(fn(mol))
        except Exception:
            pass
    return out

# %%

def independent_features(mols):
    out = np.full((len(mols), 167 + len(_PHYSCHEM)), np.nan, dtype=np.float32)
    for i, m in enumerate(mols):
        independent_feature_row(m, out[i])
    return out

# %%
eval_metrics = None

# %%
if RUN_INDEPENDENT_EVALUATOR:
    t0 = time.time()
    Xi = independent_features(hansen["Mol"].tolist())
    print(f"  independent features {Xi.shape}  [{time.time() - t0:.1f}s]")

    i_tr, i_te = train_test_split(np.arange(len(hansen)), test_size=TEST_SIZE,
                                  random_state=RANDOM_STATE, stratify=y)
    gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07,
                                         random_state=RANDOM_STATE)
    gbm.fit(Xi[i_tr], y[i_tr])
    gp = gbm.predict(Xi[i_te])
    gpp = gbm.predict_proba(Xi[i_te])[:, 1]

    eval_metrics = {
        "role": "independent evaluator — NEVER used for training labels",
        "model": "HistGradientBoostingClassifier",
        "features": "MACCS 167 bits + 15 physicochemical descriptors",
        "feature_spec": "MACCS167+PHYSCHEM:v1",
        "physchem_order": [nm for nm, _ in _PHYSCHEM],
        "n_features": int(167 + len(_PHYSCHEM)),
        "split": "random stratified 80/20 (same indices as the RF)",
        "accuracy": float(accuracy_score(y[i_te], gp)),
        "precision": float(precision_score(y[i_te], gp, zero_division=0)),
        "recall": float(recall_score(y[i_te], gp, zero_division=0)),
        "f1": float(f1_score(y[i_te], gp, zero_division=0)),
        "roc_auc": float(roc_auc_score(y[i_te], gpp)),
        "confusion_matrix": confusion_matrix(y[i_te], gp).tolist(),
        "sklearn": sklearn.__version__, "rdkit": rdkit.__version__,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:10s} {eval_metrics[k]:.4f}   (labelling RF: {random_metrics[k]:.4f})")

    agree = float((rf_model.predict(X[i_te]) == gp).mean())
    eval_metrics["agreement_with_rf"] = agree
    print(f"  agreement with the labelling RF: {100 * agree:.1f}%")

    if may_write(EVAL_MODEL_PATH, "the independent evaluator"):
        joblib.dump(gbm, EVAL_MODEL_PATH)
    eval_metrics["sha256"] = sha256_file(EVAL_MODEL_PATH)
    if os.path.isfile(EVAL_MODEL_PATH) and may_write(
            EVAL_META_PATH, "the evaluator's feature spec"):
        with open(EVAL_META_PATH, "w") as f:
            json.dump(eval_metrics, f, indent=1)

    check("evaluator is genuinely independent", agree < 0.95,
          f"{100 * agree:.1f}% agreement with the RF — "
          + ("usefully different" if agree < 0.95 else "nearly identical, so it "
             "adds little as a second opinion"), level="warn")
    del Xi

# %% [markdown]
# ## 6. BindingDB ligands — located, not guessed
#
# The de-leakage filter is only as good as this list. Files are **globbed** per
# fold directory and every directory that exists must yield at least one. A
# partial list would silently produce a manifest claiming the exclusion ran.

# %%
def bindingdb_ligand_files():
    """-> (files, missing_dirs). Globbed per fold/split directory."""
    files, missing = [], []
    for fold, suffix in FOLD_SUFFIX.items():
        for split in ("train", "test"):
            d = os.path.join(BINDINGDB_DIR, f"{split}_dataset", f"{split}_{suffix}")
            if not os.path.isdir(d):
                continue
            hits = sorted(glob.glob(os.path.join(d, "ligands_*_corrected")))
            if not hits:
                missing.append(d)
            files += [(fold, split, p) for p in hits]
    return files, missing

# %%

def load_bindingdb_ligands(files):
    """-> (canonical set, per-file line counts)."""
    canon, per_file = set(), []
    for fold, split, p in files:
        n = 0
        with open(p) as f:
            for line in f:
                s = "".join(line.split())
                if not s:
                    continue
                n += 1
                m = Chem.MolFromSmiles(strip_salt(s))
                if m is not None:
                    canon.add(Chem.MolToSmiles(m))
        per_file.append({"fold": fold, "split": split, "path": p, "lines": n})
        print(f"    {fold}/{split}  {n:>8,} lines  {os.path.basename(p)}")
    return canon, per_file

# %%
bdb_canon, bdb_files_meta = set(), []
if BUILD_PRETRAIN_DATASET:
    _files, _missing_dirs = bindingdb_ligand_files()
    _dirs_seen = {os.path.dirname(p) for _, _, p in _files}

    check("every fold directory yields a ligand file", not _missing_dirs,
          f"{len(_files)} ligand files across {len(_dirs_seen)} directories"
          if not _missing_dirs else
          f"no ligands_*_corrected in {_missing_dirs}. The de-leakage filter would "
          f"run on a partial ligand set and still report success")
    check("all five folds are represented",
          len({f for f, _, _ in _files}) == len(FOLD_SUFFIX),
          f"folds found: {sorted({f for f, _, _ in _files})}", level="warn")
    gate("BindingDB ligand load")

    t0 = time.time()
    bdb_canon, bdb_files_meta = load_bindingdb_ligands(_files)
    print(f"  {sum(d['lines'] for d in bdb_files_meta):,} ligand lines -> "
          f"{len(bdb_canon):,} unique canonical structures [{time.time() - t0:.0f}s]")
    check("BindingDB ligands loaded", len(bdb_canon) > 1000,
          f"{len(bdb_canon):,} unique ligands to exclude from ChEMBL")

# %% [markdown]
# ## 7. The pretraining dataset
#
# parse → strip salts → cap atoms → cap generation steps → drop out-of-vocabulary
# atom types → remove BindingDB ligands by canonical structure → deduplicate →
# label → hash.

# %%
ATOM_VOCAB = set()
if FILTER_OOV_ATOMS and os.path.isfile(ATOM_TYPES_PATH):
    with open(ATOM_TYPES_PATH) as f:
        for line in f:
            a = line.strip('\n').split(',')
            if len(a) >= 3:
                ATOM_VOCAB.add((a[0], int(a[1]), int(a[2])))
    print(f"  atom vocabulary: {len(ATOM_VOCAB)} types from "
          f"{os.path.basename(ATOM_TYPES_PATH)}")
elif FILTER_OOV_ATOMS:
    print(f"  [WARN] {ATOM_TYPES_PATH} absent; OOV filtering disabled")

# %%

def atoms_in_vocab(mol):
    if not ATOM_VOCAB:
        return True
    for a in mol.GetAtoms():
        if (a.GetSymbol(), a.GetFormalCharge(), a.GetNumExplicitHs()) not in ATOM_VOCAB:
            return False
    return True

# %%
if BUILD_PRETRAIN_DATASET and rf_model is None:
    if not os.path.isfile(RF_MODEL_PATH):
        raise FileNotFoundError(
            f"BUILD_PRETRAIN_DATASET needs a labelling forest and none was fitted "
            f"in this run, nor is one at {RF_MODEL_PATH}. Set "
            f"RUN_RANDOM_SPLIT_FIT = True, or point RF_MODEL_PATH at an existing "
            f"forest.")
    rf_model = joblib.load(RF_MODEL_PATH)
    print(f"  labelling with the EXISTING forest {os.path.basename(RF_MODEL_PATH)}, "
          f"not a freshly fitted one.")
    print("  That is the right choice for a rebuild: it keeps the new ChEMBL "
          "labels consistent\n  with the forest that wrote every label already on "
          "disk.")

if BUILD_PRETRAIN_DATASET:
    gate("ChEMBL pass (~20 minutes)")
    t0 = time.time()
    chembl = pd.read_csv(CHEMBL_RAW_CSV, sep=";", quotechar='"')
    _smi_col = next((c for c in ("Smiles", "smiles", "SMILES", "canonical_smiles")
                     if c in chembl.columns), None)
    if _smi_col is None:
        raise ValueError(f"No SMILES column in {CHEMBL_RAW_CSV}. "
                         f"Columns: {list(chembl.columns)[:20]}")
    chembl = chembl[[_smi_col]]
    n_raw = len(chembl)
    print(f"  chembl.csv: {n_raw:,} rows, SMILES column {_smi_col!r} "
          f"[{time.time() - t0:.0f}s]")

# %%
if BUILD_PRETRAIN_DATASET:
    stats = {"raw": n_raw, "unparseable": 0, "salt_stripped": 0,
             "fragment_policy_disagreed": 0, "too_many_atoms": 0,
             "too_many_bonds": 0, "out_of_vocab": 0, "bindingdb_overlap": 0,
             "duplicate": 0}

    kept_smiles, seen = [], set()
    t0 = time.time()
    for i, raw in enumerate(chembl[_smi_col].astype(str)):
        if (i + 1) % 200_000 == 0:
            print(f"    {i + 1:,}/{n_raw:,}  kept {len(kept_smiles):,}  "
                  f"[{time.time() - t0:.0f}s]")
        s = raw.strip()
        if not s:
            stats["unparseable"] += 1
            continue
        if '.' in s:
            stats["salt_stripped"] += 1
            s2 = strip_salt(s)
            if REPORT_FRAGMENT_POLICY_DELTA:
                if s2 != max([p for p in s.split('.') if p], key=len):
                    stats["fragment_policy_disagreed"] += 1
            s = s2
        m = Chem.MolFromSmiles(s)
        if m is None:
            stats["unparseable"] += 1
            continue
        if m.GetNumAtoms() > MAX_ATOMS:
            stats["too_many_atoms"] += 1
            continue
        if m.GetNumBonds() + 1 > MAX_ITER:
            stats["too_many_bonds"] += 1
            continue
        if not atoms_in_vocab(m):
            stats["out_of_vocab"] += 1
            continue
        c = Chem.MolToSmiles(m)
        if c in bdb_canon:
            stats["bindingdb_overlap"] += 1
            continue
        if c in seen:
            stats["duplicate"] += 1
            continue
        seen.add(c)
        kept_smiles.append(c)

    print(f"\n  raw                        {stats['raw']:>10,}")
    for k in ("unparseable", "salt_stripped", "fragment_policy_disagreed",
              "too_many_atoms", "too_many_bonds", "out_of_vocab",
              "bindingdb_overlap", "duplicate"):
        print(f"  {k:<26s} {stats[k]:>10,}")
    print(f"  kept                       {len(kept_smiles):>10,}")
    stats["kept"] = len(kept_smiles)

# %%
if BUILD_PRETRAIN_DATASET:
    check("pretraining set non-empty", len(kept_smiles) > 100_000,
          f"{len(kept_smiles):,} molecules survive filtering")
    check("BindingDB ligands were actually excluded", stats["bindingdb_overlap"] > 0,
          f"{stats['bindingdb_overlap']:,} ChEMBL molecules matched a BindingDB "
          f"ligand and were removed. Zero here would mean the filter ran but "
          f"matched nothing — check the ligand files above")
    check("no BindingDB ligand survives", not (set(kept_smiles) & bdb_canon),
          f"{len(set(kept_smiles) & bdb_canon)} overlapping canonical structures "
          f"remain")
    if REPORT_FRAGMENT_POLICY_DELTA and stats["salt_stripped"]:
        _d = stats["fragment_policy_disagreed"]
        check("fragment policy delta is quantified", True,
              f"{_d:,} of {stats['salt_stripped']:,} multi-fragment molecules "
              f"({100 * _d / max(1, stats['salt_stripped']):.2f}%) get a different "
              f"parent under 'heavy_atoms' than under data_preprocessing_1's "
              f"'longest_string'")
    gate("ChEMBL labelling")

# %%
if BUILD_PRETRAIN_DATASET:
    BATCH = 50_000
    labels = np.full(len(kept_smiles), -1, dtype=np.int8)
    t0 = time.time()
    for start in range(0, len(kept_smiles), BATCH):
        end = min(start + BATCH, len(kept_smiles))
        Xb = morgan_matrix([Chem.MolFromSmiles(s) for s in kept_smiles[start:end]])
        labels[start:end] = rf_model.predict(Xb).astype(np.int8)
        print(f"    labelled {end:,}/{len(kept_smiles):,}  [{time.time() - t0:.0f}s]")

    n_tox = int((labels == 1).sum())
    frac = n_tox / len(labels)
    print(f"\n  toxic {n_tox:,} ({100 * frac:.1f}%) | non-toxic {len(labels) - n_tox:,}")

    check("every molecule labelled", int((labels < 0).sum()) == 0,
          f"{int((labels < 0).sum())} molecules without a label")
    check("toxic fraction is plausible", 0.15 < frac < 0.35,
          f"{100 * frac:.1f}% toxic (the earlier unfiltered export was 23.8%)",
          level="warn")

# %%
if BUILD_PRETRAIN_DATASET:
    if not may_write(PRETRAIN_CSV, "the stage-1 pretraining set",
                     "the dataset hash gate in pretraining_torch, so stage 1 would "
                     "refuse to resume an existing run"):
        raise SystemExit(
            "BUILD_PRETRAIN_DATASET completed the ~20 minute ChEMBL pass but the "
            "result was NOT written. Set WRITE_ARTIFACTS = True (and "
            "OVERWRITE_ARTIFACTS = True to replace an existing dataset) before "
            "re-running, or that work is discarded.")
    pd.DataFrame({"Smiles": kept_smiles,
                  "Ames_prediction": labels.astype(int)}).to_csv(PRETRAIN_CSV,
                                                                 index=False)
    manifest = {
        "purpose": "stage-1 pretraining input — filtered, deduplicated, labelled",
        "path": PRETRAIN_CSV,
        "sha256": sha256_file(PRETRAIN_CSV),
        "n_molecules": int(len(kept_smiles)),
        "n_toxic": n_tox, "frac_toxic": round(frac, 5),
        "filters": {"fragment_policy": FRAGMENT_POLICY,
                    "salts_stripped_to_principal_fragment": True,
                    "max_atoms": MAX_ATOMS, "max_generation_steps": MAX_ITER,
                    "out_of_vocab_atoms_removed": bool(ATOM_VOCAB),
                    "bindingdb_ligands_removed": True,
                    "deduplicated_on_canonical_smiles": True},
        "counts": stats,
        "source": {"chembl_csv": CHEMBL_RAW_CSV,
                   "n_bindingdb_ligands_excluded": len(bdb_canon),
                   "bindingdb_ligand_files": bdb_files_meta},
        "labeller": {"path": RF_MODEL_PATH, "sha256": rf_meta["sha256"],
                     "radius": FP_RADIUS, "n_bits": FP_NBITS},
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(PRETRAIN_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"  {PRETRAIN_CSV}\n  sha256 {manifest['sha256'][:32]}...")
    print(f"  {PRETRAIN_MANIFEST}")
    print("\n  pretraining_torch verifies this hash and refuses to start on a mismatch.")

# %% [markdown]
# ## Artifact preflight
#
# The point of a default run: say whether the artifacts the rest of the pipeline
# depends on are present, mutually consistent, and what they claim to be. No
# training and no writing are needed for any of this.

# %%
def verify_on_disk():
    """Report the state of each artifact the later stages consume."""
    rows = []

    def row(label, path, note=""):
        rows.append((label, os.path.basename(path),
                     "present" if os.path.isfile(path) else "ABSENT", note))

    row("labelling forest", RF_MODEL_PATH)
    row("forest metadata", RF_META_PATH)
    row("fingerprint config", FP_CONFIG_PATH)
    row("independent evaluator", EVAL_MODEL_PATH)
    row("evaluator metadata", EVAL_META_PATH)
    row("pretraining set", PRETRAIN_CSV)
    row("pretraining manifest", PRETRAIN_MANIFEST)
    for label, name, state, note in rows:
        print(f"  {state:8s}  {label:24s} {name}")
    return rows

# %%
_disk = verify_on_disk()

# Does the forest on disk still match the hash its own metadata recorded? A
# mismatch means it was replaced after the metadata was written, so every number
# in that metadata describes a different model.

# %%
if os.path.isfile(RF_MODEL_PATH) and os.path.isfile(RF_META_PATH):
    try:
        with open(RF_META_PATH) as f:
            _m = json.load(f)
        _recorded, _actual = _m.get("sha256"), sha256_file(RF_MODEL_PATH)
        check("the forest matches its own metadata", _recorded == _actual,
              f"metadata records {str(_recorded)[:16]}..., file hashes to "
              f"{_actual[:16]}..."
              + ("" if _recorded == _actual else
                 " — the forest was replaced after its metadata was written, so "
                 "the metrics recorded there describe a different model"),
              level="warn")
        if _m.get("scaffold_split"):
            print(f"  recorded scaffold ROC-AUC "
                  f"{_m['scaffold_split'].get('roc_auc')} <- the honest number")
    except Exception as e:
        check("forest metadata readable", False,
              f"{type(e).__name__}: {e}", level="warn")
else:
    check("forest and metadata both present",
          os.path.isfile(RF_MODEL_PATH) and os.path.isfile(RF_META_PATH),
          "at least one is absent; 02 and 04 will refuse to reuse any cached "
          "label without them", level="warn")

# Does the pretraining set still match its manifest hash? pretraining_torch gates
# on exactly this, so checking it here is cheaper than finding out at stage 1.

# %%
if os.path.isfile(PRETRAIN_CSV) and os.path.isfile(PRETRAIN_MANIFEST):
    try:
        with open(PRETRAIN_MANIFEST) as f:
            _pm = json.load(f)
        t0 = time.time()
        _actual = sha256_file(PRETRAIN_CSV)
        _ok = _pm.get("sha256") == _actual
        check("the pretraining set matches its manifest", _ok,
              f"{_pm.get('n_molecules', '?')} molecules, "
              f"{100 * _pm.get('frac_toxic', 0):.1f}% toxic, hash verified in "
              f"{time.time() - t0:.0f}s" if _ok else
              f"manifest records {str(_pm.get('sha256'))[:16]}..., file hashes to "
              f"{_actual[:16]}... — pretraining_torch will refuse to start")
    except Exception as e:
        check("pretraining manifest readable", False, f"{type(e).__name__}: {e}",
              level="warn")
elif os.path.isfile(PRETRAIN_CSV):
    check("pretraining set has a manifest", False,
          "the CSV is there but its manifest is not, so stage 1 cannot verify "
          "what it is training on", level="warn")

# %% [markdown]
# ## Summary

# %%
print("=" * 78)
_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
for status, name, detail in sorted(CHECKS, key=lambda c: _order.get(c[0], 9)):
    print(f"  {status:5s}  {name}")
    print(f"         {detail}")

print("-" * 78)
if random_metrics:
    print(f"  Labelling RF   random  acc {random_metrics['accuracy']:.3f}  "
          f"ROC-AUC {random_metrics['roc_auc']:.3f}")
else:
    print("  Labelling RF   not fitted in this run")
if scaffold_metrics:
    print(f"                 scaffold acc {scaffold_metrics['accuracy']:.3f}  "
          f"ROC-AUC {scaffold_metrics['roc_auc']:.3f}   <- quote this one")
if eval_metrics:
    print(f"  Independent    acc {eval_metrics['accuracy']:.3f}  "
          f"ROC-AUC {eval_metrics['roc_auc']:.3f}  "
          f"agreement {100 * eval_metrics['agreement_with_rf']:.0f}%")
if BUILD_PRETRAIN_DATASET:
    print(f"  Pretraining    {stats['kept']:,} molecules, {100 * frac:.1f}% toxic")

print("-" * 78)
if WRITES:
    print("  Artifact writes this run:")
    for status, name, what in WRITES:
        print(f"    {status:9s} {name:34s} {what}")
else:
    print("  No artifact write was even attempted this run.")
if not WRITE_ARTIFACTS:
    print("  WRITE_ARTIFACTS is False, so nothing on disk was created or replaced.")

print("-" * 78)

# %%
print("""  Three toxicity numbers, three meanings:

    1. TRAINING LABELS      the RF above. What Eq. 8 optimised against.
    2. INTERNAL DIAGNOSTIC  the same RF scoring generated molecules. Partly
                            measures memorisation of its own boundary.
    3. INDEPENDENT          the GBM above, and pkCSM on the five Table-3 PDBs.
                            Only (3) is comparable to the paper's Table 3.

  The defensible claim is reduced RF-PREDICTED Ames mutagenicity. Anything
  stronger needs experimental labels this repository does not have.""")

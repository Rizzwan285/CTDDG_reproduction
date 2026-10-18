# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ames mutagenicity labeller + the definitive pretraining dataset
#
# Two jobs, in this order:
#
# 1. **Train the Ames Random Forest** that supplies `Toxic_class` to Eq. 8 in both
#    training phases, and measure it honestly.
# 2. **Build the labelled ChEMBL pretraining set** - salt-stripped, atom-capped,
#    BindingDB ligands removed by canonical structure - with a manifest and a
#    SHA-256 so stage 1 can refuse to train on the wrong file.
#
# ## What changed from the previous version
#
# | # | Change | Why |
# |---|---|---|
# | 1 | **Scaffold split reported alongside the random split** | A random split on Morgan fingerprints puts analogue series on both sides, so 0.811 / 0.890 is optimistic. The scaffold number is what any RF-derived figure is actually worth. |
# | 2 | **A second, independent classifier** (HistGradientBoosting on MACCS + physchem) | The RF writes the training labels *and* grades the generated molecules. A different algorithm on different features breaks that circularity for evaluation. |
# | 3 | **SHA-256 + manifest written beside every artifact** | `02`'s `RF_AUTO_TRAIN` can silently rebuild the forest, leaving ChEMBL labelled by one model and BindingDB by another. The hash makes that detectable. |
# | 4 | **The filtered pretraining set is built here** | `chembl.txt` has no labels and `chembl_with_ames_predictions.csv` is ambiguous. One file, built once, hashed. |
# | 5 | **Salt stripping on ChEMBL** | A disconnected graph silently corrupts its training target (see `00_phase0_diagnostics.py` check E). |
# | 6 | **Leakage verified on canonical structures, not row counts** | A row count cannot establish that the exclusion worked. |
#
# The RF's hyperparameters and the merge/dedup order are **unchanged**, so the main
# model is bit-identical to the one that wrote the existing labels.

# %%
import os
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

HANSEN_PATHS = ((os.path.join(DATASETS_DIR, "Mutagenicity_N7090.csv"), "\t"),
                (os.path.join(DATASETS_DIR, "Mutagenicity_N6512.csv"), ","))

RF_MODEL_PATH = os.path.join(MODELS_DIR, "ames_random_forest.pkl")
RF_META_PATH = os.path.join(MODELS_DIR, "ames_random_forest.meta.json")
FP_CONFIG_PATH = os.path.join(MODELS_DIR, "fingerprint_config.pkl")

# The independent evaluator. Different algorithm, different features, same data -
# so it does not share the RF's decision boundary.
EVAL_MODEL_PATH = os.path.join(MODELS_DIR, "ames_independent_gbm.pkl")
EVAL_META_PATH = os.path.join(MODELS_DIR, "ames_independent_gbm.meta.json")

# Inputs and outputs for the pretraining dataset.
CHEMBL_RAW_CSV = os.path.join(CHEMBL_DIR, "chembl.csv")
CHEMBL_ANNOTATED = os.path.join(CHEMBL_DIR, "chembl_with_ames_predictions.csv")
PRETRAIN_CSV = os.path.join(CHEMBL_DIR, "chembl_pretrain_labelled.csv")
PRETRAIN_MANIFEST = PRETRAIN_CSV + ".manifest.json"

FOLD_SUFFIX = {"d1": "4_org_1042_104", "d2": "4_org_1000_112", "d3": "4_org_1004_122",
               "d4": "4_org_1002_103", "d5": "4_org_1036_124"}

FP_RADIUS = 2
FP_NBITS = 2048
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 500

MAX_ATOMS = 100          # CDGCN/CTDDG preprocessing cap
MAX_ITER = 120           # graph_to_rnn width; a molecule needs GetNumBonds()+1 steps

# Stage toggles. The expensive one is the ChEMBL pass (~20 min).
RUN_SCAFFOLD_SPLIT = True
RUN_INDEPENDENT_EVALUATOR = True
BUILD_PRETRAIN_DATASET = True
# Also write the legacy unfiltered annotated export. Off by default: nothing should
# train on it, and it is 350 MB.
WRITE_LEGACY_ANNOTATED = False

CHECKS = []


def check(name, ok, detail, warn_only=False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    CHECKS.append({"name": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")
    if not ok and not warn_only:
        raise RuntimeError(f"{name}: {detail}")
    return ok


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


print(f"repo    {REPO_ROOT}")
print(f"models  {MODELS_DIR}")

# %%
# Preflight: fail here, not after an hour of fingerprinting.
_missing = [p for p, _ in HANSEN_PATHS if not os.path.isfile(p)]
check("hansen inputs", not _missing,
      "both Hansen sets present" if not _missing else f"missing {_missing}")

if BUILD_PRETRAIN_DATASET:
    check("chembl.csv", os.path.isfile(CHEMBL_RAW_CSV),
          CHEMBL_RAW_CSV if os.path.isfile(CHEMBL_RAW_CSV) else f"missing {CHEMBL_RAW_CSV}")

# %% [markdown]
# ## 1. Load and merge the Hansen sets
#
# Merge and dedup order is **unchanged** from the original notebook, so the model
# trained below is the same one that wrote the existing ChEMBL labels. The
# parseability filter stays where it was - *after* dedup - because moving it
# earlier shifts the dedup count and therefore the split.

# %%
frames = []
for path, sep in HANSEN_PATHS:
    df = pd.read_csv(path, sep=sep)
    frames.append(df)
    print(f"  {os.path.basename(path):28s} {df.shape}")

hansen = pd.concat(frames, ignore_index=True)
print(f"\n  concatenated              {hansen.shape}")

hansen = hansen.drop_duplicates()
print(f"  after exact dedup         {hansen.shape}")

hansen = hansen.drop_duplicates(subset=["Canonical_Smiles"], keep="first").reset_index(drop=True)
print(f"  after unique-SMILES dedup {hansen.shape}")

hansen = hansen.dropna(subset=["Canonical_Smiles", "Activity"]).reset_index(drop=True)
print(f"  after dropna              {hansen.shape}")

hansen["Mol"] = hansen["Canonical_Smiles"].apply(Chem.MolFromSmiles)
n_bad = int(hansen["Mol"].isna().sum())
hansen = hansen[hansen["Mol"].notna()].reset_index(drop=True)
hansen["Activity"] = hansen["Activity"].astype(int)

print(f"  unparseable dropped       {n_bad}")
print(f"  final                     {len(hansen)}")
print(f"  class balance             {dict(hansen['Activity'].value_counts())}")

check("hansen dedup reproduces the original", len(hansen) == 7083,
      f"{len(hansen)} molecules (original run: 7083)",
      warn_only=True)

# %% [markdown]
# ### Label semantics - easy to get backwards
#
# Hansen `Activity = 1` -> Ames positive -> mutagenic -> **toxic**. So the RF's `1`
# *is* `Toxic_class = 1` in Eq. 8, with no inversion anywhere.

# %%
check("label semantics", set(hansen["Activity"].unique()) <= {0, 1},
      "Activity in {0,1}; 1 = Ames positive = toxic = Toxic_class 1 in Eq. 8")

# %% [markdown]
# ## 2. Morgan fingerprints

# %%
def morgan_matrix(mols, radius=FP_RADIUS, n_bits=FP_NBITS):
    """[n, n_bits] uint8. Rows for None stay zero."""
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp_of = gen.GetFingerprint
    except (ImportError, AttributeError):
        fp_of = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)

    out = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, m in enumerate(mols):
        if m is not None:
            DataStructs.ConvertToNumpyArray(fp_of(m), out[i])
    return out


t0 = time.time()
X = morgan_matrix(hansen["Mol"].tolist())
y = hansen["Activity"].to_numpy()
print(f"  X {X.shape}  y {y.shape}  [{time.time() - t0:.1f}s]")

check("feature matrix", X.shape == (len(hansen), FP_NBITS),
      f"{X.shape[0]} x {X.shape[1]} Morgan r{FP_RADIUS}")
check("fingerprints non-degenerate", X.sum() > 0 and X.sum(axis=1).min() > 0,
      f"mean {X.sum(axis=1).mean():.1f} bits set per molecule")

# %% [markdown]
# ## 3a. Random split - the published number
#
# Reproduced exactly so the model and its reported metrics stay comparable to
# everything already on disk.

# %%
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"  train {X_tr.shape[0]}  test {X_te.shape[0]}")

rf_model = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                  random_state=RANDOM_STATE, n_jobs=-1)
t0 = time.time()
rf_model.fit(X_tr, y_tr)
print(f"  fitted in {time.time() - t0:.1f}s")

y_pred = rf_model.predict(X_te)
y_prob = rf_model.predict_proba(X_te)[:, 1]

random_metrics = {
    "split": "random stratified 80/20",
    "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
    "accuracy": float(accuracy_score(y_te, y_pred)),
    "precision": float(precision_score(y_te, y_pred)),
    "recall": float(recall_score(y_te, y_pred)),
    "f1": float(f1_score(y_te, y_pred)),
    "roc_auc": float(roc_auc_score(y_te, y_prob)),
    "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
}
for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
    print(f"  {k:10s} {random_metrics[k]:.4f}")
print(f"  confusion  {random_metrics['confusion_matrix']}")
print()
print(classification_report(y_te, y_pred, target_names=["Ames Negative", "Ames Positive"]))

check("random-split metrics reproduce the original",
      abs(random_metrics["accuracy"] - 0.811) < 0.01
      and abs(random_metrics["roc_auc"] - 0.890) < 0.01,
      f"acc {random_metrics['accuracy']:.3f} (was 0.811), "
      f"ROC-AUC {random_metrics['roc_auc']:.3f} (was 0.890)",
      warn_only=True)

# %% [markdown]
# ## 3b. Scaffold split - the honest number
#
# Bemis-Murcko scaffolds are assigned whole to one side, so an analogue series
# cannot straddle the split. The gap between this and the random number is how
# much of the random result was memorised series, and it sets how much weight any
# RF-derived toxicity figure can carry.

# %%
def scaffold_split(mols, test_size=TEST_SIZE, seed=RANDOM_STATE):
    """Deterministic scaffold split: largest scaffold groups fill train first."""
    groups = {}
    for i, m in enumerate(mols):
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            s = ""
        groups.setdefault(s, []).append(i)

    # Big groups first, ties broken by scaffold string so the result is stable.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n_test_target = int(round(test_size * len(mols)))
    test_idx, train_idx = [], []
    for _, idxs in ordered:
        if len(test_idx) < n_test_target and len(idxs) <= max(1, n_test_target // 4):
            test_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx)), len(groups)


scaffold_metrics = None
if RUN_SCAFFOLD_SPLIT:
    tr_i, te_i, n_scaffolds = scaffold_split(hansen["Mol"].tolist())
    print(f"  {n_scaffolds} distinct Murcko scaffolds")
    print(f"  train {len(tr_i)}  test {len(te_i)}")

    _sc_tr = {MurckoScaffold.MurckoScaffoldSmiles(mol=hansen['Mol'][i],
                                                  includeChirality=False)
              for i in tr_i}
    _sc_te = {MurckoScaffold.MurckoScaffoldSmiles(mol=hansen['Mol'][i],
                                                  includeChirality=False)
              for i in te_i}
    check("scaffold split is disjoint", len(_sc_tr & _sc_te) == 0,
          f"{len(_sc_tr & _sc_te)} scaffolds shared between the two sides")

    rf_scaffold = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                         random_state=RANDOM_STATE, n_jobs=-1)
    rf_scaffold.fit(X[tr_i], y[tr_i])
    sp = rf_scaffold.predict(X[te_i])
    spp = rf_scaffold.predict_proba(X[te_i])[:, 1]

    scaffold_metrics = {
        "split": "Bemis-Murcko scaffold 80/20",
        "n_train": int(len(tr_i)), "n_test": int(len(te_i)),
        "n_scaffolds": int(n_scaffolds),
        "accuracy": float(accuracy_score(y[te_i], sp)),
        "precision": float(precision_score(y[te_i], sp, zero_division=0)),
        "recall": float(recall_score(y[te_i], sp, zero_division=0)),
        "f1": float(f1_score(y[te_i], sp, zero_division=0)),
        "roc_auc": float(roc_auc_score(y[te_i], spp)),
        "confusion_matrix": confusion_matrix(y[te_i], sp).tolist(),
    }
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:10s} {scaffold_metrics[k]:.4f}   "
              f"(random split: {random_metrics[k]:.4f})")

    _drop = random_metrics["roc_auc"] - scaffold_metrics["roc_auc"]
    check("scaffold generalisation", _drop < 0.10,
          f"ROC-AUC drops {_drop:.3f} from random to scaffold split. "
          f"{'Modest - the RF generalises to new chemotypes' if _drop < 0.10 else 'LARGE - the random-split number was substantially memorised series; quote the scaffold number in the thesis'}",
          warn_only=True)
    del rf_scaffold

# %% [markdown]
# ## 4. Save the labelling RF
#
# `rf_model` - the random-split model - is the one that labels ChEMBL and
# BindingDB, because that is what the existing checkpoints and caches were built
# against. The scaffold model above is a measurement, not a replacement.

# %%
fingerprint_config = {"type": "Morgan", "radius": FP_RADIUS, "n_bits": FP_NBITS}
joblib.dump(fingerprint_config, FP_CONFIG_PATH)
joblib.dump(rf_model, RF_MODEL_PATH)

rf_meta = {
    "role": "training labeller - writes Toxic_class for Eq. 8 in BOTH phases",
    "sklearn": sklearn.__version__,
    "rdkit": rdkit.__version__,
    "n_molecules": int(len(hansen)),
    "radius": FP_RADIUS, "n_bits": FP_NBITS,
    "n_estimators": N_ESTIMATORS, "random_state": RANDOM_STATE,
    "random_split": random_metrics,
    "scaffold_split": scaffold_metrics,
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
rf_meta["sha256"] = sha256_file(RF_MODEL_PATH)
with open(RF_META_PATH, "w") as f:
    json.dump(rf_meta, f, indent=1)

print(f"  {RF_MODEL_PATH}")
print(f"  sha256 {rf_meta['sha256'][:32]}...")
print(f"  {RF_META_PATH}")
print()
print("  NOTE: this hash is what 02_finetune_ctddg's label cache records. If you")
print("  re-run this cell, every cached BindingDB label is invalidated and will be")
print("  rebuilt - and the ChEMBL labels below must be rebuilt too, or the two")
print("  phases end up optimising against two different definitions of toxic.")

# %% [markdown]
# ## 5. The independent evaluator
#
# The RF above writes the training labels. Scoring generated molecules with it
# measures how well the generator memorised that forest's decision boundary, not
# how toxic the molecules are. This second model uses a **different algorithm**
# (histogram gradient boosting) on **different features** (MACCS keys plus
# physicochemical descriptors), so it shares the training data but not the
# boundary.
#
# It replaces nothing: report it as a separate row beside the training labeller,
# and use pkCSM on the five Table-3 PDBs for the paper-comparable number.

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


def independent_features(mols):
    """MACCS (167) + physchem (15). Disjoint from the RF's Morgan features."""
    n = len(mols)
    out = np.full((n, 167 + len(_PHYSCHEM)), np.nan, dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        bits = np.zeros((167,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), bits)
        out[i, :167] = bits
        for j, (_, fn) in enumerate(_PHYSCHEM):
            try:
                out[i, 167 + j] = float(fn(m))
            except Exception:
                pass                       # HistGradientBoosting handles NaN natively
    return out


eval_metrics = None
if RUN_INDEPENDENT_EVALUATOR:
    t0 = time.time()
    Xi = independent_features(hansen["Mol"].tolist())
    print(f"  independent features {Xi.shape}  [{time.time() - t0:.1f}s]")

    # Use the SAME random split indices so the two models are directly comparable.
    _idx = np.arange(len(hansen))
    i_tr, i_te = train_test_split(_idx, test_size=TEST_SIZE,
                                  random_state=RANDOM_STATE, stratify=y)

    gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07,
                                         random_state=RANDOM_STATE)
    gbm.fit(Xi[i_tr], y[i_tr])
    gp = gbm.predict(Xi[i_te])
    gpp = gbm.predict_proba(Xi[i_te])[:, 1]

    eval_metrics = {
        "role": "independent evaluator - NEVER used for training labels",
        "model": "HistGradientBoostingClassifier",
        "features": "MACCS 167 bits + 15 physicochemical descriptors",
        # 04_evaluate_metrics.py rebuilds this feature vector from SMILES and checks
        # this spec before predicting. If the descriptor list or its ORDER ever
        # drifts between the two files, the model would be scoring on features it
        # was never trained on - silently, with plausible-looking output.
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

    # Agreement between the two models - how independent is "independent"?
    rf_on_te = rf_model.predict(X[i_te])
    agree = float((rf_on_te == gp).mean())
    eval_metrics["agreement_with_rf"] = agree
    print(f"  agreement with the labelling RF on the test set: {100 * agree:.1f}%")

    joblib.dump(gbm, EVAL_MODEL_PATH)
    eval_metrics["sha256"] = sha256_file(EVAL_MODEL_PATH)
    with open(EVAL_META_PATH, "w") as f:
        json.dump(eval_metrics, f, indent=1)
    print(f"  {EVAL_MODEL_PATH}")

    check("evaluator is genuinely independent", agree < 0.95,
          f"{100 * agree:.1f}% agreement with the RF - "
          f"{'usefully different' if agree < 0.95 else 'nearly identical, so it adds little'}",
          warn_only=True)
    del Xi

# %% [markdown]
# ## 6. The definitive pretraining dataset
#
# Built once, here, from `chembl.csv`, in the order CDGCN's preprocessing used:
#
# 1. parse SMILES, drop unparseable
# 2. **strip salts** - largest fragment, matching `data_preprocessing_1`
# 3. drop molecules with more than `MAX_ATOMS` atoms
# 4. drop molecules needing more than `MAX_ITER` generation steps (`bonds + 1`),
#    which the stage-1 collator cannot write into `graph_to_rnn` - this is the
#    bond cap that was missing
# 5. **remove the BindingDB ligands by canonical structure**, from all five folds
# 6. apply the Ames RF
# 7. write `chembl_pretrain_labelled.csv` plus a manifest carrying the SHA-256,
#    the counts at each step, and the RF's own hash
#
# Step 5 is the one both papers call out explicitly, and it is compared on
# canonical SMILES rather than raw strings, which a row count cannot establish.

# %%
def load_bindingdb_ligands():
    """Canonical SMILES of every BindingDB ligand across all folds, salts stripped."""
    canon, raw_lines = set(), 0
    for fold, suffix in FOLD_SUFFIX.items():
        for split in ("train", "test"):
            p = os.path.join(BINDINGDB_DIR, f"{split}_dataset", f"{split}_{suffix}",
                             f"ligands_{split}_4_org_corrected")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                for line in f:
                    s = "".join(line.split())
                    if not s:
                        continue
                    raw_lines += 1
                    if '.' in s:
                        s = max(s.split('.'), key=len)
                    m = Chem.MolFromSmiles(s)
                    if m is not None:
                        canon.add(Chem.MolToSmiles(m))
    return canon, raw_lines


if BUILD_PRETRAIN_DATASET:
    t0 = time.time()
    bdb_canon, bdb_lines = load_bindingdb_ligands()
    print(f"  BindingDB: {bdb_lines:,} ligand lines -> {len(bdb_canon):,} unique "
          f"canonical structures  [{time.time() - t0:.0f}s]")
    check("BindingDB ligands loaded", len(bdb_canon) > 0,
          f"{len(bdb_canon):,} unique ligands to exclude from ChEMBL")

# %%
if BUILD_PRETRAIN_DATASET:
    t0 = time.time()
    chembl = pd.read_csv(CHEMBL_RAW_CSV, sep=";", quotechar='"', usecols=["Smiles"])
    n_raw = len(chembl)
    print(f"  chembl.csv: {n_raw:,} rows  [{time.time() - t0:.0f}s]")

    stats = {"raw": n_raw, "unparseable": 0, "salt_stripped": 0,
             "too_many_atoms": 0, "too_many_bonds": 0, "bindingdb_overlap": 0,
             "duplicate": 0}

    kept_smiles, seen = [], set()
    t0 = time.time()
    for i, s in enumerate(chembl["Smiles"].astype(str)):
        if (i + 1) % 200_000 == 0:
            print(f"    {i + 1:,}/{n_raw:,}  kept {len(kept_smiles):,}  "
                  f"[{time.time() - t0:.0f}s]")
        s = s.strip()
        if not s:
            stats["unparseable"] += 1
            continue
        if '.' in s:
            s = max(s.split('.'), key=len)
            stats["salt_stripped"] += 1
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
        c = Chem.MolToSmiles(m)
        if c in bdb_canon:
            stats["bindingdb_overlap"] += 1
            continue
        if c in seen:
            stats["duplicate"] += 1
            continue
        seen.add(c)
        kept_smiles.append(c)

    print(f"\n  raw                 {stats['raw']:>10,}")
    for k in ("unparseable", "salt_stripped", "too_many_atoms", "too_many_bonds",
              "bindingdb_overlap", "duplicate"):
        print(f"  {k:<19s} {stats[k]:>10,}")
    print(f"  kept                {len(kept_smiles):>10,}")
    stats["kept"] = len(kept_smiles)

    check("pretraining set non-empty", len(kept_smiles) > 100_000,
          f"{len(kept_smiles):,} molecules survive filtering")
    check("BindingDB ligands excluded", stats["bindingdb_overlap"] > 0,
          f"{stats['bindingdb_overlap']:,} ChEMBL molecules matched a BindingDB ligand "
          f"and were removed. This is the leakage both papers eliminate",
          warn_only=True)

# %%
if BUILD_PRETRAIN_DATASET:
    # Label in batches so the fingerprint matrix never exceeds a few hundred MB.
    BATCH = 50_000
    labels = np.full(len(kept_smiles), -1, dtype=np.int8)
    t0 = time.time()
    for start in range(0, len(kept_smiles), BATCH):
        end = min(start + BATCH, len(kept_smiles))
        mols = [Chem.MolFromSmiles(s) for s in kept_smiles[start:end]]
        Xb = morgan_matrix(mols)
        labels[start:end] = rf_model.predict(Xb).astype(np.int8)
        print(f"    labelled {end:,}/{len(kept_smiles):,}  [{time.time() - t0:.0f}s]")

    check("every molecule labelled", int((labels < 0).sum()) == 0,
          f"{int((labels < 0).sum())} molecules without a label")

    n_tox = int((labels == 1).sum())
    frac = n_tox / len(labels)
    print(f"\n  toxic     {n_tox:,}  ({100 * frac:.1f}%)")
    print(f"  non-toxic {len(labels) - n_tox:,}")

    check("toxic fraction is plausible", 0.15 < frac < 0.35,
          f"{100 * frac:.1f}% toxic (the earlier unfiltered export was 23.8%)",
          warn_only=True)

    out = pd.DataFrame({"Smiles": kept_smiles, "Ames_prediction": labels.astype(int)})
    out.to_csv(PRETRAIN_CSV, index=False)

    manifest = {
        "purpose": "stage-1 pretraining input - filtered, deduplicated, labelled",
        "path": PRETRAIN_CSV,
        "sha256": sha256_file(PRETRAIN_CSV),
        "n_molecules": int(len(out)),
        "n_toxic": n_tox,
        "frac_toxic": round(frac, 5),
        "filters": {"salts_stripped_to_largest_fragment": True,
                    "max_atoms": MAX_ATOMS, "max_generation_steps": MAX_ITER,
                    "bindingdb_ligands_removed": True,
                    "deduplicated_on_canonical_smiles": True},
        "counts": stats,
        "source": {"chembl_csv": CHEMBL_RAW_CSV,
                   "n_bindingdb_ligands_excluded": len(bdb_canon)},
        "labeller": {"path": RF_MODEL_PATH, "sha256": rf_meta["sha256"],
                     "radius": FP_RADIUS, "n_bits": FP_NBITS},
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(PRETRAIN_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"\n  {PRETRAIN_CSV}")
    print(f"  sha256 {manifest['sha256'][:32]}...")
    print(f"  {PRETRAIN_MANIFEST}")
    print()
    print("  Point pretraining_torch.py's DATA_PATH at this file. It verifies the hash")
    print("  against this manifest and refuses to start on a mismatch.")

# %% [markdown]
# ### Verify the exclusion actually worked
#
# The filter ran on canonical SMILES, so this should be exactly zero. Checking it
# costs seconds and is the difference between believing the leakage is gone and
# knowing it.

# %%
if BUILD_PRETRAIN_DATASET:
    leak = set(kept_smiles) & bdb_canon
    check("no BindingDB ligand survives in the pretraining set", len(leak) == 0,
          f"{len(leak)} overlapping canonical structures"
          + (f", e.g. {list(leak)[:3]}" if leak else ""))

# %% [markdown]
# ## 7. Legacy annotated export (optional)
#
# The unfiltered 1.2M-row `chembl_with_ames_predictions.csv`. Nothing should train
# on it - it is kept only so older tooling that references the name still resolves.

# %%
if WRITE_LEGACY_ANNOTATED:
    t0 = time.time()
    legacy = pd.read_csv(CHEMBL_RAW_CSV, sep=";", quotechar='"')
    mols = [Chem.MolFromSmiles(s) for s in legacy["Smiles"].astype(str)]
    preds = np.full(len(legacy), np.nan)
    for start in range(0, len(legacy), 50_000):
        end = min(start + 50_000, len(legacy))
        sub = mols[start:end]
        ok = [j for j, m in enumerate(sub) if m is not None]
        if ok:
            Xb = morgan_matrix([sub[j] for j in ok])
            preds[start + np.array(ok)] = rf_model.predict(Xb)
        print(f"    {end:,}/{len(legacy):,}  [{time.time() - t0:.0f}s]")
    legacy["Ames_prediction"] = preds
    legacy = legacy.dropna(subset=["Ames_prediction"])
    legacy.to_csv(CHEMBL_ANNOTATED, index=False)
    print(f"  {CHEMBL_ANNOTATED}  ({len(legacy):,} rows)")
    print("  WARNING: unfiltered. Contains the BindingDB ligands. Do not train on it.")
else:
    print("  skipped (WRITE_LEGACY_ANNOTATED = False)")

# %% [markdown]
# ## Summary

# %%
print("=" * 78)
print("AMES LABELLER + PRETRAINING DATASET")
print("=" * 78)
for c in CHECKS:
    print(f"  {c['status']:5s}  {c['name']}")
    print(f"         {c['detail']}")

print("-" * 78)
print("  Labelling RF (writes Toxic_class for Eq. 8 in both phases)")
print(f"    random split    acc {random_metrics['accuracy']:.3f}  "
      f"ROC-AUC {random_metrics['roc_auc']:.3f}")
if scaffold_metrics:
    print(f"    scaffold split  acc {scaffold_metrics['accuracy']:.3f}  "
          f"ROC-AUC {scaffold_metrics['roc_auc']:.3f}   <- quote THIS one")
if eval_metrics:
    print("  Independent evaluator (grades generated molecules, never trains anything)")
    print(f"    random split    acc {eval_metrics['accuracy']:.3f}  "
          f"ROC-AUC {eval_metrics['roc_auc']:.3f}  "
          f"agreement with RF {100 * eval_metrics['agreement_with_rf']:.0f}%")
if BUILD_PRETRAIN_DATASET:
    print(f"  Pretraining set   {stats['kept']:,} molecules, {100 * frac:.1f}% toxic")
    print(f"                    {PRETRAIN_CSV}")

print("-" * 78)
print("""  Three toxicity numbers, three meanings - keep them apart in the write-up:

    1. TRAINING LABELS      the RF above. What Eq. 8 optimised against.
    2. INTERNAL DIAGNOSTIC  the same RF scoring generated molecules. Partly
                            measures memorisation of its own boundary; useful
                            for tracking a run, not for a claim.
    3. INDEPENDENT          the GBM above, and pkCSM on the five Table-3 PDBs.
                            Only (3) is comparable to the paper's Table 3.

  pkCSM is smaller than it looks: Table 3 is 5 PDBs x 10 runs x 100 requested,
  so roughly 1,000 unique molecules after dedup - a handful of batch uploads, not
  the full 24,787. 04_evaluate_metrics.py writes exactly that subset.""")

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
# # 04 - Evaluate generated molecules against the paper's tables
#
# Reads `outputs/samples/<fold>/run*/protein_*.csv` from `03_generate_samples.py`.
#
# | Paper | Here |
# |---|---|
# | **Table 1** drug-like properties | fully reproducible |
# | **Table 2** target awareness (docking) | `05_docking_table2.py` - needs SMINA + receptors |
# | **Table 3** % toxic by **pkCSM** | substituted, plus a *scoped* pkCSM upload file |
# | CDGCN Table 1 novelty, generation time | computed here as extras |
#
# ## What changed
#
# | # | Change | Reason |
# |---|---|---|
# | 1 | **Table 1 reports BOTH denominators** | Every row was capped at the validity rate, which made a single failed metric look like ten. Per-valid-molecule, the previous run matched or beat the paper on four of eight property rows. |
# | 2 | **The caveat list is derived from artifacts, not hardcoded** | The old report asserted stage-1 leakage from a constant while the pretraining output said otherwise. It now reads the dataset manifest, the split manifest and the checkpoint. |
# | 3 | **Independent toxicity evaluator reported separately** | The RF writes the training labels *and* grades the output. A second model (different algorithm, different features) breaks that circularity. |
# | 4 | **pkCSM upload scoped to the five Table-3 PDBs** | ~1,000 unique molecules instead of ~25,000: a handful of batch uploads, not a manual grind. |
# | 5 | **Failure census surfaced** | `03` now records why each sample failed; this reads it rather than only counting empties. |
#
# ## Aggregation
#
# Follows the paper: a percentage is computed per protein, averaged over the
# held-out proteins to give one number per run, then reported as mean +/- std over
# the 10 runs. That is what makes the published error bars as small as +/-0.3.
#
# **Denominator.** The paper's own numbers fix it: in Table 1 validity equals
# uniqueness for CDGCN and CTDDG, and every SSVAE property row equals its
# validity x uniqueness (0.884 x 58.1 = 51.4). So the headline column uses
# *molecules requested* (100). The per-valid column is reported beside it because
# it separates "how often does the model emit a molecule at all" from "when it
# does, is the molecule drug-like" - two different failures that the single
# column conflates.

# %%
import os
import re
import csv
import glob
import json
import time
import pickle
import hashlib
import warnings
from collections import Counter

import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, MACCSkeys, DataStructs

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # noqa: E402

print(f"rdkit {rdkit.__version__}")

# %% [markdown]
# ## Config

# %%
REPO_ROOT = os.environ.get("CTDDG_REPO_ROOT", "/home/muhamed/repo/CTDDG")
FOLD = os.environ.get("CTDDG_FOLD", "d1")

SAMPLES_DIR = os.path.join(REPO_ROOT, "outputs/samples", FOLD)
FINETUNE_DIR = os.path.join(REPO_ROOT, "outputs/finetune", FOLD)
BINDINGDB_DIR = os.path.join(REPO_ROOT, "datasets/data/bindingdb")
CHEMBL_DIR = os.path.join(REPO_ROOT, "datasets/data/chembl")

RF_MODEL_PATH = os.path.join(REPO_ROOT, "models/ames_random_forest.pkl")
FP_CONFIG_PATH = os.path.join(REPO_ROOT, "models/fingerprint_config.pkl")
EVAL_MODEL_PATH = os.path.join(REPO_ROOT, "models/ames_independent_gbm.pkl")
RF_META_PATH = os.path.join(REPO_ROOT, "models/ames_random_forest.meta.json")
TRAIN_AMES_META = os.path.join(BINDINGDB_DIR, "ligand_ames_labels.csv.meta.json")

PRETRAIN_MANIFEST = os.path.join(CHEMBL_DIR, "chembl_pretrain_labelled.csv.manifest.json")
SPLIT_MANIFEST = os.path.join(FINETUNE_DIR, "split_manifest.json")
FINETUNE_CKPT = os.path.join(FINETUNE_DIR, "ckpt_best.pt")

EVAL_DIR = os.path.join(REPO_ROOT, "outputs/evaluation", FOLD)
os.makedirs(EVAL_DIR, exist_ok=True)

GEN_AMES_CACHE = os.path.join(EVAL_DIR, "generated_ames_labels.csv")
GEN_GBM_CACHE = os.path.join(EVAL_DIR, "generated_independent_labels.csv")
NOVELTY_CACHE = os.path.join(EVAL_DIR, f"{FOLD}_dataset_canonical_smiles.txt")
PDB_MAP_PATH = os.path.join(EVAL_DIR, "pdb_protein_map.json")

COMPUTE_TOXICITY = True
COMPUTE_INDEPENDENT = True   # the second, non-circular evaluator
COMPUTE_NOVELTY = True

FOLD_SUFFIX = {"d1": "4_org_1042_104", "d2": "4_org_1000_112", "d3": "4_org_1004_122",
               "d4": "4_org_1002_103", "d5": "4_org_1036_124"}

THRESHOLDS = {
    'logp_lt_5':   ('LogP < 5',               lambda d: d['logp'] < 5),
    'mw_lt_500':   ('Molecular weight < 500', lambda d: d['mw'] < 500),
    'hbd_lt_5':    ('H-bond donors < 5',      lambda d: d['hbd'] < 5),
    'hba_lt_10':   ('H-bond acceptors < 10',  lambda d: d['hba'] < 10),
    'rotb_lt_10':  ('Rotatable bonds < 10',   lambda d: d['rotb'] < 10),
    'tpsa_lt_140': ('TPSA < 140',             lambda d: d['tpsa'] < 140),
    'sas_lt_6':    ('SAS < 6',                lambda d: d['sas'] < 6),
}

PAPER_TABLE1 = {
    'validity':    {'CDGCN': (91.7, 0.22), 'SSVAE': (88.4, 3.29), 'CTDDG': (94.4, 0.32)},
    'uniqueness':  {'CDGCN': (91.7, 0.22), 'SSVAE': (58.1, 4.13), 'CTDDG': (94.4, 0.32)},
    'logp_lt_5':   {'CDGCN': (68.6, 0.38), 'SSVAE': (51.4, 4.31), 'CTDDG': (74.8, 0.33)},
    'mw_lt_500':   {'CDGCN': (73.5, 0.47), 'SSVAE': (51.4, 4.31), 'CTDDG': (71.1, 0.32)},
    'hbd_lt_5':    {'CDGCN': (84.6, 0.25), 'SSVAE': (51.4, 4.31), 'CTDDG': (87.9, 0.40)},
    'hba_lt_10':   {'CDGCN': (84.5, 0.28), 'SSVAE': (51.4, 4.31), 'CTDDG': (87.2, 0.43)},
    'rotb_lt_10':  {'CDGCN': (80.3, 0.40), 'SSVAE': (51.4, 4.31), 'CTDDG': (79.9, 0.38)},
    'tpsa_lt_140': {'CDGCN': (83.5, 0.23), 'SSVAE': (51.4, 4.31), 'CTDDG': (84.1, 0.36)},
    'qed':         {'CDGCN': (0.55, 0.00), 'SSVAE': (0.44, 0.02), 'CTDDG': (0.51, 0.00)},
    'sas_lt_6':    {'CDGCN': (84.7, 0.23), 'SSVAE': (51.4, 4.31), 'CTDDG': (88.6, 0.37)},
}

PAPER_TABLE3 = {'4nos': 19.84, '2nru': 23.32, '1lqf': 20.26, '1k2r': 20.02,
                '1cqp': 13.17, 'Avg': 19.32}
PAPER_TABLE3_CDGCN = {'4nos': 24.13, '2nru': 26.89, '1lqf': 23.12, '1k2r': 22.49,
                      '1cqp': 19.88, 'Avg': 23.30}

ROW_ORDER = ['validity', 'uniqueness', 'logp_lt_5', 'mw_lt_500', 'hbd_lt_5',
             'hba_lt_10', 'rotb_lt_10', 'tpsa_lt_140', 'qed', 'sas_lt_6']
ROW_LABEL = {
    'validity': 'Validity (%)', 'uniqueness': 'Uniqueness (%)',
    'logp_lt_5': 'LogP < 5 (%)', 'mw_lt_500': 'Molecular weight < 500 (%)',
    'hbd_lt_5': 'H-bond donors < 5 (%)', 'hba_lt_10': 'H-bond acceptors < 10 (%)',
    'rotb_lt_10': 'Rotatable bonds < 10 (%)', 'tpsa_lt_140': 'TPSA < 140 (%)',
    'qed': 'QED (absolute)', 'sas_lt_6': 'SAS < 6 (%)',
}
# Rows where the per-valid-molecule normalisation is meaningful. Validity and
# uniqueness are about emitting a molecule at all, and QED is already a mean over
# valid molecules.
PER_VALID_ROWS = [k for k in ROW_ORDER if k not in ('validity', 'uniqueness', 'qed')]

CHECKS = []


def check(name, ok, detail, warn_only=True):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    CHECKS.append({"name": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")
    if not ok and not warn_only:
        raise RuntimeError(f"{name}: {detail}")
    return ok


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# %% [markdown]
# ## Provenance
#
# The caveats in the final report are built from these artifacts rather than
# hardcoded. The previous report asserted stage-1 leakage from a constant while the
# pretraining notebook's own output (`Total molecules loaded: 1090529`, the
# *filtered* count) said otherwise.

# %%
CAVEATS = []

PRETRAIN_MAN = _load_json(PRETRAIN_MANIFEST)
SPLIT_MAN = _load_json(SPLIT_MANIFEST)

if PRETRAIN_MAN:
    _f = PRETRAIN_MAN.get("filters", {})
    print(f"  pretraining set: {PRETRAIN_MAN.get('n_molecules'):,} molecules, "
          f"{100 * PRETRAIN_MAN.get('frac_toxic', 0):.1f}% toxic")
    if _f.get("bindingdb_ligands_removed"):
        check("stage-1 leakage", True,
              f"the manifest records BindingDB ligands removed "
              f"({PRETRAIN_MAN.get('counts', {}).get('bindingdb_overlap', '?')} "
              f"ChEMBL molecules excluded)")
    else:
        check("stage-1 leakage", False,
              "the manifest does NOT record the BindingDB exclusion")
        CAVEATS.append(
            "Stage 1 trained on a ChEMBL set whose manifest does not record the "
            "BindingDB exclusion, so held-out ligands may have leaked into "
            "pretraining. Both papers remove them.")
else:
    check("pretraining manifest", False,
          f"no manifest at {PRETRAIN_MANIFEST} - stage-1 data provenance is "
          f"UNVERIFIED. Build the dataset with random_forest_classifier.py")
    CAVEATS.append(
        "Stage-1 data provenance is unverified: no dataset manifest was found, so "
        "whether the BindingDB ligands were excluded from pretraining cannot be "
        "established from the artifacts.")

# %%
CKPT_META = {}
if os.path.isfile(FINETUNE_CKPT):
    import torch
    _st = torch.load(FINETUNE_CKPT, map_location='cpu', weights_only=False)
    CKPT_META = {k: _st.get(k) for k in
                 ('fold', 'epoch', 'global_counter', 'val_loss', 'val_metric',
                  'lambda_tox', 'tox_floor', 'smoke_test')} if isinstance(_st, dict) else {}
    for k, v in CKPT_META.items():
        if v is not None:
            print(f"  {k:16s} {v}")

    _vm = CKPT_META.get('val_metric')
    if _vm not in (None, 'l_nontoxic'):
        check("model selection metric", False,
              f"the checkpoint was selected on {_vm!r}, which rises when toxicity "
              f"suppression works - it picks the least-trained checkpoint")
        CAVEATS.append(
            f"The evaluated checkpoint was selected on `{_vm}`. A working CTDDG "
            f"drives that metric up by construction, so selection on it favours the "
            f"least-trained epoch. Re-run 02 with EARLY_STOP_ON = 'l_nontoxic'.")
    else:
        check("model selection metric", True, f"selected on {_vm!r}")

    if CKPT_META.get('epoch') == 1:
        CAVEATS.append(
            "Fine-tuning's best epoch was 1: validation peaked immediately and "
            "degraded after, so the model is barely fine-tuned.")
    if CKPT_META.get('smoke_test'):
        CAVEATS.append("The checkpoint is tagged smoke_test; its samples are not "
                       "reportable.")

if SPLIT_MAN:
    if not SPLIT_MAN.get('strip_salts', False):
        CAVEATS.append(
            "Fine-tuning ran without salt stripping, so every multi-fragment ligand "
            "was a silently corrupted training target (traverse_graph leaves the "
            "second component unvisited and single_reorder then mis-wires every bond).")
    _st_tr = SPLIT_MAN.get('assembly_stats', {}).get('train', {})
    if _st_tr:
        print(f"  train assembly: kept {_st_tr.get('kept'):,}, salts stripped "
              f"{_st_tr.get('salt_stripped')}, disconnected dropped "
              f"{_st_tr.get('disconnected')}")

# %% [markdown]
# ## Load generated samples

# %%
manifest_path = os.path.join(SAMPLES_DIR, "manifest.json")
if not os.path.isfile(manifest_path):
    raise FileNotFoundError(f"{manifest_path} not found - run 03_generate_samples.py "
                            f"first (with SMOKE_FIRST = False).")
with open(manifest_path) as f:
    MANIFEST = json.load(f)

N_SAMPLES = int(MANIFEST['n_samples_per_protein'])
N_RUNS = int(MANIFEST['n_runs'])

print(json.dumps({k: MANIFEST.get(k) for k in
                  ('fold', 'n_proteins_sampled', 'n_held_out_proteins_total',
                   'n_samples_per_protein', 'n_runs', 'max_decode_steps',
                   'generated_at')}, indent=1))

check("not a smoke lineage",
      not (MANIFEST.get('bundle_smoke_test')
           or MANIFEST.get('checkpoint_meta', {}).get('smoke_test')),
      "no smoke artifacts in the lineage")
check("all held-out proteins sampled",
      MANIFEST['n_proteins_sampled'] == MANIFEST['n_held_out_proteins_total'],
      f"{MANIFEST['n_proteins_sampled']} of "
      f"{MANIFEST['n_held_out_proteins_total']} proteins")

if MANIFEST.get('max_decode_steps') != MANIFEST.get('max_iter_train', 120):
    print(f"  [INFO] sampled with MAX_DECODE_STEPS = "
          f"{MANIFEST.get('max_decode_steps')} while training allowed "
          f"{MANIFEST.get('max_iter_train', 120)} (upstream behaviour)")

_fc = MANIFEST.get('failure_census')
if _fc:
    print("\n  failure census from 03:")
    for k, v in sorted(_fc.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {v:>7,}  {k}")

_ent = MANIFEST.get('first_atom_entropy')
_unif = MANIFEST.get('first_atom_uniform_entropy')
if _ent and _unif:
    check("first-atom distribution informative", _ent < 0.85 * _unif,
          f"entropy {_ent:.2f} vs uniform {_unif:.2f}")
    if _ent >= 0.85 * _unif:
        CAVEATS.append(
            f"The first-atom distribution is near-uniform over all 65 atom types "
            f"(entropy {_ent:.2f} vs uniform {_unif:.2f}), so molecules routinely "
            f"start on an exotic or charged atom. `dense_policy_0` is created fresh "
            f"at fine-tuning time and the pretrained `policy_0` is discarded.")

# %%
def load_samples():
    """-> ({(run, protein_row): [smiles ...]}, reasons Counter)."""
    out, reasons = {}, Counter()
    for run_dir in sorted(glob.glob(os.path.join(SAMPLES_DIR, "run*"))):
        run = int(re.search(r"run(\d+)$", run_dir).group(1))
        for path in sorted(glob.glob(os.path.join(run_dir, "protein_*.csv"))):
            row = int(re.search(r"protein_(\d+)\.csv$", path).group(1))
            smis = []
            with open(path) as f:
                for r in csv.DictReader(f):
                    smis.append(r['smiles'])
                    if 'reason' in r and r['reason']:
                        reasons[r['reason'].split(':')[0]] += 1
            if len(smis) != N_SAMPLES:
                print(f"  [WARN] {os.path.basename(path)} (run {run}) has {len(smis)} "
                      f"rows, expected {N_SAMPLES}")
            out[(run, row)] = smis
    return out, reasons


SAMPLES, REASONS = load_samples()
runs_present = sorted({r for r, _ in SAMPLES})
proteins_present = sorted({p for _, p in SAMPLES})
print(f"\nloaded {len(SAMPLES)} (run, protein) cells | runs {runs_present} | "
      f"{len(proteins_present)} proteins")

check("all runs present", len(runs_present) >= N_RUNS,
      f"{len(runs_present)} of {N_RUNS} runs"
      + ("" if len(runs_present) >= N_RUNS
         else "; error bars will be wider than the paper's"))

# %% [markdown]
# ## Descriptors

# %%
_DESC_CACHE = {}


def descriptors(smiles):
    """Cached per canonical SMILES. None if RDKit cannot sanitize it."""
    if smiles in _DESC_CACHE:
        return _DESC_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        _DESC_CACHE[smiles] = None
        return None
    try:
        d = {'logp': Crippen.MolLogP(mol), 'mw': Descriptors.MolWt(mol),
             'hbd': Lipinski.NumHDonors(mol), 'hba': Lipinski.NumHAcceptors(mol),
             'rotb': Lipinski.NumRotatableBonds(mol), 'tpsa': Descriptors.TPSA(mol),
             'qed': QED.qed(mol), 'sas': sascorer.calculateScore(mol),
             'n_atoms': mol.GetNumAtoms()}
    except Exception:
        d = None
    _DESC_CACHE[smiles] = d
    return d


def cell_metrics(smiles_list, n_requested=None):
    """Paper-protocol metrics for one (protein, run).

    Each property is reported twice:
      <key>           over n_requested  - the paper's denominator
      <key>_per_valid over the unique valid molecules - "given that a molecule was
                      emitted at all, is it drug-like?"
    """
    n_requested = n_requested or len(smiles_list)
    valid = [s for s in smiles_list if s and descriptors(s) is not None]
    unique = list(dict.fromkeys(valid))

    m = {'n_requested': n_requested, 'n_valid': len(valid), 'n_unique': len(unique),
         'validity': 100.0 * len(valid) / n_requested,
         'uniqueness': 100.0 * len(unique) / n_requested,
         'uniqueness_among_valid': 100.0 * len(unique) / max(1, len(valid))}

    ds = [descriptors(s) for s in unique]
    for key, (_, test) in THRESHOLDS.items():
        n_pass = sum(1 for d in ds if test(d))
        m[key] = 100.0 * n_pass / n_requested
        m[key + '_per_valid'] = 100.0 * n_pass / max(1, len(unique))
    m['qed'] = float(np.mean([d['qed'] for d in ds])) if ds else float('nan')
    m['mean_n_atoms'] = float(np.mean([d['n_atoms'] for d in ds])) if ds else float('nan')
    return m


t0 = time.time()
cells = []
for (run, row), smis in sorted(SAMPLES.items()):
    rec = {'run': run, 'protein_row': row}
    rec.update(cell_metrics(smis, N_SAMPLES))
    cells.append(rec)
CELLS = pd.DataFrame(cells)
print(f"descriptors for {len(_DESC_CACHE)} distinct SMILES in {time.time() - t0:.1f}s")
CELLS.to_csv(os.path.join(EVAL_DIR, "per_protein_per_run.csv"), index=False)

# %% [markdown]
# ## Table 1
#
# Averaged over proteins within a run, then mean +/- std across runs.
#
# **Read both numeric columns.** `This run` uses the paper's denominator, so every
# row is capped at the validity rate - if validity is 24%, no property row can
# exceed 24% no matter how drug-like the molecules are. `Per valid` divides by the
# molecules actually emitted, which separates "does the model emit a molecule"
# from "is that molecule drug-like". The paper's column is shown both ways for the
# same reason.

# %%
METRIC_COLS = (ROW_ORDER + [k + '_per_valid' for k in PER_VALID_ROWS]
               + ['uniqueness_among_valid', 'mean_n_atoms'])
per_run = CELLS.groupby('run')[METRIC_COLS].mean()


def mean_std(col):
    v = per_run[col].to_numpy(dtype=float)
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


_val_mu, _ = mean_std('validity')
_paper_validity = PAPER_TABLE1['validity']['CTDDG'][0]

table1 = []
for key in ROW_ORDER:
    mu, sd = mean_std(key)
    paper = PAPER_TABLE1[key]
    row = {
        'Property': ROW_LABEL[key],
        'This run': f"{mu:.2f} +/- {sd:.2f}" if key != 'qed' else f"{mu:.3f} +/- {sd:.3f}",
        'CTDDG (paper)': f"{paper['CTDDG'][0]} +/- {paper['CTDDG'][1]}",
        'CDGCN (paper)': f"{paper['CDGCN'][0]} +/- {paper['CDGCN'][1]}",
        'SSVAE (paper)': f"{paper['SSVAE'][0]} +/- {paper['SSVAE'][1]}",
    }
    if key in PER_VALID_ROWS:
        pv_mu, pv_sd = mean_std(key + '_per_valid')
        row['Per valid'] = f"{pv_mu:.1f}"
        row['CTDDG per valid'] = f"{100 * paper['CTDDG'][0] / _paper_validity:.1f}"
    else:
        row['Per valid'] = '-'
        row['CTDDG per valid'] = '-'
    row['_value'], row['_std'] = mu, sd
    table1.append(row)

TABLE1 = pd.DataFrame(table1)
_cols = ['Property', 'This run', 'Per valid', 'CTDDG (paper)', 'CTDDG per valid',
         'CDGCN (paper)', 'SSVAE (paper)']

print("\n=== Table 1: drug-like properties ===")
print(f"{len(proteins_present)} proteins x {len(runs_present)} runs x "
      f"{N_SAMPLES} samples")
print(TABLE1[_cols].to_string(index=False))

mu_uv, sd_uv = mean_std('uniqueness_among_valid')
mu_at, sd_at = mean_std('mean_n_atoms')
print(f"\nsecondary: uniqueness among valid {mu_uv:.2f} +/- {sd_uv:.2f} % | "
      f"mean heavy atoms {mu_at:.1f} +/- {sd_at:.1f}")

TABLE1[_cols].to_csv(os.path.join(EVAL_DIR, "table1_drug_like_properties.csv"),
                     index=False)

check("chemical validity", _val_mu >= 80.0,
      f"{_val_mu:.1f}% (paper: {_paper_validity})"
      + ("" if _val_mu >= 80.0 else
         " - this single metric caps every other row in the paper's column. Run "
         "00_phase0_diagnostics.py check A to find out whether stage 1 or stage 2 "
         "is responsible"))

# How many property rows are competitive once validity is factored out?
_wins = 0
for key in PER_VALID_ROWS:
    pv_mu, _ = mean_std(key + '_per_valid')
    if pv_mu >= 100 * PAPER_TABLE1[key]['CTDDG'][0] / _paper_validity:
        _wins += 1
check("drug-likeness of the molecules that ARE emitted",
      _wins >= len(PER_VALID_ROWS) // 2,
      f"{_wins} of {len(PER_VALID_ROWS)} property rows match or beat the paper once "
      f"normalised by validity. "
      + ("The chemistry is sound; the failure is narrow." if _wins >= len(PER_VALID_ROWS) // 2
         else "Both validity AND drug-likeness are degraded."))

# %% [markdown]
# ## Toxicity
#
# Three numbers, three meanings. Keeping them apart is the point.
#
# 1. **Training labeller (RF)** - the model that wrote `Toxic_class` for Eq. 8.
#    Scoring generated molecules with it partly measures how well the generator
#    memorised that forest's boundary. An internal diagnostic, not a claim.
# 2. **Independent evaluator (GBM)** - a different algorithm on different features,
#    trained on the same Hansen data. Shares the data, not the decision boundary.
# 3. **pkCSM** - the paper's actual Table 3 metric. Only this is comparable.
#
# **A confound worth stating:** this is measured on the molecules that survived
# sanitization. If validity is low, the surviving subset may be systematically
# simpler, and Morgan-fingerprint mutagenicity flags (nitroaromatics, aryl amines,
# fused polycyclics) are rarer in simpler molecules by construction. A low toxic
# fraction cannot be read as successful suppression until validity is fixed.

# %%
def _load_pickle(path):
    errors = []
    for how, fn in (("pickle", lambda: pickle.load(open(path, 'rb'))),
                    ("joblib", lambda: __import__('joblib').load(path))):
        try:
            return fn()
        except Exception as e:
            errors.append(f"{how}: {type(e).__name__}: {e}")
    raise RuntimeError(f"Could not load {path}\n  " + "\n  ".join(errors))


def morgan_matrix(smiles_list, radius, n_bits):
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp_of = gen.GetFingerprint
    except (ImportError, AttributeError):
        from rdkit.Chem import AllChem
        fp_of = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)

    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            DataStructs.ConvertToNumpyArray(fp_of(mol), out[i])
    return out


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


def independent_features(smiles_list):
    """Must match random_forest_classifier.py exactly: MACCS 167 + 15 physchem."""
    n = len(smiles_list)
    out = np.full((n, 167 + len(_PHYSCHEM)), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        bits = np.zeros((167,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), bits)
        out[i, :167] = bits
        for j, (_, fn) in enumerate(_PHYSCHEM):
            try:
                out[i, 167 + j] = float(fn(m))
            except Exception:
                pass
    return out


def labels_from(model_path, smiles_list, cache_path, featuriser, label_name):
    """{smiles: 0/1}, cached. `featuriser` takes a SMILES list -> feature matrix."""
    cache = {}
    if os.path.isfile(cache_path):
        df = pd.read_csv(cache_path)
        cache = dict(zip(df['smiles'].astype(str), df[label_name].astype(int)))
        print(f"  cache: {len(cache)} entries from {os.path.basename(cache_path)}")

    todo = sorted({s for s in smiles_list if s and s not in cache})
    if todo:
        mdl = _load_pickle(model_path)
        print(f"  labelling {len(todo)} molecules with "
              f"{os.path.basename(model_path)}")
        X = featuriser(todo)
        cache.update(zip(todo, mdl.predict(X).astype(int)))
        pd.DataFrame({'smiles': list(cache), label_name: list(cache.values())}) \
          .to_csv(cache_path, index=False)
    return cache


# %%
TOX = {}
if COMPUTE_TOXICITY and os.path.isfile(RF_MODEL_PATH):
    # Same-forest check against the labels Eq. 8 was trained on.
    train_fp = _load_json(TRAIN_AMES_META)
    if train_fp:
        now_sha = _sha256(RF_MODEL_PATH)
        train_sha = (train_fp.get('rf') or {}).get('sha256')
        check("same Ames RF as the BindingDB training labels",
              bool(train_sha) and train_sha == now_sha,
              "one forest wrote both" if train_sha == now_sha else
              "the RF on disk is NOT the one that labelled the training ligands")
        if train_sha and train_sha != now_sha:
            CAVEATS.append(
                "Generated molecules and training ligands were scored by DIFFERENT "
                "random forests, so the toxicity comparison is between two "
                "definitions of toxic.")

    # The ChEMBL side is NOT covered by that check. A rebuild marker means
    # 02's RF_AUTO_TRAIN replaced the forest after ChEMBL was labelled.
    if os.path.isfile(RF_META_PATH):
        _rfm = _load_json(RF_META_PATH) or {}
        if PRETRAIN_MAN:
            _lab = (PRETRAIN_MAN.get('labeller') or {}).get('sha256')
            check("ChEMBL and BindingDB share one forest",
                  _lab == _rfm.get('sha256'),
                  "the pretraining manifest records the same forest hash"
                  if _lab == _rfm.get('sha256') else
                  "the forest that labelled ChEMBL differs from the one on disk - "
                  "Eq. 8 optimised against two definitions of toxic across the two "
                  "phases")

    cfg = _load_pickle(FP_CONFIG_PATH)
    _radius = int(cfg.get('radius', 2))
    _nbits = int(cfg.get('n_bits', cfg.get('nBits', 2048)))
    all_smiles = [s for smis in SAMPLES.values() for s in smis if s]
    TOX = labels_from(RF_MODEL_PATH, all_smiles, GEN_AMES_CACHE,
                      lambda sl: morgan_matrix(sl, _radius, _nbits), 'ames')
elif COMPUTE_TOXICITY:
    check("Ames RF present", False,
          f"no RF at {RF_MODEL_PATH}; skipping the toxicity table")

EVAL_META_PATH = os.path.splitext(EVAL_MODEL_PATH)[0] + ".meta.json"
FEATURE_SPEC = "MACCS167+PHYSCHEM:v1"


def verify_feature_spec():
    """The evaluator is trained in one notebook and applied in another.

    If `_PHYSCHEM` or its ORDER drifts between the two, the model would score on
    features it was never trained on - silently, with plausible-looking output.
    The spec recorded at training time is checked here before any prediction.
    """
    meta = _load_json(EVAL_META_PATH)
    if meta is None:
        return check("independent evaluator feature spec", False,
                     f"no meta at {EVAL_META_PATH}; cannot confirm the feature "
                     f"vector matches the one the model was trained on")
    if meta.get("feature_spec") != FEATURE_SPEC:
        return check("independent evaluator feature spec", False,
                     f"model was trained with {meta.get('feature_spec')!r}, this "
                     f"notebook builds {FEATURE_SPEC!r}")
    want = [nm for nm, _ in _PHYSCHEM]
    got = meta.get("physchem_order")
    if got is not None and list(got) != want:
        return check("independent evaluator feature spec", False,
                     f"descriptor order differs. trained: {got[:4]}...  "
                     f"here: {want[:4]}...")
    n_want = 167 + len(_PHYSCHEM)
    if meta.get("n_features") not in (None, n_want):
        return check("independent evaluator feature spec", False,
                     f"trained on {meta.get('n_features')} features, this notebook "
                     f"builds {n_want}")
    return check("independent evaluator feature spec", True,
                 f"{FEATURE_SPEC}, {n_want} features, descriptor order matches")


TOX_IND = {}
if COMPUTE_INDEPENDENT and os.path.isfile(EVAL_MODEL_PATH):
    if verify_feature_spec():
        all_smiles = [s for smis in SAMPLES.values() for s in smis if s]
        TOX_IND = labels_from(EVAL_MODEL_PATH, all_smiles, GEN_GBM_CACHE,
                              independent_features, 'ames')
    else:
        print("       skipping the independent evaluator rather than reporting a "
              "number computed on the wrong features")
elif COMPUTE_INDEPENDENT:
    check("independent evaluator present", False,
          f"no model at {EVAL_MODEL_PATH} - run random_forest_classifier.py with "
          f"RUN_INDEPENDENT_EVALUATOR = True to break the circularity")

# %%
def toxicity_table(tox_map, label):
    """-> (DataFrame per protein-run, mean, std)."""
    rows = []
    for (run, row), smis in sorted(SAMPLES.items()):
        uv = list(dict.fromkeys([s for s in smis if s and descriptors(s) is not None]))
        labelled = [tox_map[s] for s in uv if s in tox_map]
        rows.append({'run': run, 'protein_row': row, 'n_scored': len(labelled),
                     'pct_toxic': 100.0 * np.mean(labelled) if labelled else float('nan')})
    df = pd.DataFrame(rows)
    per_run_tox = df.groupby('run')['pct_toxic'].mean().to_numpy(dtype=float)
    mu = float(np.mean(per_run_tox))
    sd = float(np.std(per_run_tox, ddof=1)) if len(per_run_tox) > 1 else 0.0
    return df, mu, sd


tox_mu = tox_sd = None
ind_mu = ind_sd = None

if TOX:
    TOXDF, tox_mu, tox_sd = toxicity_table(TOX, "RF")
    TOXDF.to_csv(os.path.join(EVAL_DIR, "toxicity_per_protein_per_run.csv"), index=False)
    print("\n=== Toxicity ===")
    print(f"  1. TRAINING LABELLER (Ames RF)     {tox_mu:6.2f} +/- {tox_sd:.2f} %")
    print("     partly scores the model against its own supervision")

if TOX_IND:
    INDDF, ind_mu, ind_sd = toxicity_table(TOX_IND, "GBM")
    INDDF.to_csv(os.path.join(EVAL_DIR, "toxicity_independent_per_protein_per_run.csv"),
                 index=False)
    print(f"  2. INDEPENDENT (GBM / MACCS)       {ind_mu:6.2f} +/- {ind_sd:.2f} %")
    print("     different algorithm and features; shares only the Hansen training data")

    if tox_mu is not None:
        _agree = sum(1 for s in TOX if s in TOX_IND and TOX[s] == TOX_IND[s])
        _n = sum(1 for s in TOX if s in TOX_IND)
        print(f"     the two models agree on {100 * _agree / max(1, _n):.1f}% of "
              f"generated molecules")
        check("the two toxicity models tell the same story",
              abs(tox_mu - ind_mu) < 10.0,
              f"RF {tox_mu:.1f}% vs independent {ind_mu:.1f}%"
              + ("" if abs(tox_mu - ind_mu) < 10.0 else
                 " - a large gap means the RF number is substantially memorisation "
                 "of its own boundary"))

print("\n  3. pkCSM (the paper's metric)      not run here")
print(f"     paper, 5 PDBs: CTDDG {PAPER_TABLE3['Avg']} | CDGCN "
      f"{PAPER_TABLE3_CDGCN['Avg']}")

if tox_mu is not None:
    _train_frac = None
    if SPLIT_MAN:
        _train_frac = None   # recorded in 02's log, not the manifest
    print(f"\n  CONFOUND: measured on the {_val_mu:.0f}% of samples that survived "
          f"sanitization.")
    print("  If that subset is systematically simpler, fingerprint mutagenicity "
          "flags are\n  rarer by construction. Do not read a low number as "
          "successful suppression\n  until validity is fixed.")
    CAVEATS.append(
        f"The toxic fraction is measured on the {_val_mu:.0f}% of samples that "
        f"sanitized. Until validity is fixed it cannot be separated from a "
        f"simplicity confound in the surviving subset.")

# %% [markdown]
# ### Per-PDB rows and the pkCSM upload
#
# `pkcsm_upload_table3.smi` is scoped to the five Table-3 PDBs: 5 proteins x 10
# runs x 100 requested, deduplicated, which is roughly a thousand molecules rather
# than the full ~25,000. That is a handful of batch uploads, and it is the only
# subset Table 3 is actually about.

# %%
pdb_map = _load_json(PDB_MAP_PATH)

if TOX and pdb_map:
    sub = []
    for pdb, prow in pdb_map.items():
        v = TOXDF[TOXDF['protein_row'] == int(prow)].groupby('run')['pct_toxic'].mean()
        if len(v) == 0:
            continue
        row = {'PDB': pdb,
               'This run (RF)': f"{v.mean():.2f} +/- "
                                f"{v.std(ddof=1) if len(v) > 1 else 0.0:.2f}",
               'CTDDG paper (pkCSM)': PAPER_TABLE3.get(pdb, '-'),
               'CDGCN paper (pkCSM)': PAPER_TABLE3_CDGCN.get(pdb, '-')}
        if TOX_IND:
            vi = INDDF[INDDF['protein_row'] == int(prow)].groupby('run')['pct_toxic'].mean()
            row['This run (independent)'] = f"{vi.mean():.2f}" if len(vi) else '-'
        sub.append(row)
    if sub:
        print()
        print(pd.DataFrame(sub).to_string(index=False))
        pd.DataFrame(sub).to_csv(os.path.join(EVAL_DIR, "table3_toxicity.csv"),
                                 index=False)
else:
    print(f"\n  [INFO] no {os.path.basename(PDB_MAP_PATH)}; per-PDB rows skipped. "
          f"Run 05_docking_table2.py to write it.")

# %%
uniq_all = sorted({s for smis in SAMPLES.values() for s in smis
                   if s and descriptors(s) is not None})
pk_all = os.path.join(EVAL_DIR, "pkcsm_upload_all.smi")
with open(pk_all, 'w') as f:
    f.write("\n".join(uniq_all) + "\n")
print(f"\n  {len(uniq_all):,} unique valid SMILES -> {os.path.basename(pk_all)}")

if pdb_map:
    rows_wanted = {int(v) for v in pdb_map.values()}
    uniq_t3 = sorted({s for (run, row), smis in SAMPLES.items() if row in rows_wanted
                      for s in smis if s and descriptors(s) is not None})
    pk_t3 = os.path.join(EVAL_DIR, "pkcsm_upload_table3.smi")
    with open(pk_t3, 'w') as f:
        f.write("\n".join(uniq_t3) + "\n")
    print(f"  {len(uniq_t3):,} unique valid SMILES for the 5 Table-3 PDBs -> "
          f"{os.path.basename(pk_t3)}")
    print(f"  THIS is the pkCSM upload. It is ~{len(uniq_t3) / max(1, len(uniq_all)):.0%} "
          f"of the full set, and it is the only subset Table 3 is about.")
else:
    print("  Table-3 subset not written: run 05_docking_table2.py first to map the "
          "five PDBs to protein rows.")

# %% [markdown]
# ## Novelty
#
# CDGCN Table 1's definition: exact match against the dataset, on canonical SMILES.

# %%
def dataset_canonical_smiles():
    if os.path.isfile(NOVELTY_CACHE):
        with open(NOVELTY_CACHE) as f:
            return {ln.strip() for ln in f if ln.strip()}

    suffix = FOLD_SUFFIX[FOLD]
    paths = []
    for split in ("train", "test"):
        d = os.path.join(BINDINGDB_DIR, f"{split}_dataset", f"{split}_{suffix}")
        paths += sorted(glob.glob(os.path.join(d, "ligands_*_corrected")))
    if not paths:
        print(f"  [WARN] no ligand pair files for fold {FOLD}; skipping novelty")
        return None

    raw = set()
    for p in paths:
        with open(p) as f:
            for line in f:
                s = ''.join(line.split())
                if s:
                    raw.add(s)
    print(f"  canonicalising {len(raw)} dataset ligands (once, then cached)")

    canon = set()
    for s in raw:
        if '.' in s:
            s = max(s.split('.'), key=len)       # match the training-side stripping
        m = Chem.MolFromSmiles(s)
        if m is not None:
            canon.add(Chem.MolToSmiles(m))
    with open(NOVELTY_CACHE, 'w') as f:
        f.write("\n".join(sorted(canon)) + "\n")
    return canon


novelty_mu = None
if COMPUTE_NOVELTY:
    DATASET_SMILES = dataset_canonical_smiles()
    if DATASET_SMILES:
        rows = []
        for (run, row), smis in sorted(SAMPLES.items()):
            uv = list(dict.fromkeys([s for s in smis
                                     if s and descriptors(s) is not None]))
            novel = [s for s in uv if s not in DATASET_SMILES]
            rows.append({'run': run, 'protein_row': row,
                         'novelty': 100.0 * len(novel) / max(1, len(uv))})
        NOV = pd.DataFrame(rows).groupby('run')['novelty'].mean().to_numpy(dtype=float)
        novelty_mu = float(np.mean(NOV))
        novelty_sd = float(np.std(NOV, ddof=1)) if len(NOV) > 1 else 0.0
        print(f"\nNovelty (unique valid not in the fold's ligand set): "
              f"{novelty_mu:.2f} +/- {novelty_sd:.2f} %")
        print("  CDGCN paper, novel proteins, S_1000: 99.8 +/- 0.05 %")
        print("  Novelty measures string identity, so it does not by itself clear "
              "any pretraining leak - see the provenance section above for that.")

# %% [markdown]
# ## Generation time

# %%
timing_path = os.path.join(SAMPLES_DIR, "timing.csv")
per_mol_ms = None
if os.path.isfile(timing_path):
    tdf = pd.read_csv(timing_path)
    per_mol = tdf['seconds'] / tdf['n_samples']
    per_mol_ms = per_mol.mean() * 1000
    print(f"\n{per_mol_ms:.0f} ms per molecule ({MANIFEST.get('device')}), so "
          f"S_10 ~ {per_mol.mean() * 10:.1f}s, S_100 ~ {per_mol.mean() * 100:.1f}s, "
          f"S_1000 ~ {per_mol.mean() * 1000:.0f}s")

# %% [markdown]
# ## Report

# %%
report = [
    f"# CTDDG reproduction - evaluation, fold {FOLD}",
    "",
    f"Generated {MANIFEST.get('generated_at')} | checkpoint "
    f"`{MANIFEST.get('checkpoint')}`",
    f"{len(proteins_present)} held-out proteins x {len(runs_present)} runs x "
    f"{N_SAMPLES} samples = {len(proteins_present) * len(runs_present) * N_SAMPLES} "
    f"molecules",
    "",
    "## How to read Table 1",
    "",
    "`This run` uses the paper's denominator (molecules *requested*), so every "
    f"property row is capped at the validity rate of {_val_mu:.1f}%. `Per valid` "
    "divides by the molecules actually emitted. The two separate *does the model "
    "produce a molecule at all* from *is that molecule drug-like* - which the "
    "single column conflates, and which is why a narrow failure can look like a "
    "total one.",
    "",
    "## Table 1 - drug-like properties",
    "",
    TABLE1[_cols].to_markdown(index=False),
    "",
    f"Secondary: uniqueness among valid molecules {mu_uv:.2f} +/- {sd_uv:.2f} %; "
    f"mean heavy atoms {mu_at:.1f}.",
    "",
]

if CAVEATS:
    report += ["## Why these numbers are not a reproduction of Tables 1-3", ""]
    report += [f"- {c}" for c in CAVEATS]
    report += [
        "- No CDGCN column: a conditional CDGCN fine-tune on this fold does not "
        "exist here, so the paper's central comparison cannot be made yet.",
        "",
    ]
else:
    report += ["## Caveats", "",
               "No blocking caveat was detected from the artifacts. A CDGCN column "
               "is still missing, so the paper's central comparison cannot be made.",
               ""]

if tox_mu is not None:
    report += [
        "## Toxicity - three numbers, three meanings",
        "",
        "| Source | Value | What it measures |",
        "|---|---|---|",
        f"| Training labeller (Ames RF) | **{tox_mu:.2f} +/- {tox_sd:.2f} %** | "
        f"Partly the generator's memorisation of the forest that wrote its own "
        f"supervision. Internal diagnostic. |",
    ]
    if ind_mu is not None:
        report += [f"| Independent (GBM, MACCS + physchem) | **{ind_mu:.2f} +/- "
                   f"{ind_sd:.2f} %** | Different algorithm and features; shares only "
                   f"the Hansen training data. |"]
    report += [
        f"| pkCSM (paper's Table 3) | not run | The only number comparable to "
        f"CTDDG {PAPER_TABLE3['Avg']} / CDGCN {PAPER_TABLE3_CDGCN['Avg']}. |",
        "",
        f"Measured on the {_val_mu:.0f}% of samples that sanitized. If the surviving "
        "subset is systematically simpler, fingerprint mutagenicity flags are rarer "
        "by construction, so a low number cannot be read as successful suppression "
        "until validity is fixed.",
        "",
        "`pkcsm_upload_table3.smi` holds the five-PDB subset for the real metric.",
        "",
    ]

if novelty_mu is not None:
    report += ["## Novelty", "",
               f"{novelty_mu:.2f} % of unique valid molecules are absent from the "
               "fold's ligand set (CDGCN paper: 99.8 %).", ""]

if per_mol_ms is not None:
    report += ["## Generation time", "",
               f"{per_mol_ms:.0f} ms per molecule on "
               f"{MANIFEST.get('device')}.", ""]

report += ["## Table 2 - target awareness", "",
           "Needs docking. See `05_docking_table2.py`.", ""]

if CKPT_META.get('val_metric') not in (None, 'l_nontoxic'):
    report += [
        "> **Before docking:** `02_finetune_ctddg.py`'s sanity check compares the "
        "loss under the real protein embedding against a mismatched one. If a "
        "mismatched protein gives an equal or lower loss, the conditioning path is "
        "not doing useful work and Table 2 is not measuring target awareness.",
        "",
    ]

report_path = os.path.join(EVAL_DIR, "report.md")
with open(report_path, 'w') as f:
    f.write("\n".join(report))

print(f"\nreport  {report_path}")
print(f"tables  {EVAL_DIR}")

# %%
print("=" * 78)
_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
for c in sorted(CHECKS, key=lambda r: _order.get(r["status"], 9)):
    print(f"  {c['status']:5s}  {c['name']}")
    print(f"         {c['detail']}")
if CAVEATS:
    print("-" * 78)
    print("  CAVEATS carried into the report:")
    for c in CAVEATS:
        print(f"    - {c[:100]}{'...' if len(c) > 100 else ''}")

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
# # 00 - Phase-0 diagnostics (zero GPU cost)
#
# **Run this before changing anything else.** Every check here is read-only and
# finishes in minutes. Between them they decide which of the expensive fixes are
# actually needed.
#
# | # | Check | Settles |
# |---|---|---|
# | A | Unconditional sampling from `checkpoints/ckpt.pt` | Is the validity collapse in stage 1 or stage 2? **The decisive one.** |
# | B | RDKit failure histogram | *Why* molecules fail: graph build, kekulize, or valence |
# | C | Truncation histogram (`n_atoms` vs validity) | Whether `MAX_DECODE_STEPS = 100` is implicated |
# | D | Logit range in `Policy.forward` | Whether the `clamp(+/-80)` ever binds |
# | E | Salt census + `traverse_graph` corruption demo | Size of the disconnected-graph bug |
# | F | ChEMBL file provenance | Which file stage 1 actually trained on (blocker B2) |
# | G | Checkpoint counters | Stage-1 iteration count (blocker B1) |
# | H | Ames RF identity | Whether ChEMBL and BindingDB labels came from the same forest |
#
# **Reading the result of A**, which is what everything else hangs off:
#
# * validity ~25%  => stage 1 is the cause. Fine-tuning, conditioning, embeddings,
#   the sampler and the evaluator are all exonerated. Stage 1 must be redone.
# * validity ~90%  => stage 1 is fine; the damage is in fine-tuning (salts,
#   `dense_policy_0`, conditioning) and no pretraining re-run is needed.
#
# Nothing here writes to the repository except one JSON report under
# `outputs/diagnostics/`.

# %%
import os
import csv
import glob
import json
import math
import time
import hashlib
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter

import numpy as np
from scipy import sparse

import rdkit
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

print(f"torch {torch.__version__} | rdkit {rdkit.__version__} | numpy {np.__version__}")

# %% [markdown]
# ## Config

# %%
REPO_ROOT = os.environ.get("CTDDG_REPO_ROOT", "/home/muhamed/repo/CTDDG")
FOLD = os.environ.get("CTDDG_FOLD", "d1")

ATOM_TYPES_PATH = os.path.join(REPO_ROOT, "datasets/data/atom_types.txt")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
PRETRAIN_CKPT = os.path.join(CHECKPOINT_DIR, "ckpt.pt")
PRETRAIN_TRAINER = os.path.join(CHECKPOINT_DIR, "trainer.pt")
PRETRAIN_LOG = os.path.join(CHECKPOINT_DIR, "log.out")
PRETRAIN_CONFIG = os.path.join(CHECKPOINT_DIR, "configs.json")

CHEMBL_DIR = os.path.join(REPO_ROOT, "datasets/data/chembl")
BINDINGDB_DIR = os.path.join(REPO_ROOT, "datasets/data/bindingdb")
FINETUNE_DIR = os.path.join(REPO_ROOT, "outputs/finetune", FOLD)
SAMPLES_DIR = os.path.join(REPO_ROOT, "outputs/samples", FOLD)

RF_MODEL_PATH = os.path.join(REPO_ROOT, "models/ames_random_forest.pkl")
RF_META_PATH = os.path.join(REPO_ROOT, "models/ames_random_forest.meta.json")
AMES_CACHE_META = os.path.join(BINDINGDB_DIR, "ligand_ames_labels.csv.meta.json")

OUT_DIR = os.path.join(REPO_ROOT, "outputs/diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(OUT_DIR, f"phase0_{FOLD}.json")

FOLD_SUFFIX = {"d1": "4_org_1042_104", "d2": "4_org_1000_112", "d3": "4_org_1004_122",
               "d4": "4_org_1002_103", "d5": "4_org_1036_124"}

GPU_ID = int(os.environ.get("CTDDG_GPU", "0"))
CTX = torch.device(f"cuda:{GPU_ID}") if torch.cuda.is_available() else torch.device("cpu")

# Check A: how many molecules to sample unconditionally.
N_UNCOND_SAMPLES = 2000
MAX_DECODE_STEPS = 100          # the sampler's cap, as 03 uses it
D_RECEPTIVE = 2
SEED = 42

# Expected row counts, for the provenance check.
N_CHEMBL_FILTERED = 1_090_529   # chembl.txt: >100 atoms dropped, BindingDB ligands removed
N_CHEMBL_RAW = 1_207_359        # the RF-annotated export, unfiltered

print(f"repo      {REPO_ROOT}")
print(f"fold      {FOLD}")
print(f"device    {CTX}")
print(f"report    {REPORT_PATH}")

# %% [markdown]
# ## Verdict bookkeeping
#
# Every check appends one row here instead of printing loose text, so the end of
# the notebook is a single table you can act on and a JSON file you can commit.

# %%
VERDICTS = []


def record(check_id, name, status, detail, data=None):
    """status: PASS | FAIL | WARN | INFO | SKIP."""
    row = {"check": check_id, "name": name, "status": status,
           "detail": detail, "data": data or {}}
    VERDICTS.append(row)
    mark = {"PASS": "[OK]  ", "FAIL": "[FAIL]", "WARN": "[WARN]",
            "INFO": "[INFO]", "SKIP": "[SKIP]"}.get(status, "[?]   ")
    print(f"{mark} {check_id} {name}: {detail}")
    return row


def require_file(path, what):
    """-> True if present. Records a SKIP and returns False otherwise."""
    if os.path.isfile(path):
        return True
    print(f"       missing {what}: {path}")
    return False


# %% [markdown]
# ## Molecule spec and graph utilities
#
# Byte-identical to the training notebooks. Atom-type identity is a line number in
# `atom_types.txt`, so this must be the same file the checkpoint was trained with.

# %%
class MoleculeSpec(object):
    def __init__(self, file_name=ATOM_TYPES_PATH):
        self.atom_types = []
        self.atom_symbols = []
        with open(file_name, 'r') as f:
            for line in f:
                a = line.strip('\n').split(',')
                if len(a) < 3:
                    continue
                self.atom_types.append((a[0], int(a[1]), int(a[2])))
                if a[0] not in self.atom_symbols:
                    self.atom_symbols.append(a[0])
        self.bond_orders = [Chem.BondType.AROMATIC, Chem.BondType.SINGLE,
                            Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
        self.max_iter = 120

    def get_atom_type(self, atom):
        return self.atom_types.index(
            (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetNumExplicitHs()))

    def index_to_atom(self, idx):
        symbol, charge, hs = self.atom_types[idx]
        a = Chem.Atom(symbol)
        a.SetFormalCharge(charge)
        a.SetNumExplicitHs(hs)
        return a

    def index_to_bond(self, mol, begin_id, end_id, idx):
        mol.AddBond(begin_id, end_id, self.bond_orders[idx])

    @property
    def num_atom_types(self):
        return len(self.atom_types)

    @property
    def num_bond_types(self):
        return len(self.bond_orders)


_mol_spec = None


def get_mol_spec():
    global _mol_spec
    if _mol_spec is None:
        _mol_spec = MoleculeSpec(ATOM_TYPES_PATH)
    return _mol_spec


def get_d(A, X):
    def _to_sparse(_A, _X):
        return sparse.coo_matrix(
            (np.ones([_A.shape[0] * 2], dtype=np.int64),
             (np.concatenate([_A[:, 0], _A[:, 1]], axis=0),
              np.concatenate([_A[:, 1], _A[:, 0]], axis=0))),
            shape=[_X.shape[0]] * 2)

    A_sparse = _to_sparse(A, X)
    d2 = A_sparse * A_sparse
    d3 = d2 * A_sparse

    D_2 = np.stack(d2.nonzero(), axis=1)
    D_2 = D_2[D_2[:, 0] < D_2[:, 1], :]

    D_3 = np.stack(d3.nonzero(), axis=1)
    D_3 = D_3[D_3[:, 0] < D_3[:, 1], :]

    D_3_sparse = _to_sparse(D_3, X)
    D_3_sparse = D_3_sparse - D_3_sparse.multiply(A_sparse)
    D_3 = np.stack(D_3_sparse.nonzero(), axis=1)
    D_3 = D_3[D_3[:, 0] < D_3[:, 1], :]
    return D_2, D_3


def get_mol_from_graph(X, A, sanitize=True):
    """-> (mol | None, reason). `reason` names the failure for check B."""
    try:
        mol = Chem.RWMol(Chem.Mol())
        X, A = X.tolist(), A.tolist()
        for atom_type in X:
            mol.AddAtom(get_mol_spec().index_to_atom(atom_type))
        for atom_id1, atom_id2, bond_type in A:
            get_mol_spec().index_to_bond(mol, atom_id1, atom_id2, bond_type)
    except Exception as e:
        return None, f"graph build: {type(e).__name__}"

    if not sanitize:
        return mol, "unsanitized"
    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol, "valid"
    except Exception as e:
        msg = str(e).strip().replace("\n", " ")[:60]
        return None, f"{type(e).__name__}: {msg}"


try:
    _spec = get_mol_spec()
    record("SETUP", "atom vocabulary", "PASS",
           f"{_spec.num_atom_types} atom types, {_spec.num_bond_types} bond types",
           {"n_atom_types": _spec.num_atom_types})
except Exception as e:
    record("SETUP", "atom vocabulary", "FAIL", f"{type(e).__name__}: {e}")
    raise

# %% [markdown]
# ## Check F - which ChEMBL file did stage 1 train on?
#
# Blocker B2 in `05_training_runs_and_diagnostics.md` asserts stage 1 trained on the
# unfiltered 1,207,359-row export. The pretraining notebook's own output says
# `Total molecules loaded: 1090529`, which is the **filtered** count, and no
# `[INFO] Dropped N rows` line fired - so the file it opened already had that many
# rows. Counting lines settles it.
#
# Run first because if B2 is false, a large part of the planned work disappears.

# %%
def count_rows(path, has_header=True):
    """Fast line count. -> (n_data_rows, size_bytes) or (None, None)."""
    if not os.path.isfile(path):
        return None, None
    n = 0
    with open(path, 'rb') as f:
        for _ in f:
            n += 1
    return (n - 1 if has_header and n else n), os.path.getsize(path)


chembl_candidates = {
    "chembl.txt": (os.path.join(CHEMBL_DIR, "chembl.txt"), False),
    "chembl_final.txt": (os.path.join(CHEMBL_DIR, "chembl_final.txt"), False),
    "chembl_with_ames_predictions.csv": (
        os.path.join(CHEMBL_DIR, "chembl_with_ames_predictions.csv"), True),
    "chembl_toxicity_labels.csv": (
        os.path.join(CHEMBL_DIR, "chembl_toxicity_labels.csv"), True),
}

chembl_rows = {}
for label, (path, hdr) in chembl_candidates.items():
    n, size = count_rows(path, hdr)
    chembl_rows[label] = {"path": path, "rows": n,
                          "size_mb": round(size / 1e6, 1) if size else None}
    if n is None:
        print(f"       {label:38s} absent")
    else:
        print(f"       {label:38s} {n:>10,} rows  {size / 1e6:>8.1f} MB")

# The file the pretraining port points at by default.
_trained_on = chembl_rows.get("chembl_with_ames_predictions.csv", {}).get("rows")

if _trained_on is None:
    record("F", "stage-1 data file", "SKIP",
           "chembl_with_ames_predictions.csv not on disk; cannot settle B2 by row count",
           chembl_rows)
elif _trained_on == N_CHEMBL_FILTERED:
    record("F", "stage-1 data file", "PASS",
           f"DATA_PATH holds {_trained_on:,} rows = the FILTERED count. "
           f"Blocker B2 (leakage) is most likely FALSE - confirm with check F2 below",
           chembl_rows)
elif _trained_on == N_CHEMBL_RAW:
    record("F", "stage-1 data file", "FAIL",
           f"DATA_PATH holds {_trained_on:,} rows = the UNFILTERED export. "
           f"B2 stands and stage 1 must be re-run on the filtered set",
           chembl_rows)
else:
    record("F", "stage-1 data file", "WARN",
           f"DATA_PATH holds {_trained_on:,} rows, matching neither "
           f"{N_CHEMBL_FILTERED:,} (filtered) nor {N_CHEMBL_RAW:,} (raw). Inspect it",
           chembl_rows)

# %% [markdown]
# ### F2 - the leakage check itself
#
# A row count is circumstantial. This compares actual canonical structures, which is
# the only thing that settles whether held-out ligands were in the pretraining set.

# %%
def canonical_set(smiles_iter, strip_salts=True, limit=None):
    out = set()
    for i, s in enumerate(smiles_iter):
        if limit is not None and i >= limit:
            break
        s = "".join(str(s).split())
        if not s:
            continue
        if strip_salts and '.' in s:
            s = max(s.split('.'), key=len)
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.add(Chem.MolToSmiles(m))
    return out


_pretrain_path = chembl_rows.get("chembl_with_ames_predictions.csv", {}).get("path")
_test_lig = os.path.join(BINDINGDB_DIR, "test_dataset", f"test_{FOLD_SUFFIX[FOLD]}",
                         "ligands_test_4_org_corrected")

if not (_pretrain_path and os.path.isfile(_pretrain_path) and os.path.isfile(_test_lig)):
    record("F2", "ligand leakage", "SKIP",
           "need both the pretraining CSV and the held-out ligand file")
else:
    import pandas as pd

    t0 = time.time()
    _col = None
    _hdr = pd.read_csv(_pretrain_path, nrows=0)
    for cand in ("Smiles", "smiles", "SMILES", "canonical_smiles"):
        if cand in _hdr.columns:
            _col = cand
            break
    if _col is None:
        record("F2", "ligand leakage", "SKIP",
               f"no SMILES column among {list(_hdr.columns)[:10]}")
    else:
        _pre = pd.read_csv(_pretrain_path, usecols=[_col])[_col].astype(str)
        print(f"       canonicalising {len(_pre):,} pretraining molecules "
              f"(a few minutes)...")
        pre_canon = canonical_set(_pre)

        with open(_test_lig) as f:
            lig_canon = canonical_set(line for line in f)

        overlap = lig_canon & pre_canon
        frac = len(overlap) / max(1, len(lig_canon))
        status = "PASS" if frac < 0.01 else ("WARN" if frac < 0.10 else "FAIL")
        record("F2", "ligand leakage", status,
               f"{len(overlap):,} of {len(lig_canon):,} held-out ligands "
               f"({100 * frac:.2f}%) are in the pretraining set "
               f"[{time.time() - t0:.0f}s]",
               {"n_overlap": len(overlap), "n_heldout": len(lig_canon),
                "frac": round(frac, 5), "n_pretrain_canonical": len(pre_canon)})
        del pre_canon, lig_canon, _pre

# %% [markdown]
# ## Check G - stage-1 checkpoint counters (blocker B1)
#
# `max(num_batches_tracked)` is the exact stage-1 iteration count: the only
# train-mode forward in the pretraining notebook is the one per training step, and
# it survives resumes because it lives in the state dict.

# %%
def last_log_step(path):
    """-> (last_step, n_data_rows, header)."""
    last, n, header = None, 0, None
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if header is None and not line[0].isdigit():
                    header = line
                    continue
                if line.startswith('#') or line == 'Training finished':
                    continue
                try:
                    last = int(line.split('\t')[0])
                    n += 1
                except (ValueError, IndexError):
                    continue
    except OSError:
        return None, 0, None
    return last, n, header


if not require_file(PRETRAIN_CKPT, "stage-1 checkpoint"):
    record("G", "stage-1 counters", "SKIP", "ckpt.pt absent")
    CKPT_STATE = None
else:
    CKPT_STATE = torch.load(PRETRAIN_CKPT, map_location="cpu", weights_only=False)
    _sd = CKPT_STATE['model'] if isinstance(CKPT_STATE, dict) and 'model' in CKPT_STATE \
        else CKPT_STATE

    _nbt = [int(v) for k, v in _sd.items() if k.endswith("num_batches_tracked")]
    bn_steps = max(_nbt) if _nbt else None

    trainer_counter = None
    adam_steps = None
    if os.path.isfile(PRETRAIN_TRAINER):
        _tr = torch.load(PRETRAIN_TRAINER, map_location="cpu", weights_only=False)
        if isinstance(_tr, dict):
            trainer_counter = _tr.get("global_counter")
            try:
                _st = next(iter(_tr["optimizer"]["state"].values()))["step"]
                adam_steps = int(_st)
            except Exception:
                pass

    log_last, log_rows, log_header = last_log_step(PRETRAIN_LOG)

    policy0_absmax = None
    if "policy_0" in _sd:
        policy0_absmax = float(_sd["policy_0"].abs().max())

    n_finite = sum(int(torch.isfinite(v).all()) for v in _sd.values()
                   if torch.is_tensor(v) and v.is_floating_point())
    n_float = sum(1 for v in _sd.values() if torch.is_tensor(v) and v.is_floating_point())

    print(f"       BN train-mode steps   {bn_steps}")
    print(f"       trainer.pt counter    {trainer_counter}")
    print(f"       adam step count       {adam_steps}")
    print(f"       log.out last step     {log_last}  ({log_rows} data rows)")
    print(f"       log.out header        {log_header}")
    print(f"       policy_0 absmax       {policy0_absmax}   (0.0 => never trained)")
    print(f"       finite float tensors  {n_finite}/{n_float}")

    _counters = {"bn_steps": bn_steps, "trainer_counter": trainer_counter,
                 "adam_steps": adam_steps, "log_last_step": log_last,
                 "log_rows": log_rows, "policy_0_absmax": policy0_absmax}

    _agree = len({x for x in (bn_steps, trainer_counter, log_last) if x is not None}) == 1
    if bn_steps in (None, 0):
        record("G", "stage-1 counters", "FAIL",
               "no BatchNorm steps recorded - this checkpoint was never trained",
               _counters)
    elif _agree:
        record("G", "stage-1 counters", "PASS",
               f"all counters agree at {bn_steps:,} iterations. Blocker B1 RESOLVED",
               _counters)
    else:
        record("G", "stage-1 counters", "WARN",
               f"counters disagree (bn={bn_steps}, trainer={trainer_counter}, "
               f"log={log_last}); the log probably carries rows from an earlier "
               f"SUMMARY_STEP era. The BN count is authoritative",
               _counters)

    if n_finite != n_float:
        record("G2", "checkpoint finiteness", "FAIL",
               f"{n_float - n_finite} float tensors contain NaN/inf")
    else:
        record("G2", "checkpoint finiteness", "PASS",
               f"all {n_float} float tensors finite")

# %% [markdown]
# ## Check H - did one forest write both label sets?
#
# `02_finetune_ctddg` has `RF_AUTO_TRAIN = True`, and its rebuild adds an RDKit
# parseability filter the original notebook did not have. If it fired, the ChEMBL
# labels came from the original forest and the BindingDB labels from the rebuild -
# Eq. 8 would then be optimising against two different definitions of toxic. The
# `[OK] same Ames RF` message in `04_evaluate_metrics` only checks the BindingDB
# cache; it never looks at the ChEMBL labels.

# %%
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


if not os.path.isfile(RF_MODEL_PATH):
    record("H", "Ames RF identity", "SKIP", f"no RF at {RF_MODEL_PATH}")
else:
    rf_sha = sha256_file(RF_MODEL_PATH)
    rebuilt = os.path.isfile(RF_META_PATH)
    cache_sha = None
    if os.path.isfile(AMES_CACHE_META):
        try:
            with open(AMES_CACHE_META) as f:
                cache_sha = (json.load(f).get('rf') or {}).get('sha256')
        except Exception:
            pass

    _d = {"rf_sha256": rf_sha[:16], "cache_sha256": (cache_sha or "")[:16],
          "rebuild_meta_present": rebuilt}

    if rebuilt:
        try:
            with open(RF_META_PATH) as f:
                _m = json.load(f)
            print(f"       rebuild meta: {_m.get('n_molecules')} molecules, "
                  f"acc {_m.get('accuracy')}, trained {_m.get('trained_at')}")
        except Exception:
            pass
        record("H", "Ames RF identity", "FAIL",
               "ames_random_forest.meta.json exists, so 02's RF_AUTO_TRAIN REBUILT the "
               "forest. The ChEMBL labels came from the original and the BindingDB "
               "labels from this rebuild - two definitions of toxic in one objective. "
               "Re-run the ChEMBL annotation cells of 02_random_forest_classifier.py",
               _d)
    elif cache_sha and cache_sha != rf_sha:
        record("H", "Ames RF identity", "FAIL",
               "the RF on disk is not the one that wrote the BindingDB label cache",
               _d)
    elif cache_sha:
        record("H", "Ames RF identity", "PASS",
               "one forest wrote the BindingDB cache and is still on disk; no rebuild "
               "marker, so the ChEMBL labels are from the same model", _d)
    else:
        record("H", "Ames RF identity", "WARN",
               "no cache fingerprint to compare against", _d)

# %% [markdown]
# ## Check E - salt census and the `traverse_graph` corruption
#
# `data_preprocessing_1` stripped salts into `d1_tr_ligands.txt`, a file
# `02_finetune_ctddg` never opens - it reads the raw `ligands_*_corrected` instead.
#
# On a disconnected graph `traverse_graph` walks the first component, then
# recomputes `next_nodes` from the neighbours of the **last visited node** rather
# than the remaining unvisited set, so the loop exits with every atom of the second
# component still at `step_ids = -1`. `single_reorder` then `argsort`s those `-1`s
# to the front while remapping bond endpoints through the raw `step_ids`, so every
# bond points at the wrong atom. Silent, and uncounted.

# %%
def strip_salt(smiles):
    """Largest fragment by string length - matches data_preprocessing_1's rule."""
    if '.' not in smiles:
        return smiles
    return max(smiles.split('.'), key=len)


def traverse_graph_upstream(graph, atom_ranks, current_node=None, step_ids=None,
                            p=0.8, log_p=0.0):
    """The CURRENT implementation, reproduced verbatim so the bug can be shown."""
    import random as _random
    if current_node is None:
        next_nodes = range(len(atom_ranks))
        step_ids = [-1] * len(next_nodes)
        next_node_ranks = atom_ranks
    else:
        next_nodes = list(graph.neighbors(current_node))
        next_nodes = [n for n in next_nodes if step_ids[n] < 0]
        next_node_ranks = [atom_ranks[n] for n in next_nodes]

    next_nodes = [n for n, r in sorted(zip(next_nodes, next_node_ranks), key=lambda x: x[1])]

    while len(next_nodes) > 0:
        if len(next_nodes) == 1:
            next_node = next_nodes[0]
        elif _random.random() >= (1.0 - p):
            next_node = next_nodes[0]
            log_p += np.log(p)
        else:
            next_node = next_nodes[_random.randint(1, len(next_nodes) - 1)]
            log_p += np.log((1.0 - p) / (len(next_nodes) - 1))

        step_ids[next_node] = max(step_ids) + 1
        _, log_p = traverse_graph_upstream(graph, atom_ranks, next_node, step_ids, p, log_p)
        next_nodes = [n for n in graph.neighbors(next_node) if step_ids[n] < 0]
        next_node_ranks = [atom_ranks[n] for n in next_nodes]
        next_nodes = [n for n, r in sorted(zip(next_nodes, next_node_ranks),
                                           key=lambda x: x[1])]

    return step_ids, log_p


# E1 - demonstrate the corruption on a known salt
try:
    import networkx as nx

    _demo_smi = "CC(=O)Oc1ccccc1C(=O)O.Cl"      # aspirin hydrochloride
    _m = Chem.MolFromSmiles(_demo_smi)
    _ranks = list(Chem.CanonicalRankAtoms(_m))
    _g = nx.Graph()
    _g.add_nodes_from(range(_m.GetNumAtoms()))
    _g.add_edges_from([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in _m.GetBonds()])

    _ids, _ = traverse_graph_upstream(_g, _ranks)
    _n_unvisited = sum(1 for s in _ids if s < 0)

    if _n_unvisited > 0:
        record("E1", "traverse_graph on salts", "FAIL",
               f"{_n_unvisited} of {_m.GetNumAtoms()} atoms left unvisited on "
               f"'{_demo_smi}' - every bond in that training target is mis-wired",
               {"step_ids": _ids, "n_unvisited": _n_unvisited})
    else:
        record("E1", "traverse_graph on salts", "PASS",
               "the demo salt traverses cleanly (unexpected - re-read the function)",
               {"step_ids": _ids})
except ImportError:
    record("E1", "traverse_graph on salts", "SKIP", "networkx not importable")

# %%
# E2 - how many BindingDB ligands are affected?
_salt_counts = {}
for split in ("train", "test"):
    _p = os.path.join(BINDINGDB_DIR, f"{split}_dataset",
                      f"{split}_{FOLD_SUFFIX[FOLD]}",
                      f"ligands_{split}_4_org_corrected")
    if not os.path.isfile(_p):
        _salt_counts[split] = None
        continue
    n = tot = 0
    with open(_p) as f:
        for line in f:
            s = "".join(line.split())
            if not s:
                continue
            tot += 1
            n += ('.' in s)
    _salt_counts[split] = {"multi_fragment": n, "total": tot,
                           "frac": round(n / max(1, tot), 4)}
    print(f"       {split:5s}  {n:>7,} of {tot:>7,} ligands are multi-fragment "
          f"({100 * n / max(1, tot):.1f}%)")

_worst = max((v["frac"] for v in _salt_counts.values() if v), default=0.0)
if _worst == 0.0 and any(_salt_counts.values()):
    record("E2", "salt census", "PASS", "no multi-fragment ligands in this fold",
           _salt_counts)
elif _worst >= 0.01:
    record("E2", "salt census", "FAIL",
           f"up to {100 * _worst:.1f}% of ligands are multi-fragment and every one of "
           f"them is a corrupted training target. Fix in assemble_pairs", _salt_counts)
elif any(_salt_counts.values()):
    record("E2", "salt census", "WARN",
           f"{100 * _worst:.2f}% of ligands are multi-fragment - small but still "
           f"silently corrupt", _salt_counts)
else:
    record("E2", "salt census", "SKIP", "no ligand files found for this fold")

# %%
# E3 - and the ChEMBL side
_chembl_txt = os.path.join(CHEMBL_DIR, "chembl.txt")
if os.path.isfile(_chembl_txt):
    n = tot = 0
    with open(_chembl_txt) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            tot += 1
            n += ('.' in s)
    status = "PASS" if n == 0 else "FAIL"
    record("E3", "ChEMBL salt census", status,
           f"{n:,} of {tot:,} molecules in chembl.txt are multi-fragment",
           {"multi_fragment": n, "total": tot})
else:
    record("E3", "ChEMBL salt census", "SKIP", "chembl.txt absent")

# %% [markdown]
# ## The unconditional model
#
# Exactly the stage-1 architecture (no `linear_c`, no `dense_policy_0`), so
# `ckpt.pt` loads with `strict=True`. The sampling path (`_rnn_test`,
# `_decode_step`) is added here because the pretraining notebook only ever needed
# the training path - the parameters are untouched.

# %%
def segment_sum(X, idx, num_seg):
    out = torch.zeros((num_seg,) + tuple(X.shape[1:]), dtype=X.dtype, device=X.device)
    return out.index_add(0, idx, X)


def _concat_neighbourhoods(X, A_list):
    parts = [X]
    for A in A_list:
        parts.append(torch.sparse.mm(A, X) if A is not None else torch.zeros_like(X))
    return torch.cat(parts, dim=1)


def get_activation(name):
    return {'relu': torch.relu, 'tanh': torch.tanh}[name]


class Linear_BN(nn.Sequential):
    def __init__(self, F_in, F_out):
        super(Linear_BN, self).__init__(
            nn.Linear(F_in, F_out, bias=False),
            nn.BatchNorm1d(F_out, eps=1e-5, momentum=0.1))


class GraphConv(nn.Module):
    def __init__(self, Fin, Fout, D):
        super(GraphConv, self).__init__()
        self.Fin, self.Fout, self.D = Fin, Fout, D
        self.W = nn.Parameter(torch.empty(self.Fin * (self.D + 1), self.Fout))

    def forward(self, X, A_list):
        return _concat_neighbourhoods(X, A_list) @ self.W


class Policy(nn.Module):
    """Instrumented: records the raw pre-clamp logit range for check D."""

    def __init__(self, F_in, F_h, N_A, N_B, k=1):
        super(Policy, self).__init__()
        self.F_in, self.F_h, self.N_A, self.N_B, self.k = F_in, F_h, N_A, N_B, k
        self.linear_h = Linear_BN(F_in * 2, self.F_h * k)
        self.linear_h_t = Linear_BN(F_in, self.F_h * k)
        self.linear_x = nn.Linear(self.F_h, self.N_B + self.N_B * self.N_A)
        self.linear_x_t = nn.Linear(self.F_h, 1)
        self.linear_pi = nn.Linear(self.F_in, self.k) if self.k > 1 else None
        self.logit_stats = None          # (min, max, frac_at_clamp)

    def forward(self, X, NX, NX_rep, X_end=None):
        if X_end is None:
            X_end = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X, X_end[NX_rep, :]], dim=1)

        X_h = torch.relu(self.linear_h(X)).reshape(-1, self.F_h)
        X_h_end = torch.relu(self.linear_h_t(X_end)).reshape(-1, self.F_h)

        raw_x = self.linear_x(X_h).reshape(-1, self.k, self.N_B + self.N_B * self.N_A)
        raw_x_t = self.linear_x_t(X_h_end).reshape(-1, self.k, 1)

        with torch.no_grad():
            lo = min(float(raw_x.min()), float(raw_x_t.min()))
            hi = max(float(raw_x.max()), float(raw_x_t.max()))
            n_at = int((raw_x.abs() >= 80.0).sum()) + int((raw_x_t.abs() >= 80.0).sum())
            n_all = raw_x.numel() + raw_x_t.numel()
            prev = self.logit_stats
            self.logit_stats = (min(lo, prev[0]) if prev else lo,
                                max(hi, prev[1]) if prev else hi,
                                (prev[2] + n_at) if prev else n_at,
                                (prev[3] + n_all) if prev else n_all)

        logits_x = torch.clamp(raw_x, -80.0, 80.0)
        logits_x_t = torch.clamp(raw_x_t, -80.0, 80.0)

        X_x = torch.exp(logits_x)
        X_x_end = torch.exp(logits_x_t)

        X_sum = segment_sum(X_x, NX_rep, NX.shape[0]).sum(-1, keepdim=True) + X_x_end
        X_softmax = X_x / (X_sum[NX_rep, :, :] + 1e-10)
        X_softmax_end = X_x_end / (X_sum + 1e-10)

        X_softmax = X_softmax.squeeze(1)
        X_softmax_end = X_softmax_end.squeeze(1)

        connect, append = X_softmax[:, :self.N_B], X_softmax[:, self.N_B:]
        append = append.reshape(-1, self.N_A, self.N_B)
        return append, connect, X_softmax_end.squeeze(-1)


class UMoleculeGenerator(nn.Module):
    """Unconditional generator - parameter names match the stage-1 checkpoint."""
    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation,
                 *args, **kwargs):
        super(UMoleculeGenerator, self).__init__()
        self.N_A, self.N_B, self.D = N_A, N_B, D
        self.F_e, self.F_skip, self.Fh_policy = F_e, F_skip, Fh_policy
        self.F_c = list(F_c)
        self.activation = get_activation(activation)

        self.embedding_atom = nn.Embedding(self.N_A, self.F_e)
        self.embedding_mask = nn.Embedding(3, self.F_e)
        self._build_graph_conv(*args, **kwargs)

        self.dense = nn.Sequential()
        for i, (f_in, f_out) in enumerate(zip([self.F_skip] + self.F_c[:-1], self.F_c)):
            self.dense.add_module(str(i), Linear_BN(f_in, f_out))

        self.policy_0 = nn.Parameter(torch.zeros(self.N_A))
        self.policy_h = Policy(self.F_c[-1], self.Fh_policy, self.N_A, self.N_B)
        self.mode = 'decode_step'

    @abstractmethod
    def _build_graph_conv(self, *args, **kwargs):
        pass

    def _policy_0(self):
        """Stage-1's first-atom distribution: softmax over the learnable [65] vector."""
        return torch.softmax(self.policy_0, dim=0)


class UMoleculeGenerator_RNN(UMoleculeGenerator):
    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn,
                 *args, **kwargs):
        super(UMoleculeGenerator_RNN, self).__init__(
            N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, *args, **kwargs)
        self.N_rnn = N_rnn
        self.rnn = nn.GRU(input_size=self.F_c[-1] * 2, hidden_size=self.F_c[-1],
                          num_layers=self.N_rnn, batch_first=True)

    def _rnn_test(self, X, NX, NX_rep, NX_cum, h):
        X_avg = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X_avg, X[NX_cum - 1]], dim=1)
        X, h = self.rnn(X.unsqueeze(1), h)
        return X.squeeze(1), h

    def _decode_step(self, X, A, NX, NX_rep, last_append_mask, NX_cum, h):
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        X = self._graph_conv_forward(X, A)
        X_dense = self.dense(X)
        X_mol, h = self._rnn_test(X_dense, NX, NX_rep, NX_cum, h)
        append, connect, end = self.policy_h(X_dense, NX, NX_rep, X_mol)
        return append, connect, end, h

    def forward(self, *inputs):
        if self.mode == 'decode_0':
            return self._policy_0()
        if self.mode == 'decode_step':
            return self._decode_step(*inputs)
        raise ValueError(f"Unknown mode: {self.mode}")


class UVanillaMolGen_RNN(UMoleculeGenerator_RNN):
    def __init__(self, N_A, N_B, D, F_e, F_h, F_skip, F_c, Fh_policy, activation, N_rnn):
        super(UVanillaMolGen_RNN, self).__init__(
            N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, F_h)

    def _build_graph_conv(self, F_h):
        self.F_h = list(F_h)
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i, (f_in, f_out) in enumerate(zip([self.F_e] + self.F_h[:-1], self.F_h)):
            self.conv.append(GraphConv(f_in, f_out, self.N_B + self.D))
            if i != 0:
                self.bn.append(nn.BatchNorm1d(f_in, eps=1e-5, momentum=0.1))
        self.bn_skip = nn.BatchNorm1d(sum(self.F_h), eps=1e-5, momentum=0.1)
        self.linear_skip = Linear_BN(sum(self.F_h), self.F_skip)

    def _graph_conv_forward(self, X, A):
        X_out = [X]
        for i, conv in enumerate(self.conv):
            X = X_out[-1]
            if i == 0:
                X_out.append(conv(X, A))
            else:
                X_out.append(conv(self.activation(self.bn[i - 1](X)), A))
        X_out = torch.cat(X_out[1:], dim=1)
        return self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))


# %% [markdown]
# ## Decoding driver
#
# Same as `03_generate_samples`, with the conditioning arguments dropped.

# %%
def _rand_action(append_i, connect_i, end_i):
    xs = [append_i, connect_i, np.array([end_i])]
    flat = [np.reshape(x, [-1]) for x in xs]
    lengths = np.array([f.shape[0] for f in flat], dtype=np.int64)
    begin = np.cumsum(np.pad(lengths, [[1, 0]], mode='constant')[:-1])
    end_idx = np.cumsum(lengths) - 1

    p = np.concatenate(flat)
    total = np.sum(p)
    if not np.isfinite(total) or total <= 0 or np.isnan(p).any():
        rand_index = np.random.choice(np.arange(len(p)), 1)[0]
    else:
        p = p / total
        rand_index = np.random.choice(np.arange(p.shape[0]), 1, p=p)[0]

    x_index = np.where(np.logical_and(begin <= rand_index, end_idx >= rand_index))[0][0]
    local = rand_index - begin[x_index]
    return x_index, np.unravel_index(local, xs[x_index].shape)


def decode_step(X, A, NX, NA, last_action, finished, get_init, get_action,
                n_node_types, n_edge_types):
    if X is None:
        init = get_init()                       # [N_A], a single shared distribution
        n = len(finished)
        X = np.random.choice(np.arange(init.shape[0]), size=n, p=init).astype(np.int64)
        A = np.zeros((0, 3), dtype=np.int64)
        NX = last_action = np.ones([n], dtype=np.int64)
        NA = np.zeros([n], dtype=np.int64)
        finished = np.zeros([n], dtype=bool)
        return X, A, NX, NA, last_action, finished

    unfin = np.logical_not(finished)
    X_u = X[np.repeat(unfin, NX)]
    A_u = A[np.repeat(unfin, NA), :]
    NX_u = NX[unfin]
    NA_u = NA[unfin]
    last_action_u = last_action[unfin]

    NX_rep = np.repeat(np.arange(NX_u.shape[0]), NX_u)

    if A.shape[0] == 0:
        D_2 = D_3 = np.zeros((0, 2), dtype=np.int64)
        A_u = [np.zeros((0, 2), dtype=np.int64) for _ in range(n_edge_types)] + [D_2, D_3]
    else:
        cumsum = np.cumsum(np.pad(NX_u, [[1, 0]], mode='constant')[:-1])
        shift = np.repeat(cumsum, NA_u)
        A_u[:, :2] += np.stack([shift] * 2, axis=1)
        D_2, D_3 = get_d(A_u, X_u)
        A_u = [A_u[A_u[:, 2] == i, :2] for i in range(n_edge_types)] + [D_2, D_3]

    mask = np.zeros([X_u.shape[0]], dtype=np.int64)
    last_append_index = np.cumsum(NX_u) - 1
    mask[last_append_index] = np.where(last_action_u == 1,
                                       np.ones_like(last_append_index),
                                       np.ones_like(last_append_index) * 2)

    append, connect, end = get_action([X_u, A_u, NX_u, NX_rep, mask])

    if A.shape[0] == 0:
        # Second atom: upstream takes the argmax regardless of random_sample.
        max_index = np.argmax(np.reshape(append, [-1, n_node_types * n_edge_types]), axis=1)
        atom_type, bond_type = np.unravel_index(max_index, [n_node_types, n_edge_types])
        X = np.reshape(np.stack([X, atom_type], axis=1), [-1])
        NX = np.array([2] * len(finished), dtype=np.int64)
        A = np.stack([np.zeros([len(finished)], dtype=np.int64),
                      np.ones([len(finished)], dtype=np.int64),
                      bond_type], axis=1)
        NA = np.ones([len(finished)], dtype=np.int64)
        last_action = np.ones_like(NX, dtype=np.int64)
        return X, A, NX, NA, last_action, finished

    append = np.split(append, np.cumsum(NX_u))
    connect = np.split(connect, np.cumsum(NX_u))
    end = end.tolist()

    unfinished_ids = np.where(unfin)[0].tolist()
    cumsum = np.cumsum(NX)
    cumsum_a = np.cumsum(NA)

    X_insert, X_insert_ids, A_insert, A_insert_ids, finished_ids = [], [], [], [], []

    for uid, append_i, connect_i, end_i in zip(unfinished_ids, append, connect, end):
        action_type, action_index = _rand_action(append_i, connect_i, end_i)
        if action_type == 2:
            finished_ids.append(uid)
        elif action_type == 0:
            append_pos, atom_type, bond_type = action_index
            X_insert.append(atom_type)
            X_insert_ids.append(uid)
            A_insert.append([append_pos, NX[uid], bond_type])
            A_insert_ids.append(uid)
        else:
            connect_ps, bond_type = action_index
            A_insert.append([NX[uid] - 1, connect_ps, bond_type])
            A_insert_ids.append(uid)

    if A_insert_ids:
        A = np.insert(A, cumsum_a[A_insert_ids], A_insert, axis=0)
        NA[A_insert_ids] += 1
        last_action[A_insert_ids] = 0
    if X_insert_ids:
        X = np.insert(X, cumsum[X_insert_ids], X_insert, axis=0)
        NX[X_insert_ids] += 1
        last_action[X_insert_ids] = 1
    if finished_ids:
        finished[finished_ids] = True

    return X, A, NX, NA, last_action, finished


# %% [markdown]
# ## Check A - unconditional sampling from the stage-1 checkpoint
#
# **The decisive test.** It bisects the whole pipeline: everything downstream of
# stage 1 is bypassed.

# %%
class UncondSampler(object):
    def __init__(self, ckpt_path, device, config_path=None, d_receptive=D_RECEPTIVE):
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = state['model'] if isinstance(state, dict) and 'model' in state else state

        configs = state.get('configs') if isinstance(state, dict) else None
        if configs is None and config_path and os.path.isfile(config_path):
            with open(config_path) as f:
                configs = json.load(f)
        if configs is None:
            raise RuntimeError(f"No configs in {ckpt_path} and no configs.json beside it.")
        # The unconditional model has no N_C.
        configs = {k: v for k, v in configs.items() if k != 'N_C'}

        self.mdl = UVanillaMolGen_RNN(get_mol_spec().num_atom_types,
                                      get_mol_spec().num_bond_types,
                                      D=d_receptive, **configs)
        missing, unexpected = self.mdl.load_state_dict(sd, strict=False)
        self.missing, self.unexpected = list(missing), list(unexpected)
        self.mdl.to(device).eval()
        self.device = device
        self.configs = configs

    def _to_tensor(self, inputs):
        X, A, NX, NX_rep, mask = inputs
        NX_cum = np.cumsum(NX)
        dev = self.device
        to_long = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.long,
                                            device=dev)
        X_t, NX_t, NX_rep_t, mask_t, NX_cum_t = (to_long(X), to_long(NX), to_long(NX_rep),
                                                 to_long(mask), to_long(NX_cum))
        n_nodes = int(X_t.shape[0])

        A_sparse = []
        for A_i in A:
            if A_i.shape[0] == 0:
                A_sparse.append(None)
                continue
            A_i = np.concatenate([A_i, A_i[:, [1, 0]]], axis=0)
            idx = torch.as_tensor(A_i[:, :2].T.astype(np.int64), device=dev)
            val = torch.ones(A_i.shape[0], dtype=torch.float32, device=dev)
            A_sparse.append(torch.sparse_coo_tensor(
                idx, val, (n_nodes, n_nodes), dtype=torch.float32, device=dev).coalesce())
        return X_t, A_sparse, NX_t, NX_rep_t, mask_t, NX_cum_t

    @torch.no_grad()
    def sample(self, num_samples):
        finished = np.zeros([num_samples], dtype=bool)

        def get_init():
            self.mdl.mode = 'decode_0'
            p = self.mdl().cpu().numpy().astype(np.float64)
            return p / p.sum()

        X, A, NX, NA, last_action, finished = decode_step(
            None, None, None, None, None, finished,
            get_init=get_init, get_action=None,
            n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B)

        h = np.zeros([self.mdl.N_rnn, num_samples, self.mdl.F_c[-1]], dtype=np.float32)
        count = 1

        while not np.all(finished) and count < MAX_DECODE_STEPS:
            unfin = np.logical_not(finished)

            def get_action(inputs):
                self.mdl.mode = 'decode_step'
                _h = torch.as_tensor(h[:, unfin, :], dtype=torch.float32,
                                     device=self.device).contiguous()
                _X, _A, _NX, _NX_rep, _mask, _NX_cum = self._to_tensor(inputs)
                _ap, _cn, _en, _h_new = self.mdl(_X, _A, _NX, _NX_rep, _mask, _NX_cum, _h)
                h[:, unfin, :] = _h_new.cpu().numpy()
                return (_ap.cpu().numpy(), _cn.cpu().numpy(), _en.cpu().numpy())

            X, A, NX, NA, last_action, finished = decode_step(
                X, A, NX, NA, last_action, finished,
                get_init=None, get_action=get_action,
                n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B)
            count += 1

        cum_X = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')).tolist()
        cum_A = np.cumsum(np.pad(NA, [[1, 0]], mode='constant')).tolist()
        return ([[X[xa:xb], A[aa:ab, :]]
                 for aa, ab, xa, xb in zip(cum_A[:-1], cum_A[1:], cum_X[:-1], cum_X[1:])],
                int(np.sum(np.logical_not(finished))))


# %%
UNCOND = None
if CKPT_STATE is None:
    record("A", "unconditional validity", "SKIP", "ckpt.pt absent")
else:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    try:
        UNCOND = UncondSampler(PRETRAIN_CKPT, CTX, config_path=PRETRAIN_CONFIG)
        if UNCOND.missing:
            record("A0", "checkpoint transfer", "FAIL",
                   f"{len(UNCOND.missing)} tensors missing from ckpt.pt, "
                   f"e.g. {UNCOND.missing[:5]}")
        elif UNCOND.unexpected:
            record("A0", "checkpoint transfer", "WARN",
                   f"{len(UNCOND.unexpected)} unused tensors in ckpt.pt, "
                   f"e.g. {UNCOND.unexpected[:5]}")
        else:
            record("A0", "checkpoint transfer", "PASS",
                   "every stage-1 tensor loaded into the unconditional model")

        # The pretrained first-atom distribution, for comparison with 02's
        # freshly-initialised dense_policy_0.
        with torch.no_grad():
            UNCOND.mdl.mode = 'decode_0'
            p0 = UNCOND.mdl().cpu().numpy().astype(np.float64)
            p0 = p0 / p0.sum()
        ent0 = float(-(p0 * np.log(p0 + 1e-12)).sum())
        top5 = [(get_mol_spec().atom_types[i], round(float(p0[i]), 4))
                for i in np.argsort(-p0)[:5]]
        print(f"       pretrained policy_0: entropy {ent0:.3f} "
              f"(uniform = {math.log(len(p0)):.3f})")
        print(f"       top-5 first atoms: {top5}")

        t0 = time.time()
        graphs, n_truncated = UNCOND.sample(N_UNCOND_SAMPLES)
        elapsed = time.time() - t0

        reasons = Counter()
        n_atoms_valid, n_atoms_invalid = [], []
        for Xg, Ag in graphs:
            mol, why = get_mol_from_graph(Xg, Ag, sanitize=True)
            if mol is None:
                reasons[why.split(':')[0]] += 1
                n_atoms_invalid.append(int(Xg.shape[0]))
            else:
                reasons['valid'] += 1
                n_atoms_valid.append(int(Xg.shape[0]))

        n_valid = reasons['valid']
        validity = 100.0 * n_valid / max(1, len(graphs))

        print(f"\n       {len(graphs)} unconditional samples in {elapsed:.0f}s")
        print(f"       {n_truncated} hit the {MAX_DECODE_STEPS}-step cap "
              f"without emitting `end`")
        for k, v in reasons.most_common():
            print(f"         {v:>5}  {k}")

        _d = {"n_samples": len(graphs), "validity_pct": round(validity, 2),
              "n_truncated": n_truncated,
              "failure_reasons": dict(reasons.most_common()),
              "policy_0_entropy": round(ent0, 4),
              "policy_0_uniform_entropy": round(math.log(len(p0)), 4),
              "policy_0_top5": [list(t) for t in top5],
              "mean_atoms_valid": round(float(np.mean(n_atoms_valid)), 1)
              if n_atoms_valid else None,
              "mean_atoms_invalid": round(float(np.mean(n_atoms_invalid)), 1)
              if n_atoms_invalid else None}

        if validity >= 80.0:
            record("A", "unconditional validity", "PASS",
                   f"{validity:.1f}% valid. STAGE 1 IS HEALTHY - the collapse happened "
                   f"in fine-tuning. Do NOT re-run pretraining; fix 02 (salts, model "
                   f"selection, conditioning) and re-fine-tune", _d)
        elif validity >= 50.0:
            record("A", "unconditional validity", "WARN",
                   f"{validity:.1f}% valid - stage 1 is degraded but not destroyed. "
                   f"Both stages contribute", _d)
        else:
            record("A", "unconditional validity", "FAIL",
                   f"{validity:.1f}% valid. STAGE 1 IS THE CAUSE. The unbounded Eq. 8 "
                   f"reversal damaged the generator before fine-tuning ever ran - "
                   f"stage 1 must be redone (CDGCN objective, or convert the existing "
                   f"MXNet CDGCN checkpoint)", _d)

        # Check D rides along on the same forward passes.
        ls = UNCOND.mdl.policy_h.logit_stats
        if ls is None:
            record("D", "logit clamp", "SKIP", "no logits recorded")
        else:
            lo, hi, n_at, n_all = ls
            frac = n_at / max(1, n_all)
            _dd = {"logit_min": round(lo, 2), "logit_max": round(hi, 2),
                   "frac_at_clamp": round(frac, 8), "n_logits": n_all}
            print(f"\n       logit range [{lo:.1f}, {hi:.1f}] over {n_all:,} logits; "
                  f"{100 * frac:.4f}% at the +/-80 clamp")
            if frac == 0.0:
                record("D", "logit clamp", "PASS",
                       f"logits span [{lo:.1f}, {hi:.1f}] - the +/-80 clamp NEVER binds, "
                       f"so it is not implicated. Leave Policy.forward alone", _dd)
            elif frac < 1e-5:
                record("D", "logit clamp", "WARN",
                       f"the clamp binds on {100 * frac:.5f}% of logits - marginal",
                       _dd)
            else:
                record("D", "logit clamp", "FAIL",
                       f"the clamp binds on {100 * frac:.3f}% of logits, flattening the "
                       f"action distribution and zeroing those gradients. Replace with a "
                       f"log-space max-shifted softmax in ALL THREE notebooks", _dd)

    except Exception as e:
        import traceback
        traceback.print_exc()
        record("A", "unconditional validity", "FAIL", f"{type(e).__name__}: {e}")

# %% [markdown]
# ## Check B - why do the *conditional* samples fail?
#
# Reads the CSVs `03_generate_samples` already wrote, so it costs nothing. If
# check A came back healthy, this is where the answer is.

# %%
_run1 = os.path.join(SAMPLES_DIR, "run1")
_csvs = sorted(glob.glob(os.path.join(_run1, "protein_*.csv")))

if not _csvs:
    record("B", "conditional sample census", "SKIP",
           f"no sample CSVs under {_run1}")
    SAMPLE_ROWS = None
else:
    rows = []
    for p in _csvs:
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append({"smiles": r.get("smiles", ""),
                             "n_atoms": int(r.get("n_atoms") or 0)})
    SAMPLE_ROWS = rows
    n_valid = sum(1 for r in rows if r["smiles"])
    record("B", "conditional sample census", "INFO",
           f"{n_valid:,} of {len(rows):,} samples carry a SMILES "
           f"({100 * n_valid / max(1, len(rows)):.1f}%)",
           {"n_rows": len(rows), "n_valid": n_valid})

# %% [markdown]
# ## Check C - is truncation at `MAX_DECODE_STEPS = 100` implicated?
#
# If invalid molecules pile up near the cap, raising it to `MAX_ITER = 120` helps.
# If they sit at ordinary sizes, the cap is irrelevant and should be left at the
# upstream value.

# %%
if SAMPLE_ROWS is None:
    record("C", "decode truncation", "SKIP", "no sample CSVs")
else:
    av = np.array([r["n_atoms"] for r in SAMPLE_ROWS if r["smiles"]], dtype=float)
    ai = np.array([r["n_atoms"] for r in SAMPLE_ROWS if not r["smiles"]], dtype=float)

    def _describe(a, label):
        if a.size == 0:
            print(f"       {label:9s} (none)")
            return {}
        q = np.percentile(a, [50, 90, 99])
        print(f"       {label:9s} n={a.size:>6,}  mean {a.mean():6.1f}  "
              f"median {q[0]:5.1f}  p90 {q[1]:5.1f}  p99 {q[2]:5.1f}  max {a.max():5.0f}")
        return {"n": int(a.size), "mean": round(float(a.mean()), 1),
                "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
                "max": float(a.max())}

    d_valid = _describe(av, "valid")
    d_invalid = _describe(ai, "invalid")

    near_cap = float((ai >= MAX_DECODE_STEPS - 5).mean()) if ai.size else 0.0
    _d = {"valid": d_valid, "invalid": d_invalid,
          "frac_invalid_near_cap": round(near_cap, 4),
          "max_decode_steps": MAX_DECODE_STEPS}

    if near_cap >= 0.25:
        record("C", "decode truncation", "FAIL",
               f"{100 * near_cap:.0f}% of invalid molecules sit within 5 atoms of the "
               f"{MAX_DECODE_STEPS}-step cap. Raise MAX_DECODE_STEPS to MAX_ITER = 120",
               _d)
    elif near_cap >= 0.05:
        record("C", "decode truncation", "WARN",
               f"{100 * near_cap:.0f}% of invalid molecules are near the cap - a minor "
               f"contributor", _d)
    else:
        record("C", "decode truncation", "PASS",
               f"only {100 * near_cap:.1f}% of invalid molecules are near the cap; "
               f"truncation is NOT the cause. Leave MAX_DECODE_STEPS at "
               f"{MAX_DECODE_STEPS} (upstream behaviour)", _d)

# %% [markdown]
# ## Summary

# %%
_order = {"FAIL": 0, "WARN": 1, "SKIP": 2, "INFO": 3, "PASS": 4}
print("=" * 78)
print(f"PHASE-0 DIAGNOSTICS - fold {FOLD} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 78)
for v in sorted(VERDICTS, key=lambda r: (_order.get(r["status"], 9), r["check"])):
    print(f"  {v['status']:5s}  {v['check']:5s}  {v['name']}")
    print(f"              {v['detail']}")

counts = Counter(v["status"] for v in VERDICTS)
print("-" * 78)
print("  " + "  ".join(f"{k}: {counts[k]}" for k in
                       ("FAIL", "WARN", "PASS", "INFO", "SKIP") if counts[k]))

with open(REPORT_PATH, 'w') as f:
    json.dump({"fold": FOLD, "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
               "repo_root": REPO_ROOT, "verdicts": VERDICTS}, f, indent=1)
print(f"\n  report -> {REPORT_PATH}")

# %% [markdown]
# ## What to do with this
#
# Read check **A** first; it decides the shape of the work.
#
# **A = FAIL (validity < 50%)** - stage 1 is the cause.
# 1. Re-run stage 1 with `TOX_WEIGHTING = False` (CDGCN objective), `BATCH_SIZE = 8`,
#    the bond cap and salt stripping on - or convert the existing MXNet CDGCN
#    checkpoint at `../CTDDG_new/outputs/pretrain/` (loss 15.99, same architecture,
#    same 6,986,506 parameters) and skip the GPU time entirely.
# 2. Then apply the `02` fixes and fine-tune from the clean checkpoint.
#
# **A = PASS (validity > 80%)** - stage 1 is fine; no pretraining re-run.
# 1. Fix `02`: salt stripping, `EARLY_STOP_ON = 'l_nontoxic'`, `PATIENCE = 3`,
#    the resume position.
# 2. Re-fine-tune and re-sample.
#
# **Independently of A:**
# * **F / F2 FAIL** -> leakage is real; stage 1 has to be redone on the filtered set
#   whatever else is true.
# * **E2 FAIL** -> salt stripping is mandatory in `02` before any further fine-tuning.
# * **H FAIL** -> re-run the ChEMBL annotation cells so both phases share one forest.
# * **D PASS** -> leave `Policy.forward` alone; the clamp is a red herring.
# * **C PASS** -> leave `MAX_DECODE_STEPS = 100`; it is upstream behaviour.

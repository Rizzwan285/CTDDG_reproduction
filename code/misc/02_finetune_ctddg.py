# %% [markdown]
# # Stage 2 — conditional fine-tuning (CTDDG Eq. 8)
#
# Loads the stage-1 checkpoint, adds the protein conditioning path, and trains on
# BindingDB protein-ligand pairs under the toxicity-weighted objective.
#
# **Nothing trains until `RUN_TRAINING = True`.** Everything before it is
# preflight: stage-1 provenance, FASTA order, the conditioning actually helping,
# Eq. 8 against its closed form, and the step-0 likelihood distribution.
#
# **Needs** `<fold>_<tr|te>_unique_embeddings_bepler.h5`: one average-pooled
# `(6165,)` vector per protein, keyed `p1 ... pN`. Per-residue output is rejected.

# %%
import os
import gc
import glob
import json
import math
import time
import pickle
import random
import signal
import hashlib
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.autograd.function import once_differentiable

print(f"torch {torch.__version__} | rdkit {rdkit.__version__}")

# %% [markdown]
# ## Config

# %%
FORCE_RESTART = False
RUN_TRAINING = False          # the arm switch

FOLD = os.environ.get("CTDDG_FOLD", "d1")
GPU_ID = int(os.environ.get("CTDDG_GPU", "0"))
CTX = torch.device(f"cuda:{GPU_ID}") if torch.cuda.is_available() else torch.device("cpu")

REPO_ROOT = os.environ.get("CTDDG_REPO_ROOT", "/home/muhamed/repo/CTDDG")
ATOM_TYPES_PATH = os.path.join(REPO_ROOT, "datasets/data/atom_types.txt")
BINDINGDB_DIR = os.path.join(REPO_ROOT, "datasets/data/bindingdb")
RF_MODEL_PATH = os.path.join(REPO_ROOT, "models/ames_random_forest.pkl")
FP_CONFIG_PATH = os.path.join(REPO_ROOT, "models/fingerprint_config.pkl")
PRETRAINED_CKPT = os.path.join(REPO_ROOT, "checkpoints/ckpt.pt")

OUT_DIR = os.path.join(REPO_ROOT, "outputs/finetune", FOLD)
os.makedirs(OUT_DIR, exist_ok=True)

REQUIRE_PRETRAINED = True
ALLOW_UNTRAINED_CKPT = False

FOLD_SUFFIX = {"d1": "4_org_1042_104", "d2": "4_org_1000_112", "d3": "4_org_1004_122",
               "d4": "4_org_1002_103", "d5": "4_org_1036_124"}

AMES_CACHE = os.path.join(BINDINGDB_DIR, "ligand_ames_labels.csv")
AMES_CACHE_META = AMES_CACHE + ".meta.json"

LIMIT_PAIRS = None            # a number makes this a SMOKE TEST
SMOKE_TEST = LIMIT_PAIRS is not None
_SFX = ".smoke" if SMOKE_TEST else ""

CKPT_BEST = os.path.join(OUT_DIR, "ckpt_best.pt")
CKPT_LAST = os.path.join(OUT_DIR, "ckpt_last.pt")
CKPT_TRAINER = os.path.join(OUT_DIR, "trainer.pt")
CKPT_CONFIG = os.path.join(OUT_DIR, "configs.json")
LOG_FILE = os.path.join(OUT_DIR, "log.out")
VAL_LOG_FILE = os.path.join(OUT_DIR, "val.out")
SPLIT_MANIFEST = os.path.join(OUT_DIR, f"split_manifest{_SFX}.json")
TEST_BUNDLE = os.path.join(OUT_DIR, f"test_set{_SFX}.npz")
STEP0_PATH = os.path.join(OUT_DIR, f"step0_likelihoods{_SFX}.npz")
CKPT_PREV = CKPT_LAST + ".prev"

# %% [markdown]
# ### Data and split
#
# The paper evaluates on all held-out proteins, so the whole test split is the
# test set and validation is carved out of TRAIN **by protein** — early stopping
# for a conditional model has to measure generalisation to unseen proteins.

# %%
VAL_PROTEIN_FRACTION = 0.05
SPLIT_SEED = 42

MAX_ATOMS = 100
MAX_ITER = 120
STRIP_SALTS = True
FRAGMENT_POLICY = 'heavy_atoms'

# 'complete_dfs' finishes every branch point. 'upstream_broken'
# reproduces the traversal that silently dropped atoms.

# %%
TRAVERSAL_POLICY = 'complete_dfs'

# The pair files are a SUPERSET of the fold: they hold pairs for proteins outside
# this split. The fold is defined by <fold>_<tr|te>_unique_proteins.txt, so rows
# are pooled and routed by list membership. Pooling cannot leak: the two protein
# lists are checked to be disjoint.

# %%
POOL_PAIR_FILES = True
UNMATCHED_TOLERANCE = {"train": 0.60, "test": 0.995}

# %% [markdown]
# ### Model and schedule

# %%
BATCH_SIZE = 8
BATCH_SIZE_VAL = 8
K_PATHS = 5
P_ALPHA = 0.8
LAMBDA_TOX = 1.0
N_C = 6165
ALLOW_N_C_OVERRIDE = False
EMB_EXTENSIONS = ('.h5', '.hdf5', '.npy', '.txt', '.csv')

F_E = 16
F_H = [32, 64, 128, 128, 256, 256]
F_SKIP = 256
F_C = [512]
FH_POLICY = 128
ACTIVATION = 'relu'
N_RNN = 3
D_RECEPTIVE = 2

LR = 1e-4
DECAY = 0.002
DECAY_STEP = 100
CLIP_GRAD = 3.0
SUMMARY_STEP = 200
CHECKPOINT_EVERY = 200
MAX_EPOCHS = 10
PATIENCE = 3                  # 10 with MAX_EPOCHS 10 can never fire
ES_DELTA = 1e-2

STABLE_SOFTMAX = True
VERIFY_FASTA_ORDER = True
GRACEFUL_SIGNALS = True
NUM_WORKERS = 8
SEED = 42

# %% [markdown]
# ### Model selection and the toxic branch
#
# * `l_nontoxic` — mean `l` over NON-TOXIC validation molecules. **Default.** The
#   only quantity Eq. 8 is not trying to move.
# * `nll` — a *working* CTDDG drives this UP by construction, so selecting on it
#   penalises success. This is what picked epoch 1 last time.
# * `weighted` — the Eq. 8 objective itself, unbounded below on the toxic branch.
#
# `LAMBDA_TOX = 0` does NOT recover CDGCN: the weight `1-(1+0)*tox` is 0 on toxic
# ligands, dropping them from the batch average.

# %%
EARLY_STOP_ON = 'l_nontoxic'
NLL_DIVERGENCE_MARGIN = 2.0

# A float applies `l = max(l, TOX_FLOOR)` to toxic ligands in the OBJECTIVE only;
# reported metrics stay unfloored. Do not pick a value by eye: a floor only binds
# while toxic likelihood is above it, so applied to a model that has already
# driven toxic ligands down it clamps everything from iteration 1 and the branch
# is inert while still reporting as Eq. 8. The step-0 cell measures the
# distribution the value has to come from.

# %%
TOX_FLOOR = None
MEASURE_STEP0 = True

# Warm-start FNN_0's output bias from the pretrained first-atom marginal, so at
# step 0 the conditional distribution IS that marginal. CDGCN specifies the
# substitution, not the initialisation; this is a deliberate departure from Xavier.

# %%
INIT_POLICY0_BIAS_FROM_PRETRAINED = True

# %% [markdown]
# ## Check scaffolding

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

def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()

# %%
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"device {CTX} | fold {FOLD}" + ("  [SMOKE TEST]" if SMOKE_TEST else ""))

_missing = [f"{lab}: {p}" for lab, p in
            (("atom_types.txt", ATOM_TYPES_PATH), ("bindingdb dir", BINDINGDB_DIR),
             ("fingerprint config", FP_CONFIG_PATH)) if not os.path.exists(p)]
check("required inputs", not _missing,
      "all present" if not _missing else "; ".join(_missing))

check("early stopping can fire", PATIENCE < MAX_EPOCHS,
      f"PATIENCE = {PATIENCE} < MAX_EPOCHS = {MAX_EPOCHS}" if PATIENCE < MAX_EPOCHS
      else f"PATIENCE >= MAX_EPOCHS: epoch 1 always improves, so the counter tops "
           f"out at {MAX_EPOCHS - 1} and early stopping can NEVER fire", level="warn")

check("model selection metric", EARLY_STOP_ON == 'l_nontoxic',
      f"EARLY_STOP_ON = {EARLY_STOP_ON!r}"
      + ("" if EARLY_STOP_ON == 'l_nontoxic' else
         " — 'nll' rises when CTDDG works and 'weighted' is unbounded below; both "
         "select the wrong checkpoint"), level="warn")

if not os.path.exists(PRETRAINED_CKPT) and REQUIRE_PRETRAINED:
    raise FileNotFoundError(
        f"No pretrained checkpoint at {PRETRAINED_CKPT}. CTDDG is a two-phase "
        f"method: stage 1 pretrains unconditionally on ChEMBL, stage 2 fine-tunes "
        f"that model. Starting from random weights is a different experiment.")

# %% [markdown]
# ## Restart cleanup
#
# Before any data work: a crash during assembly must not leave a resumable
# checkpoint from the previous run for the next one to pick up.

# %%
RESTART_STAMP = time.strftime('%Y%m%d-%H%M%S')
RESUMABLE = (CKPT_BEST, CKPT_LAST, CKPT_PREV, CKPT_TRAINER, LOG_FILE, VAL_LOG_FILE)

# %%

def archive_stale(paths, stamp=None, why=""):
    stamp = stamp or RESTART_STAMP
    moved = []
    for p in paths:
        if os.path.exists(p):
            dst = f"{p}.stale.{stamp}"
            os.replace(p, dst)
            moved.append(dst)
            print(f"  {os.path.basename(p)} -> {os.path.basename(dst)}"
                  + (f"  ({why})" if why else ""))
    return moved

# %%
if FORCE_RESTART:
    archive_stale(RESUMABLE, why="FORCE_RESTART")

# %% [markdown]
# ## Stage-1 provenance
#
# A state dict that loads cleanly proves the shapes match, not that any training
# happened — and not that it happened under the traversal this notebook uses.

# %%
def _last_log_step(path):
    last, n = None, 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('#', 'step\t')) or line == 'Training finished':
                    continue
                try:
                    last = int(line.split('\t')[0])
                    n += 1
                except (ValueError, IndexError):
                    continue
    except OSError:
        return None, 0

    return last, n

# %%

def pretrained_provenance(path):
    """Evidence only. `max(num_batches_tracked)` is the EXACT iteration count:
    the only train-mode forward in pretraining is the one per training step."""
    ev = {'path': path, 'bn_steps': None, 'trainer_counter': None,
          'log_last_step': None, 'log_rows': 0, 'objective': None, 'data': None}
    ckpt_dir = os.path.dirname(path)
    try:
        st = os.stat(path)
        ev['size_mb'] = st.st_size / 1e6
        ev['mtime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))
    except OSError:
        return ev

    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
        sd = state['model'] if isinstance(state, dict) and 'model' in state else state
        nbt = [int(v) for k, v in sd.items() if k.endswith('num_batches_tracked')]
        ev['bn_steps'] = max(nbt) if nbt else None
        if isinstance(state, dict):
            ev['objective'] = state.get('objective')
            ev['data'] = state.get('data')
    except Exception as e:
        print(f"[WARN] could not read {os.path.basename(path)}: {type(e).__name__}: {e}")

    tr = os.path.join(ckpt_dir, 'trainer.pt')
    if os.path.isfile(tr):
        try:
            t = torch.load(tr, map_location='cpu', weights_only=False)
            if isinstance(t, dict):
                ev['trainer_counter'] = t.get('global_counter')
                ev['objective'] = ev['objective'] or t.get('objective')
                ev['data'] = ev['data'] or t.get('data')
        except Exception:
            pass

    ev['log_last_step'], ev['log_rows'] = _last_log_step(
        os.path.join(ckpt_dir, 'log.out'))

    print(f"\n--- stage-1 provenance: {os.path.basename(path)} ---")
    print(f"  size / modified     {ev.get('size_mb', 0):.1f} MB  {ev.get('mtime')}")
    print(f"  BN train-mode steps {ev['bn_steps']}   (EXACT iteration count)")
    print(f"  trainer.pt counter  {ev['trainer_counter']}")
    print(f"  log.out             {ev['log_rows']} rows, last step {ev['log_last_step']}")
    print(f"  objective           {(ev['objective'] or {}).get('objective')}")
    print(f"  data                {(ev['data'] or {}).get('data_path')}")
    print("-" * 60)

    return ev

# %%
PRETRAIN_EVIDENCE = pretrained_provenance(PRETRAINED_CKPT) \
    if os.path.exists(PRETRAINED_CKPT) else None

# %%
if PRETRAIN_EVIDENCE:
    if PRETRAIN_EVIDENCE['bn_steps'] == 0 and not ALLOW_UNTRAINED_CKPT:
        raise RuntimeError(
            f"{PRETRAINED_CKPT} reports num_batches_tracked == 0 in every "
            f"BatchNorm layer: it has never seen a training step.")

    check("stage-1 checkpoint trained", bool(PRETRAIN_EVIDENCE['bn_steps']),
          f"{PRETRAIN_EVIDENCE['bn_steps']:,} iterations by the BatchNorm counter"
          if PRETRAIN_EVIDENCE['bn_steps'] else "no evidence of training",
          level="warn")

    _obj = PRETRAIN_EVIDENCE['objective'] or {}
    _dat = PRETRAIN_EVIDENCE['data'] or {}

    if _obj:
        check("stage-1 objective", _obj.get('tox_weighting') is False,
              f"stage 1 used {_obj.get('objective')}. "
              + ("Good — CDGCN's unweighted NLL, as upstream does."
                 if _obj.get('tox_weighting') is False else
                 "Eq. 8 in stage 1 is the ablation that collapsed validity"),
              level="warn")
        check("stage-1 softmax matches this run",
              _obj.get('stable_softmax') in (None, STABLE_SOFTMAX),
              f"stage 1 ran with stable_softmax={_obj.get('stable_softmax')}, this "
              f"run uses {STABLE_SOFTMAX}", level="warn")
    else:
        check("stage-1 objective recorded", False,
              "the checkpoint records no objective, so which one stage 1 used is "
              "UNRESOLVED from the artifacts. Only the pretraining source settles it",
              level="warn")

    _tp = _dat.get('traversal_policy')
    check("stage-1 traversal", _tp == 'complete_dfs',
          f"stage 1 ran with traversal_policy={_tp!r}. "
          + ("Targets were built by a complete DFS." if _tp == 'complete_dfs' else
             "A checkpoint predating the traversal fix was trained on targets in "
             "which ~21% of traversals left atoms unvisited and every bond of those "
             "molecules was mis-wired. Fine-tuning cannot repair that — re-run "
             "stage 1. See AUDIT.md A21"),
          level="warn" if _tp is None else ("fatal" if _tp != 'complete_dfs' else "warn"))

    check("stage-1 fragment policy matches",
          _dat.get('fragment_policy') in (None, FRAGMENT_POLICY),
          f"stage 1 used {_dat.get('fragment_policy')!r}, this run uses "

          f"{FRAGMENT_POLICY!r}", level="warn")

# %%
gate("data assembly")

# %% [markdown]
# ## Shared primitives
#
# Byte-identical to the pretraining notebook; `99_consistency_audit.py` enforces
# that. Both phases must decompose a molecule into actions the same way or the
# transferred weights mean nothing.

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
        self.max_iter = MAX_ITER

    def get_atom_type(self, atom):
        return self.atom_types.index(
            (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetNumExplicitHs()))

    def get_bond_type(self, bond):
        return self.bond_orders.index(bond.GetBondType())

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

# %%
_mol_spec = None

# %%

def get_mol_spec():
    global _mol_spec
    if _mol_spec is None:
        _mol_spec = MoleculeSpec(ATOM_TYPES_PATH)
    return _mol_spec

# %%

def get_graph_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    atom_types, atom_ranks, bonds, bond_types = [], [], [], []
    for a, r in zip(mol.GetAtoms(), Chem.CanonicalRankAtoms(mol)):
        atom_types.append(get_mol_spec().get_atom_type(a))
        atom_ranks.append(r)

    for b in mol.GetBonds():
        bonds.append([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
        bond_types.append(get_mol_spec().get_bond_type(b))

    graph = nx.Graph()
    graph.add_nodes_from(range(len(atom_types)))
    graph.add_edges_from(bonds)

    # traverse_graph cannot walk a disconnected graph: it would leave the second
    # component at step_ids = -1 and single_reorder would mis-wire every bond.
    if len(atom_types) > 1 and not nx.is_connected(graph):
        raise ValueError(f"Disconnected molecular graph (salt not stripped?): {smiles}")

    return graph, atom_types, atom_ranks, bonds, bond_types

# %%

def traverse_graph(graph, atom_ranks, current_node=None, step_ids=None,
                   p=P_ALPHA, log_p=0.0, _top=True):
    """Q_alpha proposal: canonical rank with probability p, random neighbour otherwise.

    After a branch returns, the candidate set is recomputed from `current_node`.
    The upstream version recomputed it from the node it had just recursed into,
    which abandons any branch point whose first branch is a dead end. Measured
    (`measure_traversal.py`): tert-butylbenzene lost atoms on 100% of traversals,
    13 of 20 drug-like molecules were affected, 21% of traversals overall. Those
    molecules reached `single_reorder` with `step_ids = -1` entries, which
    argsorts them to the front and mis-wires every bond - silently.

    TRAVERSAL_POLICY = 'upstream_broken' restores that behaviour for A/B only.
    """
    if current_node is None:
        step_ids = [-1] * len(atom_ranks)
        pool = list(range(len(atom_ranks)))
    else:
        pool = list(graph.neighbors(current_node))

    def _unvisited_sorted(from_pool):
        ns = [n for n in from_pool if step_ids[n] < 0]
        return [n for n, r in sorted(zip(ns, [atom_ranks[n] for n in ns]),
                                     key=lambda x: x[1])]

    next_nodes = _unvisited_sorted(pool)
    while len(next_nodes) > 0:
        if len(next_nodes) == 1:
            next_node = next_nodes[0]
        elif random.random() >= (1.0 - p):
            next_node = next_nodes[0]
            log_p += np.log(p)
        else:
            next_node = next_nodes[random.randint(1, len(next_nodes) - 1)]
            log_p += np.log((1.0 - p) / (len(next_nodes) - 1))

        step_ids[next_node] = max(step_ids) + 1
        _, log_p = traverse_graph(graph, atom_ranks, next_node, step_ids, p, log_p,
                                  _top=False)
        if TRAVERSAL_POLICY == 'upstream_broken':
            next_nodes = _unvisited_sorted(list(graph.neighbors(next_node)))
        else:
            next_nodes = _unvisited_sorted(pool)

    if _top and min(step_ids) < 0:
        raise ValueError(
            f"traverse_graph left {sum(1 for s in step_ids if s < 0)} atoms "
            f"unvisited on a connected graph. single_reorder would mis-wire every "
            f"bond. TRAVERSAL_POLICY = {TRAVERSAL_POLICY!r}")

    return step_ids, log_p

# %%

def single_reorder(X_0, A_0, step_ids):
    """Relabel atoms by traversal order and rewrite the bond endpoints.

    Validates its own contract first. This function is where a bad `step_ids`
    does its damage: an unvisited atom carries -1, `argsort` puts it at position
    0, and every bond endpoint is then remapped through a wrong table, producing
    a different molecule with no error raised (AUDIT.md A21). `traverse_graph`
    already refuses to return such a permutation, but the guard belongs here too
    -- this is the function that corrupts the target, and it must not trust its
    caller.

    The check reuses the argsort computed below, so it costs one vectorised
    comparison: measured 31% of this function's own runtime, about 83 s per
    1.09M-molecule epoch at K=5, inside the dataloader workers.
    """
    X_0, A_0 = np.copy(X_0), np.copy(A_0)
    step_ids = np.array(step_ids, dtype=np.int32)
    sorted_ids = np.argsort(step_ids)

    n_atoms = X_0.shape[0]
    if not np.array_equal(step_ids[sorted_ids], np.arange(n_atoms, dtype=np.int32)):
        n_unvisited = int((step_ids < 0).sum())
        raise ValueError(
            f"step_ids is not a permutation of 0..{n_atoms - 1}: "
            f"{n_unvisited} atom(s) unvisited, "
            f"{len(np.unique(step_ids))} distinct value(s) for {n_atoms} atoms. "
            f"Remapping bond endpoints through this would silently build a "
            f"different molecule. See AUDIT.md A21.")
    if A_0.shape[0] and int(A_0[:, :2].min()) < 0:
        raise ValueError(
            f"bond endpoint index {int(A_0[:, :2].min())} is negative; it would "
            f"wrap round the step_ids table and mis-wire the bond silently.")

    X_0 = X_0[sorted_ids]
    A_0[:, 0], A_0[:, 1] = step_ids[A_0[:, 0]], step_ids[A_0[:, 1]]

    max_b, min_b = np.amax(A_0[:, :2], axis=1), np.amin(A_0[:, :2], axis=1)
    A_0 = A_0[np.lexsort([-min_b, max_b]), :]

    max_b, min_b = np.amax(A_0[:, :2], axis=1), np.amin(A_0[:, :2], axis=1)
    is_append = np.concatenate([np.array([True]), max_b[1:] > max_b[:-1]])
    A_0 = np.concatenate([
        np.where(is_append[:, np.newaxis], np.stack([min_b, max_b], axis=1),
                 np.stack([max_b, min_b], axis=1)),
        A_0[:, -1:]
    ], axis=1)

    return X_0, A_0

# %%

def single_expand(X_0, A_0):
    X_0, A_0 = np.copy(X_0), np.copy(A_0)
    is_append_iter = np.less(A_0[:, 0], A_0[:, 1]).astype(np.int32)

    NX = np.cumsum(np.pad(is_append_iter, [[1, 0]], mode='constant', constant_values=1))
    shift = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')[:-1])
    X_index = np.arange(NX.sum(), dtype=np.int32) - np.repeat(shift, NX)
    X = X_0[X_index]

    _, A_index = np.tril_indices(A_0.shape[0])
    A = A_0[A_index, :]
    NA = np.arange(A_0.shape[0] + 1)

    action_type = 1 - is_append_iter
    atom_type = np.where(action_type == 0, X_0[A_0[:, 1]], 0)
    bond_type = A_0[:, 2]
    append_pos = np.where(action_type == 0, A_0[:, 0], 0)
    connect_pos = np.where(action_type == 1, A_0[:, 1], 0)
    actions = np.stack([action_type, atom_type, bond_type, append_pos, connect_pos],
                       axis=1)
    actions = np.append(actions, [[2, 0, 0, 0, 0]], axis=0)
    action_0 = np.array([X_0[0]], dtype=np.int32)

    last_atom_index = shift + NX - 1
    last_atom_mask = np.zeros_like(X)
    last_atom_mask[last_atom_index] = np.where(
        np.pad(is_append_iter, [[1, 0]], mode='constant', constant_values=1) == 1,
        np.ones_like(last_atom_index),
        np.ones_like(last_atom_index) * 2
    )
    return action_0, X, NX, A, NA, actions, last_atom_mask

# %%

def get_d(A, X):
    def _to_sparse(_A, _X):
        return sparse.coo_matrix(
            (np.ones([_A.shape[0] * 2], dtype=np.int32),
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

# %%

def merge_single_0(X_0, A_0, NX_0, NA_0):
    cumsum = np.cumsum(np.pad(NX_0, [[1, 0]], mode='constant')[:-1])
    A_0[:, :2] += np.stack([np.repeat(cumsum, NA_0)] * 2, axis=1)
    D_0_2, D_0_3 = get_d(A_0, X_0)

    A_split = []
    for i in range(get_mol_spec().num_bond_types):
        A_split.append(A_0[A_0[:, 2] == i, :2])
    A_split.extend([D_0_2, D_0_3])
    NX_rep_0 = np.repeat(np.arange(NX_0.shape[0]), NX_0)

    return X_0, A_split, NX_0, NX_rep_0

# %%

def merge_single(X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions,
                 last_append_mask, log_p):
    X, A, NX, NX_rep = merge_single_0(X, A, NX, NA)
    cumsum = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')[:-1])
    actions[:, -2] += cumsum * (actions[:, 0] == 0)
    actions[:, -1] += cumsum * (actions[:, 0] == 1)
    mol_ids_rep = np.repeat(mol_ids, NX)
    rep_ids_rep = np.repeat(rep_ids, NX)
    return (X, A, mol_ids_rep, rep_ids_rep, iw_ids, last_append_mask, NX, NX_rep,
            action_0, actions, log_p)

# %% [markdown]
# ### Self-tests
#
# Seconds, CPU. If these fail every training target below is suspect.

# %%
_T = []

# %%

def _t(name, ok, detail):
    _T.append((name, bool(ok), detail))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {detail}")

# %%
_cases = [("Cc1ccccc1.O=S(=O)(O)O", "Cc1ccccc1"),
          ("CC(=O)Oc1ccccc1C(=O)O.Cl", "CC(=O)Oc1ccccc1C(=O)O"), ("CCO", "CCO")]
_bad = [(s, strip_salt(s)) for s, w in _cases if strip_salt(s) != w]
_t("strip_salt picks the principal fragment", not _bad,
   f"policy={FRAGMENT_POLICY}" if not _bad else f"{_bad}")

# Branched molecules: the upstream traversal abandoned these. All must now give a
# complete permutation.

# %%
random.seed(SEED)
_branchy = ["CC(C)(C)c1ccccc1", "Cc1ccc(cc1)S(=O)(=O)N", "CC(C)(N)Cc1ccccc1",
            "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "CC(=O)Oc1ccccc1C(=O)O",
            "CN(C)CCOC(c1ccccc1)c1ccccc1"]
_inc = 0
for _smi in _branchy:
    _g, _at, _ar, _b, _bt = get_graph_from_smiles(_smi)
    for _ in range(50):
        _ids, _lp = traverse_graph(_g, _ar)
        if sorted(_ids) != list(range(len(_at))) or _lp > 0:
            _inc += 1
_t("traversal is complete on branched molecules", _inc == 0,
   f"{len(_branchy)} molecules x 50 traversals, {_inc} incomplete. These are the "
   f"molecules the upstream traversal silently corrupted")

_raised = False
try:
    get_graph_from_smiles("CC(=O)Oc1ccccc1C(=O)O.Cl")
except ValueError:
    _raised = True
_t("disconnected graphs are refused", _raised,
   "rejected before single_reorder can mis-wire it" if _raised else "passed through")

# Round trip: reorder must rebuild the same molecule.

# %%
_rt_bad = _rt_n = 0
for _smi in _branchy:
    _mol = Chem.MolFromSmiles(_smi)
    _sym = [a.GetSymbol() for a in _mol.GetAtoms()]
    _want = Chem.MolToSmiles(Chem.MolFromSmiles(_smi))
    _g, _at, _ar, _bo, _btt = get_graph_from_smiles(_smi)
    _X0 = np.arange(_mol.GetNumAtoms(), dtype=np.int32)
    _A0 = np.array([[b[0], b[1], t] for b, t in zip(_bo, _btt)], dtype=np.int32)
    for _ in range(20):
        _rt_n += 1
        _ids, _ = traverse_graph(_g, _ar)
        _Xr, _Ar = single_reorder(_X0, _A0, _ids)
        _m = Chem.RWMol(Chem.Mol())
        for _o in _Xr.tolist():
            _m.AddAtom(Chem.Atom(_sym[int(_o)]))
        for _a1, _a2, _bt2 in _Ar.tolist():
            get_mol_spec().index_to_bond(_m, int(_a1), int(_a2), int(_bt2))
        try:
            _mm = _m.GetMol()
            Chem.SanitizeMol(_mm)
            if Chem.MolToSmiles(_mm) != _want:
                _rt_bad += 1
        except Exception:
            _rt_bad += 1
_t("reorder round-trips to the original molecule", _rt_bad == 0,
   f"{_rt_n - _rt_bad}/{_rt_n} rebuilt correctly")

_failed = [n for n, ok, _ in _T if not ok]
check("self-tests", not _failed,
      f"{len(_T) - len(_failed)}/{len(_T)} passed"
      + ("" if not _failed else f"; failed: {_failed}"))
gate("fold files")

# %% [markdown]
# ## Fold files and embeddings
#
# Row `i` of the embedding matrix is HDF5 key `p{i+1}` is FASTA record `i`. Pairs
# are matched to rows by sequence, so the FASTA and the protein list must agree
# record for record. Both are checked, not assumed.

# %%
def split_dir(split):
    return os.path.join(BINDINGDB_DIR, f"{split}_dataset", f"{split}_{FOLD_SUFFIX[FOLD]}")

# %%

def find_fold_files(split):
    d = split_dir(split)
    tag = "tr" if split == "train" else "te"

    def one(pattern, what):
        hits = sorted(glob.glob(os.path.join(d, pattern)))
        if not hits:
            raise FileNotFoundError(
                f"{what}: no match for {pattern!r} in {d}\n"
                f"  Directory holds: {sorted(os.listdir(d))[:25]}")
        return hits[0]

    ligands = one("ligands_*_corrected", "ligand pairs")
    proteins = one("proteins_*_corrected", "protein pairs")
    unique = one(f"{FOLD}_{tag}_unique_proteins.txt", "unique protein list")

    expected = f"{FOLD}_{tag}_unique_embeddings_bepler.h5"
    emb = os.path.join(d, expected)
    if not os.path.isfile(emb):
        hits = [p for p in sorted(glob.glob(os.path.join(d, "*embeddings*bepler*")))
                if os.path.splitext(p)[1].lower() in EMB_EXTENSIONS]
        if not hits:
            raise FileNotFoundError(
                f"Protein embeddings missing for fold {FOLD} ({split}).\n"
                f"  Expected: {os.path.join(d, expected)}\n"
                f"  Generate with: python prose/embed_sequences.py --pool avg \\\n"
                f"      -o {expected} {FOLD}_{tag}_unique_proteins_bepler.fa\n"
                f"  --pool avg is required: one ({N_C},) vector per protein.")
        emb = hits[0]
        print(f"[WARN] {expected} not found; using {os.path.basename(emb)}")
    return ligands, proteins, unique, emb

# %%

def _as_pooled_vector(arr, name, path, width):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        squeezed = arr.squeeze()
        if squeezed.ndim == 1:
            arr = squeezed
        elif arr.ndim == 2:
            raise ValueError(
                f"{path}: protein {name!r} has shape {arr.shape} — per-residue "
                f"embeddings. Regenerate with `--pool avg`; expected ({N_C},)")
        else:
            raise ValueError(f"{path}: protein {name!r} has shape {arr.shape}")

    if width is not None and arr.shape[0] != width:
        hint = ""
        if arr.shape[0] % N_C == 0 and arr.shape[0] > N_C:
            hint = (f"\n  {arr.shape[0]} = {arr.shape[0] // N_C} x {N_C}: this looks "
                    f"like a flattened per-residue embedding. Use `--pool avg`.")
        raise ValueError(f"{path}: protein {name!r} has width {arr.shape[0]}; "
                         f"expected {width}.{hint}")
    if arr.shape[0] == 0:
        raise ValueError(f"{path}: protein {name!r} has an empty embedding.")

    return arr.astype(np.float32, copy=False)

# %%

def _h5_key_order(f, n_expected, path):
    keys = list(f.keys())
    if len(keys) != n_expected:
        shown = keys[:5] if len(keys) <= 10 else keys[:5] + ['...'] + keys[-5:]
        raise ValueError(f"{path}: {len(keys)} embedded proteins but {n_expected} "
                         f"in the fold's protein list.\n  Keys: {shown}")
    named = [f"p{i + 1}" for i in range(n_expected)]
    if all(k in f for k in named):
        return named
    # No fallback ordering. Row i must be protein i, and only the key name
    # guarantees that; any other order silently conditions every ligand on the
    # wrong protein while every shape check still passes.
    missing = [k for k in named if k not in f]
    raise ValueError(f"{os.path.basename(path)}: expected keys p1..p{n_expected}; "

                     f"{len(missing)} absent (e.g. {missing[:8]}).")

# %%

def load_embeddings(path, n_expected):
    ext = os.path.splitext(path)[1].lower()
    if ext not in EMB_EXTENSIONS:
        raise ValueError(f"{path}: unsupported embedding format {ext!r}")

    if ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            order = _h5_key_order(f, n_expected, path)
            first = _as_pooled_vector(f[order[0]][()], order[0], path,
                                      None if ALLOW_N_C_OVERRIDE else N_C)
            rows = [first] + [_as_pooled_vector(f[k][()], k, path, first.shape[0])
                              for k in order[1:]]
        C = np.stack(rows, axis=0)
    elif ext == '.npy':
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected [n_proteins, {N_C}], got {arr.shape}")
        C = arr.astype(np.float32, copy=False)
    else:
        raw = []
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                parts = line.replace(',', ' ').split()
                if not parts:
                    continue
                try:
                    float(parts[0])
                except ValueError:
                    parts = parts[1:]
                raw.append((f"line {ln}", np.array(parts, dtype=np.float32)))
        if not raw:
            raise ValueError(f"{path}: no embedding rows parsed.")
        first = _as_pooled_vector(raw[0][1], raw[0][0], path,
                                  None if ALLOW_N_C_OVERRIDE else N_C)
        C = np.stack([first] + [_as_pooled_vector(a, nm, path, first.shape[0])
                                for nm, a in raw[1:]], axis=0)

    if C.shape[0] != n_expected:
        raise ValueError(f"{path}: {C.shape[0]} rows but {n_expected} proteins listed.")
    if C.shape[1] != N_C and not ALLOW_N_C_OVERRIDE:
        raise ValueError(f"{path}: embedding width {C.shape[1]}, expected N_C = {N_C}.")
    if not np.isfinite(C).all():
        n_bad = int((~np.isfinite(C)).any(axis=1).sum())
        raise ValueError(f"{path}: {n_bad} embeddings contain NaN or inf.")
    return C


# %% [markdown]
# ## Ames labels
#
# From the repo's Random Forest, cached across folds. The cache is reused only
# when the model that wrote it is on disk and hashes the same. Mixing labels from
# two forests would mean Eq. 8 optimising against two definitions of toxic.
#
# There is deliberately **no auto-train path**: a missing model raises and points
# at `random_forest_classifier`, which rebuilds the forest *and* the ChEMBL labels
# together.

# %%
def _load_pickle(path):
    errors = []
    for how, fn in (("pickle", lambda: pickle.load(open(path, 'rb'))),
                    ("joblib", lambda: __import__('joblib').load(path))):
        try:
            return fn()
        except Exception as e:
            errors.append(f"{how}: {type(e).__name__}: {e}")
    hint = ""
    if any("incompatible dtype" in e or "node array" in e for e in errors):
        import sklearn
        hint = (f"\n  This is a scikit-learn version mismatch, not a corrupt file: "
                f"the pickle was written by an older sklearn whose tree node dtype "
                f"differs from {sklearn.__version__} here. Regenerate the model IN "
                f"THIS ENVIRONMENT by re-running random_forest_classifier. Do not "
                f"downgrade sklearn.")

    raise RuntimeError(f"Could not load {path}\n  " + "\n  ".join(errors) + hint)

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

def morgan_matrix(smiles_list, radius, n_bits):
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        out[i] = morgan_bits(Chem.MolFromSmiles(smi), radius, n_bits)
        if (i + 1) % 20000 == 0:
            print(f"    fingerprinted {i + 1}/{len(smiles_list)}")
    return out

# %%
_RF_HASH_MEMO = {}

# %%

def model_behaviour_hash(model, radius, n_bits, n_probe=512, seed=20240917):
    """Hash a labeller's DECISIONS on a fixed probe set, not its pickle bytes.

    The pickle hash is machine-dependent: `n_jobs=-1` is stored in the estimator,
    so the same forest trained on an 8-core and a 64-core box serialises to
    different bytes while predicting identically. Measured: one forest, three
    `n_jobs` values -> 3 distinct pickle hashes, 1 behavioural hash. Keying a
    label cache on the bytes therefore triggers a full relabel of every ligand
    (and, in 04, every generated molecule) for a model that has not changed.

    The probe is deterministic from `seed` and shaped like the real feature
    vector, so this changes exactly when the decision boundary does. It is a
    cache key, NOT a validation of the model.
    """
    probe = (np.random.default_rng(seed).random((n_probe, n_bits)) > 0.85
             ).astype(np.uint8)
    try:
        pred = np.asarray(model.predict(probe)).astype(np.int8)
    except Exception as e:
        return {'error': f"{type(e).__name__}: {e}"}
    return {'n_probe': int(n_probe), 'seed': int(seed),
            'sha256': hashlib.sha256(pred.tobytes()).hexdigest(),
            'frac_positive': round(float((pred == 1).mean()), 6)}

# %%

def rf_fingerprint():
    """Identifies the model+features a cached label came from.

    Carries BOTH a behavioural hash (the cache key: changes when the decision
    boundary changes) and the pickle hash (provenance only, recorded so a
    mismatch can be explained). `cache_decision` compares on behaviour.
    """
    rf = None
    try:
        st = os.stat(RF_MODEL_PATH)
        key = (RF_MODEL_PATH, st.st_size, st.st_mtime_ns)
        if key not in _RF_HASH_MEMO:
            _RF_HASH_MEMO[key] = _sha256(RF_MODEL_PATH)
        rf = {'sha256': _RF_HASH_MEMO[key], 'size': st.st_size}
    except OSError:
        rf = None
    try:
        cfg = _load_pickle(FP_CONFIG_PATH)
        fp = {'radius': int(cfg.get('radius', 2)),
              'n_bits': int(cfg.get('n_bits', cfg.get('nBits', 2048)))}
    except Exception:
        fp = None

    behaviour = None
    if rf is not None and fp is not None:
        try:
            behaviour = model_behaviour_hash(_load_pickle(RF_MODEL_PATH),
                                             fp['radius'], fp['n_bits'])
        except Exception as e:
            behaviour = {'error': f"{type(e).__name__}: {e}"}

    return {'v': 3, 'rf': rf, 'fp': fp, 'behaviour': behaviour}

# %%

def cache_decision(want, got, meta_err=None):
    """Reuse the cache only if the SAME decision boundary wrote it.

    Compares the behavioural hash and the fingerprint config. The pickle hash is
    reported but never decides: it moves when a forest is retrained on a machine
    with a different core count, which would otherwise relabel everything for an
    unchanged model.
    """
    if not (want and want.get('rf') and want.get('fp')):
        if not (want or {}).get('rf'):
            return False, ("the Random Forest is not on disk, so the cache cannot be "
                           "checked against the model that wrote it")
        return False, "the fingerprint config is unreadable"
    if got is None:
        return False, (f"fingerprint unreadable ({meta_err})" if meta_err
                       else "no fingerprint recorded beside the cache")
    if not (got.get('rf') and got.get('fp')):
        return False, "the recorded fingerprint is incomplete"

    if got.get('fp') != want.get('fp'):
        return False, (f"the fingerprint config changed: cache was written with "
                       f"{got.get('fp')}, this run builds {want.get('fp')}")

    w_b, g_b = want.get('behaviour'), got.get('behaviour')
    if not (isinstance(w_b, dict) and w_b.get('sha256')):
        return False, ("the model on disk could not be probed, so the cache cannot "
                       "be verified against it")
    if not (isinstance(g_b, dict) and g_b.get('sha256')):
        # A v2 cache predates behavioural hashing; byte equality is the only
        # evidence available, and its absence is not proof of a different model.
        if got.get('rf', {}).get('sha256') == want.get('rf', {}).get('sha256'):
            return True, ("no behavioural hash recorded (pre-v3 cache), but the "
                          "model file is byte-identical")
        return False, ("the cache predates behavioural hashing and the model file "
                       "has changed; relabelling rather than guess")
    if w_b.get('sha256') != g_b.get('sha256'):
        return False, (f"the labeller's decisions changed "
                       f"(probe positive fraction {g_b.get('frac_positive')} -> "
                       f"{w_b.get('frac_positive')})")

    same_bytes = got.get('rf', {}).get('sha256') == want.get('rf', {}).get('sha256')
    return True, ("fingerprint matches" if same_bytes else
                  "the model file differs byte for byte but predicts identically on "
                  "the probe set (a retrain with a different n_jobs does this), so "
                  "the cached labels are still valid")

# %%

def ames_labels(smiles_list, cache_path=AMES_CACHE):
    """SMILES -> {0,1} Ames mutagenicity from the repo's RF. Cached across folds."""
    cache = {}
    want = rf_fingerprint()

    if cache_path and os.path.exists(cache_path):
        got, meta_err = None, None
        if os.path.exists(AMES_CACHE_META):
            try:
                with open(AMES_CACHE_META) as f:
                    got = json.load(f)
            except Exception as e:
                meta_err = type(e).__name__
        accept, reason = cache_decision(want, got, meta_err)
        if accept:
            df = pd.read_csv(cache_path)
            cache = dict(zip(df['smiles'].astype(str), df['ames'].astype(int)))
            print(f"  cache: {len(cache)} entries ({reason})")
        else:
            archive_stale([cache_path, AMES_CACHE_META],
                          stamp=time.strftime('%Y%m%d-%H%M%S'), why="unverified cache")
            print(f"  [WARN] not reusing the cache: {reason}. Relabelling")

    todo = sorted({s for s in smiles_list if s not in cache})
    if todo:
        if not os.path.exists(RF_MODEL_PATH):
            raise FileNotFoundError(
                f"No Ames RF at {RF_MODEL_PATH}. Run random_forest_classifier, which "
                f"rebuilds the forest AND the ChEMBL labels consistently. There is no "
                f"auto-train path here on purpose: rebuilding only the forest leaves "
                f"ChEMBL and BindingDB labelled by two different models.")
        fp_cfg = _load_pickle(FP_CONFIG_PATH)
        radius = int(fp_cfg.get('radius', 2))
        n_bits = int(fp_cfg.get('n_bits', fp_cfg.get('nBits', 2048)))
        print(f"  labelling {len(todo)} new ligands | Morgan r{radius}, {n_bits} bits")

        rf = _load_pickle(RF_MODEL_PATH)
        X = morgan_matrix(todo, radius, n_bits)
        if getattr(rf, 'n_features_in_', n_bits) != n_bits:
            raise ValueError(f"RF expects {rf.n_features_in_} features, config gives "
                             f"{n_bits}.")
        cache.update(zip(todo, rf.predict(X).astype(int)))

        if cache_path:
            pd.DataFrame({'smiles': list(cache.keys()),
                          'ames': list(cache.values())}).to_csv(cache_path, index=False)
            with open(AMES_CACHE_META, 'w') as f:
                json.dump(rf_fingerprint(), f, indent=1)

    labels = [int(cache[s]) for s in smiles_list]
    n_tox = sum(labels)
    print(f"  Ames: {n_tox} toxic / {len(labels) - n_tox} non-toxic "
          f"({100.0 * n_tox / max(1, len(labels)):.1f}% toxic)")
    return labels


# %% [markdown]
# ## Pair assembly

# %%
def read_spaced(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            s = ''.join(line.split())
            if s:
                out.append(s)
            if limit is not None and len(out) >= limit:
                break

    return out

# %%

def read_fasta(path):
    records, name, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name is not None:
        records.append((name, "".join(buf)))

    return records

# %%

def verify_fasta_order(split, unique):
    """The invariant the whole conditioning path rests on: embeddings are keyed
    p1..pN by FASTA position, pairs are matched to rows by sequence content. A
    silent disagreement mis-conditions every ligand while every shape check passes."""
    d = split_dir(split)
    tag = "tr" if split == "train" else "te"
    fa = os.path.join(d, f"{FOLD}_{tag}_unique_proteins_bepler.fa")
    if not os.path.isfile(fa):
        raise FileNotFoundError(
            f"{fa} not found. It is the FASTA the embeddings were generated from, "
            f"and the only way to confirm HDF5 key p<i> really is row <i>.")

    recs = read_fasta(fa)
    if len(recs) != len(unique):
        raise ValueError(f"{os.path.basename(fa)} has {len(recs)} records but the "
                         f"protein list has {len(unique)}.")
    bad_name = [i for i, (n, _) in enumerate(recs) if n != f"p{i + 1}"]
    if bad_name:
        i = bad_name[0]
        raise ValueError(f"{os.path.basename(fa)}: record {i} is named "
                         f"{recs[i][0]!r}, expected 'p{i + 1}'.")
    mism = [i for i, ((_, s), u) in enumerate(zip(recs, unique))
            if "".join(s.split()).upper() != u.upper()]
    if mism:
        raise ValueError(
            f"{os.path.basename(fa)} and the protein list disagree at {len(mism)} "
            f"position(s); first is record {mism[0]}. Every ligand would be "
            f"conditioned on the wrong protein.")
    print(f"  [{split}] order verified: {len(recs)} proteins, FASTA == list == "
          f"p1..p{len(recs)}")

# %%

def read_pairs(split, limit=None):
    """Rows to assemble one split from. Pooled across both pair files and routed
    by protein-list membership; the two lists are checked disjoint, so pooling
    cannot leak."""
    sources = ("train", "test") if POOL_PAIR_FILES else (split,)
    seen, ligands, proteins = set(), [], []
    for src_split in sources:
        lig_path, prot_path = find_fold_files(src_split)[:2]
        lig, prot = read_spaced(lig_path), read_spaced(prot_path)
        if len(lig) != len(prot):
            raise ValueError(f"{src_split}: {len(lig)} ligand lines vs {len(prot)} "
                             f"protein lines — the pair files are not aligned.")
        n_new = 0
        for s, p in zip(lig, prot):
            key = (s, p.upper())
            if key in seen:
                continue
            seen.add(key)
            ligands.append(s)
            proteins.append(p)
            n_new += 1
        print(f"  [{split}] + {n_new} new of {len(lig)} rows from "
              f"{os.path.basename(prot_path)}")
    if limit and len(ligands) > limit:
        idx = sorted(random.Random(SPLIT_SEED).sample(range(len(ligands)), limit))
        ligands = [ligands[i] for i in idx]
        proteins = [proteins[i] for i in idx]
        print(f"  [{split}] LIMIT_PAIRS: sampled {limit} rows (SMOKE TEST)")
    return ligands, proteins

# %%

def assemble_pairs(split, limit=None):
    """-> (records, C, stats) where records = [(smiles, protein_row, ames), ...]"""
    lig_path, prot_path, uniq_path, emb_path = find_fold_files(split)
    for lab, p in (("ligands", lig_path), ("proteins", prot_path),
                   ("unique", uniq_path), ("embedding", emb_path)):
        print(f"  [{split}] {lab:9s} {os.path.basename(p)}")

    ligands, proteins = read_pairs(split, limit)

    unique = read_spaced(uniq_path)
    if not unique:
        raise ValueError(f"{uniq_path}: no protein sequences read.")
    seq_to_row = {s.upper(): i for i, s in enumerate(unique)}
    if len(seq_to_row) != len(unique):
        print(f"  [WARN] {os.path.basename(uniq_path)}: "
              f"{len(unique) - len(seq_to_row)} duplicate sequence(s); pairs for a "
              f"duplicate resolve to its LAST row.")

    if VERIFY_FASTA_ORDER:
        verify_fasta_order(split, unique)

    global N_C
    C = load_embeddings(emb_path, len(unique))
    N_C = int(C.shape[1])
    print(f"  [{split}] {len(ligands)} pairs | {len(unique)} unique proteins | "
          f"embeddings {C.shape}")

    rows = []
    n_bad = n_big = n_oov = n_unmapped = n_long = n_salt = n_disc = 0

    for smi, seq in zip(ligands, proteins):
        row = seq_to_row.get(seq.upper())
        if row is None:
            n_unmapped += 1
            continue
        # Salt stripping FIRST: a multi-fragment SMILES survives every other
        # filter below (Na, K, Cl are all in atom_types.txt) and then corrupts
        # its own training target inside traverse_graph.
        if STRIP_SALTS and '.' in smi:
            smi = strip_salt(smi)
            n_salt += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_bad += 1
            continue
        if mol.GetNumAtoms() > MAX_ATOMS:
            n_big += 1
            continue
        # graph_to_rnn is [batch, K, MAX_ITER]; a molecule needs GetNumBonds() + 1
        # steps, so anything longer cannot be written into it.
        if mol.GetNumBonds() + 1 > MAX_ITER:
            n_long += 1
            continue
        try:
            for a in mol.GetAtoms():
                get_mol_spec().get_atom_type(a)
        except ValueError:
            n_oov += 1
            continue
        try:
            _ = get_graph_from_smiles(smi)
        except ValueError:
            n_disc += 1
            continue
        rows.append((smi, row))

    print(f"  [{split}] kept {len(rows)} | salts stripped {n_salt} | dropped: "
          f"unparseable {n_bad}, >{MAX_ATOMS} atoms {n_big}, >{MAX_ITER - 1} bonds "
          f"{n_long}, out-of-vocab {n_oov}, disconnected {n_disc}, unmatched "
          f"protein {n_unmapped}")

    kept_rows = {r for _, r in rows}
    unused = sorted(set(range(len(unique))) - kept_rows)
    if unused and limit is None:
        raise RuntimeError(
            f"{len(unused)} of the {len(unique)} proteins in "
            f"{os.path.basename(uniq_path)} have no usable pair (first rows: "
            f"{unused[:8]}). Either they never appear in the pair files, or every "
            f"one of their ligands was dropped by the filters above.")
    elif unused:
        print(f"  [{split}] [SMOKE] {len(unused)} of {len(unique)} proteins have no "
              f"sampled pair")

    if n_unmapped:
        frac = n_unmapped / max(1, len(ligands))
        tol = UNMATCHED_TOLERANCE.get(split, 0.995)
        print(f"  [{split}] {n_unmapped} of {len(ligands)} pairs "
              f"({100 * frac:.1f}%) belong to proteins outside this fold")
        if frac > tol:
            raise RuntimeError(f"[{split}] dropped fraction {100 * frac:.1f}% exceeds "
                               f"its tolerance {100 * tol:.1f}%.")
    if not rows:
        raise RuntimeError(f"{split}: no usable pairs.")

    labels = ames_labels([smi for smi, _ in rows])
    records = [(smi, row, lab) for (smi, row), lab in zip(rows, labels)]
    stats = {"kept": len(rows), "salt_stripped": n_salt, "unparseable": n_bad,
             "too_many_atoms": n_big, "too_many_bonds": n_long,
             "out_of_vocab": n_oov, "disconnected": n_disc,
             "unmatched_protein": n_unmapped,
             "frac_toxic": round(sum(labels) / max(1, len(labels)), 5)}

    return records, C, stats

# %%

def split_by_protein(records, n_proteins, held_fraction, seed):
    """Split by protein row so the two parts share no protein."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_proteins)
    n_held = max(1, int(round(held_fraction * n_proteins)))
    held_rows = set(order[:n_held].tolist())
    held = [r for r in records if r[1] in held_rows]
    kept = [r for r in records if r[1] not in held_rows]
    return held, kept, sorted(held_rows), sorted(set(order[n_held:].tolist()))


# %% [markdown]
# ## Build the datasets

# %%
records_all_train, C_train, stats_train = assemble_pairs("train", LIMIT_PAIRS)
records_test, C_held, stats_test = assemble_pairs("test", LIMIT_PAIRS)

records_val, records_train, val_rows, train_rows = split_by_protein(
    records_all_train, C_train.shape[0], VAL_PROTEIN_FRACTION, SPLIT_SEED)
test_rows = sorted({r[1] for r in records_test})

print(f"\n  train {len(records_train)} pairs / {len(train_rows)} proteins")
print(f"  val   {len(records_val)} pairs / {len(val_rows)} proteins (carved from train)")
print(f"  test  {len(records_test)} pairs / {len(test_rows)} proteins (all held-out)")

check("embedding widths agree", C_train.shape[1] == C_held.shape[1],
      f"train {C_train.shape[1]}-d, held-out {C_held.shape[1]}-d")

# Protein-level leakage: the fold is built so train and held-out proteins share
# <20% similarity. Identity would be fatal, so check rather than assume.

# %%
_shared = set(read_spaced(find_fold_files("train")[2])) & \
          set(read_spaced(find_fold_files("test")[2]))
check("no identical protein across the split", not _shared,
      f"{len(_shared)} sequences identical between train and held-out" if _shared
      else "train and held-out proteins are disjoint")

check("train/val share no protein", not (set(val_rows) & set(train_rows)),
      f"{len(set(val_rows) & set(train_rows))} shared protein rows")

# The real salt invariant, not a tautology: nothing kept may still be multi-fragment.
_still_salty = [s for s, _, _ in records_train + records_test if '.' in s]
check("no multi-fragment ligand survives assembly", not _still_salty,
      f"{stats_train['salt_stripped']} train + {stats_test['salt_stripped']} test "
      f"ligands were stripped; 0 of {len(records_train) + len(records_test)} kept "
      f"records still contain '.'" if not _still_salty
      else f"{len(_still_salty)} kept records are still multi-fragment, e.g. "
           f"{_still_salty[:2]}")

_shared_lig = {r[0] for r in records_train} & {r[0] for r in records_test}
if _shared_lig:
    print(f"  [INFO] {len(_shared_lig)} ligand SMILES occur in both train and test, "
          f"paired with different proteins. The fold controls PROTEIN novelty, not "
          f"ligand novelty.")

n_held_proteins = int(C_held.shape[0])
if SMOKE_TEST:
    print(f"[SMOKE] test bundle covers {len(test_rows)}/{n_held_proteins} held-out "
          f"proteins — NOT the complete test set")
elif len(test_rows) != n_held_proteins:
    raise RuntimeError(f"test bundle covers {len(test_rows)} of {n_held_proteins} "
                       f"held-out proteins. Tables 1-3 are over all of them.")
else:
    check("test bundle complete", True,

          f"all {n_held_proteins} held-out proteins of fold {FOLD}")

# %%
with open(SPLIT_MANIFEST, 'w') as f:
    json.dump({'fold': FOLD, 'fold_suffix': FOLD_SUFFIX[FOLD], 'N_C': N_C,
               'split_seed': SPLIT_SEED,
               'val_protein_fraction': VAL_PROTEIN_FRACTION,
               'val_source': 'train split, by protein',
               'limit_pairs': LIMIT_PAIRS, 'smoke_test': SMOKE_TEST,
               'complete_held_out_test_set': not SMOKE_TEST,
               'n_held_out_proteins_total': n_held_proteins,
               'train_protein_rows': train_rows, 'val_protein_rows': val_rows,
               'test_protein_rows': test_rows,
               'n_train_pairs': len(records_train), 'n_val_pairs': len(records_val),
               'n_test_pairs': len(records_test),
               'frac_toxic': {
                   'train': round(sum(r[2] for r in records_train)
                                  / max(1, len(records_train)), 5),
                   'val': round(sum(r[2] for r in records_val)
                                / max(1, len(records_val)), 5),
                   'test': round(sum(r[2] for r in records_test)
                                 / max(1, len(records_test)), 5)},
               'assembly_stats': {'train': stats_train, 'test': stats_test},
               'strip_salts': STRIP_SALTS, 'fragment_policy': FRAGMENT_POLICY,

               'traversal_policy': TRAVERSAL_POLICY}, f, indent=1)

np.savez_compressed(
    TEST_BUNDLE,
    smiles=np.array([r[0] for r in records_test], dtype=object),
    protein_row=np.array([r[1] for r in records_test], dtype=np.int32),
    ames=np.array([r[2] for r in records_test], dtype=np.int8),
    embeddings=C_held,
    test_protein_rows=np.array(test_rows, dtype=np.int32),
    smoke_test=np.array(SMOKE_TEST),
    limit_pairs=np.array(-1 if LIMIT_PAIRS is None else LIMIT_PAIRS),

    n_held_out_proteins_total=np.array(n_held_proteins))

# %%
print(f"  test bundle -> {TEST_BUNDLE}")
gate("loader construction")

# %% [markdown]
# ## Sampler and collator

# %%
class BalancedSampler(Sampler):
    """Groups similar-length molecules into a batch to cut padding.

    Upstream behaviour kept: records beyond the last whole batch are not sampled.
    """

    def __init__(self, cost, batch_size, name="dataset"):
        if len(cost) < batch_size:
            raise ValueError(f"{name}: {len(cost)} records but batch size {batch_size}; "
                             f"this sampler would yield no batches.")
        index = np.argsort(cost).tolist()
        chunk_size = int(float(len(cost)) / batch_size)
        dropped = len(cost) - chunk_size * batch_size
        if dropped:
            print(f"  [INFO] {name}: {dropped} record(s) beyond the last whole batch "
                  f"are not sampled")
        self.index = []
        for i in range(batch_size):
            self.index.append(index[i * chunk_size:(i + 1) * chunk_size])

    def _g(self):
        for index_i in self.index:
            random.shuffle(index_i)
        for batch_index in zip(*self.index):
            yield list(batch_index)

    def __iter__(self):
        return self._g()

    def __len__(self):
        return len(self.index[0])

# %%

def process_single(record, k=K_PATHS, p=P_ALPHA):
    """Build the graph, sample K traversals, attach the ligand's Ames label."""
    smiles, tox_label = record[0], int(record[2])
    smiles = strip_salt(smiles) if STRIP_SALTS else smiles

    graph, atom_types, atom_ranks, bonds, bond_types = get_graph_from_smiles(smiles)
    X_0 = np.array(atom_types, dtype=np.int32)
    A_0 = np.concatenate([
        np.array(bonds, dtype=np.int32),
        np.array(bond_types, dtype=np.int32)[:, np.newaxis]
    ], axis=1) if len(bonds) > 0 else np.zeros((0, 3), dtype=np.int32)

    X, A, NX, NA = [], [], [], []
    mol_ids, rep_ids, iw_ids = [], [], []
    action_0, actions, last_append_mask = [], [], []
    log_p = []

    tox_class = np.full((k,), tox_label, dtype=np.int32)

    for i in range(k):
        step_ids_i, log_p_i = traverse_graph(graph, atom_ranks, p=p)
        X_i, A_i = single_reorder(X_0, A_0, step_ids_i)
        action_0_i, X_i, NX_i, A_i, NA_i, actions_i, last_mask_i = single_expand(X_i, A_i)

        X.append(X_i)
        A.append(A_i)
        NX.append(NX_i)
        NA.append(NA_i)
        action_0.append(action_0_i)
        actions.append(actions_i)
        last_append_mask.append(last_mask_i)
        mol_ids.append(np.zeros_like(NX_i, dtype=np.int32))
        rep_ids.append(np.ones_like(NX_i, dtype=np.int32) * i)
        iw_ids.append(np.ones_like(NX_i, dtype=np.int32) * i)
        log_p.append(log_p_i)

    return (np.concatenate(X, axis=0), np.concatenate(A, axis=0),
            np.concatenate(NX, axis=0), np.concatenate(NA, axis=0),
            np.concatenate(mol_ids, axis=0), np.concatenate(rep_ids, axis=0),
            np.concatenate(iw_ids, axis=0), np.concatenate(action_0, axis=0),
            np.concatenate(actions, axis=0), np.concatenate(last_append_mask, axis=0),
            np.array(log_p, dtype=np.float32), tox_class)

# %%

class CMolRNNCollate:
    """Conditional collator. `c` is gathered for surviving records only, keeping
    its rows aligned with mol_ids."""

    def __init__(self, c_matrix, k=K_PATHS, p=P_ALPHA):
        self.C, self.k, self.p = c_matrix, k, p

    def __call__(self, batch):
        valid_batch = []
        for record in batch:
            try:
                _ = get_graph_from_smiles(
                    strip_salt(record[0]) if STRIP_SALTS else record[0])
                valid_batch.append(record)
            except ValueError:
                continue
        if not valid_batch:
            return None

        shapes = [[0], [0, 3], [0], [0], [0], [0], [0], [0], [0, 5], [0], [0], [0]]
        dtypes = [np.int32] * 10 + [np.float32, np.int32]
        _build = lambda: [np.zeros(shape=s, dtype=d) for s, d in zip(shapes, dtypes)]
        _append = lambda r0, r1: [np.concatenate([a, b], axis=0) for a, b in zip(r0, r1)]

        (X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions,
         last_append_mask, log_p, tox_class) = _build()

        for n, record in enumerate(valid_batch):
            (X_i, A_i, NX_i, NA_i, mol_ids_i, rep_ids_i, iw_ids_i, action_0_i,
             actions_i, last_mask_i, log_p_i, tox_i) = process_single(
                record, self.k, self.p)
            if n != 0:
                mol_ids_i += mol_ids[-1] + 1
                iw_ids_i += iw_ids[-1] + 1
            (X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions,
             last_append_mask, log_p, tox_class) = _append(
                [X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions,
                 last_append_mask, log_p, tox_class],
                [X_i, A_i, NX_i, NA_i, mol_ids_i, rep_ids_i, iw_ids_i, action_0_i,
                 actions_i, last_mask_i, log_p_i, tox_i.reshape(-1)])

        (X, A, mol_ids_rep, rep_ids_rep, iw_ids, last_append_mask, NX, NX_rep,
         action_0, actions, log_p) = merge_single(
            X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions,
            last_append_mask, log_p)

        result_out = [X, A, mol_ids_rep, rep_ids_rep, iw_ids, last_append_mask, NX,
                      NX_rep, action_0, actions, log_p, tox_class]

        smis = [strip_salt(b[0]) if STRIP_SALTS else b[0] for b in valid_batch]
        mol_list = [Chem.MolFromSmiles(s) for s in smis]
        graph_to_rnn = np.zeros((len(valid_batch), self.k, get_mol_spec().max_iter),
                                dtype=np.int32)
        rnn_to_graph = []
        cum_sum = 0
        for i, mol_i in enumerate(mol_list):
            num_iter = mol_i.GetNumBonds() + 1
            assert num_iter <= get_mol_spec().max_iter, (
                f"{smis[i]} needs {num_iter} steps > max_iter "
                f"{get_mol_spec().max_iter}; assemble_pairs should have dropped it")
            for k_idx in range(self.k):
                graph_to_rnn[i, k_idx, :num_iter] = np.arange(num_iter) + cum_sum
                r0 = np.ones([num_iter], dtype=np.int32) * i
                rnn_to_graph.append(np.stack([r0, np.ones_like(r0) * k_idx,
                                              np.arange(num_iter)], axis=0))
                cum_sum += num_iter
        rnn_to_graph = np.concatenate(rnn_to_graph, axis=1)
        NX_cum = np.cumsum(result_out[6])

        c = self.C[[rec[1] for rec in valid_batch], :]
        return result_out + [graph_to_rnn, rnn_to_graph, NX_cum, c]

# %%

def batch_to_tensor(record, device=None):
    """`ids` maps each atom row to its molecule, which is how `c` is broadcast."""
    device = CTX if device is None else device
    *base, graph_to_rnn, rnn_to_graph, NX_cum, c = record
    (X, A, mol_ids_rep, rep_ids_rep, iw_ids, last_append_mask,
     NX, NX_rep, action_0, actions, log_p, tox_class) = base

    n_nodes = int(X.shape[0])
    X = torch.as_tensor(X, dtype=torch.long, device=device)

    A_sparse = []
    for A_i in A:
        if A_i.shape[0] == 0:
            A_sparse.append(None)
        else:
            A_i = np.concatenate([A_i, A_i[:, [1, 0]]], axis=0)
            idx = torch.as_tensor(A_i[:, :2].T.astype(np.int64), device=device)
            val = torch.ones(A_i.shape[0], dtype=torch.float32, device=device)
            A_sparse.append(torch.sparse_coo_tensor(
                idx, val, (n_nodes, n_nodes), dtype=torch.float32,
                device=device).coalesce())

    batch_size = int(mol_ids_rep.max() + 1)
    iw_size = int(rep_ids_rep.max() + 1)

    ids = torch.as_tensor(mol_ids_rep, dtype=torch.long, device=device)
    iw_ids, last_append_mask, NX, NX_rep, action_0, actions = [
        torch.as_tensor(x, dtype=torch.long, device=device)
        for x in [iw_ids, last_append_mask, NX, NX_rep, action_0, actions]]
    log_p = torch.as_tensor(log_p, dtype=torch.float32, device=device)
    tox_class = torch.as_tensor(tox_class, dtype=torch.float32, device=device)
    graph_to_rnn = torch.as_tensor(graph_to_rnn, dtype=torch.long, device=device)
    rnn_to_graph = torch.as_tensor(rnn_to_graph, dtype=torch.long, device=device)
    NX_cum = torch.as_tensor(NX_cum, dtype=torch.long, device=device)
    c = torch.as_tensor(np.ascontiguousarray(c), dtype=torch.float32, device=device)

    return [X, A_sparse, iw_ids, last_append_mask, NX, NX_rep, action_0, actions,
            log_p, batch_size, iw_size, tox_class, graph_to_rnn, rnn_to_graph,
            NX_cum, c, ids]

# %%
random.Random(17).shuffle(records_train)      # upstream Lambda.__init__ seed

loader_train = DataLoader(
    records_train,
    batch_sampler=BalancedSampler([len(r[0]) for r in records_train], BATCH_SIZE,
                                  "train"),
    num_workers=NUM_WORKERS, collate_fn=CMolRNNCollate(C_train, K_PATHS, P_ALPHA),
    persistent_workers=(NUM_WORKERS > 0))

loader_val = DataLoader(
    records_val,
    batch_sampler=BalancedSampler([len(r[0]) for r in records_val], BATCH_SIZE_VAL,
                                  "val"),
    num_workers=NUM_WORKERS,
    collate_fn=CMolRNNCollate(C_train, K_PATHS, P_ALPHA),   # val rows index C_train
    persistent_workers=(NUM_WORKERS > 0))

# Epoch length fixed once. Epoch boundaries are iteration counts, not
# StopIteration, so a mid-epoch resume cannot run a long epoch.

# %%
ITERS_PER_EPOCH = len(loader_train)
print(f"\n  {ITERS_PER_EPOCH:,} iterations per epoch "
      f"({len(records_train):,} pairs / batch {BATCH_SIZE})")

batch_cpu = next(iter(loader_train))
_b = batch_to_tensor(batch_cpu)
print(f"  batch: N={_b[9]} K={_b[10]} | X {tuple(_b[0].shape)} | c {tuple(_b[15].shape)}")
check("conditioning matrix shape", _b[15].shape == (_b[9], N_C),
      f"c is {tuple(_b[15].shape)}, expected ({_b[9]}, {N_C})")
check("atom->molecule map", int(_b[16].max()) == _b[9] - 1,
      f"ids max {int(_b[16].max())} = batch size {_b[9]} - 1")

# %% [markdown]
# ## Model
#
# Parameter names outside `linear_c.*` and `dense_policy_0.*` match the
# pretraining model, so the stage-1 checkpoint transfers by name.

# %%
def segment_sum(X, idx, num_seg):
    out = torch.zeros((num_seg,) + tuple(X.shape[1:]), dtype=X.dtype, device=X.device)
    return out.index_add(0, idx, X)

# %%

def segment_max(X, idx, num_seg):
    out = torch.full((num_seg,) + tuple(X.shape[1:]), float('-inf'),
                     dtype=X.dtype, device=X.device)
    index = idx.reshape((-1,) + (1,) * (X.dim() - 1)).expand_as(X)
    return out.scatter_reduce(0, index, X, reduce='amax', include_self=True)

# %%

def _concat_neighbourhoods(X, A_list):
    parts = [X]
    for A in A_list:
        parts.append(torch.sparse.mm(A, X) if A is not None else torch.zeros_like(X))
    return torch.cat(parts, dim=1)

# %%

def action_log_probs(raw_x, raw_x_t, NX_rep, num_mol, stable=True):
    """Joint log-softmax over every action of every atom plus `end`, per molecule.

    stable=True subtracts the per-molecule max: identical in exact arithmetic to
    the upstream exp-and-normalise, but it cannot overflow and
    d log p / d logit is bounded by 1 instead of the ~1/p the upstream
    log(p + 1e-10) produces. stable=False reproduces the upstream path.
    """
    if stable:
        m = torch.maximum(segment_max(raw_x.amax(dim=-1), NX_rep, num_mol),
                          raw_x_t.squeeze(-1))
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        e_x = torch.exp(raw_x - m[NX_rep].unsqueeze(-1))
        e_t = torch.exp(raw_x_t - m.unsqueeze(-1))
        S = segment_sum(e_x, NX_rep, num_mol).sum(-1, keepdim=True) + e_t
        log_Z = m.unsqueeze(-1) + torch.log(S)
        return raw_x - log_Z[NX_rep], raw_x_t - log_Z

    e_x = torch.exp(torch.clamp(raw_x, -80.0, 80.0))
    e_t = torch.exp(torch.clamp(raw_x_t, -80.0, 80.0))
    S = segment_sum(e_x, NX_rep, num_mol).sum(-1, keepdim=True) + e_t
    return (torch.log(e_x / (S[NX_rep] + 1e-10) + 1e-10),
            torch.log(e_t / (S + 1e-10) + 1e-10))

# %%

class EfficientGraphConvFn(torch.autograd.Function):
    """Recomputes the concatenated neighbourhood in backward instead of storing it.
    Every A is symmetric, so A^T == A and backward reuses A directly."""

    @staticmethod
    def forward(ctx, X, W, A_list):
        ctx.A_list = A_list
        ctx.save_for_backward(X, W)
        return _concat_neighbourhoods(X, A_list) @ W

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        X_out = _concat_neighbourhoods(X, ctx.A_list)
        grad_output = grad_output.contiguous()

        grad_W = X_out.t() @ grad_output
        grad_X_out = grad_output @ W.t()

        chunks = torch.split(grad_X_out, X.shape[1], dim=1)
        grad_X = chunks[0]
        for A, g in zip(ctx.A_list, chunks[1:]):
            if A is not None:
                grad_X = grad_X + torch.sparse.mm(A, g.contiguous())

        return grad_X, grad_W, None

# %%

def get_activation(name):

    return {'relu': torch.relu, 'tanh': torch.tanh}[name]

# %%

class Linear_BN(nn.Sequential):
    def __init__(self, F_in, F_out):
        super(Linear_BN, self).__init__(
            nn.Linear(F_in, F_out, bias=False),

            nn.BatchNorm1d(F_out, eps=1e-5, momentum=0.1))

# %%

class GraphConv(nn.Module):
    def __init__(self, Fin, Fout, D):
        super(GraphConv, self).__init__()
        self.Fin, self.Fout, self.D = Fin, Fout, D
        self.W = nn.Parameter(torch.empty(self.Fin * (self.D + 1), self.Fout))

    def forward(self, X, A_list):
        return EfficientGraphConvFn.apply(X, self.W, A_list)

# %%

class Policy(nn.Module):
    """Joint log-softmax over every action of every atom, plus `end`, per molecule."""

    def __init__(self, F_in, F_h, N_A, N_B, k=1):
        super(Policy, self).__init__()
        self.F_in, self.F_h, self.N_A, self.N_B, self.k = F_in, F_h, N_A, N_B, k
        self.linear_h = Linear_BN(F_in * 2, self.F_h * k)
        self.linear_h_t = Linear_BN(F_in, self.F_h * k)
        self.linear_x = nn.Linear(self.F_h, self.N_B + self.N_B * self.N_A)
        self.linear_x_t = nn.Linear(self.F_h, 1)
        self.linear_pi = nn.Linear(self.F_in, self.k) if self.k > 1 else None
        self.last_logit_absmax = None
        self.last_frac_extreme = None

    def forward(self, X, NX, NX_rep, X_end=None, return_log=False):
        if X_end is None:
            X_end = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X, X_end[NX_rep, :]], dim=1)

        X_h = torch.relu(self.linear_h(X)).reshape(-1, self.F_h)
        X_h_end = torch.relu(self.linear_h_t(X_end)).reshape(-1, self.F_h)

        raw_x = self.linear_x(X_h).reshape(-1, self.k, self.N_B + self.N_B * self.N_A)
        raw_x_t = self.linear_x_t(X_h_end).reshape(-1, self.k, 1)

        with torch.no_grad():
            self.last_logit_absmax = float(max(raw_x.abs().max(), raw_x_t.abs().max()))
            n_ext = int((raw_x.abs() >= 80.0).sum()) + int((raw_x_t.abs() >= 80.0).sum())
            self.last_frac_extreme = n_ext / max(1, raw_x.numel() + raw_x_t.numel())

        log_x, log_t = action_log_probs(raw_x, raw_x_t, NX_rep, NX.shape[0],
                                        stable=STABLE_SOFTMAX)

        if self.k > 1:
            log_pi = torch.log_softmax(self.linear_pi(X_end), dim=1).unsqueeze(-1)
            log_x = torch.logsumexp(log_x + log_pi[NX_rep, :, :], dim=1)
            log_t = torch.logsumexp(log_t + log_pi, dim=1)
        else:
            log_x, log_t = log_x.squeeze(1), log_t.squeeze(1)

        connect, append = log_x[:, :self.N_B], log_x[:, self.N_B:]
        append = append.reshape(-1, self.N_A, self.N_B)
        end = log_t.squeeze(-1)
        if return_log:
            return append, connect, end
        return torch.exp(append), torch.exp(connect), torch.exp(end)

# %%
LOG_EPS = 1e-10

LOG_FLOOR = math.log(LOG_EPS)

# %%

class MoleculeGenerator(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation,
                 *args, **kwargs):
        super(MoleculeGenerator, self).__init__()
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
        self.mode = 'loss'
        self.last_l = None            # always the UNFLOORED bound
        self.last_tox = None
        self.last_n_actions = None    # GetNumBonds() + 1, by construction
        self.last_floored = None
        self.last_frac_floor = None

    @abstractmethod
    def _build_graph_conv(self, *args, **kwargs):
        pass

    def _likelihood(self, init, append, connect, end, action_0, actions, iw_ids,
                    log_p_sigma, batch_size, iw_size, tox_class_batch):
        """`init`, `append`, `connect`, `end` arrive as LOG-probabilities, so there
        is no log(p + eps) here and no vanishing gradient on suppressed actions."""
        action_type, node_type, edge_type, append_pos, connect_pos = (
            actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3], actions[:, 4])

        def _log_mask(_logp, _mask):
            return _mask * _logp

        log_init = init.reshape(batch_size * iw_size, self.N_A)
        rows = torch.arange(action_0.shape[0], device=action_0.device)
        loss_init = log_init[rows, action_0]

        loss_end = _log_mask(end, (action_type == 2).to(end.dtype))
        loss_append = _log_mask(append[append_pos, node_type, edge_type],
                                (action_type == 0).to(end.dtype))
        loss_connect = _log_mask(connect[connect_pos, edge_type],
                                 (action_type == 1).to(end.dtype))

        log_p_x = loss_end + loss_append + loss_connect
        with torch.no_grad():
            self.last_frac_floor = float((log_p_x <= LOG_FLOOR).float().mean())

        log_p_x = segment_sum(log_p_x.unsqueeze(-1), iw_ids,
                              batch_size * iw_size).squeeze(-1)
        log_p_x = log_p_x + loss_init

        log_p_x = log_p_x.reshape(batch_size, iw_size)
        log_p_sigma = log_p_sigma.reshape(batch_size, iw_size)

        l = log_p_x - log_p_sigma
        l = torch.logsumexp(l, dim=1) - math.log(float(iw_size))      # Eq. 3

        if tox_class_batch.numel() != batch_size * iw_size:
            raise RuntimeError(
                f"tox_class has {tox_class_batch.numel()} entries, expected "
                f"batch_size*K = {batch_size * iw_size}; labels would be misaligned "
                f"with molecules.")
        tox_labels = tox_class_batch.reshape(batch_size, iw_size)[:, 0]

        # Action count straight from the batch, so it cannot drift out of
        # alignment with `l` the way a separately-built array does.
        with torch.no_grad():
            self.last_n_actions = torch.bincount(
                iw_ids, minlength=batch_size * iw_size
            ).reshape(batch_size, iw_size)[:, 0].to(l.dtype)

        l_raw = l
        if TOX_FLOOR is not None:
            l = torch.where(tox_labels > 0, torch.clamp(l, min=TOX_FLOOR), l)
            self.last_floored = ((tox_labels > 0) & (l_raw < TOX_FLOOR)).detach()
        else:
            self.last_floored = torch.zeros_like(tox_labels, dtype=torch.bool)

        l_weighted = l * (1.0 - (1.0 + LAMBDA_TOX) * tox_labels)      # Eq. 8
        self.last_l = l_raw.detach()
        self.last_tox = tox_labels.detach()
        return l_weighted

# %%

class MoleculeGenerator_RNN(MoleculeGenerator):
    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn,
                 *args, **kwargs):
        super(MoleculeGenerator_RNN, self).__init__(
            N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, *args, **kwargs)
        self.N_rnn = N_rnn
        self.rnn = nn.GRU(input_size=self.F_c[-1] * 2, hidden_size=self.F_c[-1],
                          num_layers=self.N_rnn, batch_first=True)

    def _rnn_train(self, X, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum):
        X_avg = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X_avg, X[NX_cum - 1]], dim=1)
        X = X[graph_to_rnn]
        batch_size, iw_size, length, num_features = X.shape
        out, _ = self.rnn(X.reshape(batch_size * iw_size, length, num_features))
        out = out.reshape(batch_size, iw_size, length, -1)
        return out[rnn_to_graph[0], rnn_to_graph[1], rnn_to_graph[2]]

    def _rnn_test(self, X, NX, NX_rep, NX_cum, h):
        X_avg = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X_avg, X[NX_cum - 1]], dim=1)
        X, h = self.rnn(X.unsqueeze(1), h)

        return X.squeeze(1), h

# %%

class _TwoLayerDense(nn.Module):
    """FNN_0: protein embedding -> first-atom LOGITS.

    New at fine-tuning time: CDGCN 2.2 replaces the learnable `policy_0` vector
    with FNN_0(c). The paper does not specify the initialisation, and `loss_init`
    is one term in ~34 per molecule so it learns slowly — hence the optional warm
    start from the pretrained marginal.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(_TwoLayerDense, self).__init__()
        self.input = nn.Linear(input_size, hidden_size, bias=False)
        self.bn_input = nn.BatchNorm1d(hidden_size, eps=1e-5, momentum=0.1)
        self.output = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, c):
        return self.output(torch.relu(self.bn_input(self.input(c))))

# %%

class CMoleculeGenerator_RNN(MoleculeGenerator_RNN):
    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, N_C, D, F_e, F_skip, F_c, Fh_policy, activation,
                 N_rnn, *args, **kwargs):
        self.N_C = N_C
        super(CMoleculeGenerator_RNN, self).__init__(
            N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn,
            *args, **kwargs)
        self.dense_policy_0 = _TwoLayerDense(self.N_C, self.N_A * 3, self.N_A)

    @abstractmethod
    def _graph_conv_forward(self, X, A, c, ids):
        pass

    def _policy_0(self, c):
        # 0.0 * policy_0 keeps the unused pretraining parameter in the graph
        return torch.softmax(self.dense_policy_0(c), dim=-1) + 0.0 * self.policy_0

    def _log_policy_0(self, c):
        return torch.log_softmax(self.dense_policy_0(c), dim=-1) + 0.0 * self.policy_0

    def _policy(self, X, A, NX, NX_rep, last_append_mask,
                graph_to_rnn, rnn_to_graph, NX_cum, c, ids):
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        X = self._graph_conv_forward(X, A, c, ids)
        X_dense = self.dense(X)
        X_mol = self._rnn_train(X_dense, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum)
        return self.policy_h(X_dense, NX, NX_rep, X_mol, return_log=True)

    def _decode_step(self, X, A, NX, NX_rep, last_append_mask, NX_cum, h, c, ids):
        """One sampling step — unused in training, needed by the sampler."""
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        X = self._graph_conv_forward(X, A, c, ids)
        X_dense = self.dense(X)
        X_mol, h = self._rnn_test(X_dense, NX, NX_rep, NX_cum, h)
        return self.policy_h(X_dense, NX, NX_rep, X_mol) + (h,)

    def forward(self, *inputs):
        if self.mode in ('loss', 'likelihood'):
            (X, A, iw_ids, last_append_mask, NX, NX_rep, action_0, actions, log_p,
             batch_size, iw_size, tox_class, graph_to_rnn, rnn_to_graph, NX_cum,
             c, ids) = inputs
            init = self._log_policy_0(c).unsqueeze(1).expand(-1, iw_size, -1)
            append, connect, end = self._policy(
                X, A, NX, NX_rep, last_append_mask, graph_to_rnn, rnn_to_graph,
                NX_cum, c, ids)
            l_weighted = self._likelihood(init, append, connect, end, action_0,
                                          actions, iw_ids, log_p, batch_size,
                                          iw_size, tox_class)
            return l_weighted if self.mode == 'likelihood' else -l_weighted.mean()
        elif self.mode == 'decode_0':
            return self._policy_0(*inputs)
        elif self.mode == 'decode_step':
            return self._decode_step(*inputs)

        raise ValueError(f"Unknown mode: {self.mode}")

# %%

class CVanillaMolGen_RNN(CMoleculeGenerator_RNN):
    def __init__(self, N_A, N_B, N_C, D, F_e, F_h, F_skip, F_c, Fh_policy,
                 activation, N_rnn):
        super(CVanillaMolGen_RNN, self).__init__(
            N_A, N_B, N_C, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, F_h)

    def _build_graph_conv(self, F_h):
        self.F_h = list(F_h)
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()            # bn[i-1] belongs to conv[i]
        for i, (f_in, f_out) in enumerate(zip([self.F_e] + self.F_h[:-1], self.F_h)):
            self.conv.append(GraphConv(f_in, f_out, self.N_B + self.D))
            if i != 0:
                self.bn.append(nn.BatchNorm1d(f_in, eps=1e-5, momentum=0.1))
        self.bn_skip = nn.BatchNorm1d(sum(self.F_h), eps=1e-5, momentum=0.1)
        self.linear_skip = Linear_BN(sum(self.F_h), self.F_skip)
        # FNN_1..FNN_l: the per-layer protein projectors
        self.linear_c = nn.ModuleList(
            [nn.Linear(self.N_C, f_out, bias=False) for f_out in self.F_h])

    def _graph_conv_forward(self, X, A, c, ids):
        X_out = [X]
        for i, conv in enumerate(self.conv):
            X = X_out[-1]
            c_i = self.linear_c[i](c)[ids, :]
            X_out.append(conv(X if i == 0 else self.activation(self.bn[i - 1](X)), A)
                         + c_i)
        X_out = torch.cat(X_out[1:], dim=1)
        return self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))


# %% [markdown]
# ## Init, early stopping, checkpointing

# %%
def init_weights_xavier(module):
    """MXNet's Xavier (uniform, avg, magnitude 3) == xavier_uniform_(gain=1)."""
    for name, p in module.named_parameters():
        if p.dim() < 2:
            continue
        nn.init.xavier_uniform_(p, gain=1.0)
    for name, p in module.named_parameters():
        if 'bias' in name.split('.')[-1]:
            nn.init.zeros_(p)
    for m in module.modules():
        if isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.reset_running_stats()
    if hasattr(module, 'policy_0'):
        nn.init.zeros_(module.policy_0)
    return module

# %%

class EarlyStopping(object):
    def __init__(self, patience=5, delta=1e-2):
        self.patience, self.delta = patience, delta
        self.best, self.counter = math.inf, 0
        self.improved = False

    def step(self, loss):
        if loss < self.best - self.delta:
            self.best, self.counter, self.improved = loss, 0, True
        else:
            self.counter += 1
            self.improved = False
        return self.counter >= self.patience

    def state_dict(self):
        return {'best': self.best, 'counter': self.counter}

    def load_state_dict(self, s):
        self.best = s.get('best', self.best)
        self.counter = s.get('counter', self.counter)

# %%

def atomic_save(obj, path, keep_prev=True):
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            torch.save(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        if keep_prev and os.path.exists(path):
            try:
                os.replace(path, path + ".prev")
            except OSError:
                pass
        os.replace(tmp, path)
        dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):

            os.remove(tmp)

# %%

def robust_load(path):
    """Load `path`, falling back to its rotated .prev copy."""
    for candidate, which in ((path, "current"), (path + ".prev", "previous")):
        if not os.path.isfile(candidate):
            continue
        try:
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            if which == "previous":
                print(f"  [WARN] {os.path.basename(path)} was unreadable; recovered "
                      f"the rotated backup. At most one interval is lost.")
            return state, which
        except Exception as e:
            print(f"  [WARN] {os.path.basename(candidate)} unreadable "
                  f"({type(e).__name__}: {e})")

    return None, None

# %%

def rng_state():
    s = {"python": random.getstate(), "numpy": np.random.get_state(),
         "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()

    return s

# %%

def set_rng_state(s):
    if not s:
        return
    try:
        random.setstate(s["python"])
        np.random.set_state(s["numpy"])
        torch.set_rng_state(torch.as_tensor(s["torch"], dtype=torch.uint8))
        if torch.cuda.is_available() and "cuda" in s:
            torch.cuda.set_rng_state_all([torch.as_tensor(x, dtype=torch.uint8)
                                          for x in s["cuda"]])
    except Exception as e:
        print(f"  [WARN] could not restore RNG state ({type(e).__name__})")


# %% [markdown]
# ## Resume state

# %%
is_continuous = False
global_counter = 0
epoch_no = 1
resume_state = None
elapsed_offset = 0.0
best_val_nll = math.inf

if FORCE_RESTART:
    print("  FORCE_RESTART — the previous run was archived before data work")
else:
    resume_state, _ = robust_load(CKPT_LAST)
    if resume_state is not None and "optimizer" not in resume_state:
        tstate, _ = robust_load(CKPT_TRAINER)
        if tstate is not None:
            resume_state = {**resume_state, **tstate}
            print("  recovered a legacy ckpt_last.pt + trainer.pt pair")
        else:
            print("  [WARN] ckpt_last.pt has weights but no optimizer state — "
                  "restarting from the pretrained checkpoint")
            resume_state = None
    is_continuous = resume_state is not None

# %%

def objective_signature():
    """Settings whose change makes a resumed run incomparable to the checkpoint."""
    return {'lambda_tox': LAMBDA_TOX, 'tox_floor': TOX_FLOOR,
            'early_stop_on': EARLY_STOP_ON, 'fold': FOLD, 'n_c': N_C,
            'split_seed': SPLIT_SEED, 'val_protein_fraction': VAL_PROTEIN_FRACTION,
            'stable_softmax': STABLE_SOFTMAX}

# %%

def data_signature():
    return {'pool_pair_files': POOL_PAIR_FILES, 'limit_pairs': LIMIT_PAIRS,
            'max_atoms': MAX_ATOMS, 'max_iter': MAX_ITER, 'batch_size': BATCH_SIZE,
            'strip_salts': STRIP_SALTS, 'fragment_policy': FRAGMENT_POLICY,
            'traversal_policy': TRAVERSAL_POLICY,
            'iters_per_epoch': ITERS_PER_EPOCH}

# %%
configs = {'N_C': N_C, 'F_e': F_E, 'F_h': F_H, 'F_skip': F_SKIP, 'F_c': F_C,

           'Fh_policy': FH_POLICY, 'activation': ACTIVATION, 'N_rnn': N_RNN}

# %%
if is_continuous:
    configs = resume_state.get("configs") or (
        json.load(open(CKPT_CONFIG)) if os.path.isfile(CKPT_CONFIG) else configs)

    old_sig, new_sig = resume_state.get('objective'), objective_signature()
    if old_sig is None:
        print("  [WARN] checkpoint predates objective tracking")
    else:
        changed = {k: (old_sig[k], v) for k, v in new_sig.items()
                   if k in old_sig and old_sig[k] != v}
        if changed:
            raise RuntimeError(
                "Settings changed since the checkpoint: "
                + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in changed.items())
                + "\n  Resuming would carry an optimiser state and an early-stopping "
                  "best belonging to the previous settings. Revert, or set "
                  "FORCE_RESTART = True.")
        absent = [k for k in new_sig if k not in old_sig]
        if absent:
            print(f"  [WARN] checkpoint does not record {absent}")

    old_data = resume_state.get('data')
    if old_data:
        d = {k: (old_data[k], v) for k, v in data_signature().items()
             if k in old_data and old_data[k] != v}
        if 'limit_pairs' in d:
            raise RuntimeError(f"The checkpoint was trained with LIMIT_PAIRS = "
                               f"{d['limit_pairs'][0]!r}, this run uses "
                               f"{d['limit_pairs'][1]!r}. A smoke run and a full run "
                               f"are different experiments.")
        for fatal in ('iters_per_epoch', 'strip_salts', 'traversal_policy',
                      'fragment_policy'):
            if fatal in d:
                raise RuntimeError(
                    f"The training set changed ({fatal}: {d[fatal][0]!r} -> "
                    f"{d[fatal][1]!r}). Epoch boundaries and the early-stopping best "
                    f"no longer line up. Set FORCE_RESTART = True.")
        if d:
            print(f"  [WARN] training-set settings changed: {list(d)}")
    print("  resuming an existing fine-tuning run")
else:
    print("  fresh fine-tuning run")

    archive_stale(RESUMABLE, why="fresh run")

# %%
check("embedding width matches the checkpoint", configs.get("N_C") == N_C,

      f"checkpoint N_C = {configs.get('N_C')}, this fold's embeddings are {N_C}-d")

with open(CKPT_CONFIG, 'w') as f:

    json.dump(configs, f, indent=1)

model = CVanillaMolGen_RNN(get_mol_spec().num_atom_types,
                           get_mol_spec().num_bond_types, D=D_RECEPTIVE, **configs)

# %%

def load_pretrained(model, path):
    """Transfer phase-1 weights. Only the conditioning path may be missing."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)

    new = ('linear_c.', 'dense_policy_0.')
    unexplained = [k for k in missing if not k.startswith(new)]
    if unexplained:
        raise RuntimeError(f"Pretrained checkpoint missing non-conditioning tensors: "
                           f"{unexplained[:10]} — the two phases have drifted apart.")
    print(f"  transferred {len(sd) - len(unexpected)}/{len(sd)} pretrained tensors; "
          f"{len(missing)} conditioning tensors newly initialised")
    print("  a clean transfer means the SHAPES match — it is not evidence that the "
          "checkpoint was trained.")
    if unexpected:
        print(f"  [WARN] unused tensors in the checkpoint: {unexpected[:10]}")

    if INIT_POLICY0_BIAS_FROM_PRETRAINED and 'policy_0' in sd:
        with torch.no_grad():
            model.dense_policy_0.output.bias.copy_(
                torch.log_softmax(sd['policy_0'].float().to(
                    model.dense_policy_0.output.bias.device), dim=0))
            model.dense_policy_0.output.weight.zero_()
        print("  FNN_0 warm start: output bias = log-softmax(pretrained policy_0), "
              "output weight zeroed. At step 0 the conditional first-atom "
              "distribution IS the pretrained marginal; protein dependence is "
              "learned from there. Deliberate departure from Xavier.")

    return missing

# %%
if is_continuous:
    model.load_state_dict(resume_state['model'])
else:
    init_weights_xavier(model)
    if os.path.exists(PRETRAINED_CKPT):
        load_pretrained(model, PRETRAINED_CKPT)
    elif REQUIRE_PRETRAINED:
        raise FileNotFoundError(f"No pretrained checkpoint at {PRETRAINED_CKPT}.")
    else:
        print("  [WARN] REQUIRE_PRETRAINED off — training from random init, which is "

              "NOT the two-phase protocol")

model.to(CTX)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
early_stopping = EarlyStopping(PATIENCE, ES_DELTA)

# %%
if is_continuous:
    optimizer.load_state_dict(resume_state['optimizer'])
    for g in optimizer.param_groups:
        g['lr'] = resume_state.get('lr', LR)
    global_counter = int(resume_state.get('global_counter', 0))
    epoch_no = int(resume_state.get('epoch_no', 1))
    early_stopping.load_state_dict(resume_state.get('early_stopping', {}))
    elapsed_offset = float(resume_state.get('elapsed_hours', 0.0))
    best_val_nll = float(resume_state.get('best_val_nll', math.inf))
    set_rng_state(resume_state.get('rng'))

    for st in optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(CTX)

    derived_epoch = global_counter // ITERS_PER_EPOCH + 1
    if derived_epoch != epoch_no:
        print(f"  [INFO] epoch counter {epoch_no} -> {derived_epoch} (derived from "
              f"iteration {global_counter:,} / {ITERS_PER_EPOCH:,})")
        epoch_no = derived_epoch

    print(f"  iteration {global_counter:,}, epoch {epoch_no}, "
          f"lr {optimizer.param_groups[0]['lr']:.3e}, best {early_stopping.best:.4f}, "
          f"patience {early_stopping.counter}/{PATIENCE}")
    del resume_state
    gc.collect()

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_cond = sum(p.numel() for n, p in model.named_parameters()
             if n.startswith(('linear_c.', 'dense_policy_0.')))
print(f"\n  {n_trainable:,} trainable parameters ({n_cond:,} = "
      f"{100 * n_cond / n_trainable:.0f}% in the conditioning path)")

# %% [markdown]
# ## Sanity checks
#
# Eq. 8 against its closed form; the conditioning actually *helping* rather than
# merely moving the loss; and the scale of the conditioning term against the graph
# signal it is added to.

# %%
def sanity_check(model, batch_cpu, n_perm=8):
    prev = model.mode
    model.mode = 'likelihood'
    model.eval()
    try:
        with torch.no_grad():
            inputs = batch_to_tensor(batch_cpu)
            lw = model(*inputs)
            l, tox = model.last_l, model.last_tox

        l_eff = l if TOX_FLOOR is None else torch.where(
            tox > 0, torch.clamp(l, min=TOX_FLOOR), l)
        err = float((lw - l_eff * (1.0 - (1.0 + LAMBDA_TOX) * tox)).abs().max())
        n_tox = int(tox.sum())
        print(f"  Eq. 8  | {tox.numel()} molecules, {n_tox} toxic | max error {err:.2e}")
        check("Eq. 8 matches its closed form", err < 1e-3, f"max error {err:.2e}")

        if n_tox:
            flipped = int((torch.sign(lw[tox > 0]) == -torch.sign(l_eff[tox > 0])).sum())
            check("sign flips on toxic molecules", flipped == n_tox, f"{flipped}/{n_tox}")
        else:
            print("  [WARN] no toxic molecule in this batch — the sign flip is not "
                  "exercised here")

        print(f"         per-action log-probs below {LOG_FLOOR:.2f}: "
              f"{100 * model.last_frac_floor:.3f}%")
        print(f"         max |logit| {model.policy_h.last_logit_absmax:.1f} | "
              f"{100 * model.policy_h.last_frac_extreme:.4f}% beyond +/-80")
    finally:
        model.mode = prev
        model.train()

# %%
sanity_check(model, batch_cpu)

# %%

def conditioning_check(model, batch_cpu, n_perm=8):
    """Does the RIGHT protein beat a wrong one?

    A single permutation of 8 embeddings is a one-bit verdict on the check that
    decides whether Table 2 means anything, and a permutation can be near-identity.
    This averages over several and reports the fraction the true pairing wins.
    """
    prev = model.mode
    model.mode = 'loss'
    model.eval()
    try:
        inputs = batch_to_tensor(batch_cpu)
        with torch.no_grad():
            real = model(*inputs).item()
            mismatched = []
            g = torch.Generator(device='cpu').manual_seed(SEED)
            n = inputs[15].shape[0]
            for _ in range(n_perm):
                perm = torch.randperm(n, generator=g).to(inputs[15].device)
                if bool((perm == torch.arange(n, device=perm.device)).all()):
                    continue
                shuffled = list(inputs)
                shuffled[15] = inputs[15][perm]      # real embeddings, wrong proteins
                mismatched.append(model(*shuffled).item())

            noise = list(inputs)
            noise[15] = torch.randn_like(inputs[15]) * float(inputs[15].std())
            random_c = model(*noise).item()

        wins = sum(1 for m in mismatched if real < m)
        med = float(np.median(mismatched)) if mismatched else float('nan')
        print(f"  cond   | real {real:.4f} | mismatched median {med:.4f} over "
              f"{len(mismatched)} shuffles | random {random_c:.4f}")
        check("the condition has some effect", abs(real - random_c) > 1e-6,
              f"delta vs random embedding {abs(real - random_c):.4f}")
        check("the RIGHT protein beats a mismatched one",
              wins >= 0.75 * max(1, len(mismatched)),
              f"real loss is lower than {wins}/{len(mismatched)} mismatched pairings. "
              + ("The model is using protein identity." if
                 wins >= 0.75 * max(1, len(mismatched)) else
                 "Pairing each ligand with the WRONG protein is as good or better. "
                 "The conditioning path is not doing useful work yet; Table 2 "
                 "(target awareness) is meaningless until this flips"), level="warn")
    finally:
        model.mode = prev

        model.train()

# %%
conditioning_check(model, batch_cpu)

# %%

def scale_check(model, batch_cpu):
    """Scale of the additive condition against the graph signal it is added to."""
    model.eval()
    try:
        with torch.no_grad():
            inputs = batch_to_tensor(batch_cpu)
            X, A, c, ids = inputs[0], inputs[1], inputs[15], inputs[16]
            h = model.embedding_atom(X) + model.embedding_mask(inputs[3])
            print(f"  scale  | std(c) = {c.std().item():.3f}")
            worst = 0.0
            for i, conv in enumerate(model.conv):
                inp = h if i == 0 else model.activation(model.bn[i - 1](h))
                g = conv(inp, A)
                c_i = model.linear_c[i](c)[ids, :]
                r = c_i.std().item() / max(1e-9, g.std().item())
                worst = max(worst, r)
                print(f"         layer {i}: std(conv) {g.std().item():7.3f}  "
                      f"std(c_i) {c_i.std().item():7.3f}  ratio {r:6.1f}x")
                h = g + c_i
            check("conditioning does not swamp the graph signal", worst <= 10,
                  f"worst ratio {worst:.1f}x"
                  + ("" if worst <= 10 else
                     " — at this ratio the pretrained graph weights contribute little "
                     "early in fine-tuning. The step-0 measurement below settles "
                     "whether it costs likelihood"), level="warn")
    finally:
        model.train()

# %%
scale_check(model, batch_cpu)

# %% [markdown]
# ## Evaluation

# %%
def evaluate(model, loader, collect=False):
    """Molecule-weighted means: (Eq. 8 loss, plain NLL, mean l non-toxic, mean l toxic).

    With collect=True also returns per-molecule `l`, toxicity and **action count**,
    all gathered from the same batches so they cannot fall out of alignment.
    """
    prev = model.mode
    model.mode = 'likelihood'
    model.eval()
    sum_w = sum_l = 0.0
    n_mol = 0
    sum_non, n_non, sum_tox, n_tox = 0.0, 0, 0.0, 0
    all_l, all_tox, all_act = [], [], []
    try:
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                lw = model(*batch_to_tensor(batch))
                l, tox, act = model.last_l, model.last_tox, model.last_n_actions
                sum_w += float(lw.sum())
                sum_l += float(l.sum())
                n_mol += int(l.numel())
                if (tox == 0).any():
                    sum_non += float(l[tox == 0].sum())
                    n_non += int((tox == 0).sum())
                if (tox > 0).any():
                    sum_tox += float(l[tox > 0].sum())
                    n_tox += int((tox > 0).sum())
                if collect:
                    all_l.append(l.cpu().numpy())
                    all_tox.append(tox.cpu().numpy())
                    all_act.append(act.cpu().numpy())
    finally:
        model.mode = prev
        model.train()

    if n_mol == 0:
        raise RuntimeError("Validation loader produced no molecules; early stopping "
                           "would silently see NaN.")

    out = (-sum_w / n_mol, -sum_l / n_mol,
           sum_non / n_non if n_non else float('nan'),
           sum_tox / n_tox if n_tox else float('nan'))
    if collect:
        return (out, np.concatenate(all_l), np.concatenate(all_tox),
                np.concatenate(all_act))

    return out

# %%

def select_metric(val_w, val_nll, val_non):
    """The quantity early stopping minimises. `val_non` is a log-likelihood
    (negative, higher is better), so the minimised form is its negation."""
    if EARLY_STOP_ON == 'weighted':
        return val_w
    if EARLY_STOP_ON == 'nll':
        return val_nll
    if EARLY_STOP_ON == 'l_nontoxic':
        return -val_non
    raise ValueError(f"Unknown EARLY_STOP_ON: {EARLY_STOP_ON!r}")


# %% [markdown]
# ## Step-0 measurement
#
# The likelihood distribution **before the first optimizer step**. One validation
# pass, and it answers four questions at once: baseline checkpoint quality on
# *this* data with no domain-shift confound; whether the toxic/non-toxic gap is
# inherited suppression or a **molecule-size confound**; the only defensible basis
# for a `TOX_FLOOR`; and whether the conditioning initialisation costs likelihood,
# which a variance ratio cannot settle.

# %%
if MEASURE_STEP0 and not is_continuous:
    print("  measuring step-0 likelihoods (one validation pass)...")
    t0 = time.time()
    (s0_w, s0_nll, s0_non, s0_tox), s0_l, s0_toxv, s0_act = evaluate(
        model, loader_val, collect=True)
    print(f"  [{time.time() - t0:.0f}s]  weighted {s0_w:.3f} | nll {s0_nll:.3f} | "
          f"l_nontox {s0_non:.2f} | l_tox {s0_tox:.2f}")

    non = s0_l[s0_toxv == 0]
    tox = s0_l[s0_toxv > 0]
    for label, a in (("non-toxic", non), ("toxic", tox)):
        if a.size:
            q = np.percentile(a, [1, 5, 50, 95])
            print(f"  {label:9s} n={a.size:>5}  mean {a.mean():8.2f}  p1 {q[0]:8.2f}  "
                  f"p5 {q[1]:8.2f}  median {q[2]:8.2f}  p95 {q[3]:8.2f}")

# %%
if MEASURE_STEP0 and not is_continuous:
    # `l` grows with action count, so a raw toxic/non-toxic gap can be molecule
    # size rather than suppression. Decompose it by least squares:
    #
    #     l ~ a * action_count + b + c * toxic
    #
    # then  mean l_tox - mean l_non  ==  a * delta_actions  +  c  exactly, so `c`
    # is the gap at MATCHED molecule size and `a * delta_actions` is the size
    # component. A per-action ratio cannot do this: dividing by each molecule's own
    # action count and multiplying back by the pooled mean recovers only
    # c * mean(act)/mean(act_tox) of the real effect, understating it whenever the
    # toxic molecules are the larger ones.
    #
    # Linearity in action count is a CONTROL, not a claim; r2 below says how well
    # it holds on this data.
    def size_controlled_gap(l_arr, tox_arr, act_arr):
        """-> (size_component, matched_size_gap, r2) in nats, or None."""
        mask = np.isfinite(l_arr) & np.isfinite(act_arr)
        l_arr, tox_arr, act_arr = l_arr[mask], tox_arr[mask], act_arr[mask]
        if (tox_arr > 0).sum() < 2 or (tox_arr == 0).sum() < 2:
            return None
        t_bin = (tox_arr > 0).astype(np.float64)
        M = np.stack([act_arr.astype(np.float64),
                      np.ones_like(t_bin), t_bin], axis=1)
        if np.linalg.matrix_rank(M) < 3:
            return None
        coef, *_ = np.linalg.lstsq(M, l_arr.astype(np.float64), rcond=None)
        a, _b, c = coef
        resid = l_arr - M @ coef
        var = float(((l_arr - l_arr.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / var if var > 0 else float('nan')
        d_act = float(act_arr[t_bin > 0].mean() - act_arr[t_bin == 0].mean())
        return float(a) * d_act, float(c), r2

    print(f"  mean actions: non-toxic {s0_act[s0_toxv == 0].mean():.1f} | "
          f"toxic {s0_act[s0_toxv > 0].mean():.1f}")
    _decomp = size_controlled_gap(s0_l, s0_toxv, s0_act)
    if _decomp is None:
        check("the toxic/non-toxic gap is suppression, not molecule size", True,
              "not enough molecules on one branch to separate the two", level="warn")
    else:
        _size_part, _matched, _r2 = _decomp
        _total = _size_part + _matched
        print(f"  gap decomposition (toxic - non-toxic, nats): total {_total:+.1f} "
              f"= size {_size_part:+.1f} + matched-size {_matched:+.1f}  "
              f"(r2 {_r2:.3f})")
        check("the toxic/non-toxic gap is suppression, not molecule size",
              abs(_matched) > abs(_size_part),
              f"at matched molecule size the gap is {_matched:+.1f} nats, against "
              f"{_size_part:+.1f} nats explained by action count. "
              + ("The gap survives the size control, so it is real suppression "
                 "inherited from stage 1."
                 if abs(_matched) > abs(_size_part) else
                 "Most of the gap is explained by toxic ligands simply being larger "
                 "molecules — it is largely a size confound, not suppression.")
              + (f" The linear control only explains r2 = {_r2:.2f} of the variance "
                 f"in l, so read the split loosely." if _r2 < 0.5 else ""),
              level="warn")

    if tox.size:
        # A defensible floor sits well below the typical NON-toxic bound (so it
        # never binds on the molecules we want to model) but above the toxic p5 at
        # step 0 (so it has room to bind later rather than being inert from
        # iteration 1).
        cand = float(np.floor(min(non.mean() if non.size else tox.mean(),
                                  np.percentile(tox, 5)) - 20.0))
        print(f"\n  TOX_FLOOR candidate: {cand:.1f}")
        print(f"    mean non-toxic l is {non.mean():.1f}, so the floor is "
              f"{abs(cand - non.mean()):.0f} nats below it")
        print(f"    toxic p5 at step 0 is {np.percentile(tox, 5):.1f}, so it starts "
              f"out binding on under 5% of toxic molecules")
        print(f"    Set TOX_FLOOR = {cand:.1f} ONLY for a deliberate stabilisation "
              f"experiment; None is the paper-faithful baseline.")

    np.savez_compressed(STEP0_PATH, l=s0_l, tox=s0_toxv, action_count=s0_act,
                        val_weighted=s0_w, val_nll=s0_nll,
                        val_l_nontoxic=s0_non, val_l_toxic=s0_tox)
    print(f"  -> {STEP0_PATH}")
else:

    print("  skipped (resuming, or MEASURE_STEP0 = False)")

# %%
gate("training")

# %% [markdown]
# ## Training loop

# %%
TRAIN_HEADER = ('step\ttime(h)\tloss\tlr\tl_nontoxic\tl_toxic\tepoch\tfrac_toxic'

                '\tfrac_floored\tgrad_norm\tfrac_clip\tfrac_logfloor\n')
VAL_HEADER = ('step\ttime(h)\tepoch\tval_weighted\tval_nll\tval_l_nontoxic'

              '\tval_l_toxic\tes_metric\tes_counter\tbest\n')

# %%

def ensure_header(path, header):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        with open(path, 'w') as f:

            f.write(header)

# %%
ensure_header(LOG_FILE, TRAIN_HEADER)

ensure_header(VAL_LOG_FILE, VAL_HEADER)

t0 = time.time() if not is_continuous else time.time() - elapsed_offset * 3600.0

# %%

def save_state(best=False, reason=""):
    state = {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'lr': optimizer.param_groups[0]['lr'],
        'global_counter': global_counter, 'epoch_no': epoch_no,
        'iters_per_epoch': ITERS_PER_EPOCH,
        'early_stopping': early_stopping.state_dict(),
        'best_val_nll': best_val_nll,
        'elapsed_hours': (time.time() - t0) / 3600.0,
        'rng': rng_state(), 'configs': configs, 'fold': FOLD, 'N_C': N_C,
        'lambda_tox': LAMBDA_TOX, 'tox_floor': TOX_FLOOR,
        'objective': objective_signature(), 'data': data_signature(),
        'smoke_test': SMOKE_TEST,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'reason': reason,
    }
    atomic_save(state, CKPT_LAST)
    atomic_save({k: state[k] for k in ('optimizer', 'lr', 'global_counter',
                                       'epoch_no', 'early_stopping')},
                CKPT_TRAINER, keep_prev=False)
    if best:
        atomic_save({'model': model.state_dict(), 'configs': configs, 'fold': FOLD,
                     'epoch': epoch_no - 1, 'global_counter': global_counter,
                     'val_loss': early_stopping.best, 'val_metric': EARLY_STOP_ON,
                     'N_C': N_C, 'lambda_tox': LAMBDA_TOX, 'tox_floor': TOX_FLOOR,
                     'objective': objective_signature(), 'data': data_signature(),
                     'smoke_test': SMOKE_TEST,

                     'saved_at': state['saved_at']}, CKPT_BEST)

# %%

def die_nonfinite(what, value):
    """Stop WITHOUT touching ckpt_last.pt — the last good state must survive."""
    path = os.path.join(OUT_DIR, f"nonfinite_{time.strftime('%Y%m%d-%H%M%S')}.pt")
    try:
        atomic_save({'model': model.state_dict(), 'global_counter': global_counter,
                     'epoch_no': epoch_no, 'what': what}, path, keep_prev=False)
    except Exception as e:
        print(f"  [WARN] could not write {path}: {type(e).__name__}")
    raise RuntimeError(
        f"Non-finite {what} ({value}) at iteration {global_counter}. ckpt_last.pt is "
        f"untouched, so re-running resumes from the last good state; the failing "
        f"weights are in {os.path.basename(path)}. This is a real numerical failure, "
        f"unlike an 'nan' in the l_toxic column, which only means a logging window "

        f"held no toxic molecule.")

# %%

def grad_diagnostics(model, clip_value):
    """-> (pre-clip global norm, fraction of elements the clip will rewrite)."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return float('nan'), float('nan')
    norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads)))
    n_clip = sum(int((g.detach().abs() > clip_value).sum()) for g in grads)
    return norm, n_clip / max(1, sum(g.numel() for g in grads))

# %%
_stop = {'requested': False, 'name': None}

# %%

def _request_stop(signum, frame):
    if _stop['requested']:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        return
    _stop['requested'] = True
    _stop['name'] = signal.Signals(signum).name
    print(f"\n[SIGNAL] {_stop['name']} — finishing this iteration, checkpointing, "
          f"then exiting. Send it again to abort immediately.", flush=True)

# %%
if GRACEFUL_SIGNALS:
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _request_stop)
        except (ValueError, OSError):
            pass

# %%

def done():
    if epoch_no > MAX_EPOCHS:
        return f"MAX_EPOCHS = {MAX_EPOCHS} reached"
    if early_stopping.counter >= PATIENCE:
        return f"early stopping triggered ({early_stopping.counter}/{PATIENCE})"
    return None

# %%
print(f"Model selection: {EARLY_STOP_ON} | patience {PATIENCE}/{MAX_EPOCHS}")
print(f"Objective: Eq. 8, lambda={LAMBDA_TOX}, floor={TOX_FLOOR} | "

      f"softmax {'exact log-space' if STABLE_SOFTMAX else 'legacy'}")

print(f"RUN_TRAINING {RUN_TRAINING}" + ("  [SMOKE TEST]" if SMOKE_TEST else ""))

# %%
if not RUN_TRAINING:
    print("\nRUN_TRAINING is False — preflight only, nothing was trained.")
    print(f"Every check above passed the gate. Set RUN_TRAINING = True to start "
          f"{MAX_EPOCHS} epochs of {ITERS_PER_EPOCH:,} iterations.")
elif done():
    print(f"Nothing to do: {done()}. Raise MAX_EPOCHS, or set FORCE_RESTART = True.")
else:
    model.mode = 'loss'
    model.train()
    it_train = iter(loader_train)
    interrupted = False
    win = {'loss': 0.0, 'n_batch': 0, 'sum_non': 0.0, 'n_non': 0, 'sum_tox': 0.0,
           'n_tox': 0, 'n_floored': 0, 'logfloor': 0.0, 'grad_norm': 0.0,
           'frac_clip': 0.0, 'n_diag': 0}
    print(f"Training from iteration {global_counter:,}, epoch {epoch_no}")

    while not done():
        try:
            inputs_cpu = next(it_train)
        except StopIteration:
            # Epoch boundaries come from the iteration count below, NOT from this,
            # so a mid-epoch resume cannot produce a long epoch.
            it_train = iter(loader_train)
            continue
        if inputs_cpu is None:
            continue

        global_counter += 1
        is_summary = (global_counter % SUMMARY_STEP == 0)

        optimizer.zero_grad(set_to_none=True)    # Gluon overwrites; torch accumulates
        loss = model(*batch_to_tensor(inputs_cpu))
        if not torch.isfinite(loss):
            die_nonfinite("loss", loss.item())
        loss.backward()

        # clip_grad_value_ clamps, and clamping NaN returns NaN, so a non-finite
        # gradient would pass straight into optimizer.step().
        _grads = [p.grad for p in model.parameters() if p.grad is not None]
        if _grads and not torch.isfinite(torch.stack([g.sum() for g in _grads]).sum()):
            die_nonfinite("gradient", float('nan'))

        if is_summary:
            gn, fc = grad_diagnostics(model, CLIP_GRAD)
            win['grad_norm'] += gn
            win['frac_clip'] += fc
            win['n_diag'] += 1

        torch.nn.utils.clip_grad_value_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        if global_counter % DECAY_STEP == 0:
            for g in optimizer.param_groups:
                g['lr'] *= (1.0 - DECAY)

        if global_counter % CHECKPOINT_EVERY == 0:
            save_state(reason=f"periodic@{global_counter}")

        _l, _tox = model.last_l, model.last_tox       # the raw, unfloored bound
        win['loss'] += loss.item()
        win['n_batch'] += 1
        win['sum_non'] += float(_l[_tox == 0].sum())
        win['n_non'] += int((_tox == 0).sum())
        win['sum_tox'] += float(_l[_tox > 0].sum())
        win['n_tox'] += int((_tox > 0).sum())
        win['n_floored'] += int(model.last_floored.sum())
        win['logfloor'] += model.last_frac_floor

        if is_summary:
            lr_now = optimizer.param_groups[0]['lr']
            hours = (time.time() - t0) / 3600.0
            mean_loss = win['loss'] / max(1, win['n_batch'])
            l_non = win['sum_non'] / win['n_non'] if win['n_non'] else float('nan')
            l_tox = win['sum_tox'] / win['n_tox'] if win['n_tox'] else float('nan')
            frac_tox = win['n_tox'] / max(1, win['n_tox'] + win['n_non'])
            frac_floor = win['n_floored'] / max(1, win['n_tox'])
            g_norm = win['grad_norm'] / max(1, win['n_diag'])
            f_clip = win['frac_clip'] / max(1, win['n_diag'])
            f_logfloor = win['logfloor'] / max(1, win['n_batch'])

            with open(LOG_FILE, 'a') as f:
                f.write(f"{global_counter}\t{hours:.4f}\t{mean_loss:.4f}\t{lr_now}"
                        f"\t{l_non:.4f}\t{l_tox:.4f}\t{epoch_no}\t{frac_tox:.4f}"
                        f"\t{frac_floor:.4f}\t{g_norm:.4f}\t{f_clip:.6f}"
                        f"\t{f_logfloor:.6f}\n")

            print(f"iter {global_counter:06d} | ep {epoch_no} | loss {mean_loss:8.3f} "
                  f"| l_nontox {l_non:8.2f} | l_tox {l_tox:9.2f} | "
                  f"tox {100 * frac_tox:4.1f}%"
                  + (f" | floored {100 * frac_floor:5.1f}%" if TOX_FLOOR is not None
                     else "")
                  + f" | |g| {g_norm:7.1f} | clip {100 * f_clip:4.1f}%"
                    f" | lr {lr_now:.2e} | {hours:.2f}h", flush=True)

            if TOX_FLOOR is not None and win['n_tox'] and frac_floor > 0.98:
                print("  [WARN] essentially every toxic molecule is at the floor: the "
                      "toxic branch contributes no gradient, equivalent to "
                      "LAMBDA_TOX = 0. Lower the floor, or accept that toxic ligands "
                      "are simply being ignored.")
            if f_clip > 0.5:
                print("  [WARN] over half the gradient elements are being clipped, so "
                      "only the gradient SIGN survives for those parameters.")
            if f_logfloor > 0.10:
                print("  [WARN] more than 10% of per-action log-probabilities are "
                      "below ln(1e-10): the objective is driving actions to "
                      "numerically zero probability.")

            for k in win:
                win[k] = 0.0
            del loss, inputs_cpu
            gc.collect()

        if global_counter % ITERS_PER_EPOCH == 0:
            finished = epoch_no
            epoch_no += 1

            val_w, val_nll, val_non, val_tox = evaluate(model, loader_val)
            metric = select_metric(val_w, val_nll, val_non)
            stop = early_stopping.step(metric)
            hours = (time.time() - t0) / 3600.0

            with open(VAL_LOG_FILE, 'a') as f:
                f.write(f"{global_counter}\t{hours:.4f}\t{finished}\t{val_w:.4f}"
                        f"\t{val_nll:.4f}\t{val_non:.4f}\t{val_tox:.4f}"
                        f"\t{metric:.4f}\t{early_stopping.counter}"
                        f"\t{early_stopping.best:.4f}\n")

            print(f"epoch {finished} | val weighted {val_w:.4f} | nll {val_nll:.4f} | "
                  f"l_nontox {val_non:.3f} | l_tox {val_tox:.3f} | "
                  f"{EARLY_STOP_ON} {metric:.4f} | best {early_stopping.best:.4f} | "
                  f"patience {early_stopping.counter}/{PATIENCE}"
                  f"{'  <- new best' if early_stopping.improved else ''}")

            # Plain NLL watched against its own best, independently of the selection
            # metric. Under a working CTDDG this RISES by construction, so it is a
            # diagnostic, not a target.
            if val_nll < best_val_nll:
                best_val_nll = val_nll
            elif val_nll > best_val_nll + NLL_DIVERGENCE_MARGIN:
                print(f"  [note] val NLL {val_nll:.2f} is "
                      f"{val_nll - best_val_nll:.2f} above its best "
                      f"({best_val_nll:.2f}). Expected when toxicity suppression is "
                      f"working. Judge the run on l_nontoxic ({val_non:.2f}).")

            save_state(best=early_stopping.improved, reason=f"epoch{finished}")
            if stop:
                print(f"Early stopping after epoch {finished}.")

        if _stop['requested']:
            save_state(reason=f"signal:{_stop['name']}@{global_counter}")
            with open(LOG_FILE, 'a') as f:
                f.write(f"# interrupted by {_stop['name']} at {global_counter}\n")
            print(f"[SIGNAL] checkpointed at iteration {global_counter}.")
            interrupted = True
            break

    if not interrupted:
        save_state(reason="final")
        if not os.path.exists(CKPT_BEST):
            save_state(best=True)      # no epoch improved; keep the final weights
        with open(LOG_FILE, 'a') as f:
            f.write('# finished\n')
        print(f"\nDone. Best validation {EARLY_STOP_ON}: {early_stopping.best:.4f}")

# %% [markdown]
# ## Summary

# %%
print("=" * 78)
_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
for status, name, detail in sorted(CHECKS, key=lambda c: _order.get(c[0], 9)):
    print(f"  {status:5s}  {name}")
    print(f"         {detail}")

print("-" * 78)
for name, path in (("best weights", CKPT_BEST), ("last weights", CKPT_LAST),
                   ("configs", CKPT_CONFIG), ("test bundle", TEST_BUNDLE),
                   ("split manifest", SPLIT_MANIFEST),
                   ("step-0 likelihoods", STEP0_PATH),
                   ("train log", LOG_FILE), ("val log", VAL_LOG_FILE)):
    print(f"  {name:20s} {path}")
if SMOKE_TEST:
    print("\n[SMOKE TEST] checkpoints, bundle and metrics are pipeline artifacts, "
          "not results.")
print("\nNext: 03_generate_samples (start with SMOKE_FIRST = True).")

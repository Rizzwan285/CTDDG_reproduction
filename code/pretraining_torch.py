# %% [markdown]
# # Stage 1 — unconditional pretraining
#
# Learns to generate chemically valid molecules from ChEMBL, with no protein.
# Produces `checkpoints/ckpt.pt`, which stage 2 fine-tunes.
#
# **Nothing trains until `RUN_TRAINING = True`.** Everything above that switch is
# preflight: dataset hash, collator census, objective verification. A 480,000-
# iteration run costs days; the checks cost seconds.
#
# ## Why stage 1 does not run Eq. 8
#
# The published `pretraining.ipynb` never implemented Eq. 8 — `_likelihood`
# accepts `tox_class_batch` and ignores it — so whatever produced the paper's
# 94.4% validity was *CDGCN-objective stage 1 + Eq. 8 at stage 2 only*.
#
# The previous run here applied Eq. 8 for all 480,000 pretraining iterations.
# For a toxic molecule the objective is `min l`, unbounded below; the logged
# `l_tox` reached −637. `TOX_WEIGHTING = True` reproduces that deliberately.

# %%
import os
import gc
import time
import json
import math
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
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.autograd.function import once_differentiable

print(f"torch {torch.__version__} | rdkit {rdkit.__version__} | numpy {np.__version__}")

# %% [markdown]
# ## Config

# %%
FORCE_RESTART = os.environ.get("CTDDG_FORCE_RESTART", "0") == "1"
RUN_TRAINING = True          # the arm switch; flip only after the preflight is clean

GPU_ID = int(os.environ.get("CTDDG_GPU", "0"))
CTX = torch.device(f"cuda:{GPU_ID}") if torch.cuda.is_available() else torch.device("cpu")

REPO_ROOT = os.environ.get("CTDDG_REPO_ROOT", "/home/muhamed/repo/CTDDG")
DATA_PATH = os.environ.get(
    "CTDDG_DATA_PATH",
    os.path.join(REPO_ROOT, "datasets/data/chembl/chembl_pretrain_labelled.csv"))
ATOM_TYPES_PATH = os.environ.get(
    "CTDDG_ATOM_TYPES_PATH", os.path.join(REPO_ROOT, "datasets/data/atom_types.txt"))
CHECKPOINT_DIR = os.environ.get(
    "CTDDG_CHECKPOINT_DIR", os.path.join(REPO_ROOT, "checkpoints"))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CKPT_PARAMS_FILE = os.path.join(CHECKPOINT_DIR, "ckpt.pt")
CKPT_TRAINER_FILE = os.path.join(CHECKPOINT_DIR, "trainer.pt")
CKPT_CONFIG_FILE = os.path.join(CHECKPOINT_DIR, "configs.json")
LOG_FILE = os.path.join(CHECKPOINT_DIR, "log.out")
VAL_LOG_FILE = os.path.join(CHECKPOINT_DIR, "val.out")

VERIFY_DATA_HASH = True
REQUIRE_DATA_HASH_MATCH = True
SMILES_COL = None
TOX_COL = None
LIMIT_MOLECULES = None

# %% [markdown]
# ### Objective
#
# `False` = CDGCN's Eq. 6, weight 1 on every molecule — the upstream stage-1
# objective. `True` = CTDDG Eq. 8 in stage 1 as well, which is the ablation that
# collapsed validity. `LAMBDA_TOX = 0` is neither: it drops toxic molecules from
# the batch average.

# %%
TOX_WEIGHTING = False
LAMBDA_TOX = 1.0
TOX_FLOOR = None

# Exact max-shifted joint log-softmax. False restores the upstream
# clamp-at-±80-then-exponentiate path for an A/B comparison.

# %%
STABLE_SOFTMAX = True
FRAGMENT_POLICY = 'heavy_atoms'

# 'complete_dfs' finishes every branch point. 'upstream_broken'
# reproduces the traversal that silently dropped atoms.

# %%
TRAVERSAL_POLICY = 'complete_dfs'

# %% [markdown]
# ### Architecture and schedule

# %%
BATCH_SIZE = 8
K_PATHS = 5
P_ALPHA = 0.8

F_E = 16
F_H = [32, 64, 128, 128, 256, 256]
F_SKIP = 256
F_C = [512]
FH_POLICY = 128
ACTIVATION = 'relu'
N_RNN = 3
D_RECEPTIVE = 2
MAX_ITER = 120
MAX_ATOMS = 100

LR = 1e-3
DECAY = 1e-3
DECAY_STEP = 100
CLIP_GRAD = 3.0
SUMMARY_STEP = 500
CHECKPOINT_STEP = 500
TOTAL_ITERATIONS = 480000

VAL_MOLECULES = 2000          # held out from training; 0 disables
VAL_EVERY = 5000
VAL_SEED = 1234

NUM_WORKERS = 8
SEED = 42
ALLOW_TF32 = True
DIE_ON_NONFINITE = True
GRACEFUL_SIGNALS = True

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

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()

# %%
OBJECTIVE_NAME = "CTDDG Eq.8" if TOX_WEIGHTING else "CDGCN Eq.6 (unweighted NLL)"
print(f"objective: {OBJECTIVE_NAME} | softmax: "
      f"{'exact log-space' if STABLE_SOFTMAX else 'legacy clamp+exp'}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32

check("data file", os.path.exists(DATA_PATH), DATA_PATH)
check("atom types", os.path.exists(ATOM_TYPES_PATH), ATOM_TYPES_PATH)
check("batch size matches the paper", BATCH_SIZE == 8,
      f"BATCH_SIZE = {BATCH_SIZE} (paper: 8)", level="warn")

_t = torch.zeros((1,), device=CTX)
_ = _t.cpu().numpy()
check("device responsive", True,
      f"{CTX}" + (f" ({torch.cuda.get_device_name(GPU_ID)})" if CTX.type == "cuda" else ""))

if TOX_WEIGHTING:
    check("Eq. 8 in stage 1", False,
          "this is the ablation that collapsed validity to ~24%; the published "
          "stage-1 code does not do it. Proceed only if that is the experiment",
          level="warn")

# %% [markdown]
# ## Dataset provenance
#
# The cheapest check in the pipeline and the most consequential. Stage 1 refuses
# to start on a file that is not the one `random_forest_classifier` built.

# %%
DATA_MANIFEST = None
_man_path = DATA_PATH + ".manifest.json"

# %%
if not VERIFY_DATA_HASH:
    check("dataset hash", True, "verification disabled", level="warn")
elif not os.path.isfile(_man_path):
    check("dataset manifest", False,
          f"no manifest at {_man_path}. Build the dataset with "
          f"random_forest_classifier, or set VERIFY_DATA_HASH = False to train on "
          f"an unverified file",
          level="fatal" if REQUIRE_DATA_HASH_MATCH else "warn")
else:
    with open(_man_path) as f:
        DATA_MANIFEST = json.load(f)
    actual, expected = sha256_file(DATA_PATH), DATA_MANIFEST.get("sha256")
    check("dataset hash", actual == expected,
          f"{actual[:16]}... " + ("matches the manifest" if actual == expected
                                  else f"!= manifest {str(expected)[:16]}..."),
          level="fatal" if REQUIRE_DATA_HASH_MATCH else "warn")

    _f = DATA_MANIFEST.get("filters", {})
    print(f"       {DATA_MANIFEST.get('n_molecules'):,} molecules, "
          f"{100 * DATA_MANIFEST.get('frac_toxic', 0):.1f}% toxic")
    print(f"       fragment policy {_f.get('fragment_policy')} | max atoms "
          f"{_f.get('max_atoms')} | max steps {_f.get('max_generation_steps')}")

    check("BindingDB ligands excluded", bool(_f.get("bindingdb_ligands_removed")),
          "the manifest confirms the exclusion both papers specify"
          if _f.get("bindingdb_ligands_removed") else
          "the manifest does NOT record the exclusion — this is the leakage that "
          "makes Tables 1-3 incomparable")
    check("bond cap matches this run",
          _f.get("max_generation_steps") == MAX_ITER,
          f"dataset built with max_generation_steps = "
          f"{_f.get('max_generation_steps')}, this run uses {MAX_ITER}", level="warn")
    check("fragment policy matches this run",
          _f.get("fragment_policy") in (None, FRAGMENT_POLICY),
          f"dataset built with {_f.get('fragment_policy')!r}, this run uses "
          f"{FRAGMENT_POLICY!r}", level="warn")

# %%
gate("data load")

# %% [markdown]
# ## Molecule spec and traversal

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
# ### Traversal self-test
#
# Seconds. If this fails, every training target downstream is suspect.

# %%
_ok, _why = True, ""
try:
    random.seed(SEED)
    for _smi in ("CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)cccn2", "C1CC2CCC1CC2"):
        _g, _at, _ar, _b, _bt = get_graph_from_smiles(_smi)
        for _ in range(10):
            _ids, _ = traverse_graph(_g, _ar)
            if min(_ids) < 0 or sorted(_ids) != list(range(len(_at))):
                _ok, _why = False, f"{_smi}: step_ids {_ids}"
                break
except Exception as e:
    _ok, _why = False, f"{type(e).__name__}: {e}"
check("traversal visits every atom exactly once", _ok,
      "3 molecules x 10 traversals, step_ids are a permutation" if _ok else _why)

_raised = False
try:
    get_graph_from_smiles("CC(=O)Oc1ccccc1C(=O)O.Cl")
except ValueError:
    _raised = True
check("disconnected graphs are refused", _raised,
      "get_graph_from_smiles raises on an unstripped salt rather than silently "
      "mis-wiring its bonds" if _raised else "a salt passed through unrejected")

# %% [markdown]
# ## Data ingestion

# %%
_PREFERRED_SMILES = ["Smiles", "smiles", "SMILES", "canonical_smiles",

                     "Canonical_Smiles", "CanonicalSmiles"]
_PREFERRED_TOX = ["Ames_prediction", "Ames_Prediction", "ames_prediction",

                  "Toxic_class", "tox_class", "Toxicity", "toxicity"]

# %%

def _resolve_columns(columns):
    cols = list(columns)
    smi = SMILES_COL or next((c for c in _PREFERRED_SMILES if c in cols), None)
    if smi is None:
        smi = next((c for c in cols if 'smiles' in c.lower()), None)
    if smi is None:
        raise ValueError(
            f"No SMILES column among {cols[:40]}. Set SMILES_COL explicitly. "
            f"(The previous fallback took column 0, which would train on whatever "
            f"happened to be first.)")

    tox = TOX_COL or next((c for c in _PREFERRED_TOX if c in cols), None)
    if tox is None:
        tox = next((c for c in cols
                    if ('ames' in c.lower() or 'toxic' in c.lower()) and c != smi), None)
    if tox is None:
        raise ValueError(f"No toxicity column among {cols[:40]}. Set TOX_COL explicitly.")

    return smi, tox

# %%

def load_chembl_dataset(file_path):
    header = pd.read_csv(file_path, nrows=0)
    smiles_col, tox_col = _resolve_columns(header.columns)
    print(f"  columns: SMILES={smiles_col!r}  toxicity={tox_col!r}")

    df = pd.read_csv(file_path, usecols=[smiles_col, tox_col])
    n_before = len(df)
    df = df.dropna(subset=[smiles_col, tox_col])
    if len(df) != n_before:
        print(f"  dropped {n_before - len(df)} rows with a missing SMILES or label")

    records = list(zip(df[smiles_col].astype(str).tolist(),
                       df[tox_col].astype(float).astype(int).tolist()))
    if LIMIT_MOLECULES is not None:
        records = records[:LIMIT_MOLECULES]
        print(f"  [WARN] LIMIT_MOLECULES: using only {len(records)} molecules")
    return records

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


# %% [markdown]
# ## Collator
#
# Three filters per record, applied before anything expensive: out-of-vocabulary
# atoms, oversized molecules, and molecules needing more than `MAX_ITER` steps —
# the last is the bond cap whose absence crashed a DataLoader worker hours in.

# %%
def process_single(record, k=K_PATHS, p=P_ALPHA):
    if isinstance(record, (tuple, list)):
        smiles, tox_label = record[0], int(record[1])
    elif " " in record:
        parts = record.split(" ")
        smiles, tox_label = parts[0], int(parts[1])
    else:
        smiles, tox_label = record, 0

    smiles = strip_salt(smiles)
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

class MolRNNCollate:
    """Standalone and picklable, so it survives DataLoader worker start-up."""

    def __init__(self, k=K_PATHS, p=P_ALPHA, max_iter=MAX_ITER, max_atoms=MAX_ATOMS):
        self.k, self.p = k, p
        self.max_iter, self.max_atoms = max_iter, max_atoms

    def _admissible(self, smiles):
        smiles = strip_salt(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "unparseable"
        if mol.GetNumAtoms() > self.max_atoms:
            return False, "too_many_atoms"
        if mol.GetNumBonds() + 1 > self.max_iter:
            return False, "too_many_bonds"
        try:
            for a in mol.GetAtoms():
                get_mol_spec().get_atom_type(a)
        except ValueError:
            return False, "out_of_vocab"
        try:
            _ = get_graph_from_smiles(smiles)
        except ValueError:
            return False, "disconnected"
        return True, "ok"

    def __call__(self, batch):
        valid_batch = []
        for record in batch:
            smi = record[0] if isinstance(record, (tuple, list)) else \
                (record.split()[0] if " " in record else record)
            if self._admissible(smi)[0]:
                valid_batch.append(record)
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

        smiles_batch = [strip_salt(b[0] if isinstance(b, (tuple, list))
                                   else b.split()[0]) for b in valid_batch]
        mol_list = [Chem.MolFromSmiles(s) for s in smiles_batch]

        graph_to_rnn = np.zeros((len(valid_batch), self.k, self.max_iter), dtype=np.int32)
        rnn_to_graph = []
        cum_sum = 0
        for i, mol_i in enumerate(mol_list):
            num_iter = mol_i.GetNumBonds() + 1
            assert num_iter <= self.max_iter, (
                f"{smiles_batch[i]} needs {num_iter} steps > max_iter {self.max_iter}")
            for k_idx in range(self.k):
                graph_to_rnn[i, k_idx, :num_iter] = np.arange(num_iter) + cum_sum
                r0 = np.ones([num_iter], dtype=np.int32) * i
                rnn_to_graph.append(np.stack([r0, np.ones_like(r0) * k_idx,
                                              np.arange(num_iter)], axis=0))
                cum_sum += num_iter

        rnn_to_graph = np.concatenate(rnn_to_graph, axis=1)
        NX_cum = np.cumsum(result_out[6])
        return result_out + [graph_to_rnn, rnn_to_graph, NX_cum]

# %%

def batch_to_tensor(record, device=None):
    device = CTX if device is None else device
    *base, graph_to_rnn, rnn_to_graph, NX_cum = record
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

    iw_ids, last_append_mask, NX, NX_rep, action_0, actions = [
        torch.as_tensor(x, dtype=torch.long, device=device)
        for x in [iw_ids, last_append_mask, NX, NX_rep, action_0, actions]]
    log_p = torch.as_tensor(log_p, dtype=torch.float32, device=device)
    tox_class = torch.as_tensor(tox_class, dtype=torch.float32, device=device)
    graph_to_rnn = torch.as_tensor(graph_to_rnn, dtype=torch.long, device=device)
    rnn_to_graph = torch.as_tensor(rnn_to_graph, dtype=torch.long, device=device)
    NX_cum = torch.as_tensor(NX_cum, dtype=torch.long, device=device)

    return [X, A_sparse, iw_ids, last_append_mask, NX, NX_rep, action_0, actions,
            log_p, batch_size, iw_size, tox_class, graph_to_rnn, rnn_to_graph, NX_cum]


# %% [markdown]
# ## Build the dataset and loaders

# %%
t0 = time.time()
dataset_all = load_chembl_dataset(DATA_PATH)
print(f"  {len(dataset_all):,} molecules loaded  [{time.time() - t0:.0f}s]")

check("dataset size", len(dataset_all) > 1000, f"{len(dataset_all):,} molecules")
check("labels are binary", set(r[1] for r in dataset_all) <= {0, 1},
      f"label values {sorted(set(r[1] for r in dataset_all))}")
if DATA_MANIFEST:
    check("row count matches the manifest",
          len(dataset_all) == DATA_MANIFEST.get("n_molecules"),
          f"{len(dataset_all):,} loaded vs "
          f"{DATA_MANIFEST.get('n_molecules'):,} in the manifest", level="warn")

if VAL_MOLECULES and VAL_MOLECULES < len(dataset_all) // 10:
    _perm = np.random.default_rng(VAL_SEED).permutation(len(dataset_all))
    _val_idx = set(_perm[:VAL_MOLECULES].tolist())
    dataset = [r for i, r in enumerate(dataset_all) if i not in _val_idx]
    dataset_val = [dataset_all[i] for i in sorted(_val_idx)]
else:
    dataset, dataset_val = dataset_all, []

labels = [r[1] for r in dataset]
num_toxic = sum(labels)
frac_toxic = num_toxic / max(1, len(labels))
print(f"  train {len(dataset):,} | held out {len(dataset_val):,}")
print(f"  non-toxic {len(labels) - num_toxic:,} | toxic {num_toxic:,} "
      f"({100 * frac_toxic:.1f}%)")

check("held-out slice exists", len(dataset_val) > 0,
      f"{len(dataset_val):,} molecules held out, evaluated every {VAL_EVERY:,} "
      f"iterations. This is the only signal that says the run is healthy before "
      f"it finishes" if dataset_val else
      "VAL_MOLECULES = 0 — the run will have no held-out signal at all until "
      "00_phase0_diagnostics check A is run on the finished checkpoint",
      level="warn")

collate_fn = MolRNNCollate(k=K_PATHS, p=P_ALPHA)

# %%
loader_train = DataLoader(
    dataset, batch_sampler=BalancedSampler([len(r[0]) for r in dataset],
                                           BATCH_SIZE, "chembl-train"),
    num_workers=NUM_WORKERS, collate_fn=collate_fn,
    persistent_workers=(NUM_WORKERS > 0))

loader_val = None
if dataset_val:
    loader_val = DataLoader(
        dataset_val, batch_sampler=BalancedSampler([len(r[0]) for r in dataset_val],
                                                   BATCH_SIZE, "chembl-val"),
        num_workers=min(2, NUM_WORKERS), collate_fn=collate_fn,
        persistent_workers=(NUM_WORKERS > 0))

batch_cpu = next(iter(loader_train))
check("collator produced a batch", batch_cpu is not None, "first batch is not None")

# %% [markdown]
# ### Collator rejection census
#
# A non-zero `too_many_bonds` is the crash the previous collator would have hit.

# %%
_probe = random.Random(SEED).sample(range(len(dataset)), min(20000, len(dataset)))
_reasons = {}
for i in _probe:
    ok, why = collate_fn._admissible(dataset[i][0])
    if not ok:
        _reasons[why] = _reasons.get(why, 0) + 1

print(f"  probed {len(_probe):,} molecules")
for k, v in sorted(_reasons.items(), key=lambda kv: -kv[1]):
    print(f"    {k:18s} {v:>6,}  ({100 * v / len(_probe):.2f}%)")
if not _reasons:
    print("    (nothing rejected)")

_drop_rate = sum(_reasons.values()) / len(_probe)
check("collator rejection rate", _drop_rate < 0.02,
      f"{100 * _drop_rate:.2f}% rejected; too_many_bonds = "
      f"{_reasons.get('too_many_bonds', 0)} (these would have crashed a worker "
      f"in the previous collator)", level="warn")

# %% [markdown]
# ## Model

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


# %% [markdown]
# ### Generator and objective

# %%
LOG_EPS = 1e-10

LOG_FLOOR = math.log(LOG_EPS)       # -23.0259, kept as the diagnostic threshold

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

        self.last_l = None                 # the UNFLOORED per-molecule bound
        self.last_tox = None               # the REAL labels, even when not weighting
        self.last_n_actions = None         # GetNumBonds() + 1, by construction
        self.last_frac_floor = None
        self.last_floored = None

    @abstractmethod
    def _build_graph_conv(self, *args, **kwargs):
        pass

    @abstractmethod
    def _graph_conv_forward(self, X, A):
        pass

    def _policy_0(self):
        return torch.softmax(self.policy_0, dim=0)

    def _log_policy_0(self):
        return torch.log_softmax(self.policy_0, dim=0)

# %%

def _likelihood_body(self, init, append, connect, end, action_0, actions, iw_ids,
                     log_p_sigma, batch_size, iw_size, tox_class_batch):
    """`init`, `append`, `connect`, `end` arrive as LOG-probabilities."""
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

    # Fraction of per-action log-probabilities below ln(1e-10). Under the legacy
    # path that value was a hard floor AND a dead gradient; here nothing is
    # floored, so this is a free measurement of how far the objective is driving
    # actions toward zero probability.
    with torch.no_grad():
        self.last_frac_floor = float((log_p_x <= LOG_FLOOR).float().mean())

    log_p_x = segment_sum(log_p_x.unsqueeze(-1), iw_ids,
                          batch_size * iw_size).squeeze(-1)
    log_p_x = log_p_x + loss_init

    log_p_x = log_p_x.reshape(batch_size, iw_size)
    log_p_sigma = log_p_sigma.reshape(batch_size, iw_size)

    # Eq. 3: the importance-sampled per-molecule bound.
    l = log_p_x - log_p_sigma
    l = torch.logsumexp(l, dim=1) - math.log(float(iw_size))

    if tox_class_batch.numel() != batch_size * iw_size:
        raise RuntimeError(
            f"tox_class has {tox_class_batch.numel()} entries, expected "
            f"batch_size*K = {batch_size * iw_size}; labels would be misaligned.")
    real_tox = tox_class_batch.reshape(batch_size, iw_size)[:, 0]

    with torch.no_grad():
        self.last_n_actions = torch.bincount(
            iw_ids, minlength=batch_size * iw_size
        ).reshape(batch_size, iw_size)[:, 0].to(l.dtype)

    return l, real_tox

# %%
MoleculeGenerator._likelihood_body = _likelihood_body

# %%

def _likelihood(self, init, append, connect, end, action_0, actions, iw_ids,
                log_p_sigma, batch_size, iw_size, tox_class_batch):
    l, real_tox = self._likelihood_body(init, append, connect, end, action_0,
                                        actions, iw_ids, log_p_sigma, batch_size,
                                        iw_size, tox_class_batch)

    # With TOX_WEIGHTING False the effective label is zero everywhere, giving
    # weight 1 on every molecule — CDGCN's Eq. 6. NOT the same as LAMBDA_TOX = 0,
    # which gives weight 0 on toxic molecules and drops them from the average.
    eff_tox = real_tox if TOX_WEIGHTING else torch.zeros_like(real_tox)

    l_raw = l
    if TOX_WEIGHTING and TOX_FLOOR is not None:
        l = torch.where(real_tox > 0, torch.clamp(l, min=TOX_FLOOR), l)
        self.last_floored = ((real_tox > 0) & (l_raw < TOX_FLOOR)).detach()
    else:
        self.last_floored = torch.zeros_like(real_tox, dtype=torch.bool)

    l_weighted = l * (1.0 - (1.0 + LAMBDA_TOX) * eff_tox)
    self.last_l = l_raw.detach()
    self.last_tox = real_tox.detach()

    return l_weighted

# %%
MoleculeGenerator._likelihood = _likelihood

# %%

def _mg_forward(self, X, A, iw_ids, last_append_mask, NX, NX_rep, action_0, actions,
                log_p, batch_size, iw_size, tox_class, *rnn_args):
    init = self._log_policy_0().unsqueeze(0).expand(batch_size * iw_size, -1)
    append, connect, end = self._policy(X, A, NX, NX_rep, last_append_mask, *rnn_args)
    l_weighted = self._likelihood(init, append, connect, end, action_0, actions,
                                  iw_ids, log_p, batch_size, iw_size, tox_class)
    if self.mode == 'likelihood':
        return l_weighted

    return -l_weighted.mean()

# %%
MoleculeGenerator.forward = _mg_forward

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

    def _policy(self, X, A, NX, NX_rep, last_append_mask, graph_to_rnn, rnn_to_graph,
                NX_cum):
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        X = self._graph_conv_forward(X, A)
        X_dense = self.dense(X)          # computed once, unlike the MXNet original
        X_mol = self._rnn_train(X_dense, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum)

        return self.policy_h(X_dense, NX, NX_rep, X_mol, return_log=True)

# %%

class VanillaMolGen_RNN(MoleculeGenerator_RNN):
    def __init__(self, N_A, N_B, D, F_e, F_h, F_skip, F_c, Fh_policy, activation, N_rnn):
        super(VanillaMolGen_RNN, self).__init__(
            N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, F_h)

    def _build_graph_conv(self, F_h):
        self.F_h = list(F_h)
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()          # bn[i-1] belongs to conv[i]
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
            X_out.append(conv(X if i == 0 else self.activation(self.bn[i - 1](X)), A))
        X_out = torch.cat(X_out[1:], dim=1)
        return self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))


# %% [markdown]
# ## Initialisation and signatures

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

def objective_signature():
    return {"tox_weighting": TOX_WEIGHTING, "lambda_tox": LAMBDA_TOX,
            "tox_floor": TOX_FLOOR, "objective": OBJECTIVE_NAME,
            "stable_softmax": STABLE_SOFTMAX}

# %%

def data_signature():
    return {"data_path": DATA_PATH,
            "data_sha256": (DATA_MANIFEST or {}).get("sha256"),
            "n_molecules": len(dataset_all), "n_train": len(dataset),
            "n_val": len(dataset_val), "val_seed": VAL_SEED,
            "batch_size": BATCH_SIZE, "k_paths": K_PATHS, "p_alpha": P_ALPHA,
            "max_atoms": MAX_ATOMS, "max_iter": MAX_ITER,
            "fragment_policy": FRAGMENT_POLICY,
            "traversal_policy": TRAVERSAL_POLICY,
            "limit_molecules": LIMIT_MOLECULES}


# %% [markdown]
# ## Build and resume

# %%
configs = {'F_e': F_E, 'F_h': F_H, 'F_skip': F_SKIP, 'F_c': F_C,
           'Fh_policy': FH_POLICY, 'activation': ACTIVATION, 'N_rnn': N_RNN}

is_continuous = False
global_counter = 0
elapsed_offset = 0.0
resume_state = None
_have = all(os.path.isfile(f) for f in (CKPT_PARAMS_FILE, CKPT_TRAINER_FILE))

if FORCE_RESTART:
    stamp = time.strftime('%Y%m%d-%H%M%S')
    for p in (CKPT_PARAMS_FILE, CKPT_TRAINER_FILE, LOG_FILE, VAL_LOG_FILE):
        if os.path.exists(p):
            os.replace(p, f"{p}.stale.{stamp}")
            print(f"  {os.path.basename(p)} -> .stale.{stamp}")
    with open(CKPT_CONFIG_FILE, 'w') as f:
        json.dump(configs, f, indent=1)
elif _have:
    is_continuous = True
    print("  existing checkpoint found - resuming")
    if os.path.isfile(CKPT_CONFIG_FILE):
        with open(CKPT_CONFIG_FILE) as f:
            _saved = json.load(f)
        configs = {k: v for k, v in _saved.items() if k in configs}
else:
    print("  fresh run")
    with open(CKPT_CONFIG_FILE, 'w') as f:
        json.dump(configs, f, indent=1)

model = VanillaMolGen_RNN(get_mol_spec().num_atom_types,
                          get_mol_spec().num_bond_types, D=D_RECEPTIVE, **configs)

if is_continuous:
    resume_state = torch.load(CKPT_PARAMS_FILE, map_location='cpu', weights_only=False)
    model.load_state_dict(resume_state['model'])
model.to(CTX)

# %%
if not is_continuous:
    init_weights_xavier(model)
    model.to(CTX)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)

# %%
if is_continuous:
    tstate = torch.load(CKPT_TRAINER_FILE, map_location='cpu', weights_only=False)
    optimizer.load_state_dict(tstate['optimizer'])
    for g in optimizer.param_groups:
        g['lr'] = tstate.get('lr', LR)
    for st in optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(CTX)

    global_counter = int(tstate.get('global_counter', 0))
    elapsed_offset = float(tstate.get('elapsed_hours', 0.0))

    _nbt = [int(v) for k, v in resume_state['model'].items()
            if k.endswith('num_batches_tracked')]
    bn_steps = max(_nbt) if _nbt else None
    check("resume counters agree", bn_steps == global_counter,
          f"trainer.pt says {global_counter:,}, BatchNorm says {bn_steps}. The "
          f"BatchNorm count is authoritative", level="warn")

    old_obj = tstate.get('objective')
    if old_obj is None:
        print("  [WARN] checkpoint predates objective tracking")
    else:
        changed = {k: (old_obj[k], v) for k, v in objective_signature().items()
                   if k in old_obj and old_obj[k] != v}
        if changed:
            raise RuntimeError(
                "The objective changed since this checkpoint: "
                + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in changed.items())
                + "\n  Resuming would produce a model trained half under one "
                  "objective and half under another. Set FORCE_RESTART = True.")

    old_data = tstate.get('data')
    if old_data:
        d = {k: (old_data[k], v) for k, v in data_signature().items()
             if k in old_data and old_data[k] != v}
        for fatal in ('data_sha256', 'n_val', 'val_seed', 'fragment_policy'):
            if fatal in d:
                raise RuntimeError(
                    f"The training data changed since this checkpoint ({fatal}: "
                    f"{d[fatal][0]!r} -> {d[fatal][1]!r}). Set FORCE_RESTART = True.")
        if d:
            print(f"  [WARN] data settings changed: {list(d)}")

    print(f"  resuming at iteration {global_counter:,}, "
          f"lr {optimizer.param_groups[0]['lr']:.3e}, {elapsed_offset:.2f} h spent")
    del resume_state
    gc.collect()

# %%
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_bn = sum(b.numel() for name, b in model.named_buffers()
           if name.endswith('running_mean') or name.endswith('running_var'))
print(f"  {n_trainable:,} trainable + {n_bn:,} BatchNorm running stats "
      f"= {n_trainable + n_bn:,}")
check("parameter count matches MXNet", n_trainable + n_bn == 6_986_506,
      f"{n_trainable + n_bn:,} (MXNet reported 6,986,506)", level="warn")

_b = batch_to_tensor(batch_cpu)
check("batch shape", _b[9] == BATCH_SIZE and _b[10] == K_PATHS,
      f"N={_b[9]} (expected {BATCH_SIZE}), K={_b[10]} (expected {K_PATHS})")
check("adjacency blocks", len(_b[1]) == get_mol_spec().num_bond_types + D_RECEPTIVE,
      f"{len(_b[1])} = N_B {get_mol_spec().num_bond_types} + D {D_RECEPTIVE}")
check("tox_class alignment", _b[11].numel() == _b[9] * _b[10],
      f"{_b[11].numel()} entries = batch {_b[9]} x K {_b[10]}")

# %% [markdown]
# ## Objective verification
#
# Against the closed form, on a real batch, before any optimizer step.

# %%
def verify_objective(model, batch_cpu):
    prev = model.mode
    model.mode = 'likelihood'
    model.eval()
    try:
        with torch.no_grad():
            lw = model(*batch_to_tensor(batch_cpu))
            l, tox = model.last_l, model.last_tox
            n_act = model.last_n_actions

        eff = tox if TOX_WEIGHTING else torch.zeros_like(tox)
        l_eff = l
        if TOX_WEIGHTING and TOX_FLOOR is not None:
            l_eff = torch.where(tox > 0, torch.clamp(l, min=TOX_FLOOR), l)
        err = float((lw - l_eff * (1.0 - (1.0 + LAMBDA_TOX) * eff)).abs().max())
        n_tox = int(tox.sum())

        print(f"  {tox.numel()} molecules, {n_tox} toxic")
        check("objective matches its closed form", err < 1e-3, f"max error {err:.2e}")

        if TOX_WEIGHTING and n_tox:
            flipped = int((torch.sign(lw[tox > 0]) == -torch.sign(l_eff[tox > 0])).sum())
            check("sign flips on toxic molecules", flipped == n_tox, f"{flipped}/{n_tox}")
        elif not TOX_WEIGHTING:
            same = int((torch.sign(lw) == torch.sign(l)).sum())
            check("no sign reversal under the CDGCN objective", same == l.numel(),
                  f"{same}/{l.numel()} molecules keep their sign — weight is 1 "
                  f"everywhere, including on the {n_tox} toxic ones")

        # n_actions must equal GetNumBonds() + 1 by construction.
        smis = [strip_salt(b[0] if isinstance(b, (tuple, list)) else b.split()[0])
                for b in [dataset[i] for i in next(iter(
                    BalancedSampler([len(r[0]) for r in dataset], BATCH_SIZE, "probe")))]]
        print(f"  action counts {n_act.int().tolist()[:8]}")
        check("action counts are integral and positive",
              bool((n_act > 0).all()) and float((n_act - n_act.round()).abs().max()) == 0,
              f"min {int(n_act.min())}, max {int(n_act.max())} steps per molecule")

        print(f"  per-action log-probabilities below {LOG_FLOOR:.2f}: "
              f"{100 * model.last_frac_floor:.3f}%")
        print(f"  max |logit| {model.policy_h.last_logit_absmax:.1f} | "
              f"{100 * model.policy_h.last_frac_extreme:.4f}% beyond +/-80")
        check("logits are not saturated", model.policy_h.last_frac_extreme < 1e-4,
              f"{100 * model.policy_h.last_frac_extreme:.4f}% of logits exceed "
              f"+/-80. Harmless under STABLE_SOFTMAX; under the legacy path these "
              f"are exactly the ones that were distorted", level="warn")
    finally:
        model.mode = prev

        model.train()

# %%
verify_objective(model, batch_cpu)

if STABLE_SOFTMAX:
    _p = model.policy_h
    _g = torch.Generator().manual_seed(0)
    _rx = torch.randn(9, 1, _p.N_B + _p.N_B * _p.N_A, generator=_g)
    _rt = torch.randn(3, 1, 1, generator=_g)
    _rep = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    _a, _b2 = action_log_probs(_rx, _rt, _rep, 3, stable=True)
    _a0, _b0 = action_log_probs(_rx, _rt, _rep, 3, stable=False)
    _tot = segment_sum(_a.exp(), _rep, 3).sum(-1) + _b2.exp().squeeze(-1)
    check("stable and legacy softmax agree on ordinary logits",
          float((_a - _a0).abs().max()) < 1e-4,
          f"max |delta log p| {float((_a - _a0).abs().max()):.2e}; probabilities "
          f"sum to 1 within {float((_tot - 1).abs().max()):.2e}")

    _rx2, _rt2 = _rx * 3e4, _rt * 3e4
    _a2, _b3 = action_log_probs(_rx2, _rt2, _rep, 3, stable=True)
    _tot2 = segment_sum(_a2.exp(), _rep, 3).sum(-1) + _b3.exp().squeeze(-1)
    check("stable softmax survives extreme logits",
          bool(torch.isfinite(_a2).all())
          and float((_tot2 - 1).abs().max()) < 1e-4,
          f"finite at +/-3e4 and still normalised to "
          f"{float((_tot2 - 1).abs().max()):.2e}")

gate("training")

# %% [markdown]
# ## Checkpointing
#
# Atomic: write to tmp, fsync, rotate, rename. A plain `torch.save` killed
# mid-write leaves a truncated file, and stage 1 writes one every 500 steps for
# days on a SLURM node.

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

def save_checkpoint(global_counter, elapsed_hours):
    atomic_save({'model': model.state_dict(), 'configs': configs,
                 'objective': objective_signature(), 'data': data_signature(),
                 'global_counter': global_counter,
                 'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')},
                CKPT_PARAMS_FILE)
    atomic_save({'optimizer': optimizer.state_dict(),
                 'lr': optimizer.param_groups[0]['lr'],
                 'global_counter': global_counter, 'elapsed_hours': elapsed_hours,
                 'objective': objective_signature(), 'data': data_signature()},

                CKPT_TRAINER_FILE, keep_prev=False)

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
    print(f"\n[SIGNAL] {_stop['name']} - finishing this iteration, checkpointing, "
          f"then exiting. Send it again to abort immediately.", flush=True)

# %%
if GRACEFUL_SIGNALS:
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _request_stop)
        except (ValueError, OSError):
            pass

# %% [markdown]
# ## Held-out evaluation

# %%
@torch.no_grad()
def evaluate(model, loader):
    """-> (NLL, mean l | non-toxic, mean l | toxic, n molecules)."""
    prev = model.mode
    model.mode = 'likelihood'
    model.eval()
    sum_l = sum_non = sum_tox = 0.0
    n_mol = n_non = n_tox = 0
    try:
        for batch in loader:
            if batch is None:
                continue
            model(*batch_to_tensor(batch))
            l, tox = model.last_l, model.last_tox
            sum_l += float(l.sum())
            n_mol += int(l.numel())
            if (tox == 0).any():
                sum_non += float(l[tox == 0].sum())
                n_non += int((tox == 0).sum())
            if (tox > 0).any():
                sum_tox += float(l[tox > 0].sum())
                n_tox += int((tox > 0).sum())
    finally:
        model.mode = prev
        model.train()
    return (-sum_l / max(1, n_mol),
            sum_non / n_non if n_non else float('nan'),
            sum_tox / n_tox if n_tox else float('nan'), n_mol)


# %% [markdown]
# ## Training loop
#
# `grad_norm`, `frac_clip` and `frac_floor` are the three columns that would have
# made the previous failure visible at iteration 5,000 rather than at evaluation.
# `frac_floor` is the one that matters most: it counts actions driven below
# `p = 1e-10`, which is where the old implementation stopped learning entirely.

# %%
TRAIN_HEADER = ('step\ttime(h)\tloss\tlr\tl_nontoxic\tl_toxic\tfrac_toxic'
                '\tgrad_norm\tfrac_clip\tfrac_floor\n')
VAL_HEADER = 'step\ttime(h)\tval_nll\tval_l_nontoxic\tval_l_toxic\tn_molecules\n'

# %%

def ensure_header(path, header):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        with open(path, 'w') as f:
            f.write(header)

# %%
ensure_header(LOG_FILE, TRAIN_HEADER)
if loader_val is not None:
    ensure_header(VAL_LOG_FILE, VAL_HEADER)

print("-" * 70)
print(f"  objective    {OBJECTIVE_NAME}")
print(f"  softmax      {'exact log-space' if STABLE_SOFTMAX else 'legacy clamp+exp'}")
print(f"  device       {CTX}")
print(f"  batch / K    {BATCH_SIZE} / {K_PATHS}   alpha {P_ALPHA}")
print(f"  lr           {LR} (decay {DECAY} every {DECAY_STEP})")
print(f"  from         iteration {global_counter:,} of {TOTAL_ITERATIONS:,}")
if not TOX_WEIGHTING:
    print("  NOTE: the `loss` column IS a negative log-likelihood under this")
    print("        objective. The MXNet CDGCN run's 15.99 is the nearest reference")
    print("        point, but it used a different framework and a differently")
    print("        filtered ChEMBL set, so treat it as indicative, not a target.")
else:
    print("  NOTE: the `loss` column is the Eq. 8 WEIGHTED loss. It goes negative,")
    print("        and a falling value means the toxic term is growing — not progress.")
print(f"  RUN_TRAINING {RUN_TRAINING}")
print("-" * 70)

# %%
if not RUN_TRAINING:
    print("RUN_TRAINING is False — preflight only, nothing was trained.")
    print(f"Every check above passed the gate. Set RUN_TRAINING = True to start "
          f"{TOTAL_ITERATIONS - global_counter:,} iterations.")
else:
    model.mode = 'loss'
    model.train()
    it_train = iter(loader_train)
    t0 = time.time() - elapsed_offset * 3600.0
    win = {'loss': 0.0, 'n': 0, 'sum_non': 0.0, 'n_non': 0, 'sum_tox': 0.0,
           'n_tox': 0, 'frac_floor': 0.0, 'grad_norm': 0.0, 'frac_clip': 0.0,
           'n_diag': 0}

    while global_counter < TOTAL_ITERATIONS:
        try:
            inputs_cpu = next(it_train)
        except StopIteration:
            it_train = iter(loader_train)
            continue
        if inputs_cpu is None:
            continue

        global_counter += 1
        is_summary = (global_counter % SUMMARY_STEP == 0)

        optimizer.zero_grad(set_to_none=True)     # Gluon overwrites; torch accumulates
        loss = model(*batch_to_tensor(inputs_cpu))

        if DIE_ON_NONFINITE and not torch.isfinite(loss):
            save_checkpoint(global_counter - 1, (time.time() - t0) / 3600.0)
            raise RuntimeError(f"Non-finite loss ({loss.item()}) at iteration "
                               f"{global_counter}. The checkpoint holds the last "
                               f"good state.")
        loss.backward()

        # Every step, not only summary steps: clamping a NaN returns NaN, so it
        # would pass straight into optimizer.step() and poison every parameter.
        if DIE_ON_NONFINITE:
            _g = [p.grad for p in model.parameters() if p.grad is not None]
            if _g and not torch.isfinite(torch.stack([x.sum() for x in _g]).sum()):
                save_checkpoint(global_counter - 1, (time.time() - t0) / 3600.0)
                raise RuntimeError(f"Non-finite gradient at iteration {global_counter}.")

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

        _l, _tox = model.last_l, model.last_tox
        win['loss'] += loss.item()
        win['n'] += 1
        win['sum_non'] += float(_l[_tox == 0].sum())
        win['n_non'] += int((_tox == 0).sum())
        win['sum_tox'] += float(_l[_tox > 0].sum())
        win['n_tox'] += int((_tox > 0).sum())
        win['frac_floor'] += model.last_frac_floor

        if global_counter % CHECKPOINT_STEP == 0:
            save_checkpoint(global_counter, (time.time() - t0) / 3600.0)

        if is_summary:
            lr_now = optimizer.param_groups[0]['lr']
            hours = (time.time() - t0) / 3600.0
            mean_loss = win['loss'] / max(1, win['n'])
            l_non = win['sum_non'] / win['n_non'] if win['n_non'] else float('nan')
            l_tox = win['sum_tox'] / win['n_tox'] if win['n_tox'] else float('nan')
            f_tox = win['n_tox'] / max(1, win['n_tox'] + win['n_non'])
            f_floor = win['frac_floor'] / max(1, win['n'])
            g_norm = win['grad_norm'] / max(1, win['n_diag'])
            f_clip = win['frac_clip'] / max(1, win['n_diag'])

            with open(LOG_FILE, 'a') as f:
                f.write(f"{global_counter}\t{hours:.4f}\t{mean_loss:.4f}\t{lr_now}"
                        f"\t{l_non:.4f}\t{l_tox:.4f}\t{f_tox:.4f}"
                        f"\t{g_norm:.4f}\t{f_clip:.6f}\t{f_floor:.6f}\n")

            print(f"iter {global_counter:06d} | loss {mean_loss:9.3f} | "
                  f"l_nontox {l_non:8.2f} | l_tox {l_tox:9.2f} | "
                  f"tox {100 * f_tox:4.1f}% | |g| {g_norm:8.1f} | "
                  f"clip {100 * f_clip:5.1f}% | floor {100 * f_floor:5.2f}% | "
                  f"lr {lr_now:.2e} | {hours:.2f}h", flush=True)

            if f_clip > 0.5:
                print("  [WARN] over half the gradient elements are being clipped, so "
                      "only the gradient SIGN matters — which is how an unbounded "
                      "branch takes over training.")
            if f_floor > 0.10:
                print("  [WARN] more than 10% of per-action log-probabilities are "
                      "below ln(1e-10): actions are being driven to zero probability.")

            for k in win:
                win[k] = 0.0
            del loss, inputs_cpu
            gc.collect()

        if loader_val is not None and global_counter % VAL_EVERY == 0:
            v_nll, v_non, v_tox, v_n = evaluate(model, loader_val)
            hours = (time.time() - t0) / 3600.0
            with open(VAL_LOG_FILE, 'a') as f:
                f.write(f"{global_counter}\t{hours:.4f}\t{v_nll:.4f}\t{v_non:.4f}"
                        f"\t{v_tox:.4f}\t{v_n}\n")
            print(f"  [val] iter {global_counter:,} | nll {v_nll:.3f} | "
                  f"l_nontox {v_non:.2f} | l_tox {v_tox:.2f} | {v_n} molecules",
                  flush=True)

        if _stop['requested']:
            save_checkpoint(global_counter, (time.time() - t0) / 3600.0)
            with open(LOG_FILE, 'a') as f:
                f.write(f"# interrupted by {_stop['name']} at {global_counter}\n")
            print(f"[SIGNAL] checkpointed at iteration {global_counter}. Re-run to "
                  f"continue.")
            break

    if not _stop['requested']:
        save_checkpoint(global_counter, (time.time() - t0) / 3600.0)
        with open(LOG_FILE, 'a') as f:
            f.write('# finished\n')
        print(f"\n  reached {global_counter:,} iterations")

# %% [markdown]
# ## Summary

# %%
print("=" * 70)
_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
for status, name, detail in sorted(CHECKS, key=lambda c: _order.get(c[0], 9)):
    print(f"  {status:5s}  {name}: {detail}")

print("-" * 70)
print(f"  objective  {OBJECTIVE_NAME}")
print(f"  artifacts  {CKPT_PARAMS_FILE}")
print("  Next: 00_phase0_diagnostics check A measures unconditional validity on")
print("  this checkpoint. Do that before committing to fine-tuning.")

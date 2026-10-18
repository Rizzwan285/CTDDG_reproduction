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
# # Stage 1 - unconditional pretraining (PyTorch)
#
# Learns to generate chemically valid molecules from ChEMBL, with no protein.
# Produces `checkpoints/ckpt.pt`, which `02_finetune_ctddg.py` fine-tunes.
#
# ## What changed, and why
#
# | # | Change | Reason |
# |---|---|---|
# | 1 | **`TOX_WEIGHTING = False` by default** - stage 1 trains CDGCN's plain NLL | See below. This is the strategic change. |
# | 2 | `BATCH_SIZE` 16 -> **8** | The paper's value in both phases; 16 broke comparability with Tables 1-3. |
# | 3 | **Bond cap in the collator** | `graph_to_rnn` is `[batch, K, 120]`; a molecule needing more than 120 steps raises a broadcast error inside a DataLoader worker and kills the run. |
# | 4 | **Salt stripping + a connectivity assertion** | `traverse_graph` leaves the second component of a disconnected graph unvisited at `step_ids = -1`, and `single_reorder` then mis-wires every bond. Silent today; now impossible. |
# | 5 | **Dataset hash verified against its manifest** | Stage 1 refuses to start unless `DATA_PATH` is the file `random_forest_classifier.py` built and hashed. |
# | 6 | **Gradient norm, clip fraction and log-probability floor logged** | These three columns would have made the failure obvious at iteration 5,000 instead of at evaluation. |
# | 7 | **Resume reads `trainer.pt`, not the last line of `log.out`** | The log carries rows from earlier `SUMMARY_STEP` eras; parsing it shifted the counter. |
#
# ## Why stage 1 no longer runs Eq. 8
#
# The published repository's `pretraining.ipynb` **never implemented Eq. 8** - its
# `_likelihood` accepts `tox_class_batch` and ignores it, so upstream stage 1 is
# plain CDGCN NLL. Whatever produced the paper's 94.4% validity was therefore
# *CDGCN-objective stage 1 + Eq. 8 at stage 2 only*.
#
# The previous run here applied Eq. 8 for all 480,000 pretraining iterations, and
# the mechanism is destructive. For a toxic molecule the objective is `min l_i`,
# unbounded below; `_log_mask` floors each action's log-probability at
# `ln(1e-10) = -23.026`, and the logged `l_tox` reached **-637**, i.e. ~28 of ~34
# actions pinned at that floor. The gradient there is `1/(p + 1e-10)` ~ `1e10`,
# so with ~24% of each batch toxic the summed gradient's **sign** is set entirely
# by the toxic branch, and `clip_grad_value_(3.0)` then rewrites its magnitude to
# exactly +/-3.0. Every parameter takes a maximal step in the direction "make
# toxic molecules impossible", and the non-toxic signal is numerically
# annihilated. Stage-1 `l_nontoxic` ended between -30 and -135 against CDGCN's
# -15.99 on the same data: roughly -1.5 nats per action instead of -0.5, i.e.
# p ~ 0.21 per action instead of 0.61. A policy that diffuse produces about one
# valid molecule in four, which is what was measured.
#
# Set `TOX_WEIGHTING = True` only to reproduce that failure deliberately.
#
# > **Cheaper option.** A completed 480k CDGCN-objective checkpoint already exists
# > at `../CTDDG_new/outputs/pretrain/` (loss 15.99, identical architecture,
# > identical 6,986,506 parameters). Converting those MXNet weights costs no GPU
# > time. Run `00_phase0_diagnostics.py` check A first: if it returns ~90%
# > unconditional validity, stage 1 is already fine and neither option is needed.

# %%
import os
import gc
import time
import json
import math
import random
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
FORCE_RESTART = False

GPU_ID = int(os.environ.get("CTDDG_GPU", "0"))
CTX = torch.device(f"cuda:{GPU_ID}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Active compute context: {CTX}")

REPO_ROOT = os.environ.get("CTDDG_REPO_ROOT", "/home/muhamed/repo/CTDDG")

# The dataset random_forest_classifier.py builds: filtered, deduplicated, labelled,
# with a manifest carrying its SHA-256.
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

# Verify DATA_PATH against the manifest written beside it. Turn off only for a
# deliberate experiment on another file.
VERIFY_DATA_HASH = True
# Strict: refuse to start on a hash mismatch. False downgrades it to a warning.
REQUIRE_DATA_HASH_MATCH = True

SMILES_COL = None        # None = auto-detect; set explicitly to skip detection
TOX_COL = None
LIMIT_MOLECULES = None   # e.g. 50_000 for a smoke test; None = everything

# %% [markdown]
# ### The objective

# %%
# False  -> CDGCN's Eq. 6: weight 1 on EVERY molecule. The upstream stage-1
#           objective, and what the paper's numbers were produced with.
# True   -> CTDDG's Eq. 8 in stage 1 as well. Reproduces the collapse documented
#           above. Only set this for a deliberate ablation.
#
# NOTE: `LAMBDA_TOX = 0` does NOT give the CDGCN objective. The weight
# `1 - (1+0)*tox` is 0 on toxic molecules, which DROPS them from the batch average
# rather than training on them normally. That is a different experiment - a useful
# control, but not the predecessor model.
TOX_WEIGHTING = False
LAMBDA_TOX = 1.0

# Review item F13. A float applies `l = max(l, TOX_FLOOR)` to toxic molecules in
# the OBJECTIVE only; reported metrics stay unfloored. Irrelevant while
# TOX_WEIGHTING is False, since the toxic branch carries weight +1 then.
#
# Do not pick a value by eye. A floor only binds while toxic `l` is above it, so
# it must be set from iteration 0, and choosing it needs the toxic-`l`
# distribution at step 0 - which 02_finetune_ctddg.py now measures before its
# first optimizer step.
TOX_FLOOR = None

# %% [markdown]
# ### Architecture and schedule

# %%
BATCH_SIZE = 8           # the paper's value (was 16, which broke comparability)
K_PATHS = 5              # K: importance-sampling paths per molecule
P_ALPHA = 0.8            # alpha: coefficient of randomness in Q_alpha

F_E = 16                 # atom embedding width
F_H = [32, 64, 128, 128, 256, 256]   # the 6 GCN layer widths
F_SKIP = 256
F_C = [512]
FH_POLICY = 128
ACTIVATION = 'relu'
N_RNN = 3
D_RECEPTIVE = 2
MAX_ITER = 120           # cap on generation steps; also graph_to_rnn's width

MAX_ATOMS = 100          # belt-and-braces; the dataset is already filtered

LR = 1e-3
DECAY = 1e-3             # lr *= (1 - DECAY) every DECAY_STEP iterations
DECAY_STEP = 100
CLIP_GRAD = 3.0          # element-wise, matching MXNet's optimizer clip_gradient
SUMMARY_STEP = 500
CHECKPOINT_STEP = 500
TOTAL_ITERATIONS = 480000

NUM_WORKERS = 8
SEED = 42
ALLOW_TF32 = True

# Fail fast if the loss or gradients go non-finite, instead of silently poisoning
# every parameter. The previous version had no such guard.
DIE_ON_NONFINITE = True

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


OBJECTIVE_NAME = "CTDDG Eq.8" if TOX_WEIGHTING else "CDGCN Eq.6 (unweighted NLL)"
print(f"\nObjective: {OBJECTIVE_NAME}")
if TOX_WEIGHTING:
    print("  [WARN] Eq. 8 in stage 1 is what collapsed validity to 23.9% last time.")
    print("         The published stage-1 code does not do this. See the header.")

# %% [markdown]
# ## Preflight

# %%
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32

check("data file", os.path.exists(DATA_PATH), DATA_PATH)
check("atom types", os.path.exists(ATOM_TYPES_PATH), ATOM_TYPES_PATH)

_t = torch.zeros((1,), device=CTX)
_ = _t.cpu().numpy()
check("device responsive", True,
      f"{CTX}" + (f" ({torch.cuda.get_device_name(GPU_ID)})"
                  if CTX.type == "cuda" else ""))

check("batch size matches the paper", BATCH_SIZE == 8,
      f"BATCH_SIZE = {BATCH_SIZE} (paper: 8)", warn_only=True)

# %% [markdown]
# ### Dataset provenance
#
# The single most consequential thing to get right, and the cheapest to verify.
# `random_forest_classifier.py` writes a manifest beside the dataset recording its
# SHA-256, the filters applied and the labelling model's own hash.

# %%
DATA_MANIFEST = None
_man_path = DATA_PATH + ".manifest.json"

if not VERIFY_DATA_HASH:
    check("dataset hash", True, "verification disabled", warn_only=True)
elif not os.path.isfile(_man_path):
    check("dataset manifest", False,
          f"no manifest at {_man_path}. Build the dataset with "
          f"random_forest_classifier.py, or set VERIFY_DATA_HASH = False to train "
          f"on an unverified file",
          warn_only=not REQUIRE_DATA_HASH_MATCH)
else:
    with open(_man_path) as f:
        DATA_MANIFEST = json.load(f)
    actual = sha256_file(DATA_PATH)
    expected = DATA_MANIFEST.get("sha256")
    check("dataset hash", actual == expected,
          f"{actual[:16]}... "
          + ("matches the manifest" if actual == expected
             else f"!= manifest {str(expected)[:16]}... - the file changed since it "
                  f"was built"),
          warn_only=not REQUIRE_DATA_HASH_MATCH)

    _f = DATA_MANIFEST.get("filters", {})
    print(f"       {DATA_MANIFEST.get('n_molecules'):,} molecules, "
          f"{100 * DATA_MANIFEST.get('frac_toxic', 0):.1f}% toxic")
    print(f"       salts stripped: {_f.get('salts_stripped_to_largest_fragment')}  |  "
          f"max atoms: {_f.get('max_atoms')}  |  "
          f"max steps: {_f.get('max_generation_steps')}")
    print(f"       BindingDB ligands removed: {_f.get('bindingdb_ligands_removed')}")

    check("BindingDB ligands excluded",
          bool(_f.get("bindingdb_ligands_removed")),
          "the manifest confirms the exclusion both papers specify"
          if _f.get("bindingdb_ligands_removed")
          else "the manifest does NOT record the exclusion - this is the leakage "
               "that makes Tables 1-3 incomparable",
          warn_only=True)
    check("bond cap applied upstream",
          _f.get("max_generation_steps") == MAX_ITER,
          f"dataset built with max_generation_steps = {_f.get('max_generation_steps')}, "
          f"this run uses MAX_ITER = {MAX_ITER}", warn_only=True)

# %% [markdown]
# ## Molecule spec and graph traversal
#
# Pure NumPy / NetworkX / RDKit. Identical to the fine-tuning notebook - both
# phases must build graphs the same way or the transferred weights mean nothing.

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


_mol_spec = None


def get_mol_spec():
    global _mol_spec
    if _mol_spec is None:
        _mol_spec = MoleculeSpec(ATOM_TYPES_PATH)
    return _mol_spec


def strip_salt(smiles):
    """Largest fragment by string length - the rule data_preprocessing_1 used.

    A disconnected graph silently corrupts its own training target (see
    traverse_graph below), so this is a correctness fix, not a tidiness one.
    """
    if '.' not in smiles:
        return smiles
    return max(smiles.split('.'), key=len)


def get_graph_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    atom_types, atom_ranks, bonds, bond_types = [], [], [], []
    for a, r in zip(mol.GetAtoms(), Chem.CanonicalRankAtoms(mol)):
        atom_types.append(get_mol_spec().get_atom_type(a))
        atom_ranks.append(r)

    for b in mol.GetBonds():
        idx_1, idx_2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = get_mol_spec().get_bond_type(b)
        bonds.append([idx_1, idx_2])
        bond_types.append(bt)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(atom_types)))
    graph.add_edges_from(bonds)

    # A disconnected graph cannot be traversed by the DFS below; it would leave
    # the second component at step_ids = -1 and mis-wire every bond in
    # single_reorder. Refuse it here rather than train on a corrupted target.
    if len(atom_types) > 1 and not nx.is_connected(graph):
        raise ValueError(f"Disconnected molecular graph (salt not stripped?): {smiles}")

    return graph, atom_types, atom_ranks, bonds, bond_types


def traverse_graph(graph, atom_ranks, current_node=None, step_ids=None,
                   p=P_ALPHA, log_p=0.0, _top=True):
    """Q_alpha proposal: canonical rank with probability p, random neighbour otherwise."""
    if current_node is None:
        next_nodes = range(len(atom_ranks))
        step_ids = [-1] * len(next_nodes)
        next_node_ranks = atom_ranks
    else:
        next_nodes = list(graph.neighbors(current_node))
        next_nodes = [n for n in next_nodes if step_ids[n] < 0]
        next_node_ranks = [atom_ranks[n] for n in next_nodes]

    next_nodes = [n for n, r in sorted(zip(next_nodes, next_node_ranks),
                                       key=lambda x: x[1])]

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
        next_nodes = [n for n in graph.neighbors(next_node) if step_ids[n] < 0]
        next_node_ranks = [atom_ranks[n] for n in next_nodes]
        next_nodes = [n for n, r in sorted(zip(next_nodes, next_node_ranks),
                                           key=lambda x: x[1])]

    # Unreachable now that get_graph_from_smiles rejects disconnected graphs, which
    # is exactly why it is worth asserting: if it ever fires, something upstream
    # changed and the alternative is silent corruption.
    if _top and min(step_ids) < 0:
        raise ValueError(
            f"traverse_graph left {sum(1 for s in step_ids if s < 0)} atoms unvisited. "
            f"The graph is disconnected; single_reorder would mis-wire every bond.")

    return step_ids, log_p


# %%
def single_reorder(X_0, A_0, step_ids):
    X_0, A_0 = np.copy(X_0), np.copy(A_0)
    step_ids = np.array(step_ids, dtype=np.int32)
    sorted_ids = np.argsort(step_ids)
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
# ## Data ingestion
#
# Column detection is strict: the original one-liner
# `next(c for c in df.columns if 'label' in c.lower())` matched ChEMBL's
# **`BAO Label`** column before ever reaching `Ames_prediction`.

# %%
_PREFERRED_SMILES = ["Smiles", "smiles", "SMILES", "canonical_smiles",
                     "Canonical_Smiles", "CanonicalSmiles"]
_PREFERRED_TOX = ["Ames_prediction", "Ames_Prediction", "ames_prediction",
                  "Toxic_class", "tox_class", "Toxicity", "toxicity"]


def _resolve_columns(columns):
    cols = list(columns)
    smi = SMILES_COL
    if smi is None:
        for cand in _PREFERRED_SMILES:
            if cand in cols:
                smi = cand
                break
    if smi is None:
        smi = next((c for c in cols if 'smiles' in c.lower()), cols[0])

    tox = TOX_COL
    if tox is None:
        for cand in _PREFERRED_TOX:
            if cand in cols:
                tox = cand
                break
    if tox is None:
        # Narrow fallback: 'ames'/'toxic' only, never bare 'label' or 'class'.
        tox = next((c for c in cols
                    if ('ames' in c.lower() or 'toxic' in c.lower()) and c != smi), None)
    if tox is None:
        raise ValueError(f"No toxicity column among {cols[:40]}. Set TOX_COL explicitly.")
    if smi not in cols or tox not in cols:
        raise ValueError(f"Resolved columns absent. smiles={smi!r} tox={tox!r}")
    return smi, tox


def load_chembl_dataset(file_path):
    """-> [(smiles, tox_label)]. Salts stripped; unparseable and oversized dropped."""
    header = pd.read_csv(file_path, nrows=0)
    smiles_col, tox_col = _resolve_columns(header.columns)
    print(f"  columns: SMILES={smiles_col!r}  toxicity={tox_col!r}")

    # usecols keeps the other columns (including any huge Fingerprint field) out of RAM.
    df = pd.read_csv(file_path, usecols=[smiles_col, tox_col])
    n_before = len(df)
    df = df.dropna(subset=[smiles_col, tox_col])
    if len(df) != n_before:
        print(f"  dropped {n_before - len(df)} rows with a missing SMILES or label")

    smiles_list = df[smiles_col].astype(str).tolist()
    tox_list = df[tox_col].astype(float).astype(int).tolist()

    records = list(zip(smiles_list, tox_list))
    if LIMIT_MOLECULES is not None:
        records = records[:LIMIT_MOLECULES]
        print(f"  [WARN] LIMIT_MOLECULES set: using only {len(records)} molecules")
    return records


class BalancedSampler(Sampler):
    """Groups similar-length molecules into a batch to cut padding in the GCN/RNN.

    Upstream behaviour kept: the records beyond the last whole batch are never
    sampled. The guard is new - with fewer records than batch_size the sampler
    silently yields nothing.
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
                  f"are not sampled (upstream behaviour)")
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
# Three filters, applied per record before anything expensive happens:
# out-of-vocabulary atoms, oversized molecules, and - new - molecules needing more
# than `MAX_ITER` generation steps. The last one is blocker B6: `graph_to_rnn` is
# `[batch, K, 120]`, and `graph_to_rnn[i, k, :num_iter] = arange(num_iter) + cum`
# raises a broadcast `ValueError` inside a worker when `num_iter > 120`, killing
# the whole run hours in.

# %%
def process_single(record, k=K_PATHS, p=P_ALPHA):
    """Build the graph, sample K traversals, attach the molecule's Ames label."""
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


class MolRNNCollate:
    """Standalone and picklable, so it survives DataLoader worker start-up."""

    def __init__(self, k=K_PATHS, p=P_ALPHA, max_iter=MAX_ITER, max_atoms=MAX_ATOMS):
        self.k = k
        self.p = p
        self.max_iter = max_iter
        self.max_atoms = max_atoms

    def _admissible(self, smiles):
        """-> (ok, reason). Everything that would break the collator, checked once."""
        smiles = strip_salt(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "unparseable"
        if mol.GetNumAtoms() > self.max_atoms:
            return False, "too_many_atoms"
        # Blocker B6: a molecule needs GetNumBonds() + 1 steps, and graph_to_rnn
        # is only max_iter wide.
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
            ok, _ = self._admissible(smi)
            if ok:
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
            # _admissible guarantees this; assert rather than truncate silently.
            assert num_iter <= self.max_iter, (
                f"{smiles_batch[i]} needs {num_iter} steps > max_iter {self.max_iter}")
            for k_idx in range(self.k):
                graph_to_rnn[i, k_idx, :num_iter] = np.arange(num_iter) + cum_sum
                r0 = np.ones([num_iter], dtype=np.int32) * i
                r1 = np.ones_like(r0) * k_idx
                r2 = np.arange(num_iter)
                rnn_to_graph.append(np.stack([r0, r1, r2], axis=0))
                cum_sum += num_iter

        rnn_to_graph = np.concatenate(rnn_to_graph, axis=1)
        NX_cum = np.cumsum(result_out[6])
        return result_out + [graph_to_rnn, rnn_to_graph, NX_cum]


def batch_to_tensor(record, device=None):
    """Move one collated batch onto the compute device."""
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
            A_i = np.concatenate([A_i, A_i[:, [1, 0]]], axis=0)     # symmetrise
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
# ## Build the loader and smoke-test one batch

# %%
t0 = time.time()
dataset = load_chembl_dataset(DATA_PATH)
print(f"  {len(dataset):,} molecules loaded  [{time.time() - t0:.0f}s]")

labels = [rec[1] for rec in dataset]
num_toxic = sum(labels)
frac_toxic = num_toxic / max(1, len(labels))
print(f"  non-toxic {len(labels) - num_toxic:,}  |  toxic {num_toxic:,} "
      f"({100 * frac_toxic:.1f}%)")

check("dataset size", len(dataset) > 1000, f"{len(dataset):,} molecules")
check("labels are binary", set(labels) <= {0, 1}, f"label values {sorted(set(labels))}")
if DATA_MANIFEST:
    check("row count matches the manifest",
          len(dataset) == DATA_MANIFEST.get("n_molecules"),
          f"{len(dataset):,} loaded vs {DATA_MANIFEST.get('n_molecules'):,} in the "
          f"manifest", warn_only=True)

# %%
sampler_train = BalancedSampler(cost=[len(rec[0]) for rec in dataset],
                                batch_size=BATCH_SIZE, name="chembl")
collate_fn = MolRNNCollate(k=K_PATHS, p=P_ALPHA)
loader_train = DataLoader(dataset, batch_sampler=sampler_train,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn,
                          persistent_workers=(NUM_WORKERS > 0))

batch_cpu = next(iter(loader_train))
check("collator produced a batch", batch_cpu is not None,
      "first batch is not None")

batch_gpu = batch_to_tensor(batch_cpu)
(_X, _A, _iw, _mask, _NX, _NXr, _a0, _acts, _lp, _bs, _iws, _tox,
 _g2r, _r2g, _nxc) = batch_gpu

print(f"  batch N={_bs}  K={_iws}  |  X {tuple(_X.shape)}  |  "
      f"adjacency blocks {len(_A)}")
check("batch shape", _bs == BATCH_SIZE and _iws == K_PATHS,
      f"N={_bs} (expected {BATCH_SIZE}), K={_iws} (expected {K_PATHS})")
check("adjacency blocks", len(_A) == get_mol_spec().num_bond_types + D_RECEPTIVE,
      f"{len(_A)} = N_B {get_mol_spec().num_bond_types} + D {D_RECEPTIVE}")
check("tox_class alignment", _tox.numel() == _bs * _iws,
      f"{_tox.numel()} entries = batch {_bs} x K {_iws}")

# %% [markdown]
# ### Collator rejection census
#
# How many molecules the three filters drop. If `too_many_bonds` is non-zero, the
# previous version of this notebook would have crashed on exactly those.

# %%
_probe = random.Random(SEED).sample(range(len(dataset)), min(20000, len(dataset)))
_reasons = {}
for i in _probe:
    smi = dataset[i][0]
    ok, why = collate_fn._admissible(smi)
    if not ok:
        _reasons[why] = _reasons.get(why, 0) + 1

print(f"  probed {len(_probe):,} molecules")
for k, v in sorted(_reasons.items(), key=lambda kv: -kv[1]):
    print(f"    {k:18s} {v:>6,}  ({100 * v / len(_probe):.2f}%)")
if not _reasons:
    print("    (nothing rejected)")

_drop_rate = sum(_reasons.values()) / len(_probe)
check("collator rejection rate", _drop_rate < 0.02,
      f"{100 * _drop_rate:.2f}% of molecules rejected. "
      f"too_many_bonds = {_reasons.get('too_many_bonds', 0)} "
      f"(these would have crashed the previous collator)",
      warn_only=True)

# %% [markdown]
# ## Model
#
# Architecture untouched. `EfficientGraphConvFn` recomputes the concatenated
# neighbourhood in backward instead of storing it; every `A` is symmetric, so
# `A^T == A` and backward reuses it directly.

# %%
def segment_sum(X, idx, num_seg):
    """out[idx[i]] += X[i]. Replaces MXNet's SegmentSumFn sparse-matmul trick."""
    out = torch.zeros((num_seg,) + tuple(X.shape[1:]), dtype=X.dtype, device=X.device)
    return out.index_add(0, idx, X)


def _concat_neighbourhoods(X, A_list):
    parts = [X]
    for A in A_list:
        parts.append(torch.sparse.mm(A, X) if A is not None else torch.zeros_like(X))
    return torch.cat(parts, dim=1)


class EfficientGraphConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, A_list):
        ctx.A_list = A_list
        ctx.save_for_backward(X, W)
        return _concat_neighbourhoods(X, A_list) @ W

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        A_list = ctx.A_list
        X_out = _concat_neighbourhoods(X, A_list)
        grad_output = grad_output.contiguous()

        grad_W = X_out.t() @ grad_output
        grad_X_out = grad_output @ W.t()

        chunks = torch.split(grad_X_out, X.shape[1], dim=1)
        grad_X = chunks[0]
        for A, g in zip(A_list, chunks[1:]):
            if A is not None:
                grad_X = grad_X + torch.sparse.mm(A, g.contiguous())
        return grad_X, grad_W, None


def get_activation(name):
    return {'relu': torch.relu, 'tanh': torch.tanh}[name]


class Linear_BN(nn.Sequential):
    """Dense(no bias) -> BatchNorm. MXNet BatchNorm(momentum=0.9) == torch(0.1)."""

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
        return EfficientGraphConvFn.apply(X, self.W, A_list)


class Policy(nn.Module):
    """Joint softmax over every action of every atom, plus `end`, per molecule.

    The +/-80 clamp before exp() is upstream behaviour, kept. Check D of
    00_phase0_diagnostics.py measures whether it ever binds; it records the raw
    pre-clamp range here so a long run can be audited without re-sampling.
    """

    def __init__(self, F_in, F_h, N_A, N_B, k=1):
        super(Policy, self).__init__()
        self.F_in, self.F_h, self.N_A, self.N_B, self.k = F_in, F_h, N_A, N_B, k
        self.linear_h = Linear_BN(F_in * 2, self.F_h * k)
        self.linear_h_t = Linear_BN(F_in, self.F_h * k)
        self.linear_x = nn.Linear(self.F_h, self.N_B + self.N_B * self.N_A)
        self.linear_x_t = nn.Linear(self.F_h, 1)
        self.linear_pi = nn.Linear(self.F_in, self.k) if self.k > 1 else None
        self.last_logit_absmax = None
        self.last_frac_clamped = None

    def forward(self, X, NX, NX_rep, X_end=None):
        if X_end is None:
            X_end = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X, X_end[NX_rep, :]], dim=1)

        X_h = torch.relu(self.linear_h(X)).reshape(-1, self.F_h)
        X_h_end = torch.relu(self.linear_h_t(X_end)).reshape(-1, self.F_h)

        raw_x = self.linear_x(X_h).reshape(-1, self.k, self.N_B + self.N_B * self.N_A)
        raw_x_t = self.linear_x_t(X_h_end).reshape(-1, self.k, 1)

        with torch.no_grad():
            self.last_logit_absmax = float(max(raw_x.abs().max(), raw_x_t.abs().max()))
            n_at = int((raw_x.abs() >= 80.0).sum()) + int((raw_x_t.abs() >= 80.0).sum())
            self.last_frac_clamped = n_at / max(1, raw_x.numel() + raw_x_t.numel())

        X_x = torch.exp(torch.clamp(raw_x, -80.0, 80.0))
        X_x_end = torch.exp(torch.clamp(raw_x_t, -80.0, 80.0))

        X_sum = segment_sum(X_x, NX_rep, NX.shape[0]).sum(-1, keepdim=True) + X_x_end
        X_softmax = X_x / (X_sum[NX_rep, :, :] + 1e-10)
        X_softmax_end = X_x_end / (X_sum + 1e-10)

        if self.k > 1:
            pi = torch.softmax(self.linear_pi(X_end), dim=1).unsqueeze(-1)
            X_softmax = (X_softmax * pi[NX_rep, :, :]).sum(dim=1)
            X_softmax_end = (X_softmax_end * pi).sum(dim=1)
        else:
            X_softmax = X_softmax.squeeze(1)
            X_softmax_end = X_softmax_end.squeeze(1)

        connect, append = X_softmax[:, :self.N_B], X_softmax[:, self.N_B:]
        append = append.reshape(-1, self.N_A, self.N_B)
        return append, connect, X_softmax_end.squeeze(-1)


# %% [markdown]
# ## Generator and the objective

# %%
LOG_EPS = 1e-10
LOG_FLOOR = math.log(LOG_EPS)       # -23.0259: the per-action log-probability floor


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

        # Diagnostics, filled every forward. All detached.
        self.last_l = None                 # the UNFLOORED per-molecule bound
        self.last_tox = None               # the REAL labels, even when not weighting
        self.last_frac_floor = None        # fraction of actions at the log floor
        self.last_floored = None

    @abstractmethod
    def _build_graph_conv(self, *args, **kwargs):
        pass

    @abstractmethod
    def _graph_conv_forward(self, X, A):
        pass

    def _policy_0(self):
        return torch.softmax(self.policy_0, dim=0)

    def _likelihood(self, init, append, connect, end, action_0, actions, iw_ids,
                    log_p_sigma, batch_size, iw_size, tox_class_batch):
        action_type, node_type, edge_type, append_pos, connect_pos = (
            actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3], actions[:, 4])

        def _log_mask(_x, _mask):
            return _mask * torch.log(_x + LOG_EPS) + (1 - _mask) * torch.zeros_like(_x)

        init = init.reshape(batch_size * iw_size, self.N_A)
        rows = torch.arange(action_0.shape[0], device=action_0.device)
        loss_init = torch.log(init[rows, action_0] + LOG_EPS)

        loss_end = _log_mask(end, (action_type == 2).to(end.dtype))
        loss_append = _log_mask(append[append_pos, node_type, edge_type],
                                (action_type == 0).to(end.dtype))
        loss_connect = _log_mask(connect[connect_pos, edge_type],
                                 (action_type == 1).to(end.dtype))

        log_p_x = loss_end + loss_append + loss_connect

        # How many per-action log-probabilities are pinned at ln(1e-10)? Exactly one
        # of the three terms is active per row, so this is well defined. A rising
        # number means the objective is driving actions to numerically zero
        # probability, where the gradient is 1/(p + 1e-10) ~ 1e10 - which is how the
        # previous run's toxic branch swamped everything else.
        with torch.no_grad():
            self.last_frac_floor = float((log_p_x <= LOG_FLOOR + 1e-3).float().mean())

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

        # The weight. With TOX_WEIGHTING False the effective label is zero for every
        # molecule, giving weight 1 everywhere - CDGCN's Eq. 6.
        #
        # This is NOT the same as LAMBDA_TOX = 0, which gives weight 0 on toxic
        # molecules and drops them from the batch average.
        eff_tox = real_tox if TOX_WEIGHTING else torch.zeros_like(real_tox)

        l_raw = l
        if TOX_WEIGHTING and TOX_FLOOR is not None:
            l = torch.where(real_tox > 0, torch.clamp(l, min=TOX_FLOOR), l)
            self.last_floored = ((real_tox > 0) & (l_raw < TOX_FLOOR)).detach()
        else:
            self.last_floored = torch.zeros_like(real_tox, dtype=torch.bool)

        l_weighted = l * (1.0 - (1.0 + LAMBDA_TOX) * eff_tox)

        # Always the unfloored bound and the REAL labels, so the l_nontoxic /
        # l_toxic columns stay meaningful in both objectives: under CDGCN you can
        # still watch the toxic branch without training on it.
        self.last_l = l_raw.detach()
        self.last_tox = real_tox.detach()
        return l_weighted

    def _policy(self, X, A, NX, NX_rep, last_append_mask):
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        X = self._graph_conv_forward(X, A)
        X = self.dense(X)
        return self.policy_h(X, NX, NX_rep)

    def forward(self, X, A, iw_ids, last_append_mask, NX, NX_rep, action_0, actions,
                log_p, batch_size, iw_size, tox_class, *rnn_args):
        init = self._policy_0().unsqueeze(0).expand(batch_size * iw_size, -1)
        append, connect, end = self._policy(X, A, NX, NX_rep, last_append_mask,
                                            *rnn_args)
        l_weighted = self._likelihood(init, append, connect, end, action_0, actions,
                                      iw_ids, log_p, batch_size, iw_size, tox_class)
        if self.mode == 'likelihood':
            return l_weighted
        return -l_weighted.mean()


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
        """Unused in training; present so the diagnostic sampler can drive this model."""
        X_avg = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X_avg, X[NX_cum - 1]], dim=1)
        X, h = self.rnn(X.unsqueeze(1), h)
        return X.squeeze(1), h

    def _policy(self, X, A, NX, NX_rep, last_append_mask, graph_to_rnn, rnn_to_graph,
                NX_cum):
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        X = self._graph_conv_forward(X, A)
        # Computed once, unlike the MXNet original which ran it twice and so
        # updated the BatchNorm running statistics twice per step.
        X_dense = self.dense(X)
        X_mol = self._rnn_train(X_dense, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum)
        return self.policy_h(X_dense, NX, NX_rep, X_mol)


class VanillaMolGen_RNN(MoleculeGenerator_RNN):
    def __init__(self, N_A, N_B, D, F_e, F_h, F_skip, F_c, Fh_policy, activation, N_rnn):
        super(VanillaMolGen_RNN, self).__init__(
            N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, F_h)

    def _build_graph_conv(self, F_h):
        self.F_h = list(F_h)
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()          # bn[i-1] belongs to conv[i]; conv[0] has none
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
# ## Initialisation
#
# MXNet's `mx.init.Xavier()` default is uniform / factor_type='avg' / magnitude=3,
# giving bounds `+/-sqrt(6/(fan_in+fan_out))` - algebraically identical to
# `xavier_uniform_(w, gain=1)`. Biases, `policy_0` and BatchNorm parameters keep
# their own MXNet initialisers (zeros / ones).

# %%
def init_weights_xavier(module):
    for name, p in module.named_parameters():
        if p.dim() < 2:
            continue                                  # biases, BN gamma/beta, policy_0
        if 'bn' in name.split('.')[-1]:
            continue
        nn.init.xavier_uniform_(p, gain=1.0)

    for name, p in module.named_parameters():
        if name.endswith('bias') or ('bias' in name.split('.')[-1]):
            nn.init.zeros_(p)

    for m in module.modules():
        if isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.reset_running_stats()

    if hasattr(module, 'policy_0'):
        nn.init.zeros_(module.policy_0)
    return module


# %% [markdown]
# ## Objective and data signatures
#
# Resuming a run under a different objective, batch size or dataset silently
# produces a model that is neither one thing nor the other. These signatures are
# written into the checkpoint and compared on resume.

# %%
def objective_signature():
    return {"tox_weighting": TOX_WEIGHTING, "lambda_tox": LAMBDA_TOX,
            "tox_floor": TOX_FLOOR, "objective": OBJECTIVE_NAME}


def data_signature():
    return {"data_path": DATA_PATH,
            "data_sha256": (DATA_MANIFEST or {}).get("sha256"),
            "n_molecules": len(dataset),
            "batch_size": BATCH_SIZE, "k_paths": K_PATHS, "p_alpha": P_ALPHA,
            "max_atoms": MAX_ATOMS, "max_iter": MAX_ITER,
            "limit_molecules": LIMIT_MOLECULES}


# %% [markdown]
# ## Build, resume

# %%
configs = {'F_e': F_E, 'F_h': F_H, 'F_skip': F_SKIP, 'F_c': F_C,
           'Fh_policy': FH_POLICY, 'activation': ACTIVATION, 'N_rnn': N_RNN}

is_continuous = False
global_counter = 0
elapsed_offset = 0.0
resume_state = None

_have = all(os.path.isfile(f) for f in (CKPT_PARAMS_FILE, CKPT_TRAINER_FILE))

if FORCE_RESTART:
    print("  FORCE_RESTART set - archiving the previous run")
    stamp = time.strftime('%Y%m%d-%H%M%S')
    for p in (CKPT_PARAMS_FILE, CKPT_TRAINER_FILE, LOG_FILE):
        if os.path.exists(p):
            os.replace(p, f"{p}.stale.{stamp}")
            print(f"    {os.path.basename(p)} -> .stale.{stamp}")
    with open(CKPT_CONFIG_FILE, 'w') as f:
        json.dump(configs, f, indent=1)
elif _have:
    is_continuous = True
    print("  existing checkpoint found - resuming")
    if os.path.isfile(CKPT_CONFIG_FILE):
        with open(CKPT_CONFIG_FILE) as f:
            _saved = json.load(f)
        configs = {k: v for k, v in _saved.items()
                   if k in ('F_e', 'F_h', 'F_skip', 'F_c', 'Fh_policy',
                            'activation', 'N_rnn')}
else:
    print("  fresh run")
    with open(CKPT_CONFIG_FILE, 'w') as f:
        json.dump(configs, f, indent=1)

model = VanillaMolGen_RNN(get_mol_spec().num_atom_types,
                          get_mol_spec().num_bond_types,
                          D=D_RECEPTIVE, **configs)

if is_continuous:
    state = torch.load(CKPT_PARAMS_FILE, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.to(CTX)
    resume_state = state
else:
    init_weights_xavier(model)
    model.to(CTX)

# MXNet's mx.optimizer.Adam defaults (beta1=0.9, beta2=0.999, eps=1e-8) match
# torch's. `wd` was 0; DECAY above is a learning-rate schedule, not weight decay.
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

    # The counter comes from trainer.pt, not the last line of log.out. The log
    # carries rows from earlier SUMMARY_STEP eras, so parsing it shifted the
    # counter on every resume.
    global_counter = int(tstate.get('global_counter', 0))
    elapsed_offset = float(tstate.get('elapsed_hours', 0.0))

    # Cross-check against the BatchNorm counter, which is exact: the only
    # train-mode forward is the one per training iteration.
    _nbt = [int(v) for k, v in resume_state['model'].items()
            if k.endswith('num_batches_tracked')]
    bn_steps = max(_nbt) if _nbt else None
    print(f"  trainer counter {global_counter:,}  |  BN steps {bn_steps}")
    check("resume counters agree", bn_steps == global_counter,
          f"trainer.pt says {global_counter:,}, BatchNorm says {bn_steps}. "
          f"The BatchNorm count is authoritative", warn_only=True)

    old_obj = tstate.get('objective')
    new_obj = objective_signature()
    if old_obj is None:
        print("  [WARN] checkpoint predates objective tracking")
    else:
        changed = {k: (old_obj[k], v) for k, v in new_obj.items()
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
        if 'data_sha256' in d:
            raise RuntimeError(
                f"The training data changed since this checkpoint "
                f"({str(d['data_sha256'][0])[:12]}... -> "
                f"{str(d['data_sha256'][1])[:12]}...). Set FORCE_RESTART = True.")
        if d:
            print(f"  [WARN] data settings changed: {list(d)}")

    print(f"  resuming at iteration {global_counter:,}, "
          f"lr {optimizer.param_groups[0]['lr']:.3e}, {elapsed_offset:.2f} h spent")
    del resume_state
    gc.collect()

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_bn = sum(b.numel() for name, b in model.named_buffers()
           if name.endswith('running_mean') or name.endswith('running_var'))
print(f"\n  {n_trainable:,} trainable parameters + {n_bn:,} BatchNorm running stats "
      f"= {n_trainable + n_bn:,}")
check("parameter count matches MXNet", n_trainable + n_bn == 6_986_506,
      f"{n_trainable + n_bn:,} (MXNet reported 6,986,506)", warn_only=True)

# %% [markdown]
# ## Sanity checks before the loop
#
# The objective against its closed form, on a real batch.

# %%
def verify_objective(model, batch_cpu):
    prev = model.mode
    model.mode = 'likelihood'
    model.eval()
    try:
        with torch.no_grad():
            inputs = batch_to_tensor(batch_cpu)
            lw = model(*inputs)
            l, tox = model.last_l, model.last_tox

        eff = tox if TOX_WEIGHTING else torch.zeros_like(tox)
        l_eff = l
        if TOX_WEIGHTING and TOX_FLOOR is not None:
            l_eff = torch.where(tox > 0, torch.clamp(l, min=TOX_FLOOR), l)
        expected = l_eff * (1.0 - (1.0 + LAMBDA_TOX) * eff)
        err = float((lw - expected).abs().max())

        n_tox = int(tox.sum())
        print(f"  {tox.numel()} molecules, {n_tox} toxic")
        print(f"  max |l_weighted - closed form| = {err:.3e}")
        check("objective matches its closed form", err < 1e-3, f"max error {err:.2e}")

        if TOX_WEIGHTING:
            if n_tox == 0:
                print("  [WARN] no toxic molecule in this batch - the sign flip is "
                      "not exercised. Re-run on another batch")
            else:
                flipped = int((torch.sign(lw[tox > 0]) ==
                               -torch.sign(l_eff[tox > 0])).sum())
                check("sign flips on toxic molecules", flipped == n_tox,
                      f"{flipped}/{n_tox} toxic molecules had their sign reversed")
        else:
            same = int((torch.sign(lw) == torch.sign(l)).sum())
            check("no sign reversal under the CDGCN objective", same == l.numel(),
                  f"{same}/{l.numel()} molecules keep their sign - weight is 1 "
                  f"everywhere, including on the {n_tox} toxic ones")

        print(f"  per-action log-probabilities at the {LOG_FLOOR:.3f} floor: "
              f"{100 * model.last_frac_floor:.3f}%")
        print(f"  max |logit| {model.policy_h.last_logit_absmax:.1f}  "
              f"(clamp binds at 80)")
    finally:
        model.mode = prev
        model.train()


verify_objective(model, batch_cpu)

# %% [markdown]
# ## Training loop
#
# Three columns are new. Together they would have made the previous failure
# obvious at iteration 5,000 rather than at the evaluation table:
#
# * **`grad_norm`** - the pre-clip global gradient norm. Under the unbounded Eq. 8
#   this climbs without limit; `clip_grad_value_` hides that from the loss.
# * **`frac_clip`** - fraction of gradient elements the element-wise clip rewrote.
#   Near 1.0 means every parameter is taking a maximal step and the actual
#   gradient magnitudes no longer matter - only their signs, which the toxic
#   branch then sets.
# * **`frac_floor`** - fraction of per-action log-probabilities pinned at
#   `ln(1e-10)`. Rising means actions are being driven to numerically zero
#   probability, where the gradient is ~1e10.

# %%
TRAIN_HEADER = ('step\ttime(h)\tloss\tlr\tl_nontoxic\tl_toxic\tfrac_toxic'
                '\tgrad_norm\tfrac_clip\tfrac_floor\n')


def ensure_header(path, header):
    """Create with its header if absent or empty. Never truncates an existing log."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        with open(path, 'w') as f:
            f.write(header)


ensure_header(LOG_FILE, TRAIN_HEADER)
t0 = time.time() - elapsed_offset * 3600.0

print("-" * 70)
print(f"  objective    {OBJECTIVE_NAME}")
print(f"  device       {CTX}")
print(f"  batch / K    {BATCH_SIZE} / {K_PATHS}   alpha {P_ALPHA}")
print(f"  lr           {LR} (decay {DECAY} every {DECAY_STEP})")
print(f"  from         iteration {global_counter:,} of {TOTAL_ITERATIONS:,}")
if not TOX_WEIGHTING:
    print("  NOTE: the `loss` column IS a negative log-likelihood under this")
    print("        objective, so it is directly comparable to the MXNet CDGCN")
    print("        run's final 15.99 on the same architecture and data.")
else:
    print("  NOTE: the `loss` column is the Eq. 8 WEIGHTED loss. It is not an NLL,")
    print("        it goes negative, and a falling value is not evidence of")
    print("        progress - it means the toxic term is growing.")
print("-" * 70)

# %%
def save_checkpoint(global_counter, elapsed_hours):
    torch.save({'model': model.state_dict(),
                'configs': configs,
                'objective': objective_signature(),
                'data': data_signature(),
                'global_counter': global_counter,
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')},
               CKPT_PARAMS_FILE)
    torch.save({'optimizer': optimizer.state_dict(),
                'lr': optimizer.param_groups[0]['lr'],
                'global_counter': global_counter,
                'elapsed_hours': elapsed_hours,
                'objective': objective_signature(),
                'data': data_signature()},
               CKPT_TRAINER_FILE)


def grad_diagnostics(model, clip_value):
    """-> (pre-clip global norm, fraction of elements the clip will rewrite).

    One extra reduction, so this is called only on summary steps.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return float('nan'), float('nan')
    norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads)))
    n_clip = sum(int((g.detach().abs() > clip_value).sum()) for g in grads)
    n_tot = sum(g.numel() for g in grads)
    return norm, n_clip / max(1, n_tot)


model.mode = 'loss'
model.train()
it_train = iter(loader_train)

win = {'loss': 0.0, 'n': 0, 'sum_non': 0.0, 'n_non': 0, 'sum_tox': 0.0, 'n_tox': 0,
       'frac_floor': 0.0, 'grad_norm': 0.0, 'frac_clip': 0.0, 'n_diag': 0}

while global_counter < TOTAL_ITERATIONS:
    try:
        inputs_cpu = next(it_train)
    except StopIteration:
        it_train = iter(loader_train)
        continue

    if inputs_cpu is None:          # whole batch rejected by the collator filters
        continue

    global_counter += 1
    is_summary = (global_counter % SUMMARY_STEP == 0)

    inputs_gpu = batch_to_tensor(inputs_cpu)

    # Gluon overwrites gradients; torch accumulates. zero_grad is required.
    optimizer.zero_grad(set_to_none=True)
    loss = model(*inputs_gpu)

    if DIE_ON_NONFINITE and not torch.isfinite(loss):
        save_checkpoint(global_counter - 1, (time.time() - t0) / 3600.0)
        raise RuntimeError(
            f"Non-finite loss ({loss.item()}) at iteration {global_counter}. "
            f"The checkpoint holds the last good state.")

    loss.backward()

    if is_summary:
        gn, fc = grad_diagnostics(model, CLIP_GRAD)
        win['grad_norm'] += gn
        win['frac_clip'] += fc
        win['n_diag'] += 1

        if DIE_ON_NONFINITE and not math.isfinite(gn):
            save_checkpoint(global_counter - 1, (time.time() - t0) / 3600.0)
            raise RuntimeError(
                f"Non-finite gradient at iteration {global_counter}. Clamping a NaN "
                f"returns NaN, so it would have passed straight into optimizer.step() "
                f"and poisoned every parameter.")

    torch.nn.utils.clip_grad_value_(model.parameters(), CLIP_GRAD)
    optimizer.step()

    if global_counter % DECAY_STEP == 0:
        for g in optimizer.param_groups:
            g['lr'] *= (1.0 - DECAY)

    # Accumulate over the window. A single batch made l_tox read 'nan' whenever it
    # happened to hold no toxic molecule - at batch 8 and ~24% toxic that is
    # 0.76^8 = 11% of batches, which is noise, not divergence.
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
            f.flush()

        print(f"iter {global_counter:06d} | loss {mean_loss:9.3f} | "
              f"l_nontox {l_non:8.2f} | l_tox {l_tox:9.2f} | tox {100 * f_tox:4.1f}% | "
              f"|g| {g_norm:8.1f} | clip {100 * f_clip:5.1f}% | "
              f"floor {100 * f_floor:5.2f}% | lr {lr_now:.2e} | {hours:.2f}h",
              flush=True)

        if f_clip > 0.5:
            print("  [WARN] over half the gradient elements are being clipped. Every "
                  "parameter is taking a maximal step, so only the gradient SIGN "
                  "matters - which is how an unbounded branch takes over training.")
        if f_floor > 0.10:
            print("  [WARN] more than 10% of per-action log-probabilities are pinned "
                  "at ln(1e-10), where the gradient is ~1e10. Under Eq. 8 this is the "
                  "toxic branch running away.")

        for k in win:
            win[k] = 0.0
        del loss, inputs_cpu, inputs_gpu
        gc.collect()

# %% [markdown]
# ## Final save

# %%
save_checkpoint(global_counter, (time.time() - t0) / 3600.0)
with open(LOG_FILE, 'a') as f:
    f.write('# finished\n')

print(f"\n  reached {global_counter:,} iterations")
print(f"  {CKPT_PARAMS_FILE}")
print(f"  {CKPT_TRAINER_FILE}")
print(f"  {LOG_FILE}")
print()
print(f"  objective: {OBJECTIVE_NAME}")
print("  Next: run 00_phase0_diagnostics.py check A against this checkpoint to")
print("  measure unconditional validity before committing to fine-tuning.")

# %%
print("=" * 70)
for c in CHECKS:
    print(f"  {c['status']:5s}  {c['name']}: {c['detail']}")

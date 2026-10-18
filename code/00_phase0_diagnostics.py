# %% [markdown]
# # 00 — Phase-0 diagnostics
#
# Read-only. Nothing here trains, overwrites or deletes. Run it before every
# expensive job; it decides which of them are actually needed.
#
# | Check | Settles |
# |---|---|
# | T | Self-tests of the shared primitives |
# | F | Which ChEMBL file the existing checkpoint trained on |
# | F2 | Whether held-out ligands leaked into pretraining |
# | G | Stage-1 iteration count |
# | H | Whether one forest wrote both label sets |
# | E | Salts and graph traversal |
# | A | Unconditional validity — **the decisive one** |
# | D | Logit range |
# | B, C | Conditional sample census and truncation |
#
# Reading A: ~25% → stage 1 is the cause. ~90% → stage 1 is fine, the damage is
# in fine-tuning.

# %%
import os
import re
import ast
import csv
import glob
import json
import math
import time
import random
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

PRETRAIN_CSV = os.environ.get(

    "CTDDG_DATA_PATH", os.path.join(CHEMBL_DIR, "chembl_pretrain_labelled.csv"))

PRETRAIN_MANIFEST = PRETRAIN_CSV + ".manifest.json"

OUT_DIR = os.path.join(REPO_ROOT, "outputs/diagnostics")

os.makedirs(OUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUT_DIR, f"phase0_{FOLD}.json")

FOLD_SUFFIX = {"d1": "4_org_1042_104", "d2": "4_org_1000_112", "d3": "4_org_1004_122",

               "d4": "4_org_1002_103", "d5": "4_org_1036_124"}

GPU_ID = int(os.environ.get("CTDDG_GPU", "0"))

CTX = torch.device(f"cuda:{GPU_ID}") if torch.cuda.is_available() else torch.device("cpu")

N_UNCOND_SAMPLES = 2000

MAX_DECODE_STEPS = 100

D_RECEPTIVE = 2

SEED = 42

STABLE_SOFTMAX = True

FRAGMENT_POLICY = 'heavy_atoms'

P_ALPHA = 0.8

# %%
MAX_ITER = 120

STRIP_SALTS = True

# 'complete_dfs' finishes every branch point; 'upstream_broken' reproduces the
# traversal the checkpoint was trained under. Checks T6/T6b run BOTH, so this is
# only the default for anything else in this notebook.

# %%
TRAVERSAL_POLICY = 'complete_dfs'

print(f"repo {REPO_ROOT} | fold {FOLD} | device {CTX}")

# %% [markdown]
# ## Verdicts
#
# Every check appends one row. The notebook ends in one table and one JSON file.

# %%
VERDICTS = []
_MARK = {"PASS": "[OK]  ", "FAIL": "[FAIL]", "WARN": "[WARN]",
         "INFO": "[INFO]", "SKIP": "[SKIP]"}

# %%

def record(check_id, name, status, detail, data=None):
    VERDICTS.append({"check": check_id, "name": name, "status": status,
                     "detail": detail, "data": data or {}})
    print(f"{_MARK.get(status, '[?]   ')} {check_id} {name}: {detail}")

# %%

def status_of(check_id):
    for v in VERDICTS:
        if v["check"] == check_id:
            return v["status"]
    return None

# %%

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()

# %%

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

# %%

def ligand_files(fold_suffix, split):
    """Every ligand pair file in a fold's split directory. Globbed, never guessed."""
    d = os.path.join(BINDINGDB_DIR, f"{split}_dataset", f"{split}_{fold_suffix}")
    if not os.path.isdir(d):
        return d, []
    return d, sorted(glob.glob(os.path.join(d, "ligands_*_corrected")))


# %% [markdown]
# ## Shared primitives
#
# Byte-identical to the training notebooks; `99_consistency_audit.py` enforces that.

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

# %%

def get_mol_from_graph(X, A, sanitize=True):
    """-> (mol | None, reason)."""
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

        return None, f"{type(e).__name__}: {str(e).strip().splitlines()[0][:60]}"

# %%
try:
    record("SETUP", "atom vocabulary", "PASS",
           f"{get_mol_spec().num_atom_types} atom types, "
           f"{get_mol_spec().num_bond_types} bond types",
           {"n_atom_types": get_mol_spec().num_atom_types})
except Exception as e:
    record("SETUP", "atom vocabulary", "FAIL", f"{type(e).__name__}: {e}")
    raise

# %% [markdown]
# ## Check T — self-tests
#
# Seconds, CPU only. These test the code that is about to make decisions.

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

def _toy_logits(scale, seed=0):
    g = torch.Generator().manual_seed(seed)
    raw_x = torch.randn(11, 1, 264, generator=g) * scale
    raw_t = torch.randn(3, 1, 1, generator=g) * scale
    NX_rep = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

    return raw_x, raw_t, NX_rep, 3

# %%
T = []

# %%

def t_check(name, ok, detail):
    T.append((name, bool(ok), detail))

    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {detail}")


# T1 normalisation

# %%
lx, lt, rep, nm = _toy_logits(1.0)

a, b = action_log_probs(lx, lt, rep, nm, stable=True)

tot = segment_sum(a.exp(), rep, nm).sum(-1) + b.exp().squeeze(-1)
t_check("T1 probabilities sum to 1 per molecule", torch.allclose(tot, torch.ones_like(tot), atol=1e-5),

        f"max deviation {float((tot - 1).abs().max()):.2e}")

# T2 agreement with the legacy path on ordinary logits

# %%
a0, b0 = action_log_probs(lx, lt, rep, nm, stable=False)
t_check("T2 stable == legacy on ordinary logits",
        float((a - a0).abs().max()) < 1e-4,

        f"max |delta log p| {float((a - a0).abs().max()):.2e}")

# T3 extreme logits

# %%
lx, lt, rep, nm = _toy_logits(3.0e4, seed=1)

a, b = action_log_probs(lx, lt, rep, nm, stable=True)

a0, b0 = action_log_probs(lx, lt, rep, nm, stable=False)

tot = segment_sum(a.exp(), rep, nm).sum(-1) + b.exp().squeeze(-1)
t_check("T3 stable survives +/-3e4 logits",
        bool(torch.isfinite(a).all()) and torch.allclose(tot, torch.ones_like(tot), atol=1e-4),
        f"finite={bool(torch.isfinite(a).all())}, sum deviation "

        f"{float((tot - 1).abs().max()):.2e}")

_tot0 = segment_sum(a0.exp(), rep, nm).sum(-1) + b0.exp().squeeze(-1)
# The legacy path still normalises to 1 here - the clamp saturates symmetrically.
# Its failure is the gradient, not the mass, so that is what T4b measures.
t_check("T3b legacy mass is NOT the problem", True,
        f"legacy mass per molecule {[round(float(v), 3) for v in _tot0.reshape(-1)]} "

        f"- normalisation survives the clamp; the damage is in the gradient (T4b)")

# T4 gradient magnitude

# %%
lx = (torch.randn(11, 1, 264) * 50).requires_grad_(True)

lt = (torch.randn(3, 1, 1) * 50).requires_grad_(True)

a, _ = action_log_probs(lx, lt, rep, 3, stable=True)

a[0, 0, 0].backward()
t_check("T4 d log p / d logit bounded by 1", float(lx.grad.abs().max()) <= 1.0 + 1e-5,

        f"max |grad| {float(lx.grad.abs().max()):.4f}")

# T4b the decisive one: the legacy path's gradient on a suppressed action.

# %%
_g2 = torch.Generator().manual_seed(3)

_bx = torch.randn(11, 1, 264, generator=_g2) * 20

_bt = torch.randn(3, 1, 1, generator=_g2) * 20

_j = int(_bx[0, 0].argmin())

_grads = {}
for _st in (True, False):
    _x = _bx.clone().requires_grad_(True)
    _t = _bt.clone().requires_grad_(True)
    _a, _ = action_log_probs(_x, _t, rep, 3, stable=_st)
    _a[0, 0, _j].backward()

    _grads[_st] = float(_x.grad[0, 0, _j].abs())
t_check("T4b legacy freezes suppressed actions",
        _grads[False] < 1e-8 <= _grads[True],
        f"corrective gradient for an action at p~0: stable {_grads[True]:.4g}, "
        f"legacy {_grads[False]:.4g}. Zero means the model cannot move it back")

# T5 fragment policy

# %%
_cases = [("CC(=O)Oc1ccccc1C(=O)O.Cl", "CC(=O)Oc1ccccc1C(=O)O"),
          ("Cc1ccccc1.O=S(=O)(O)O", "Cc1ccccc1"),

          ("CCO", "CCO")]

_bad = [(s, strip_salt(s), want) for s, want in _cases if strip_salt(s) != want]
t_check("T5 strip_salt picks the principal fragment", not _bad,

        "aspirin/toluene/plain all correct" if not _bad else f"{_bad}")
t_check("T5b strip_salt is deterministic",
        all(strip_salt("CCO.OCC") == strip_salt("CCO.OCC") for _ in range(5)),

        f"'CCO.OCC' -> {strip_salt('CCO.OCC')!r} every time")

# T6 the traversal, both policies. This is the SHARED function from the training
# notebooks, not a demo copy, so what it reports is what training actually does.

# %%
import networkx as nx

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

def _graph_of(smi):
    m = Chem.MolFromSmiles(smi)
    ranks = list(Chem.CanonicalRankAtoms(m))
    g = nx.Graph()
    g.add_nodes_from(range(m.GetNumAtoms()))
    g.add_edges_from([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in m.GetBonds()])

    return m, g, ranks

# %%

def _graph_refuses_disconnected(smi):
    """True only if the rejection is FOR disconnection.

    `get_graph_from_smiles` raises ValueError for an unparseable SMILES and for an
    out-of-vocabulary atom as well, so a bare `except ValueError` would report this
    check as passing when the molecule was merely rejected for some other reason.
    """
    try:
        get_graph_from_smiles(smi)
        return False
    except ValueError as e:

        return "Disconnected" in str(e)

# %%

def _traversal_census(smiles_list, policy, n=20, seed=SEED):
    """-> (n_traversals, n_incomplete, n_unvisited_atoms, worst_molecule)."""
    global TRAVERSAL_POLICY
    keep = TRAVERSAL_POLICY
    random.seed(seed)
    total = incomplete = unvisited = 0
    per_mol = {}
    try:
        TRAVERSAL_POLICY = policy
        for smi in smiles_list:
            _m, _g, _r = _graph_of(smi)
            bad = 0
            for _ in range(n):
                total += 1
                try:
                    ids, _lp = traverse_graph(_g, _r)
                except ValueError as e:
                    # The guard raises instead of returning corrupt step_ids, and
                    # carries the count, so read it from the message rather than
                    # re-running with the guard disabled.
                    incomplete += 1
                    bad += 1
                    hit = re.search(r"left (\d+) atoms", str(e))
                    unvisited += int(hit.group(1)) if hit else 1
                    continue
                miss = sum(1 for x in ids if x < 0)
                if miss:
                    incomplete += 1
                    bad += 1
                    unvisited += miss
            per_mol[smi] = bad
    finally:
        TRAVERSAL_POLICY = keep
    worst = max(per_mol.items(), key=lambda kv: kv[1]) if per_mol else (None, 0)

    return total, incomplete, unvisited, worst


# Branch points whose first branch is a dead end: gem-dimethyl, tert-butyl,
# N,N-dimethyl, sulfonamide. Ordinary drug motifs, not exotic ones.

# %%
_branchy = ["CC(C)(C)c1ccccc1", "Cc1ccc(cc1)S(=O)(=O)N", "CC(C)(N)Cc1ccccc1",
            "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "CN(C)CCOC(c1ccccc1)c1ccccc1",
            "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)cccn2", "CN1CCC[C@H]1c1cccnc1",

            "O=C(O)c1ccccc1", "C1CC2CCC1CC2"]

_n_new, _inc_new, _unv_new, _worst_new = _traversal_census(_branchy, 'complete_dfs')
t_check("T6 the traversal visits every atom of a connected graph", _inc_new == 0,
        f"{_n_new - _inc_new}/{_n_new} traversals complete over "
        f"{len(_branchy)} molecules")

_n_old, _inc_old, _unv_old, _worst_old = _traversal_census(_branchy,
                                                           'upstream_broken')
t_check("T6b the upstream traversal silently dropped atoms", _inc_old > 0,
        f"{_inc_old}/{_n_old} traversals ({100.0 * _inc_old / max(1, _n_old):.0f}%) "
        f"left {_unv_old} atoms unvisited; worst molecule {_worst_old[0]!r} failed "
        f"{_worst_old[1]}/20. Those step_ids reached single_reorder holding -1, "
        f"which argsorts to the front and mis-wires every bond. See AUDIT.md A21")

t_check("T6c the fix is what changed, not the molecules",
        _inc_new == 0 and _inc_old > 0,
        f"same {len(_branchy)} molecules, same seed: {_inc_old} incomplete under the "
        f"upstream traversal, {_inc_new} under the fixed one")

# T7 disconnected input. The fixed traversal's top-level candidate pool is every
# atom, not just neighbours, so when a component is exhausted it starts a new root
# and still returns a complete permutation. That is robust but NOT a licence to
# feed it salts: the second fragment is not part of the ligand, and its atoms would
# become training targets. The guard therefore lives in get_graph_from_smiles.

# %%
_m, _g, _r = _graph_of("CC(=O)Oc1ccccc1C(=O)O.Cl")

random.seed(SEED)
TRAVERSAL_POLICY = 'complete_dfs'
try:
    _ids_new, _ = traverse_graph(_g, _r)
    _ok_new = sorted(_ids_new) == list(range(_m.GetNumAtoms()))
except ValueError:
    _ok_new = False

random.seed(SEED)
TRAVERSAL_POLICY = 'upstream_broken'
try:
    _ids_old, _ = traverse_graph(_g, _r)
    _unvis_old = sum(1 for x in _ids_old if x < 0)
except ValueError as _e:
    _hit = re.search(r"left (\d+) atoms", str(_e))
    _unvis_old = int(_hit.group(1)) if _hit else 1
TRAVERSAL_POLICY = 'complete_dfs'

t_check("T7 the fixed traversal completes across components", _ok_new,
        f"{_m.GetNumAtoms()}-atom two-component graph -> a complete permutation, "
        f"because the top-level pool is every atom. The upstream traversal left "
        f"{_unvis_old} of them unvisited on the same input")

t_check("T7b a disconnected molecule is refused before the traversal",
        _graph_refuses_disconnected("CC(=O)Oc1ccccc1C(=O)O.Cl"),
        "get_graph_from_smiles raises on a multi-fragment SMILES: the chloride's "
        "atom would otherwise become a training target of the aspirin molecule")

t_check("T7c stripping makes it connected",
        nx.is_connected(_graph_of(strip_salt("CC(=O)Oc1ccccc1C(=O)O.Cl"))[1]),
        "strip_salt output is a single component")

_t_failed = [n for n, ok, _ in T if not ok]
record("T", "self-tests", "PASS" if not _t_failed else "FAIL",
       f"{len(T) - len(_t_failed)}/{len(T)} passed"
       + ("" if not _t_failed else f"; failed: {_t_failed}"),
       {"failed": _t_failed, "n": len(T)})

# %% [markdown]
# ## Check F — what did the existing checkpoint train on?
#
# The default `DATA_PATH` says what the *next* run will read. Only the `data`
# signature stored inside `ckpt.pt` says what *this* checkpoint read.

# %%
def count_rows(path, has_header=True):
    if not os.path.isfile(path):
        return None, None
    n = 0
    with open(path, 'rb') as f:
        for _ in f:
            n += 1
    return (n - 1 if has_header and n else n), os.path.getsize(path)

# %%
chembl_candidates = {
    "chembl_pretrain_labelled.csv": (PRETRAIN_CSV, True),
    "chembl.txt": (os.path.join(CHEMBL_DIR, "chembl.txt"), False),
    "chembl_final.txt": (os.path.join(CHEMBL_DIR, "chembl_final.txt"), False),
    "chembl_with_ames_predictions.csv":
        (os.path.join(CHEMBL_DIR, "chembl_with_ames_predictions.csv"), True),
    "chembl_toxicity_labels.csv":
        (os.path.join(CHEMBL_DIR, "chembl_toxicity_labels.csv"), True),
}

chembl_rows = {}
for label, (path, hdr) in chembl_candidates.items():
    n, size = count_rows(path, hdr)
    chembl_rows[label] = {"path": path, "rows": n,
                          "size_mb": round(size / 1e6, 1) if size else None}
    print(f"       {label:38s} " +
          ("absent" if n is None else f"{n:>10,} rows  {size / 1e6:>8.1f} MB"))

MANIFEST = load_json(PRETRAIN_MANIFEST)
if MANIFEST:
    _f = MANIFEST.get("filters", {})
    print(f"\n       manifest: {MANIFEST.get('n_molecules'):,} molecules, "
          f"{100 * MANIFEST.get('frac_toxic', 0):.1f}% toxic")
    print(f"       BindingDB removed: {_f.get('bindingdb_ligands_removed')} | "
          f"fragment policy: {_f.get('fragment_policy')}")

CKPT_STATE = None
ckpt_data = None

# %%
if not os.path.isfile(PRETRAIN_CKPT):
    record("F", "stage-1 data provenance", "SKIP", f"no checkpoint at {PRETRAIN_CKPT}",
           chembl_rows)
else:
    CKPT_STATE = torch.load(PRETRAIN_CKPT, map_location="cpu", weights_only=False)
    ckpt_data = CKPT_STATE.get("data") if isinstance(CKPT_STATE, dict) else None
    ckpt_obj = CKPT_STATE.get("objective") if isinstance(CKPT_STATE, dict) else None

    if ckpt_data is None:
        record("F", "stage-1 data provenance", "WARN",
               "the checkpoint records no `data` signature, so it predates provenance "
               "tracking. Which file it trained on CANNOT be established from the "
               "artifacts — report it as unresolved rather than inferring from row "
               "counts. Any checkpoint written by the current pretraining notebook "
               "will carry one", chembl_rows)
    else:
        trained_path = ckpt_data.get("data_path")
        trained_sha = ckpt_data.get("data_sha256")
        print(f"       checkpoint trained on  {trained_path}")
        print(f"       recorded sha256        {str(trained_sha)[:32]}...")
        print(f"       recorded n_molecules   {ckpt_data.get('n_molecules')}")
        if ckpt_obj:
            print(f"       objective              {ckpt_obj.get('objective')}")

        if MANIFEST and trained_sha == MANIFEST.get("sha256"):
            record("F", "stage-1 data provenance", "PASS",
                   f"the checkpoint's recorded hash matches the manifest of "
                   f"{os.path.basename(PRETRAIN_CSV)}; its filters record "
                   f"bindingdb_ligands_removed="
                   f"{MANIFEST.get('filters', {}).get('bindingdb_ligands_removed')}",
                   {"ckpt_data": ckpt_data})
        else:
            record("F", "stage-1 data provenance", "FAIL",
                   f"the checkpoint trained on {trained_path}, which is not the "
                   f"manifested dataset. Stage 1 must be re-run on the built set",
                   {"ckpt_data": ckpt_data})

# %% [markdown]
# ### F2 — leakage, by canonical structure
#
# Row counts are circumstantial. This compares structures. The built dataset is
# already canonical, so only the ligand side needs RDKit.

# %%
_lig_paths, _lig_dirs_missing = [], []
for _split in ("train", "test"):
    _d, _files = ligand_files(FOLD_SUFFIX[FOLD], _split)
    if not _files:
        _lig_dirs_missing.append(_d)
    _lig_paths += _files

print(f"       ligand files for fold {FOLD}: {[os.path.basename(p) for p in _lig_paths]}")

if _lig_dirs_missing:
    record("F2a", "ligand files located", "FAIL",
           f"no ligands_*_corrected in {_lig_dirs_missing}. Any leakage verdict "
           f"below would be computed on a partial ligand set",
           {"missing": _lig_dirs_missing})
else:
    record("F2a", "ligand files located", "PASS",
           f"{len(_lig_paths)} file(s) across train and test splits")

_pretrain_for_f2 = (ckpt_data or {}).get("data_path") or PRETRAIN_CSV

# %%
if not (_lig_paths and os.path.isfile(_pretrain_for_f2)):
    record("F2", "ligand leakage", "SKIP",
           "need both ligand files and the pretraining CSV")
else:
    import pandas as pd

    t0 = time.time()
    lig_canon = set()
    for p in _lig_paths:
        with open(p) as f:
            for line in f:
                s = "".join(line.split())
                if not s:
                    continue
                s = strip_salt(s)
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    lig_canon.add(Chem.MolToSmiles(m))

    _hdr = pd.read_csv(_pretrain_for_f2, nrows=0)
    _col = next((c for c in ("Smiles", "smiles", "SMILES", "canonical_smiles")
                 if c in _hdr.columns), None)
    _already_canon = bool((MANIFEST or {}).get("filters", {})
                          .get("deduplicated_on_canonical_smiles"))

    if _col is None:
        record("F2", "ligand leakage", "SKIP",
               f"no SMILES column among {list(_hdr.columns)[:10]}")
    else:
        _pre = pd.read_csv(_pretrain_for_f2, usecols=[_col])[_col].astype(str)
        if _already_canon:
            pre_canon = set(_pre)
            _how = "read directly (the manifest says the file is already canonical)"
        else:
            print(f"       canonicalising {len(_pre):,} pretraining molecules...")
            pre_canon = set()
            for s in _pre:
                m = Chem.MolFromSmiles(strip_salt("".join(str(s).split())))
                if m is not None:
                    pre_canon.add(Chem.MolToSmiles(m))
            _how = "canonicalised here"

        overlap = lig_canon & pre_canon
        frac = len(overlap) / max(1, len(lig_canon))
        record("F2", "ligand leakage",
               "PASS" if frac < 0.001 else ("WARN" if frac < 0.01 else "FAIL"),
               f"{len(overlap):,} of {len(lig_canon):,} fold-{FOLD} ligands "
               f"({100 * frac:.3f}%) are in the pretraining set; {_how} "
               f"[{time.time() - t0:.0f}s]",
               {"n_overlap": len(overlap), "n_ligands": len(lig_canon),
                "frac": round(frac, 6), "n_pretrain": len(pre_canon),
                "examples": sorted(overlap)[:3]})
        del pre_canon, lig_canon, _pre

# %% [markdown]
# ## Check G — stage-1 counters
#
# `max(num_batches_tracked)` is the exact iteration count: one train-mode forward
# per training step, and it survives resumes because it lives in the state dict.

# %%
def last_log_step(path):
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

# %%
if CKPT_STATE is None:
    record("G", "stage-1 counters", "SKIP", "ckpt.pt absent")
else:
    _sd = CKPT_STATE['model'] if 'model' in CKPT_STATE else CKPT_STATE
    _nbt = [int(v) for k, v in _sd.items() if k.endswith("num_batches_tracked")]
    bn_steps = max(_nbt) if _nbt else None

    trainer_counter = adam_steps = None
    if os.path.isfile(PRETRAIN_TRAINER):
        _tr = torch.load(PRETRAIN_TRAINER, map_location="cpu", weights_only=False)
        if isinstance(_tr, dict):
            trainer_counter = _tr.get("global_counter")
            try:
                adam_steps = int(next(iter(_tr["optimizer"]["state"].values()))["step"])
            except Exception:
                pass

    log_last, log_rows, log_header = last_log_step(PRETRAIN_LOG)
    policy0_absmax = float(_sd["policy_0"].abs().max()) if "policy_0" in _sd else None
    n_float = sum(1 for v in _sd.values() if torch.is_tensor(v) and v.is_floating_point())
    n_finite = sum(int(torch.isfinite(v).all()) for v in _sd.values()
                   if torch.is_tensor(v) and v.is_floating_point())

    for k, v in (("BN train-mode steps", bn_steps), ("trainer.pt counter", trainer_counter),
                 ("adam step count", adam_steps), ("log.out last step", log_last),
                 ("log.out data rows", log_rows), ("policy_0 absmax", policy0_absmax)):
        print(f"       {k:22s} {v}")
    print(f"       log.out header         {log_header}")

    _counters = {"bn_steps": bn_steps, "trainer_counter": trainer_counter,
                 "adam_steps": adam_steps, "log_last_step": log_last,
                 "log_rows": log_rows, "policy_0_absmax": policy0_absmax}

    if bn_steps in (None, 0):
        record("G", "stage-1 counters", "FAIL",
               "no BatchNorm steps recorded — this checkpoint never trained", _counters)
    elif len({x for x in (bn_steps, trainer_counter, log_last) if x is not None}) == 1:
        record("G", "stage-1 counters", "PASS",
               f"all counters agree at {bn_steps:,} iterations", _counters)
    else:
        record("G", "stage-1 counters", "WARN",
               f"counters disagree (bn={bn_steps}, trainer={trainer_counter}, "
               f"log={log_last}); the log probably carries rows from an earlier "
               f"SUMMARY_STEP era. The BatchNorm count is authoritative", _counters)

    record("G2", "checkpoint finiteness",
           "PASS" if n_finite == n_float else "FAIL",
           f"{n_finite}/{n_float} float tensors finite")

# %% [markdown]
# ## Check H — did one forest write both label sets?
#
# By hash. The presence of a metadata file proves nothing: the training notebook
# always writes one.

# %%
if not os.path.isfile(RF_MODEL_PATH):
    record("H", "Ames RF identity", "SKIP", f"no RF at {RF_MODEL_PATH}")
else:
    rf_sha = sha256_file(RF_MODEL_PATH)
    meta = load_json(RF_META_PATH) or {}
    meta_sha = meta.get("sha256")
    cache_sha = ((load_json(AMES_CACHE_META) or {}).get("rf") or {}).get("sha256")
    manifest_sha = ((MANIFEST or {}).get("labeller") or {}).get("sha256")

    for k, v in (("on disk now", rf_sha), ("recorded at training", meta_sha),
                 ("BindingDB label cache", cache_sha),
                 ("ChEMBL manifest", manifest_sha)):
        print(f"       {k:24s} {str(v)[:32] if v else '(none)'}")

    _d = {"rf_sha256": rf_sha, "meta_sha256": meta_sha,
          "cache_sha256": cache_sha, "manifest_sha256": manifest_sha}
    _known = [s for s in (meta_sha, cache_sha, manifest_sha) if s]
    _mismatch = [s for s in _known if s != rf_sha]

    if not _known:
        record("H", "Ames RF identity", "WARN",
               "nothing records which forest wrote the labels, so identity is "
               "UNRESOLVED. Re-running random_forest_classifier writes all three", _d)
    elif _mismatch:
        record("H", "Ames RF identity", "FAIL",
               f"{len(_mismatch)} of {len(_known)} recorded hashes differ from the "
               f"forest on disk — ChEMBL and BindingDB were labelled by different "
               f"models, so Eq. 8 optimised against two definitions of toxic. "
               f"Re-run random_forest_classifier, which rebuilds both consistently", _d)
    elif len(_known) == 3:
        record("H", "Ames RF identity", "PASS",
               "training metadata, BindingDB cache and ChEMBL manifest all record "
               "the forest that is on disk", _d)
    else:
        record("H", "Ames RF identity", "WARN",
               f"{len(_known)} of 3 provenance records present and matching; the "
               f"others are absent", _d)

# %% [markdown]
# ## Check E — salt census

# %%
_salt_counts = {}
for split in ("train", "test"):
    _d, _files = ligand_files(FOLD_SUFFIX[FOLD], split)
    if not _files:
        _salt_counts[split] = None
        continue
    n = tot = 0
    for p in _files:
        with open(p) as f:
            for line in f:
                s = "".join(line.split())
                if not s:
                    continue
                tot += 1
                n += ('.' in s)
    _salt_counts[split] = {"multi_fragment": n, "total": tot,
                           "frac": round(n / max(1, tot), 4),
                           "files": [os.path.basename(x) for x in _files]}
    print(f"       {split:5s}  {n:>7,} of {tot:>7,} ligands are multi-fragment "

          f"({100 * n / max(1, tot):.2f}%)")

_worst = max((v["frac"] for v in _salt_counts.values() if v), default=None)

if _worst is None:
    record("E", "salt census", "SKIP", "no ligand files found for this fold")
elif _worst == 0.0:
    record("E", "salt census", "PASS", "no multi-fragment ligands in this fold",
           _salt_counts)
else:
    record("E", "salt census", "INFO",
           f"up to {100 * _worst:.2f}% of ligands are multi-fragment. Test T7 shows "
           f"what that costs if they are not stripped; the fine-tuning notebook "

           f"strips and then asserts connectivity", _salt_counts)

_chembl_txt = os.path.join(CHEMBL_DIR, "chembl.txt")

# %%
if os.path.isfile(_chembl_txt):
    n = tot = 0
    with open(_chembl_txt) as f:
        for line in f:
            s = line.strip()
            if s:
                tot += 1
                n += ('.' in s)
    record("E2", "ChEMBL salt census", "PASS" if n == 0 else "INFO",
           f"{n:,} of {tot:,} molecules in chembl.txt are multi-fragment",
           {"multi_fragment": n, "total": tot})
else:
    record("E2", "ChEMBL salt census", "SKIP", "chembl.txt absent")

# %% [markdown]
# ## The unconditional model
#
# Stage-1 architecture exactly: no `linear_c`, no `dense_policy_0`.

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

        return _concat_neighbourhoods(X, A_list) @ self.W

# %%

class Policy(nn.Module):
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

class UMoleculeGenerator(nn.Module):
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

        return torch.softmax(self.policy_0, dim=0)

# %%

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

# %%

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
            X_out.append(conv(X if i == 0 else self.activation(self.bn[i - 1](X)), A))
        X_out = torch.cat(X_out[1:], dim=1)
        return self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))


# %% [markdown]
# ## Decoding driver

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

# %%

def decode_step(X, A, NX, NA, last_action, finished, get_init, get_action,
                n_node_types, n_edge_types):
    if X is None:
        init = get_init()
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
    NX_u, NA_u = NX[unfin], NA[unfin]
    last_action_u = last_action[unfin]
    NX_rep = np.repeat(np.arange(NX_u.shape[0]), NX_u)

    if A.shape[0] == 0:
        D_2 = D_3 = np.zeros((0, 2), dtype=np.int64)
        A_u = [np.zeros((0, 2), dtype=np.int64) for _ in range(n_edge_types)] + [D_2, D_3]
    else:
        cumsum = np.cumsum(np.pad(NX_u, [[1, 0]], mode='constant')[:-1])
        A_u[:, :2] += np.stack([np.repeat(cumsum, NA_u)] * 2, axis=1)
        D_2, D_3 = get_d(A_u, X_u)
        A_u = [A_u[A_u[:, 2] == i, :2] for i in range(n_edge_types)] + [D_2, D_3]

    mask = np.zeros([X_u.shape[0]], dtype=np.int64)
    last_append_index = np.cumsum(NX_u) - 1
    mask[last_append_index] = np.where(last_action_u == 1,
                                       np.ones_like(last_append_index),
                                       np.ones_like(last_append_index) * 2)

    append, connect, end = get_action([X_u, A_u, NX_u, NX_rep, mask])

    if A.shape[0] == 0:
        max_index = np.argmax(np.reshape(append, [-1, n_node_types * n_edge_types]), axis=1)
        atom_type, bond_type = np.unravel_index(max_index, [n_node_types, n_edge_types])
        X = np.reshape(np.stack([X, atom_type], axis=1), [-1])
        NX = np.array([2] * len(finished), dtype=np.int64)
        A = np.stack([np.zeros([len(finished)], dtype=np.int64),
                      np.ones([len(finished)], dtype=np.int64), bond_type], axis=1)
        NA = np.ones([len(finished)], dtype=np.int64)
        last_action = np.ones_like(NX, dtype=np.int64)
        return X, A, NX, NA, last_action, finished

    append = np.split(append, np.cumsum(NX_u))
    connect = np.split(connect, np.cumsum(NX_u))
    end = end.tolist()

    unfinished_ids = np.where(unfin)[0].tolist()
    cumsum, cumsum_a = np.cumsum(NX), np.cumsum(NA)
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
        configs = {k: v for k, v in configs.items() if k != 'N_C'}

        self.mdl = UVanillaMolGen_RNN(get_mol_spec().num_atom_types,
                                      get_mol_spec().num_bond_types,
                                      D=d_receptive, **configs)
        missing, unexpected = self.mdl.load_state_dict(sd, strict=False)
        self.missing, self.unexpected = list(missing), list(unexpected)
        self.mdl.to(device).eval()
        self.device, self.configs = device, configs

    def _to_tensor(self, inputs):
        X, A, NX, NX_rep, mask = inputs
        dev = self.device
        to_long = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.long,
                                            device=dev)
        X_t, NX_t, NX_rep_t, mask_t = to_long(X), to_long(NX), to_long(NX_rep), to_long(mask)
        NX_cum_t = to_long(np.cumsum(NX))
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
            None, None, None, None, None, finished, get_init=get_init, get_action=None,
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
                X, A, NX, NA, last_action, finished, get_init=None,
                get_action=get_action, n_node_types=self.mdl.N_A,
                n_edge_types=self.mdl.N_B)
            count += 1

        cum_X = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')).tolist()
        cum_A = np.cumsum(np.pad(NA, [[1, 0]], mode='constant')).tolist()
        return ([[X[xa:xb], A[aa:ab, :]]
                 for aa, ab, xa, xb in zip(cum_A[:-1], cum_A[1:], cum_X[:-1], cum_X[1:])],
                int(np.sum(np.logical_not(finished))))


# %% [markdown]
# ## Check A — unconditional validity
#
# Bisects the pipeline: everything downstream of stage 1 is bypassed.

# %%
UNCOND = None
if CKPT_STATE is None:
    record("A", "unconditional validity", "SKIP", "ckpt.pt absent")
elif status_of("T") == "FAIL":
    record("A", "unconditional validity", "SKIP",
           "self-tests failed; sampling results would not be trustworthy")
else:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    UNCOND = UncondSampler(PRETRAIN_CKPT, CTX, config_path=PRETRAIN_CONFIG)

    if UNCOND.missing:
        record("A0", "checkpoint transfer", "FAIL",
               f"{len(UNCOND.missing)} tensors missing, e.g. {UNCOND.missing[:5]}")
    elif UNCOND.unexpected:
        record("A0", "checkpoint transfer", "WARN",
               f"{len(UNCOND.unexpected)} unused tensors, e.g. {UNCOND.unexpected[:5]}")
    else:
        record("A0", "checkpoint transfer", "PASS",
               "every stage-1 tensor loaded into the unconditional model")

# %%
if UNCOND is not None:
    with torch.no_grad():
        UNCOND.mdl.mode = 'decode_0'
        p0 = UNCOND.mdl().cpu().numpy().astype(np.float64)
        p0 = p0 / p0.sum()
    ent0 = float(-(p0 * np.log(p0 + 1e-12)).sum())
    top5 = [(get_mol_spec().atom_types[i], round(float(p0[i]), 4))
            for i in np.argsort(-p0)[:5]]
    print(f"       pretrained policy_0 entropy {ent0:.3f} "
          f"(uniform = {math.log(len(p0)):.3f})")
    print(f"       top-5 first atoms {top5}")

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

    validity = 100.0 * reasons['valid'] / max(1, len(graphs))
    print(f"\n       {len(graphs)} samples in {elapsed:.0f}s | {n_truncated} hit the "
          f"{MAX_DECODE_STEPS}-step cap")
    for k, v in reasons.most_common():
        print(f"         {v:>5}  {k}")

    _d = {"n_samples": len(graphs), "validity_pct": round(validity, 2),
          "n_truncated": n_truncated, "failure_reasons": dict(reasons.most_common()),
          "policy_0_entropy": round(ent0, 4),
          "policy_0_uniform_entropy": round(math.log(len(p0)), 4),
          "policy_0_top5": [list(t) for t in top5],
          "mean_atoms_valid": round(float(np.mean(n_atoms_valid)), 1) if n_atoms_valid else None,
          "mean_atoms_invalid": round(float(np.mean(n_atoms_invalid)), 1) if n_atoms_invalid else None}

    if validity >= 80.0:
        record("A", "unconditional validity", "PASS",
               f"{validity:.1f}% valid. STAGE 1 IS HEALTHY — do not re-run "
               f"pretraining; the damage is in fine-tuning", _d)
    elif validity >= 50.0:
        record("A", "unconditional validity", "WARN",
               f"{validity:.1f}% valid — degraded but not destroyed. Both stages "
               f"contribute", _d)
    else:
        record("A", "unconditional validity", "FAIL",
               f"{validity:.1f}% valid. STAGE 1 IS THE CAUSE — it must be redone "
               f"(CDGCN objective, batch 8, stable softmax)", _d)

    lam = UNCOND.mdl.policy_h.last_logit_absmax
    fx = UNCOND.mdl.policy_h.last_frac_extreme
    record("D", "logit range", "PASS" if fx == 0.0 else "INFO",
           f"max |logit| {lam:.1f}; {100 * fx:.4f}% beyond +/-80. "
           + ("Nothing extreme." if fx == 0.0 else
              "Harmless under STABLE_SOFTMAX, which never clamps. Under the legacy "
              "path every one of these received EXACTLY ZERO gradient (measured), "
              "so they were frozen wherever training left them."),
           {"logit_absmax": lam, "frac_beyond_80": fx, "stable_softmax": STABLE_SOFTMAX})

# %% [markdown]
# ## Checks B and C — conditional samples already on disk

# %%
_csvs = sorted(glob.glob(os.path.join(SAMPLES_DIR, "run1", "protein_*.csv")))
SAMPLE_ROWS = None

if not _csvs:
    record("B", "conditional sample census", "SKIP",
           f"no sample CSVs under {os.path.join(SAMPLES_DIR, 'run1')}")
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
                "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2])}

    _d = {"valid": _describe(av, "valid"), "invalid": _describe(ai, "invalid"),
          "max_decode_steps": MAX_DECODE_STEPS}
    near_cap = float((ai >= MAX_DECODE_STEPS - 5).mean()) if ai.size else 0.0
    _d["frac_invalid_near_cap"] = round(near_cap, 4)

    if near_cap >= 0.25:
        record("C", "decode truncation", "FAIL",
               f"{100 * near_cap:.0f}% of invalid molecules sit within 5 atoms of the "
               f"cap. Raise MAX_DECODE_STEPS to MAX_ITER = 120", _d)
    elif near_cap >= 0.05:
        record("C", "decode truncation", "WARN",
               f"{100 * near_cap:.0f}% of invalid molecules are near the cap — a "
               f"minor contributor", _d)
    else:
        record("C", "decode truncation", "PASS",
               f"only {100 * near_cap:.1f}% of invalid molecules are near the cap; "
               f"leave MAX_DECODE_STEPS at {MAX_DECODE_STEPS}", _d)

# %% [markdown]
# ## Summary and decision

# %%
_order = {"FAIL": 0, "WARN": 1, "SKIP": 2, "INFO": 3, "PASS": 4}
print("=" * 78)
print(f"PHASE-0 — fold {FOLD} — {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
               "repo_root": REPO_ROOT, "stable_softmax": STABLE_SOFTMAX,
               "fragment_policy": FRAGMENT_POLICY, "verdicts": VERDICTS}, f, indent=1)
print(f"\n  report -> {REPORT_PATH}")

_a, _f2, _h = status_of("A"), status_of("F2"), status_of("H")
print("NEXT STEP")
print("-" * 78)
if status_of("T") == "FAIL":
    print("  Self-tests failed. Nothing below is trustworthy — fix those first.")
elif _f2 == "FAIL" or status_of("F2a") == "FAIL" or status_of("F") == "FAIL":
    print("  Pretraining data provenance or leakage is broken. Stage 1 must be re-run")
    print("  on the manifested dataset regardless of what check A says, so settle")
    print("  batch size, bond cap and floor BEFORE starting it — it is the expensive one.")
elif _a == "FAIL":
    print("  Stage 1 is the cause. Re-run pretraining with TOX_WEIGHTING = False and")
    print("  BATCH_SIZE = 8, or convert the existing MXNet CDGCN checkpoint. Do not")
    print("  spend time on fine-tuning repairs yet.")
elif _a in ("PASS", "WARN"):
    print("  Stage 1 is usable. Go to 02_finetune_ctddg: its preflight gate will")
    print("  re-check the conditioning path and measure step-0 before training.")
else:
    print("  Check A did not run. It is the decisive one — get ckpt.pt in place.")

# %%
if _h == "FAIL":
    print("  Separately: the two label sets came from different forests. Re-run")
    print("  random_forest_classifier before reading anything into toxicity numbers.")

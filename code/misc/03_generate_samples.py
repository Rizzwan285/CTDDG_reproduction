# %% [markdown]
# # Stage 3 — conditional generation
#
# Loads `ckpt_best.pt` and the test bundle written by stage 2, then samples
# molecules for each held-out protein.
#
# **The numerics here must match training exactly.** The model was optimised under
# one normalisation of the action distribution; sampling under another sends it
# into states its training never scored. `STABLE_SOFTMAX` is read from the
# checkpoint rather than set here, and a parity cell proves the two agree before
# anything is generated.
#
# **Nothing expensive runs until `SMOKE_FIRST = False`.** The smoke pass is one
# protein with a failure census, which is where the diagnosis actually is.

# %%
import os
import gc
import csv
import json
import math
import time
import random
import hashlib
from abc import ABCMeta, abstractmethod
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

import rdkit
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

print(f"torch {torch.__version__} | rdkit {rdkit.__version__}")

# %% [markdown]
# ## Config

# %%
REPO_ROOT = os.environ.get("CTDDG_REPO_ROOT", "/home/muhamed/repo/CTDDG")
FOLD = os.environ.get("CTDDG_FOLD", "d1")

ATOM_TYPES_PATH = os.path.join(REPO_ROOT, "datasets/data/atom_types.txt")
FINETUNE_DIR = os.path.join(REPO_ROOT, "outputs/finetune", FOLD)
CKPT_PATH = os.path.join(FINETUNE_DIR, "ckpt_best.pt")
TEST_BUNDLE = os.path.join(FINETUNE_DIR, "test_set.npz")
CKPT_CONFIG = os.path.join(FINETUNE_DIR, "configs.json")

OUT_DIR = os.path.join(REPO_ROOT, "outputs/samples", FOLD)
os.makedirs(OUT_DIR, exist_ok=True)

GPU_ID = int(os.environ.get("CTDDG_GPU", "0"))
CTX = torch.device(f"cuda:{GPU_ID}") if torch.cuda.is_available() else torch.device("cpu")

# %% [markdown]
# ### Protocol
#
# 100 molecules x 10 runs per held-out protein, as the paper reports.

# %%
N_SAMPLES = int(os.environ.get("CTDDG_N_SAMPLES", "100"))
N_RUNS = int(os.environ.get("CTDDG_N_RUNS", "10"))
LIMIT_PROTEINS = os.environ.get("CTDDG_LIMIT_PROTEINS")
LIMIT_PROTEINS = int(LIMIT_PROTEINS) if LIMIT_PROTEINS else None

# A molecule needs GetNumBonds() + 1 actions, and training admitted anything up to
# MAX_ITER_TRAIN of them. A decode cap BELOW that truncates molecules the training
# distribution contains, manufacturing invalid samples that say nothing about the
# model. The check below enforces cap >= training budget.

# %%
MAX_DECODE_STEPS = int(os.environ.get("CTDDG_MAX_DECODE_STEPS", "120"))
MAX_ITER = 120                  # the shared MoleculeSpec reads this
MAX_ITER_TRAIN = MAX_ITER

D_RECEPTIVE = 2
BASE_SEED = 42
OVERWRITE = False          # False = skip (protein, run) pairs already on disk

SMOKE_FIRST = True         # the arm switch
SMOKE_N = 200

# Read from the checkpoint in the parity cell; this is only the fallback for a
# checkpoint that predates the field.

# %%
STABLE_SOFTMAX = True

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
print(f"device {CTX} | fold {FOLD}")

_missing = [f"{lab}: {p}" for lab, p in
            (("atom_types.txt", ATOM_TYPES_PATH), ("checkpoint", CKPT_PATH),
             ("test bundle", TEST_BUNDLE)) if not os.path.exists(p)]
check("required inputs", not _missing,
      "all present" if not _missing else "; ".join(_missing))

check("decode cap covers the training action budget",
      MAX_DECODE_STEPS >= MAX_ITER_TRAIN,
      f"MAX_DECODE_STEPS = {MAX_DECODE_STEPS} >= MAX_ITER_TRAIN = {MAX_ITER_TRAIN}"
      if MAX_DECODE_STEPS >= MAX_ITER_TRAIN else
      f"MAX_DECODE_STEPS = {MAX_DECODE_STEPS} is BELOW the {MAX_ITER_TRAIN}-action "
      f"training budget, so molecules the training set admitted are cut off "
      f"mid-build and counted as invalid. Raise it to {MAX_ITER_TRAIN}")

if _missing:
    raise FileNotFoundError(
        "Stage 3 reads what stage 2 wrote:\n  " + "\n  ".join(_missing)
        + "\n  Run 02_finetune_ctddg with RUN_TRAINING = True first.")

# %% [markdown]
# ## Shared primitives
#
# Byte-identical to stages 1 and 2 — `99_consistency_audit.py` enforces it. For
# the sampler this is not tidiness: `Policy` IS the action distribution, so a
# divergence here means generating from a different model than the one trained.

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

def get_mol_from_graph(X, A, sanitize=True):
    """-> (mol | None, reason).

    `reason` is what turns "26/100 valid" into a diagnosis: a KekulizeException
    census points somewhere different from an AtomValenceException census.
    """
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

def get_mol_from_graph_list(graph_list, sanitize=True):
    return [get_mol_from_graph(X, A, sanitize) for X, A in graph_list]

# %%
check("atom vocabulary", get_mol_spec().num_atom_types > 0,
      f"{get_mol_spec().num_atom_types} atom types, "
      f"{get_mol_spec().num_bond_types} bond types")

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
# ## Model
#
# The stage-2 classes with the training-only paths removed, so `ckpt_best.pt`
# loads strictly by name.

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
        self.mode = 'decode_0'

    @abstractmethod
    def _build_graph_conv(self, *args, **kwargs):

        pass

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

    def _rnn_test(self, X, NX, NX_rep, NX_cum, h):
        X_avg = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X_avg, X[NX_cum - 1]], dim=1)
        X, h = self.rnn(X.unsqueeze(1), h)
        return X.squeeze(1), h

# %%

class _TwoLayerDense(nn.Module):
    """FNN_0: protein embedding -> first-atom LOGITS.

    Returns logits, not probabilities, and `_policy_0` applies the softmax. This
    matches stage 2, where `_log_policy_0` takes a log_softmax of the same output.
    A module that normalises internally would make the two stages disagree while
    loading each other's weights without complaint.
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
        return torch.softmax(self.dense_policy_0(c), dim=-1) + 0.0 * self.policy_0

    def _decode_step(self, X, A, NX, NX_rep, last_append_mask, NX_cum, h, c, ids):
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        X = self._graph_conv_forward(X, A, c, ids)
        X_dense = self.dense(X)
        X_mol, h = self._rnn_test(X_dense, NX, NX_rep, NX_cum, h)
        append, connect, end = self.policy_h(X_dense, NX, NX_rep, X_mol)
        return append, connect, end, h

    def forward(self, *inputs):
        if self.mode == 'decode_0':
            return self._policy_0(*inputs)
        if self.mode == 'decode_step':
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
        self.bn = nn.ModuleList()
        for i, (f_in, f_out) in enumerate(zip([self.F_e] + self.F_h[:-1], self.F_h)):
            self.conv.append(GraphConv(f_in, f_out, self.N_B + self.D))
            if i != 0:
                self.bn.append(nn.BatchNorm1d(f_in, eps=1e-5, momentum=0.1))
        self.bn_skip = nn.BatchNorm1d(sum(self.F_h), eps=1e-5, momentum=0.1)
        self.linear_skip = Linear_BN(sum(self.F_h), self.F_skip)
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
# ## Decoding driver
#
# Port of the upstream `_decode_step`. Two upstream behaviours are kept
# deliberately: the second atom is chosen by argmax even when `random_sample` is
# true, and the loop stops at `MAX_DECODE_STEPS`. Changing either makes this a
# different sampler from the paper's.

# %%
def _rand_action(append_i, connect_i, end_i):
    """One draw from the joint distribution over append / connect / end for one
    molecule. `Policy` already normalises jointly, so the renormalisation here is
    a guard against drift, not part of the model."""
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

def _greedy_action(append_i, connect_i, end_i):
    _argmax = lambda _x: np.unravel_index(np.argmax(_x), _x.shape)
    append_id, append_val = _argmax(append_i), np.max(append_i)
    connect_id, connect_val = _argmax(connect_i), np.max(connect_i)
    if end_i >= append_val and end_i >= connect_val:
        return 2, None
    if append_val >= connect_val:
        return 0, append_id
    return 1, connect_id

# %%

def _decode_step(X, A, NX, NA, last_action, finished, get_init, get_action,
                 random_sample=True, n_node_types=None, n_edge_types=None):
    n_node_types = n_node_types or get_mol_spec().num_atom_types
    n_edge_types = n_edge_types or get_mol_spec().num_bond_types

    if X is None:
        init = get_init()
        if random_sample:
            X = np.array([np.random.choice(np.arange(init.shape[1]), 1, p=init[i, :])[0]
                          for i in range(init.shape[0])], dtype=np.int64)
        else:
            X = np.argmax(init, axis=1)
        A = np.zeros((0, 3), dtype=np.int64)
        NX = last_action = np.ones([X.shape[0]], dtype=np.int64)
        NA = np.zeros([X.shape[0]], dtype=np.int64)
        finished = np.zeros([X.shape[0]], dtype=bool)
        return X, A, NX, NA, last_action, finished

    unfin = np.logical_not(finished)
    X_u = X[np.repeat(unfin, NX)]
    A_u = A[np.repeat(unfin, NA), :]
    NX_u = NX[unfin]
    NA_u = NA[unfin]
    last_action_u = last_action[unfin]

    mol_ids_rep = NX_rep = np.repeat(np.arange(NX_u.shape[0]), NX_u)

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

    append, connect, end = get_action([X_u, A_u, NX_u, NX_rep, mask, mol_ids_rep])

    if A.shape[0] == 0:
        # Second atom. Upstream takes the argmax here regardless of `random_sample`;
        # diversity at this step comes entirely from the sampled first atom.
        max_index = np.argmax(np.reshape(append, [-1, n_node_types * n_edge_types]),
                              axis=1)
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
        if random_sample:
            action_type, action_index = _rand_action(append_i, connect_i, end_i)
        else:
            action_type, action_index = _greedy_action(append_i, connect_i, end_i)

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
# ## Sampler

# %%
class CTDDGSampler(object):
    def __init__(self, ckpt_path, device, config_fallback=None,
                 d_receptive=D_RECEPTIVE):
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = state['model'] if isinstance(state, dict) and 'model' in state else state

        configs = state.get('configs') if isinstance(state, dict) else None
        if configs is None and config_fallback and os.path.isfile(config_fallback):
            with open(config_fallback) as f:
                configs = json.load(f)
        if configs is None:
            raise RuntimeError(f"No configs in {ckpt_path} and no configs.json "
                               f"beside it.")

        self.meta = {k: state.get(k) for k in
                     ('fold', 'epoch', 'global_counter', 'val_loss', 'val_metric',
                      'N_C', 'lambda_tox', 'tox_floor', 'smoke_test', 'saved_at')} \
            if isinstance(state, dict) else {}
        self.objective = state.get('objective') if isinstance(state, dict) else None
        self.data = state.get('data') if isinstance(state, dict) else None

        self.mdl = CVanillaMolGen_RNN(get_mol_spec().num_atom_types,
                                      get_mol_spec().num_bond_types,
                                      D=d_receptive, **configs)
        self.mdl.load_state_dict(sd, strict=True)
        self.mdl.to(device).eval()
        self.device = device
        self.configs = configs
        self.N_C = int(configs['N_C'])
        self.n_truncated = 0
        # Accumulated across the whole run. Kept here rather than in `Policy`, which
        # must stay byte-identical across notebooks.
        self.logit_absmax = 0.0
        self.n_extreme = 0.0
        self.n_policy_calls = 0

    def _note_logits(self):
        p = self.mdl.policy_h
        if p.last_logit_absmax is not None:
            self.logit_absmax = max(self.logit_absmax, p.last_logit_absmax)
            self.n_extreme += p.last_frac_extreme
            self.n_policy_calls += 1

    @property
    def frac_extreme(self):
        return self.n_extreme / max(1, self.n_policy_calls)

    def _to_tensor(self, inputs):
        X, A, NX, NX_rep, mask, _ = inputs
        NX_cum = np.cumsum(NX)
        dev = self.device
        to_long = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.long,
                                            device=dev)
        X_t, NX_t, NX_rep_t, mask_t, NX_cum_t = (to_long(X), to_long(NX),
                                                 to_long(NX_rep), to_long(mask),
                                                 to_long(NX_cum))
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
                idx, val, (n_nodes, n_nodes), dtype=torch.float32,
                device=dev).coalesce())
        return X_t, A_sparse, NX_t, NX_rep_t, mask_t, NX_cum_t

    @torch.no_grad()
    def first_atom_distribution(self, c):
        """FNN_0's output for one protein. eval() mode, so BatchNorm uses its
        running statistics and a batch of one is well-defined."""
        self.mdl.mode = 'decode_0'
        _c = torch.as_tensor(np.asarray(c, dtype=np.float32)[None, :],
                             dtype=torch.float32, device=self.device)
        p = self.mdl(_c).cpu().numpy()[0].astype(np.float64)
        return p / p.sum()

    @torch.no_grad()
    def sample(self, num_samples, c, output_type='graph', sanitize=True,
               random_sample=True):
        """c: a single (N_C,) embedding, broadcast to every sample."""
        c = np.asarray(c, dtype=np.float32)
        if c.ndim == 1:
            c = np.stack([c] * num_samples, axis=0)
        if c.shape[1] != self.N_C:
            raise ValueError(
                f"Conditioning vector is {c.shape[1]}-d but the model expects "
                f"{self.N_C}-d. Average-pooled ProSE/Bepler embeddings are 6165-d; "
                f"ProtTrans-BERT is 1024-d.")

        finished = np.zeros([num_samples], dtype=bool)

        def get_init():
            self.mdl.mode = 'decode_0'
            _c = torch.as_tensor(c, dtype=torch.float32, device=self.device)
            p = self.mdl(_c).cpu().numpy().astype(np.float64)
            return p / p.sum(axis=1, keepdims=True)

        X, A, NX, NA, last_action, finished = _decode_step(
            None, None, None, None, None, finished,
            get_init=get_init, get_action=None, random_sample=random_sample,
            n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B)

        h = np.zeros([self.mdl.N_rnn, num_samples, self.mdl.F_c[-1]], dtype=np.float32)
        count = 1

        while not np.all(finished) and count < MAX_DECODE_STEPS:
            unfin = np.logical_not(finished)

            def get_action(inputs):
                self.mdl.mode = 'decode_step'
                _h = torch.as_tensor(h[:, unfin, :], dtype=torch.float32,
                                     device=self.device).contiguous()
                _c = torch.as_tensor(c[unfin, :], dtype=torch.float32,
                                     device=self.device)
                _X, _A, _NX, _NX_rep, _mask, _NX_cum = self._to_tensor(inputs)
                _ap, _cn, _en, _h_new = self.mdl(
                    _X, _A, _NX, _NX_rep, _mask, _NX_cum, _h, _c, _NX_rep)
                h[:, unfin, :] = _h_new.cpu().numpy()
                self._note_logits()
                return (_ap.cpu().numpy(), _cn.cpu().numpy(), _en.cpu().numpy())

            X, A, NX, NA, last_action, finished = _decode_step(
                X, A, NX, NA, last_action, finished,
                get_init=None, get_action=get_action, random_sample=random_sample,
                n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B)
            count += 1

        n_trunc = int(np.sum(np.logical_not(finished)))
        self.n_truncated += n_trunc

        cum_X = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')).tolist()
        cum_A = np.cumsum(np.pad(NA, [[1, 0]], mode='constant')).tolist()
        graph_list = [[X[xa:xb], A[aa:ab, :]]
                      for aa, ab, xa, xb in zip(cum_A[:-1], cum_A[1:],
                                                cum_X[:-1], cum_X[1:])]

        if output_type == 'graph':
            return graph_list, n_trunc
        mol_list = get_mol_from_graph_list(graph_list, sanitize)
        if output_type == 'mol':
            return mol_list, n_trunc
        if output_type == 'smiles':
            return ([Chem.MolToSmiles(m) if m is not None else None
                     for m, _ in mol_list], n_trunc)
        raise ValueError(f"Unrecognized output type: {output_type}")


# %% [markdown]
# ## Load the test bundle

# %%
bundle = np.load(TEST_BUNDLE, allow_pickle=True)
C_test = bundle['embeddings'].astype(np.float32)
test_rows = bundle['test_protein_rows'].tolist()
known_test_smiles = [str(s) for s in bundle['smiles'].tolist()]
bundle_smoke = bool(bundle['smoke_test']) if 'smoke_test' in bundle.files else False
n_total_proteins = int(bundle['n_held_out_proteins_total']) if \
    'n_held_out_proteins_total' in bundle.files else C_test.shape[0]

print(f"test bundle: {C_test.shape[0]} proteins x {C_test.shape[1]}-d | "
      f"{len(test_rows)} with pairs | {len(known_test_smiles)} known ligands")

check("test bundle is not a smoke subset", not bundle_smoke,
      "written by a full run" if not bundle_smoke else
      "this bundle came from a LIMIT_PAIRS run and is NOT the paper's test set",
      level="warn")
check("test bundle covers every held-out protein",
      len(test_rows) == n_total_proteins,
      f"{len(test_rows)} of {n_total_proteins} proteins", level="warn")

# %% [markdown]
# ## Load the checkpoint

# %%
sampler = CTDDGSampler(CKPT_PATH, CTX, config_fallback=CKPT_CONFIG)
print(f"checkpoint: {CKPT_PATH}")
for k, v in sampler.meta.items():
    if v is not None:
        print(f"  {k:16s} {v}")

check("conditioning width matches the bundle", sampler.N_C == C_test.shape[1],
      f"model expects {sampler.N_C}-d, bundle holds {C_test.shape[1]}-d")

check("checkpoint is not a smoke artifact", not sampler.meta.get('smoke_test'),
      "smoke_test flag is False" if not sampler.meta.get('smoke_test')
      else "this checkpoint is tagged smoke_test — its samples are not reportable",
      level="warn")

_vm = sampler.meta.get('val_metric')
check("model selection metric", _vm in (None, 'l_nontoxic'),
      f"selected on {_vm!r}"
      + ("" if _vm in (None, 'l_nontoxic') else
         " — `nll` RISES when toxicity suppression works, so selecting on it picks "
         "the LEAST-trained checkpoint. Re-run 02 with EARLY_STOP_ON = 'l_nontoxic'"),
      level="warn")

check("not stuck at epoch 1", sampler.meta.get('epoch') not in (1,),
      f"best epoch {sampler.meta.get('epoch')}"
      + ("" if sampler.meta.get('epoch') not in (1,) else
         " — validation peaked immediately and degraded after; see "
         "05_training_runs_and_diagnostics.md before reading anything into these "
         "metrics"),
      level="warn")

_tp = (sampler.data or {}).get('traversal_policy')
check("the checkpoint was trained under a complete traversal", _tp == 'complete_dfs',
      f"traversal_policy = {_tp!r}"
      + ("" if _tp == 'complete_dfs' else
         " — targets in which ~21% of traversals left atoms unvisited and those "
         "molecules' bonds were mis-wired. Samples from this checkpoint measure the "
         "bug, not the method. See AUDIT.md A21"),
      level="warn" if _tp is None else ("fatal" if _tp != 'complete_dfs' else "warn"))

# %% [markdown]
# ## Numerics parity
#
# The single most important check here. `STABLE_SOFTMAX` is taken from the
# checkpoint, not from this notebook's config, and then the two normalisations are
# compared on the model's own logits. Sampling under a normalisation the model was
# not trained under is silent: every shape matches, every probability sums to one,
# and the samples come from a different distribution.

# %%
_ckpt_stable = (sampler.objective or {}).get('stable_softmax')
if _ckpt_stable is None:
    check("checkpoint records its softmax", False,
          f"the checkpoint predates the field; falling back to STABLE_SOFTMAX = "
          f"{STABLE_SOFTMAX}, which is a GUESS. Only the stage-2 source settles it",
          level="warn")
else:
    if bool(_ckpt_stable) != STABLE_SOFTMAX:
        print(f"  STABLE_SOFTMAX {STABLE_SOFTMAX} -> {bool(_ckpt_stable)} "
              f"(from the checkpoint)")
    STABLE_SOFTMAX = bool(_ckpt_stable)
    check("sampling matches the trained normalisation", True,
          f"STABLE_SOFTMAX = {STABLE_SOFTMAX}, read from the checkpoint")

# %%

def softmax_parity(sampler, n_atoms=24, n_mol=3, seed=0):
    """Compare the two normalisations on real `Policy` logits.

    `Policy` returns probabilities when return_log=False and log-probabilities
    otherwise; both go through `action_log_probs`, so exp(log) == prob is a
    self-consistency check, and legacy-vs-stable is the one that matters.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    F = sampler.mdl.F_c[-1]
    X = torch.randn(n_atoms, F, generator=g).to(sampler.device)
    NX = torch.full((n_mol,), n_atoms // n_mol, dtype=torch.long,
                    device=sampler.device)
    NX_rep = torch.arange(n_mol, device=sampler.device).repeat_interleave(NX)
    X_mol = torch.randn(n_mol, F, generator=g).to(sampler.device)

    pol = sampler.mdl.policy_h
    with torch.no_grad():
        global STABLE_SOFTMAX
        keep = STABLE_SOFTMAX
        try:
            STABLE_SOFTMAX = True
            ap_s, cn_s, en_s = pol(X, NX, NX_rep, X_mol)
            lap, lcn, len_ = pol(X, NX, NX_rep, X_mol, return_log=True)
            STABLE_SOFTMAX = False
            ap_l, cn_l, en_l = pol(X, NX, NX_rep, X_mol)
        finally:
            STABLE_SOFTMAX = keep

    def total(ap, cn, en):
        flat = torch.cat([ap.reshape(n_atoms, -1), cn.reshape(n_atoms, -1)], dim=1)
        return segment_sum(flat, NX_rep, n_mol).sum(-1) + en.reshape(n_mol, -1).sum(-1)

    return {
        'mass_stable': total(ap_s, cn_s, en_s).cpu().numpy(),
        'mass_legacy': total(ap_l, cn_l, en_l).cpu().numpy(),
        'exp_log_vs_prob': float(max((lap.exp() - ap_s).abs().max(),
                                     (lcn.exp() - cn_s).abs().max(),
                                     (len_.exp() - en_s).abs().max())),
        'stable_vs_legacy': float(max((ap_s - ap_l).abs().max(),
                                      (cn_s - cn_l).abs().max(),
                                      (en_s - en_l).abs().max())),
        'logit_absmax': pol.last_logit_absmax,

    }

# %%
_par = softmax_parity(sampler)
print(f"  total action mass per molecule: stable "
      f"{np.round(_par['mass_stable'], 6).tolist()} | legacy "

      f"{np.round(_par['mass_legacy'], 6).tolist()}")

print(f"  max |logit| on this probe: {_par['logit_absmax']:.1f}")

check("the action distribution normalises",
      bool(np.abs(_par['mass_stable'] - 1.0).max() < 1e-4),

      f"max deviation from 1.0 is {np.abs(_par['mass_stable'] - 1.0).max():.2e}")
check("exp(log-prob) == prob",
      _par['exp_log_vs_prob'] < 1e-5,
      f"max difference {_par['exp_log_vs_prob']:.2e} — the same code path serves "

      f"training (log) and sampling (prob)")
check("stable and legacy agree at ordinary logit scale",
      _par['stable_vs_legacy'] < 1e-5,
      f"max difference {_par['stable_vs_legacy']:.2e} at max |logit| "
      f"{_par['logit_absmax']:.1f}. The two normalisations are identical in exact "
      f"arithmetic; they diverge only once the +/-80 clamp binds", level="warn")

# %% [markdown]
# ### First-atom distribution
#
# `dense_policy_0` is new at fine-tuning time, and `loss_init` is one term out of
# ~34 per molecule, so it learns slowly. An entropy near `log(N_A)` means it is
# still near-uniform over every atom type, which starts a sizeable share of
# molecules on an exotic or charged atom.

# %%
_p0 = sampler.first_atom_distribution(C_test[test_rows[0]])
_ent = float(-(_p0 * np.log(_p0 + 1e-12)).sum())
_unif = math.log(len(_p0))
_top = [(get_mol_spec().atom_types[i], round(float(_p0[i]), 4))
        for i in np.argsort(-_p0)[:8]]

print(f"  entropy {_ent:.3f}  (uniform over {len(_p0)} types = {_unif:.3f})")
print(f"  top-8 first atoms: {_top}")

check("first-atom distribution is informative", _ent < 0.85 * _unif,
      f"entropy {_ent:.2f} vs uniform {_unif:.2f}"
      + ("" if _ent < 0.85 * _unif else
         f" — near-uniform over all {len(_p0)} atom types, so molecules routinely "
         f"start on an exotic or charged atom. dense_policy_0 is undertrained"),
      level="warn")

# Does the first-atom distribution depend on the protein at all? If FNN_0's output
# weight is still ~zero (the stage-2 warm start sets it to zero deliberately), every
# protein gets the same distribution and the conditioning does nothing at step 0.

# %%
_p0_all = np.stack([sampler.first_atom_distribution(C_test[r])
                    for r in test_rows[:min(8, len(test_rows))]], axis=0)
_spread = float(np.abs(_p0_all - _p0_all.mean(axis=0, keepdims=True)).max())
print(f"  across {_p0_all.shape[0]} proteins, max deviation from the mean "
      f"first-atom distribution: {_spread:.2e}")
check("the first atom depends on the protein", _spread > 1e-4,
      f"max spread {_spread:.2e}"
      + ("" if _spread > 1e-4 else
         " — every protein gets the SAME first-atom distribution. FNN_0's output "
         "weight has not moved from its zero warm start, so step 0 is unconditional"),
      level="warn")

gate("generation")

# %% [markdown]
# ## Generate
#
# Resumable: a `(protein, run)` CSV already on disk is skipped unless `OVERWRITE`.
# Every requested sample gets a row; invalid ones carry an empty SMILES and their
# failure reason.

# %%
FAILURES = Counter()

# %%

def sample_one(protein_row, run, n_samples=N_SAMPLES):
    seed = BASE_SEED + run * 100003 + int(protein_row)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)

    t0 = time.time()
    graphs, n_trunc = sampler.sample(n_samples, c=C_test[protein_row],
                                     output_type='graph')
    results = get_mol_from_graph_list(graphs, sanitize=True)
    elapsed = time.time() - t0

    rows = []
    for i, (m, reason) in enumerate(results):
        smi = ''
        if m is not None:
            try:
                smi = Chem.MolToSmiles(m)
            except Exception as e:
                smi = ''
                reason = f"MolToSmiles: {type(e).__name__}"
        FAILURES[reason.split(':')[0]] += 1
        rows.append({'sample_idx': i, 'smiles': smi,
                     'n_atoms': int(graphs[i][0].shape[0]) if i < len(graphs) else 0,
                     'reason': reason})
    return rows, elapsed, n_trunc

# %%

def write_rows(path, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample_idx', 'smiles', 'n_atoms', 'reason'])
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


# %% [markdown]
# ### Smoke pass
#
# One protein, `SMOKE_N` molecules, with the failure census. Minutes instead of the
# full run's hours, and it is where the diagnosis is: "26/100 valid" says nothing;
# a histogram of `KekulizeException` against `AtomValenceException` against
# graph-build failures says where to look.

# %%
if SMOKE_FIRST:
    _row = test_rows[0]
    print(f"  smoke: protein row {_row}, {SMOKE_N} molecules")
    _before = Counter(FAILURES)
    rows, elapsed, n_trunc = sample_one(_row, run=0, n_samples=SMOKE_N)
    _delta = Counter(FAILURES)
    _delta.subtract(_before)

    n_valid = sum(1 for r in rows if r['smiles'])
    validity = 100.0 * n_valid / SMOKE_N
    print(f"\n  {n_valid}/{SMOKE_N} valid ({validity:.1f}%) in {elapsed:.1f}s")
    print(f"  {n_trunc} hit the {MAX_DECODE_STEPS}-step cap without emitting `end`")
    print("\n  failure census:")
    for k, v in _delta.most_common():
        if v > 0:
            print(f"    {v:>5}  ({100.0 * v / SMOKE_N:5.1f}%)  {k}")

# %%
if SMOKE_FIRST:
    _av = [r['n_atoms'] for r in rows if r['smiles']]
    _ai = [r['n_atoms'] for r in rows if not r['smiles']]
    if _av:
        print(f"  valid   mean {np.mean(_av):5.1f} atoms  max {max(_av)}")
    if _ai:
        _near = sum(1 for a in _ai if a >= MAX_DECODE_STEPS - 5) / len(_ai)
        print(f"  invalid mean {np.mean(_ai):5.1f} atoms  max {max(_ai)}  "
              f"| {100 * _near:.0f}% within 5 atoms of the cap")
        check("truncation is not the cause", _near < 0.25,
              f"{100 * _near:.0f}% of invalid molecules sit near the decode cap"
              + (f" — raise MAX_DECODE_STEPS above {MAX_DECODE_STEPS}"
                 if _near >= 0.25 else ""), level="warn")

    print(f"  logits: max |z| {sampler.logit_absmax:.1f}, "
          f"{100 * sampler.frac_extreme:.5f}% beyond +/-80")
    check("the smoke pass stays inside the clamp range",
          sampler.frac_extreme < 1e-5,
          f"{100 * sampler.frac_extreme:.5f}% of logits exceeded +/-80"
          + ("" if sampler.frac_extreme < 1e-5 else
             " — this is where the two normalisations diverge, so these samples "
             "depend on which one was used"), level="warn")

    check("smoke validity", validity >= 80.0,
          f"{validity:.1f}% valid. "
          + ("Healthy." if validity >= 80.0 else
             f"The full run will reproduce this. Stop here rather than generating "
             f"{len(test_rows) * N_RUNS * N_SAMPLES:,} molecules at this rate — run "
             f"00_phase0_diagnostics check A to find out whether stage 1 or stage 2 "
             f"is responsible."), level="warn")

    print("\n  Set SMOKE_FIRST = False to run the full protocol.")

# %% [markdown]
# ## Full run

# %%
targets = test_rows if LIMIT_PROTEINS is None else test_rows[:LIMIT_PROTEINS]
timing = []

if SMOKE_FIRST:
    print(f"SMOKE_FIRST is True — the full run ("
          f"{len(targets) * N_RUNS * N_SAMPLES:,} molecules) was NOT started.")
else:
    print(f"{len(targets)} proteins x {N_RUNS} runs x {N_SAMPLES} samples = "
          f"{len(targets) * N_RUNS * N_SAMPLES:,} molecules")

    for run in range(1, N_RUNS + 1):
        run_dir = os.path.join(OUT_DIR, f"run{run}")
        os.makedirs(run_dir, exist_ok=True)
        t_run = time.time()
        n_done = 0

        for j, row in enumerate(targets, 1):
            out_csv = os.path.join(run_dir, f"protein_{row:04d}.csv")
            if os.path.isfile(out_csv) and not OVERWRITE:
                continue
            rows, elapsed, n_trunc = sample_one(row, run)
            write_rows(out_csv, rows)
            timing.append({'run': run, 'protein_row': row, 'n_samples': N_SAMPLES,
                           'seconds': elapsed, 'n_truncated': n_trunc})
            n_done += 1
            if j % 10 == 0 or j == len(targets):
                n_valid = sum(1 for r in rows if r['smiles'])
                print(f"  run {run} | {j}/{len(targets)} | last protein {row}: "
                      f"{n_valid}/{N_SAMPLES} valid in {elapsed:.1f}s", flush=True)
            if j % 25 == 0:
                gc.collect()

        print(f"run {run} done ({n_done} newly sampled) in "
              f"{(time.time() - t_run) / 60:.1f} min")

# %% [markdown]
# ## Failure census

# %%
if not SMOKE_FIRST and timing:
    tdf = pd.DataFrame(timing)
    tdf.to_csv(os.path.join(OUT_DIR, "timing.csv"), index=False)
    print(f"mean {tdf['seconds'].mean():.2f}s per {N_SAMPLES} molecules "
          f"({tdf['seconds'].mean() / N_SAMPLES * 1000:.0f} ms each)")
    print(f"{int(tdf['n_truncated'].sum()):,} molecules hit the "
          f"{MAX_DECODE_STEPS}-step cap without emitting `end`")

    total = sum(FAILURES.values())
    print(f"\nfailure census over {total:,} samples:")
    census = []
    for k, v in FAILURES.most_common():
        print(f"  {v:>7,}  ({100 * v / max(1, total):5.2f}%)  {k}")
        census.append({'reason': k, 'count': v, 'pct': 100 * v / max(1, total)})
    pd.DataFrame(census).to_csv(os.path.join(OUT_DIR, "failure_census.csv"),
                                index=False)

    print(f"\nlogits: max |z| {sampler.logit_absmax:.1f}, "
          f"{100 * sampler.frac_extreme:.5f}% beyond +/-80")
    check("the logit clamp does not bind", sampler.frac_extreme < 1e-5,
          f"{100 * sampler.frac_extreme:.5f}% of logits exceeded +/-80"
          + ("" if sampler.frac_extreme < 1e-5 else
             " — the legacy normalisation flattens the action distribution there, so "
             "these samples depend on which normalisation ran. Re-run with "
             "STABLE_SOFTMAX = True in ALL notebooks"), level="warn")

# %% [markdown]
# ## Manifest

# %%
if not SMOKE_FIRST:
    manifest = {
        'fold': FOLD,
        'checkpoint': CKPT_PATH,
        'checkpoint_sha256': hashlib.sha256(
            open(CKPT_PATH, 'rb').read()).hexdigest(),
        'checkpoint_meta': {k: (v if isinstance(v, (str, int, float, bool, type(None)))
                                else str(v)) for k, v in sampler.meta.items()},
        'checkpoint_objective': sampler.objective,
        'checkpoint_data': sampler.data,
        'stable_softmax': STABLE_SOFTMAX,
        'test_bundle': TEST_BUNDLE,
        'bundle_smoke_test': bundle_smoke,
        'n_proteins_sampled': len(targets),
        'n_held_out_proteins_total': n_total_proteins,
        'protein_rows': [int(r) for r in targets],
        'n_samples_per_protein': N_SAMPLES,
        'n_runs': N_RUNS,
        'max_decode_steps': MAX_DECODE_STEPS,
        'max_iter_train': MAX_ITER_TRAIN,
        'base_seed': BASE_SEED,
        'N_C': sampler.N_C,
        'device': str(CTX),
        'first_atom_entropy': round(_ent, 4),
        'first_atom_uniform_entropy': round(_unif, 4),
        'first_atom_protein_spread': float(_spread),
        'logit_absmax': float(sampler.logit_absmax),
        'frac_logits_beyond_80': float(sampler.frac_extreme),
        'n_truncated_total': int(sum(t['n_truncated'] for t in timing)) if timing else 0,
        'failure_census': dict(FAILURES.most_common()),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=1, default=str)

    np.savez_compressed(os.path.join(OUT_DIR, "reference_ligands.npz"),
                        smiles=np.array(known_test_smiles, dtype=object))

    print(f"samples   {OUT_DIR}")
    print(f"manifest  {os.path.join(OUT_DIR, 'manifest.json')}")

# %% [markdown]
# ## Summary

# %%
print("=" * 78)
_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
for status, name, detail in sorted(CHECKS, key=lambda c: _order.get(c[0], 9)):
    print(f"  {status:5s}  {name}")
    print(f"         {detail}")
print("-" * 78)
print("Next: 04_evaluate_metrics." if not SMOKE_FIRST else
      "Set SMOKE_FIRST = False once the smoke pass looks right.")

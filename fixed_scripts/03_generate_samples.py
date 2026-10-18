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
# # 03 - Sample molecules from the fine-tuned CTDDG checkpoint
#
# PyTorch conditional sampler. Replaces the MXNet `CVanilla_RNN_Builder`, which
# cannot load `ckpt_best.pt`.
#
# **Inputs** `outputs/finetune/<fold>/ckpt_best.pt` + `test_set.npz` (written by
# `02_finetune_ctddg.py`). The bundle carries the `[n_proteins, 6165]` ProSE/Bepler
# matrix in the exact row order fine-tuning used, so conditioning cannot drift.
#
# Do **not** feed this ProtTrans-BERT `.txt` embeddings: those are 1024-d, this
# model is 6165-d, and they were built from a different fold.
#
# **Output** `outputs/samples/<fold>/run<r>/protein_<row>.csv`, one row per
# *requested* sample. Invalid samples are kept as empty SMILES so validity stays
# measurable - do not dedup or resample here.
#
# ## What changed
#
# | # | Change | Reason |
# |---|---|---|
# | 1 | **Per-sample failure reason recorded** | The previous version reported only "26/100 valid". *Why* a molecule fails - graph build, kekulize, valence - is what localises the problem, and it costs nothing to record. |
# | 2 | **First-atom distribution reported** | `dense_policy_0` is new at fine-tuning time and starts near-uniform over all 65 atom types while the pretrained `policy_0` is discarded. Its entropy is printed so an undertrained init is visible before 104,000 molecules are generated. |
# | 3 | **`MAX_DECODE_STEPS` is a deliberate choice, with the evidence next to it** | It is 100 upstream while training allows 120. `00_phase0_diagnostics.py` check C measures whether that matters; the default stays at the upstream value. |
# | 4 | **Checkpoint selection metric surfaced loudly** | A checkpoint chosen on `val_nll` is the least-trained one, because a working CTDDG drives that metric up. |
# | 5 | **A smoke run is offered before the full 104,000** | One protein, one run, with the failure census - minutes instead of 17. |

# %%
import os
import gc
import csv
import json
import math
import time
import random
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

GPU_ID = int(os.environ.get("CTDDG_GPU", "0"))
CTX = torch.device(f"cuda:{GPU_ID}") if torch.cuda.is_available() else torch.device("cpu")

# Paper protocol: 10 runs of 100 molecules for each of the 104 held-out proteins.
N_SAMPLES = int(os.environ.get("CTDDG_N_SAMPLES", "100"))
N_RUNS = int(os.environ.get("CTDDG_N_RUNS", "10"))
LIMIT_PROTEINS = os.environ.get("CTDDG_LIMIT_PROTEINS")
LIMIT_PROTEINS = int(LIMIT_PROTEINS) if LIMIT_PROTEINS else None

# The upstream sampler's cap. Training allows MAX_ITER = 120 steps; the published
# sampler stops at 100, so very large molecules are truncated mid-build.
#
# Keep 100 unless check C of 00_phase0_diagnostics.py shows invalid molecules
# clustering within a few atoms of the cap. Raising it makes this a different
# sampler from the paper's, which is a deviation worth recording rather than
# making silently.
MAX_DECODE_STEPS = int(os.environ.get("CTDDG_MAX_DECODE_STEPS", "100"))
MAX_ITER_TRAIN = 120

D_RECEPTIVE = 2
BASE_SEED = 42
OVERWRITE = False          # False = skip (protein, run) pairs already on disk

# Generate one protein x one run first, print the failure census, and stop.
# Flip to False for the full protocol.
SMOKE_FIRST = True
SMOKE_N = 200              # molecules for the smoke pass

os.makedirs(OUT_DIR, exist_ok=True)

CHECKS = []


def check(name, ok, detail, warn_only=False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    CHECKS.append({"name": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")
    if not ok and not warn_only:
        raise RuntimeError(f"{name}: {detail}")
    return ok


print(f"device {CTX} | fold {FOLD} | {N_RUNS} runs x {N_SAMPLES} samples")
check("decode cap vs training cap", MAX_DECODE_STEPS == 100,
      f"MAX_DECODE_STEPS = {MAX_DECODE_STEPS}; training allowed {MAX_ITER_TRAIN}. "
      + ("100 is the upstream sampler's value" if MAX_DECODE_STEPS == 100
         else "this is a DEVIATION from the published sampler - record it"),
      warn_only=True)

# %% [markdown]
# ## Molecule spec and graph utilities
#
# Identical to both training notebooks. Atom-type identity is a line number in
# `atom_types.txt`, so this must be the file the checkpoint was trained with.

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
        self.max_iter = MAX_ITER_TRAIN

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
    """Distance-2 and distance-3 neighbourhoods, as in training."""
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
    """-> (mol | None, reason).

    `reason` is new: 'valid', 'graph build: ...', or the RDKit exception class and
    message. Recording it is what turns "26/100 valid" into a diagnosis.
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


def get_mol_from_graph_list(graph_list, sanitize=True):
    return [get_mol_from_graph(X, A, sanitize) for X, A in graph_list]


check("atom vocabulary", get_mol_spec().num_atom_types > 0,
      f"{get_mol_spec().num_atom_types} atom types, "
      f"{get_mol_spec().num_bond_types} bond types")

# %% [markdown]
# ## Model
#
# The same classes as `02_finetune_ctddg.py`, minus the training-only paths, so
# `ckpt_best.pt` loads strictly by name.

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
        # No autograd.Function needed - sampling runs under no_grad.
        return _concat_neighbourhoods(X, A_list) @ self.W


class Policy(nn.Module):
    def __init__(self, F_in, F_h, N_A, N_B, k=1):
        super(Policy, self).__init__()
        self.F_in, self.F_h, self.N_A, self.N_B, self.k = F_in, F_h, N_A, N_B, k
        self.linear_h = Linear_BN(F_in * 2, self.F_h * k)
        self.linear_h_t = Linear_BN(F_in, self.F_h * k)
        self.linear_x = nn.Linear(self.F_h, self.N_B + self.N_B * self.N_A)
        self.linear_x_t = nn.Linear(self.F_h, 1)
        self.linear_pi = nn.Linear(self.F_in, self.k) if self.k > 1 else None
        self.logit_absmax = 0.0
        self.n_clamped = 0
        self.n_logits = 0

    def forward(self, X, NX, NX_rep, X_end=None):
        if X_end is None:
            X_end = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X, X_end[NX_rep, :]], dim=1)

        X_h = torch.relu(self.linear_h(X)).reshape(-1, self.F_h)
        X_h_end = torch.relu(self.linear_h_t(X_end)).reshape(-1, self.F_h)

        raw_x = self.linear_x(X_h).reshape(-1, self.k, self.N_B + self.N_B * self.N_A)
        raw_x_t = self.linear_x_t(X_h_end).reshape(-1, self.k, 1)

        # Accumulated across the whole run so the clamp can be audited afterwards.
        self.logit_absmax = max(self.logit_absmax,
                                float(max(raw_x.abs().max(), raw_x_t.abs().max())))
        self.n_clamped += int((raw_x.abs() >= 80.0).sum()) + \
            int((raw_x_t.abs() >= 80.0).sum())
        self.n_logits += raw_x.numel() + raw_x_t.numel()

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
        self.mode = 'decode_step'

    @abstractmethod
    def _build_graph_conv(self, *args, **kwargs):
        pass


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


class _TwoLayerDense(nn.Module):
    """FNN_0: protein embedding -> first-atom distribution."""

    def __init__(self, input_size, hidden_size, output_size):
        super(_TwoLayerDense, self).__init__()
        self.input = nn.Linear(input_size, hidden_size, bias=False)
        self.bn_input = nn.BatchNorm1d(hidden_size, eps=1e-5, momentum=0.1)
        self.output = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, c):
        return torch.softmax(self.output(torch.relu(self.bn_input(self.input(c)))),
                             dim=-1)


class CMoleculeGenerator_RNN(MoleculeGenerator_RNN):
    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, N_C, D, F_e, F_skip, F_c, Fh_policy, activation,
                 N_rnn, *args, **kwargs):
        self.N_C = N_C
        super(CMoleculeGenerator_RNN, self).__init__(
            N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, *args, **kwargs)
        self.dense_policy_0 = _TwoLayerDense(self.N_C, self.N_A * 3, self.N_A)

    @abstractmethod
    def _graph_conv_forward(self, X, A, c, ids):
        pass

    def _policy_0(self, c):
        return self.dense_policy_0(c) + 0.0 * self.policy_0

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
            if i == 0:
                X_out.append(conv(X, A) + c_i)
            else:
                X_out.append(conv(self.activation(self.bn[i - 1](X)), A) + c_i)
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


def _rand_action(append_i, connect_i, end_i):
    """One draw from the joint softmax over append / connect / end for one molecule."""
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


def _greedy_action(append_i, connect_i, end_i):
    _argmax = lambda _x: np.unravel_index(np.argmax(_x), _x.shape)
    append_id, append_val = _argmax(append_i), np.max(append_i)
    connect_id, connect_val = _argmax(connect_i), np.max(connect_i)
    if end_i >= append_val and end_i >= connect_val:
        return 2, None
    if append_val >= connect_val:
        return 0, append_id
    return 1, connect_id


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
            raise RuntimeError(f"No configs in {ckpt_path} and no configs.json beside it.")

        self.meta = {k: state.get(k) for k in
                     ('fold', 'epoch', 'global_counter', 'val_loss', 'val_metric',
                      'N_C', 'lambda_tox', 'tox_floor', 'smoke_test', 'saved_at')} \
            if isinstance(state, dict) else {}

        self.mdl = CVanillaMolGen_RNN(get_mol_spec().num_atom_types,
                                      get_mol_spec().num_bond_types,
                                      D=d_receptive, **configs)
        self.mdl.load_state_dict(sd, strict=True)
        self.mdl.to(device).eval()
        self.device = device
        self.configs = configs
        self.N_C = int(configs['N_C'])
        self.n_truncated = 0

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
                idx, val, (n_nodes, n_nodes), dtype=torch.float32, device=dev).coalesce())
        return X_t, A_sparse, NX_t, NX_rep_t, mask_t, NX_cum_t

    @torch.no_grad()
    def first_atom_distribution(self, c):
        """The FNN_0 output for one protein, for the init diagnostic."""
        self.mdl.mode = 'decode_0'
        _c = torch.as_tensor(np.stack([np.asarray(c, dtype=np.float32)] * 2, axis=0),
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
            return self.mdl(_c).cpu().numpy()

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
                return (_ap.cpu().numpy(), _cn.cpu().numpy(), _en.cpu().numpy())

            X, A, NX, NA, last_action, finished = _decode_step(
                X, A, NX, NA, last_action, finished,
                get_init=None, get_action=get_action, random_sample=random_sample,
                n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B)
            count += 1

        # Molecules that hit the cap without emitting `end` are truncated mid-build.
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
# ## Load the test bundle and the checkpoint

# %%
if not os.path.isfile(TEST_BUNDLE):
    raise FileNotFoundError(
        f"{TEST_BUNDLE} not found. It is written by 02_finetune_ctddg.py and carries "
        f"the held-out embeddings in fine-tuning row order. A `.smoke` variant is a "
        f"smoke subset, not the paper's test set.")

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
      warn_only=True)
check("test bundle covers every held-out protein",
      len(test_rows) == n_total_proteins,
      f"{len(test_rows)} of {n_total_proteins} proteins", warn_only=True)

sampler = CTDDGSampler(CKPT_PATH, CTX, config_fallback=CKPT_CONFIG)
print(f"\ncheckpoint: {CKPT_PATH}")
for k, v in sampler.meta.items():
    if v is not None:
        print(f"  {k:16s} {v}")

check("checkpoint is not a smoke artifact", not sampler.meta.get('smoke_test'),
      "smoke_test flag is False" if not sampler.meta.get('smoke_test')
      else "this checkpoint is tagged smoke_test - its samples are not reportable",
      warn_only=True)

_vm = sampler.meta.get('val_metric')
check("model selection metric", _vm in (None, 'l_nontoxic'),
      f"selected on {_vm!r}"
      + ("" if _vm in (None, 'l_nontoxic') else
         " - `nll` RISES when toxicity suppression works, so selecting on it picks "
         "the LEAST-trained checkpoint. Re-run 02 with EARLY_STOP_ON = 'l_nontoxic'"),
      warn_only=True)

check("not stuck at epoch 1", sampler.meta.get('epoch') not in (1,),
      f"best epoch {sampler.meta.get('epoch')}"
      + ("" if sampler.meta.get('epoch') not in (1,) else
         " - validation peaked immediately and degraded after; see "
         "05_training_runs_and_diagnostics.md before reading anything into these "
         "metrics"),
      warn_only=True)

# %% [markdown]
# ### First-atom distribution
#
# `dense_policy_0` is created fresh at fine-tuning time and the pretrained
# `policy_0` is multiplied by zero, so the init distribution has to be relearned
# from `loss_init` - one term out of ~34 per molecule. An entropy close to
# `log(65) = 4.17` means it is still near-uniform over all 65 atom types, which
# puts exotic and charged atoms at the start of a quarter of all molecules.

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
         " - near-uniform over all 65 atom types, so molecules routinely start on "
         "an exotic or charged atom. dense_policy_0 is undertrained"),
      warn_only=True)

# %% [markdown]
# ## Generate
#
# Resumable: a `(protein, run)` CSV already on disk is skipped unless `OVERWRITE`.
# Every requested sample gets a row; invalid ones carry an empty SMILES and their
# failure reason.

# %%
FAILURES = Counter()


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
# full run's ~17, and it is where the diagnosis actually is: "26/100 valid" says
# nothing, a histogram of `KekulizeException` versus `AtomValenceException` versus
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
            print(f"    {v:>5}  {k}")

    _av = [r['n_atoms'] for r in rows if r['smiles']]
    _ai = [r['n_atoms'] for r in rows if not r['smiles']]
    if _av:
        print(f"\n  valid   mean {np.mean(_av):5.1f} atoms  max {max(_av)}")
    if _ai:
        _near = sum(1 for a in _ai if a >= MAX_DECODE_STEPS - 5) / len(_ai)
        print(f"  invalid mean {np.mean(_ai):5.1f} atoms  max {max(_ai)}  "
              f"| {100 * _near:.0f}% within 5 atoms of the cap")
        check("truncation is not the cause", _near < 0.25,
              f"{100 * _near:.0f}% of invalid molecules sit near the decode cap"
              + (" - raise MAX_DECODE_STEPS to 120" if _near >= 0.25 else ""),
              warn_only=True)

    check("smoke validity", validity >= 80.0,
          f"{validity:.1f}% valid. "
          + ("Healthy." if validity >= 80.0 else
             "The full run will reproduce this. Stop here and fix the model rather "
             "than spending 17 minutes generating 104,000 molecules at this rate - "
             "run 00_phase0_diagnostics.py check A to find out whether stage 1 or "
             "stage 2 is responsible."),
          warn_only=True)

    print("\n  Set SMOKE_FIRST = False to run the full protocol.")

# %%
targets = test_rows if LIMIT_PROTEINS is None else test_rows[:LIMIT_PROTEINS]
timing = []

if SMOKE_FIRST:
    print("SMOKE_FIRST is True - skipping the full run.")
else:
    print(f"\n{len(targets)} proteins x {N_RUNS} runs x {N_SAMPLES} samples = "
          f"{len(targets) * N_RUNS * N_SAMPLES} molecules")

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
# ## Failure census and manifest

# %%
if not SMOKE_FIRST and timing:
    import pandas as pd

    tdf = pd.DataFrame(timing)
    tdf.to_csv(os.path.join(OUT_DIR, "timing.csv"), index=False)
    print(f"\nmean {tdf['seconds'].mean():.2f}s per {N_SAMPLES} molecules "
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

    _clamped = sampler.mdl.policy_h.n_clamped / max(1, sampler.mdl.policy_h.n_logits)
    print(f"\nlogits: max |z| {sampler.mdl.policy_h.logit_absmax:.1f}, "
          f"{100 * _clamped:.5f}% at the +/-80 clamp")
    check("the logit clamp does not bind", _clamped < 1e-5,
          f"{100 * _clamped:.5f}% of logits reached the clamp"
          + ("" if _clamped < 1e-5 else
             " - it flattens the action distribution and zeroes those gradients. "
             "Replace with a log-space max-shifted softmax in ALL THREE notebooks"),
          warn_only=True)

# %%
if not SMOKE_FIRST:
    manifest = {
        'fold': FOLD,
        'checkpoint': CKPT_PATH,
        'checkpoint_meta': {k: (v if isinstance(v, (str, int, float, bool, type(None)))
                                else str(v)) for k, v in sampler.meta.items()},
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
        'n_truncated_total': int(sum(t['n_truncated'] for t in timing)) if timing else 0,
        'failure_census': dict(FAILURES.most_common()),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=1)

    np.savez_compressed(os.path.join(OUT_DIR, "reference_ligands.npz"),
                        smiles=np.array(known_test_smiles, dtype=object))

    print(f"\nsamples   {OUT_DIR}")
    print(f"manifest  {os.path.join(OUT_DIR, 'manifest.json')}")
    print("Next: 04_evaluate_metrics.py")

# %%
print("=" * 70)
for c in CHECKS:
    print(f"  {c['status']:5s}  {c['name']}: {c['detail']}")

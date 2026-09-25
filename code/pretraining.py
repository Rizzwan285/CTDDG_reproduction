'''

import sys
print(sys.version)

import os
import sys
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, QED
from rdkit.Chem import rdmolfiles
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem

from rdkit import RDLogger

# Suppress RDKit warnings and errors
#RDLogger.DisableLog('rdApp.*')

mol = Chem.MolFromSmiles('CO[C@@H]1CC[C@@]2(CC1)Cc1c([C@]32COC(=[NH+]3)N)cc(cc1)c1cncc(c1)C')
mol

"""# Importing required packages"""

import os
import gc
import sys
import time
import json
import math
import random
import argparse
import itertools
import numpy as np
import mxnet as mx
import pandas as pd
import networkx as nx
from scipy import sparse
from mxnet.gluon import nn
from collections import Counter
from mxnet.autograd import Function
from mxnet.gluon.data import Dataset
from mxnet import gluon, autograd, nd
from mxnet.gluon.data import DataLoader
from abc import ABCMeta, abstractmethod
from mxnet.gluon.data.sampler import Sampler

"""### Testing the gpu on mxnet"""

import mxnet as mx

def check_gpu_availability():
    num_gpus = mx.context.num_gpus()
    if num_gpus > 0:
        print(f"GPU is available. Number of GPUs: {num_gpus}")
    else:
        print("No GPU available.")

check_gpu_availability()

import mxnet as mx
import numpy as np

# def matrix_multiplication_on_gpu():
#     ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

#     # Create random matrices on the GPU
#     a = mx.nd.random.uniform(shape=(1000, 1000), ctx=ctx)
#     b = mx.nd.random.uniform(shape=(1000, 1000), ctx=ctx)

#     # Perform matrix multiplication on the GPU
#     result = mx.nd.dot(a, b)

#     # Transfer the result back to CPU for printing (optional)
#     result_cpu = result.asnumpy()

#     print("Matrix multiplication result:")
#     print(result_cpu)

# matrix_multiplication_on_gpu()

"""# Initializing hyperparameters"""

batch_size = 8   # training batch size
batch_size_test = 8   # test batch size
k = 5   # number of generation paths
p = 0.8   # randomness parameter alpha
F_e = 16    # initial hidden embedding size for each node in a graph
F_h = [32,64,128,128,256,256]    # output sizes of each GCN layer
F_skip = 256    # size of skip connection layer
F_c = [512, ]   # hidden sizes of fully connected layers after graph convolution
Fh_policy = 128   # hidden size for policy layer
activation = 'relu'   # activation function
lr = 1e-3   # initial learning rate
decay = 1e-3    # initial weight decay
decay_step = 100    # perform decay after 100 steps
clip_grad = 3.0    # gradient clipping factor
summary_step = 500    # store model and training metrics after every certain no. of steps
N_rnn = 3   # number of layers used in GRUs
is_continuous = False   # load previous model or not
model_name = 'base_cdgcn'
# model_dir = f'/workspace/Toxicity_experiment/March_experiment/{model_name}'
model_dir = f'/workspace/Toxicity_experiment/March_experiment/just_test'
os.makedirs(model_dir, exist_ok=True)
# ckpt_dir = f'/workspace/Toxicity_experiment/March_experiment/{model_name}/logs/'    # logs directory
ckpt_dir = f'{model_dir}/logs/'
os.makedirs(ckpt_dir, exist_ok=True)
"""# Reading data file"""

def read_data(file_name):
    smiles = []
    with open(file_name) as f:
        for line in f:
            smiles.append(line.strip())
    return smiles





ChEMBL = '/workspace/Toxicity_experiment/chembl_final.txt'
dataset = read_data(ChEMBL)

print(len(dataset))
print(type(dataset))
print(dataset[:5])
iterations = (len(dataset)//batch_size)*5    # training iterations
single_traverse = len(dataset)//batch_size
#print("Iterations taken to traverse the whole dataset for one time: ",single_traverse)
print(iterations)

"""# Creating mini-batches"""

class BalancedSampler(Sampler):

    def __init__(self, cost, batch_size):
        index = np.argsort(cost).tolist()
        chunk_size = int(float(len(cost))/batch_size)
        self.index = []
        for i in range(batch_size):
            self.index.append(index[i*chunk_size:(i + 1)*chunk_size])

    def _g(self):
        # shuffle data
        for index_i in self.index:
            random.shuffle(index_i)

        for batch_index in zip(*self.index):
            yield batch_index

    def __iter__(self):
        return self._g()

    def __len__(self):
        return len(self.index[0])

"""# Obtaining all molecular properties"""


class MoleculeSpec(object):

    def __init__(self, file_name='/workspace/binding_data/atom_types.txt'):
        self.atom_types = []
        self.atom_symbols = []
        with open(file_name) as f:
            for line in f:
                atom_type_i = line.strip('\n').split(',')
                self.atom_types.append((atom_type_i[0], int(atom_type_i[1]), int(atom_type_i[2])))
                if atom_type_i[0] not in self.atom_symbols:
                    self.atom_symbols.append(atom_type_i[0])
        self.bond_orders = [Chem.BondType.AROMATIC,
                            Chem.BondType.SINGLE,
                            Chem.BondType.DOUBLE,
                            Chem.BondType.TRIPLE]
        self.max_iter = 120

    def get_atom_type(self, atom):
        atom_symbol = atom.GetSymbol()
        atom_charge = atom.GetFormalCharge()
        atom_hs = atom.GetNumExplicitHs()
        return self.atom_types.index((atom_symbol, atom_charge, atom_hs))

    def get_bond_type(self, bond):
        return self.bond_orders.index(bond.GetBondType())

    def index_to_atom(self, idx):
        atom_symbol, atom_charge, atom_hs = self.atom_types[idx]
        a = Chem.Atom(atom_symbol)
        a.SetFormalCharge(atom_charge)
        a.SetNumExplicitHs(atom_hs)
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
        _mol_spec = MoleculeSpec()
    return _mol_spec



# Load ML classifier



"""# Defining utility functions for data preprocessing and postprocessing"""

import mxnet as mx
from mxnet import gluon, nd, autograd
import numpy as np
from rdkit import Chem




def get_graph_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # build graph
    atom_types, atom_ranks, bonds, bond_types = [], [], [], []
    for a, r in zip(mol.GetAtoms(), Chem.CanonicalRankAtoms(mol)):
        atom_types.append(get_mol_spec().get_atom_type(a))
        atom_ranks.append(r)
    for b in mol.GetBonds():
        idx_1, idx_2, bt = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), get_mol_spec().get_bond_type(b)
        bonds.append([idx_1, idx_2])
        bond_types.append(bt)

    # build nx graph
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atom_types)))
    graph.add_edges_from(bonds)

    return graph, atom_types, atom_ranks, bonds, bond_types


def get_graph_from_smiles_list(smiles_list):
    graph_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)

        # build graph
        atom_types, bonds, bond_types = [], [], []
        for a in mol.GetAtoms():
            atom_types.append(get_mol_spec().get_atom_type(a))
        for b in mol.GetBonds():
            idx_1, idx_2, bt = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), get_mol_spec().get_bond_type(b)
            bonds.append([idx_1, idx_2])
            bond_types.append(bt)

        X_0 = np.array(atom_types, dtype=np.int32)
        A_0 = np.concatenate([np.array(bonds, dtype=np.int32),
                              np.array(bond_types, dtype=np.int32)[:, np.newaxis]],
                             axis=1)
        graph_list.append([X_0, A_0])
    return graph_list






def traverse_graph(graph, atom_ranks, current_node=None, step_ids=None, p=0.9, log_p=0.0):
    if current_node is None:
        #print(f"Traversal has started")
        next_nodes = range(len(atom_ranks))
        step_ids = [-1, ] * len(next_nodes)
        next_node_ranks = atom_ranks
    else:
        next_nodes = graph.neighbors(current_node)  # get neighbor nodes
        next_nodes = [n for n in next_nodes if step_ids[n] < 0] # filter visited nodes
        next_node_ranks = [atom_ranks[n] for n in next_nodes] # get ranks for neighbors
    next_nodes = [n for n, r in sorted(zip(next_nodes, next_node_ranks), key=lambda _x:_x[1])] # sort by rank

    # iterate through neighbors
    while len(next_nodes) > 0:
        if len(next_nodes)==1:
            next_node = next_nodes[0]
        elif random.random() >= (1 - p):
            next_node = next_nodes[0]
            log_p += np.log(p)
        else:
            next_node = next_nodes[random.randint(1, len(next_nodes) - 1)]
            log_p += np.log((1.0 - p) / (len(next_nodes) - 1))
        step_ids[next_node] = max(step_ids) + 1
        _, log_p = traverse_graph(graph, atom_ranks, next_node, step_ids, p, log_p)
        next_nodes = [n for n in next_nodes if step_ids[n] < 0] # filter visited nodes

    
    return step_ids, log_p


def single_reorder(X_0, A_0, step_ids):
    
    X_0, A_0 = np.copy(X_0), np.copy(A_0)
    

    step_ids = np.array(step_ids, dtype=np.int32)

    # sort by step_ids
    sorted_ids = np.argsort(step_ids)
    X_0 = X_0[sorted_ids]
    A_0[:, 0], A_0[:, 1] = step_ids[A_0[:, 0]], step_ids[A_0[:, 1]]
    # max_b is the max_index of each bond,min_b is the min index of each index
    max_b, min_b = np.amax(A_0[:, :2], axis=1), np.amin(A_0[:, :2], axis=1)
    A_0 = A_0[np.lexsort([-min_b, max_b]), :]

    # separate append and connect
    max_b, min_b = np.amax(A_0[:, :2], axis=1), np.amin(A_0[:, :2], axis=1)
    is_append = np.concatenate([np.array([True]), max_b[1:] > max_b[:-1]])
    A_0 = np.concatenate([np.where(is_append[:, np.newaxis],
                                 np.stack([min_b, max_b], axis=1),
                                 np.stack([max_b, min_b], axis=1)),
                        A_0[:, -1:]], axis=1)
    
    return X_0, A_0


def single_expand(X_0, A_0):
    #print(f"We are in single expand function")
    X_0, A_0 = np.copy(X_0), np.copy(A_0)

    # expand X

    is_append_iter = np.less(A_0[:, 0], A_0[:, 1]).astype(np.int32)
    
    
    NX = np.cumsum(np.pad(is_append_iter, [[1, 0]], mode='constant', constant_values=1))
    shift = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')[:-1])
    X_index = np.arange(NX.sum(), dtype=np.int32) - np.repeat(shift, NX)

    
    X = X_0[X_index]
   
    _, A_index = np.tril_indices(A_0.shape[0])
   
    A = A_0[A_index, :]
   
    NA = np.arange(A_0.shape[0] + 1)                  # Num_of_bonds + 1
    
    action_type = 1 - is_append_iter
    atom_type = np.where(action_type == 0, X_0[A_0[:, 1]], 0)
    bond_type = A_0[:, 2]
    append_pos = np.where(action_type == 0, A_0[:, 0], 0)
    connect_pos = np.where(action_type == 1, A_0[:, 1], 0)
    actions = np.stack([action_type, atom_type, bond_type, append_pos, connect_pos],
                       axis=1)
    last_action = [[2, 0, 0, 0, 0]]
    actions = np.append(actions, last_action, axis=0)
    #print(f"actions is {actions}")
    action_0 = np.array([X_0[0]], dtype=np.int32)
    #print(f"action_0 is {action_0}")

    
    last_atom_index = shift + NX - 1
    last_atom_mask = np.zeros_like(X)
    last_atom_mask[last_atom_index] = np.where(
        np.pad(is_append_iter, [[1, 0]], mode='constant', constant_values=1) == 1,
        np.ones_like(last_atom_index),
        np.ones_like(last_atom_index) * 2)
    

    return action_0, X, NX, A, NA, actions, last_atom_mask


def get_d(A, X):
    
    _to_sparse = lambda _A, _X: sparse.coo_matrix((np.ones([_A.shape[0] * 2], dtype=np.int32),
                                                   (np.concatenate([_A[:, 0], _A[:, 1]], axis=0),
                                                    np.concatenate([_A[:, 1], _A[:, 0]], axis=0))),
                                                    shape=[_X.shape[0], ] * 2)   # this is creating a square matrtix of size len(X)


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

    A_0[:, :2] += np.stack([np.repeat(cumsum, NA_0), ] * 2, axis=1)

    D_0_2, D_0_3 = get_d(A_0, X_0)

    A_split = []
    for i in range(get_mol_spec().num_bond_types):
        A_i = A_0[A_0[:, 2] == i, :2]
        A_split.append(A_i)
    A_split.extend([D_0_2, D_0_3])
    A_0 = A_split


    NX_rep_0 = np.repeat(np.arange(NX_0.shape[0]), NX_0)


    return X_0, A_0, NX_0, NX_rep_0


def merge_single(X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions, last_append_mask, log_p):
    
    X, A, NX, NX_rep = merge_single_0(X, A, NX, NA)
    cumsum = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')[:-1])
    actions[:, -2] += cumsum * (actions[:, 0] == 0)
    actions[:, -1] += cumsum * (actions[:, 0] == 1)
    mol_ids_rep = np.repeat(mol_ids, NX)
    rep_ids_rep = np.repeat(rep_ids, NX)

    return X, A,\
           mol_ids_rep, rep_ids_rep, iw_ids,\
           last_append_mask,\
           NX, NX_rep,\
           action_0, actions, \
           log_p



def process_single(smiles, k, p):

    smiles, smiles_class = smiles.split(" ")
    smiles_class = int(smiles_class)
    graph, atom_types, atom_ranks, bonds, bond_types = get_graph_from_smiles(smiles)

    try:
        graph, atom_types, atom_ranks, bonds, bond_types = get_graph_from_smiles(smiles)
        X_0 = np.array(atom_types, dtype=np.int32)
        A_0 = np.concatenate([np.array(bonds, dtype=np.int32),
                                  np.array(bond_types, dtype=np.int32)[:, np.newaxis]], axis=1)


    except Exception as e:
            print(f"Error processing SMILES: {smiles}")
            print(f"Error message: {e}")





    """m = Chem.MolFromSmiles(smiles)
    qed_smiles = QED.qed(m)
    if qed_smiles > 0.50:
        prop_class = 0
    else :
        prop_class = 1"""   
    
    
    X, A = [], []
    NX, NA = [], []
    mol_ids, rep_ids, iw_ids = [], [], []
    action_0, actions = [], []
    last_append_mask = []
    log_p = []
    tox_class = [1] * k

    # random sampling decoding route
    #print(f"We are starting sampling traversals for this smiles {smiles}")

    for i in range(k):
        

        step_ids_i, log_p_i = traverse_graph(graph, atom_ranks, p=p)
        
        X_i, A_i = single_reorder(X_0, A_0, step_ids_i)
        action_0_i, X_i, NX_i, A_i, NA_i, actions_i, last_atom_mask_i = single_expand(X_i, A_i)

        # appends
        X.append(X_i)
        A.append(A_i)
        NX.append(NX_i)
        NA.append(NA_i)
        action_0.append(action_0_i)
        actions.append(actions_i)
        last_append_mask.append(last_atom_mask_i)

        mol_ids.append(np.zeros_like(NX_i, dtype=np.int32))
        rep_ids.append(np.ones_like(NX_i, dtype=np.int32) * i)
        iw_ids.append(np.ones_like(NX_i, dtype=np.int32) * i)

        log_p.append(log_p_i)
        #tox_class.append(tox_class_i)

    # concatenate
    X = np.concatenate(X, axis=0)
    A = np.concatenate(A, axis = 0)
    NX = np.concatenate(NX, axis = 0)
    NA = np.concatenate(NA, axis = 0)
    action_0 = np.concatenate(action_0, axis = 0)
    actions = np.concatenate(actions, axis = 0)
    last_append_mask = np.concatenate(last_append_mask, axis = 0)
    mol_ids = np.concatenate(mol_ids, axis = 0)
    rep_ids = np.concatenate(rep_ids, axis = 0)
    iw_ids = np.concatenate(iw_ids, axis = 0)
    log_p = np.array(log_p, dtype=np.float32)
    tox_class = np.array(tox_class)

    return X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions, last_append_mask, log_p, tox_class


# noinspection PyArgumentList
def get_mol_from_graph(X, A, sanitize=True):
    try:
        mol = Chem.RWMol(Chem.Mol())

        X, A = X.tolist(), A.tolist()
        for i, atom_type in enumerate(X):
            mol.AddAtom(get_mol_spec().index_to_atom(atom_type))

        for atom_id1, atom_id2, bond_type in A:
            get_mol_spec().index_to_bond(mol, atom_id1, atom_id2, bond_type)
    except:
        return None

    if sanitize:
        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        except:
            return None
    else:
        return mol

def get_mol_from_graph_list(graph_list, sanitize=True):
    mol_list = [get_mol_from_graph(X, A, sanitize) for X, A in graph_list]
    return mol_list


# Defining dataloader

class MolLoader(DataLoader):
    """Load graph based molecule representation from SMILES"""
    def __init__(self, dataset, batch_size=10, num_workers=0,
                 k=10, p=0.9, shuffle=False, sampler=None, batch_sampler=None):
        self.k = k
        self.p = p

        # batch_sampler, sampler and shuffle are mutually exclusive
        if batch_sampler is not None:
            super(MolLoader, self).__init__(dataset, batch_sampler=batch_sampler,
                                            num_workers=num_workers, batchify_fn=self._collate_fn)
        elif sampler is not None:
            super(MolLoader, self).__init__(dataset, sampler=sampler,
                                            num_workers=num_workers, batchify_fn=self._collate_fn,
                                            last_batch='rollover')
        else:
            super(MolLoader, self).__init__(dataset, batch_size, shuffle=shuffle,
                                            num_workers=num_workers, batchify_fn=self._collate_fn,
                                            last_batch='rollover')


    def _collate_fn(self, batch):
        # names = X, A,
        #         NX, NA,
        #         mol_ids, rep_ids, iw_ids,
        #         action_0, actions,
        #         last_append_mask,
        #         log_p

        shapes = [[0], [0, 3],
                  [0], [0],
                  [0], [0], [0],
                  [0], [0, 5],
                  [0],
                  [0], [0]]
        dtypes = [np.int32, np.int32,
                  np.int32, np.int32,
                  np.int32, np.int32, np.int32,
                  np.int32, np.int32,
                  np.int32,
                  np.float32, np.int32]

        _build = lambda: [np.zeros(shape=s, dtype=d) for s, d in zip(shapes, dtypes)]
        _append = lambda _r0, _r1: [np.concatenate([__r0, __r1], axis=0)
                                    for __r0, __r1 in zip(_r0, _r1)]

        X, A, \
        NX, NA, \
        mol_ids, rep_ids, iw_ids, \
        action_0, actions, \
        last_append_mask, \
        log_p, tox_class = _build()

        # Processing batch of molecules
        #print(f"This is new batch")
        for i, record_in in enumerate(batch):
            #print(f"{i}th smiles is {record_in}")
            smiles = record_in

            # Using process single
            X_i, A_i, \
            NX_i, NA_i, \
            mol_ids_i, rep_ids_i, iw_ids_i, \
            action_0_i, actions_i, \
            last_append_mask_i, log_p_i, tox_class_i = process_single(smiles, self.k, self.p)

            if i != 0:
                mol_ids_i += mol_ids[-1] + 1
                iw_ids_i += iw_ids[-1] + 1

            # appending to the list
            tox_class_i = tox_class_i.reshape(-1)


            X, A, \
            NX, NA, \
            mol_ids, rep_ids, iw_ids, \
            action_0, actions, \
            last_append_mask, \
            log_p, tox_class = _append([X, A,
                             NX, NA,
                             mol_ids, rep_ids, iw_ids,
                             action_0, actions,
                             last_append_mask,
                             log_p, tox_class],
                            [X_i, A_i,
                             NX_i, NA_i,
                             mol_ids_i, rep_ids_i, iw_ids_i,
                             action_0_i, actions_i,
                             last_append_mask_i,
                             log_p_i, tox_class_i])

        
        # using merge single
        X, A, \
        mol_ids_rep, rep_ids_rep, iw_ids, \
        last_append_mask, \
        NX, NX_rep, \
        action_0, actions, \
        log_p = merge_single(X, A,
                                   NX, NA,
                                   mol_ids, rep_ids, iw_ids,
                                   action_0, actions,
                                   last_append_mask,
                                   log_p)
        

        # returning all the data structures combined as list
        result_out = [X, A,
                      mol_ids_rep, rep_ids_rep, iw_ids,
                      last_append_mask,
                      NX, NX_rep,
                      action_0, actions,
                      log_p, tox_class]

        return result_out

    @staticmethod
    def from_numpy_to_tensor(record):
        """Convert numpy to tensor and place it to a specific device"""
        [X, A,
         mol_ids_rep, rep_ids_rep, iw_ids,
         last_append_mask,
         NX, NX_rep,
         action_0, actions,
         log_p, tox_class] = record

        
        X = nd.array(X, ctx=mx.gpu(), dtype='int32')

        # They are tryign
        A_sparse = []
        #print(f"This is a batch for numpy_to_tensor")
        for A_i in A:
            
            if A_i.shape[0] == 0:
                A_sparse.append(None)
                #print(f"We had to append none to A_sparse")
            else:
                # transpose may not be supported in gpu
                A_i = np.concatenate([A_i, A_i[:, [1, 0]]], axis=0)

                # construct csr matrix ...
                data = np.ones((A_i.shape[0], ), dtype=np.float32)
                row, col = A_i[:, 0], A_i[:, 1]
                A_sparse_i = nd.sparse.csr_matrix((data, (row, col)),
                                                  shape=tuple([int(X.shape[0]), ]*2),
                                                  ctx=mx.gpu(),
                                                  dtype='float32')

                
                A_sparse.append(A_sparse_i)

        batch_size, iw_size = (mol_ids_rep.max() + 1).item(), \
                              (rep_ids_rep.max() + 1).item()

        mol_ids_rep, rep_ids_rep, iw_ids, \
        last_append_mask, \
        NX, NX_rep, action_0, actions = [nd.array(_x, ctx=mx.gpu(), dtype='int32')
                                         for _x in [mol_ids_rep, rep_ids_rep, iw_ids,
                                                    last_append_mask,
                                                    NX, NX_rep, action_0, actions]]

        log_p = nd.array(log_p, ctx=mx.gpu(), dtype='float32')
        tox_class = nd.array(tox_class, ctx=mx.gpu())
        #print(f"Length of log_p is {log_p}")
        record = [X, A_sparse, iw_ids, last_append_mask,
                  NX, NX_rep, action_0, actions, log_p,
                  batch_size, iw_size, tox_class]


        return record


class MolRNNLoader(MolLoader):

    def _collate_fn(self, batch):
        result_out = super(MolRNNLoader, self)._collate_fn(batch)

        # things ready for rnn
        batch = [smile.split(" ")[0] for smile in batch]
        
        mol_list = [Chem.MolFromSmiles(batch_i) for batch_i in batch]
        
        graph_to_rnn = np.zeros((len(batch), self.k, get_mol_spec().max_iter), dtype=np.int32)
        rnn_to_graph = []
        cum_sum = 0
        for i, mol_i in enumerate(mol_list):
            num_iter = mol_i.GetNumBonds() + 1
            for k in range(self.k):
                graph_to_rnn[i, k, :num_iter] = (np.arange(num_iter) + cum_sum)

                rnn_to_graph_0 = np.ones([num_iter,], dtype=np.int32) * i
                rnn_to_graph_1 = np.ones_like(rnn_to_graph_0) * k
                rnn_to_graph_2 = np.arange(num_iter)
                rnn_to_graph.append(np.stack([rnn_to_graph_0, rnn_to_graph_1, rnn_to_graph_2], axis=0))

                cum_sum += num_iter
        rnn_to_graph = np.concatenate(rnn_to_graph, axis=1)
        NX_cum = np.cumsum(result_out[6])

        result_out = result_out + [graph_to_rnn, rnn_to_graph, NX_cum]

        return result_out

    @staticmethod
    def from_numpy_to_tensor(record):
        [X, A,
         mol_ids_rep, rep_ids_rep, iw_ids,
         last_append_mask,
         NX, NX_rep,
         action_0, actions,
         log_p, tox_class,
         graph_to_rnn, rnn_to_graph, NX_cum] = record

        output = MolLoader.from_numpy_to_tensor([X, A,
                                                 mol_ids_rep, rep_ids_rep, iw_ids,
                                                 last_append_mask,
                                                 NX, NX_rep,
                                                 action_0, actions,
                                                 log_p, tox_class])

        graph_to_rnn, rnn_to_graph, NX_cum =\
            nd.array(graph_to_rnn, ctx=mx.gpu(), dtype='int32'),\
            nd.array(rnn_to_graph, ctx=mx.gpu(), dtype='int32'), \
            nd.array(NX_cum, ctx=mx.gpu(), dtype='int32')

        output = output + [graph_to_rnn, rnn_to_graph, NX_cum]

        return output

"""# Defining graph convolution and other functions"""

class GraphConvFn(Function):

    def __init__(self, A):

        self.A = A # type: nd.sparse.CSRNDArray
        self.A_T = self.A # assume symmetric
        super(GraphConvFn, self).__init__()

    def forward(self, X):
        

        if self.A is not None:
            #print(f"Shape of A is {self.A.shape}")

            if len(X.shape) > 2:
                X_resized = X.reshape((X.shape[0], -1))
                output = nd.sparse.dot(self.A, X_resized)      # dot product of adjacency and features matrix
                output = output.reshape([-1, ] + [X.shape[i] for i in range(1, len(X.shape))])
            else:
                output = nd.sparse.dot(self.A, X)
            return output
        else:
            return nd.zeros_like(X)

    def backward(self, grad_output):

        if self.A is not None:
            if len(grad_output.shape) > 2:
                grad_output_resized = grad_output.reshape((grad_output.shape[0], -1))
                grad_input = nd.sparse.dot(self.A_T, grad_output_resized)
                grad_input = grad_input.reshape([-1] + [grad_output.shape[i]
                                                        for i in range(1, len(grad_output.shape))])
            else:
                grad_input = nd.sparse.dot(self.A_T, grad_output)
            return grad_input
        else:
            return nd.zeros_like(grad_output)


class EfficientGraphConvFn(Function):
    """Save memory by re-computation"""

    def __init__(self, A_list):
        self.A_list = A_list
        super(EfficientGraphConvFn, self).__init__()

    def forward(self, X, W):
       # print(f"We are in efficient_graph_convolution")
       # print(f"length of A_list is {len(self.A_list)}")
        X_list = [X]
        for A in self.A_list:
            if A is not None:
                X_list.append(nd.sparse.dot(A, X))
            else:
                X_list.append(nd.zeros_like(X))
        X_out = nd.concat(*X_list, dim=1)
        self.save_for_backward(X, W)
        # print(f"X_out shape is{X_out.shape} and W shape is {W.shape}")
        return nd.dot(X_out, W)

    def backward(self, grad_output):
        X, W = self.saved_tensors

        # recompute X_out
        X_list = [X, ]
        for A in self.A_list:
            if A is not None:
                X_list.append(nd.sparse.dot(A, X))
            else:
                X_list.append(nd.zeros_like(X))
        X_out = nd.concat(*X_list, dim=1)

        grad_W = nd.dot(X_out.T, grad_output)

        grad_X_out = nd.dot(grad_output, W.T)
        grad_X_out_list = nd.split(grad_X_out, num_outputs=len(self.A_list) + 1)


        grad_X = [grad_X_out_list[0], ]
        for A, grad_X_out in zip(self.A_list, grad_X_out_list[1:]):
            if A is not None:
                grad_X.append(nd.sparse.dot(A, grad_X_out))
            else:
                grad_X.append(nd.zeros_like(grad_X_out))

        grad_X = sum(grad_X)

        return grad_X, grad_W


class SegmentSumFn(GraphConvFn):

    def __init__(self, idx, num_seg):
        # build A
        # construct coo
        data = nd.ones(idx.shape[0], ctx=idx.context, dtype='int32')
        row, col = idx, nd.arange(idx.shape[0], ctx=idx.context, dtype='int32')
        shape = (num_seg, int(idx.shape[0]))
        sparse = nd.sparse.csr_matrix((data, (row, col)), shape=shape,
                                      ctx=idx.context, dtype='float32')
        super(SegmentSumFn, self).__init__(sparse)

        sparse = nd.sparse.csr_matrix((data, (col, row)), shape=(shape[1], shape[0]),
                                      ctx=idx.context, dtype='float32')
        self.A_T = sparse


def squeeze(input, axis):
    assert input.shape[axis] == 1

    new_shape = list(input.shape)
    del new_shape[axis]

    return input.reshape(new_shape)


def unsqueeze(input, axis):
    return nd.expand_dims(input, axis=axis)


def logsumexp(inputs, axis=None, keepdims=False):
    """Numerically stable logsumexp.
    Args:
        inputs: A Variable with any shape.
        axis: An integer.
        keepdims: A boolean.
    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    Adopted from: https://github.com/pytorch/pytorch/issues/2591
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if axis is None:
        inputs = inputs.reshape([-1])
        axis = 0
    s = nd.max(inputs, axis=axis, keepdims=True)
    outputs = s + (inputs - s).exp().sum(axis=axis, keepdims=True).log()
    if not keepdims:
        outputs = nd.sum(outputs, axis=axis, keepdims=False)
    return outputs


def get_activation(name):
    activation_dict = {
        'relu':nd.relu,
        'tanh':nd.tanh
    }
    return activation_dict[name]

"""# Defining neural networks"""

class Linear_BN(nn.Sequential):
    def __init__(self, F_in, F_out):
        super(Linear_BN, self).__init__()
       # print(f"Dense layer of input {F_in} and output {F_out} is created ")
        self.add(nn.Dense(F_out, in_units=F_in, use_bias=False))
        self.add(BatchNorm(in_channels=F_out))


class GraphConv(nn.Block):

    def __init__(self, Fin, Fout, D):
        super(GraphConv, self).__init__()

        # model settings
        self.Fin = Fin
        self.Fout = Fout
        self.D = D
       
        # model parameters
        self.W = self.params.get('w', shape=(self.Fin * (self.D + 1), self.Fout),
                                 init=None, allow_deferred_init=False)

    def forward(self, X, A_list):
        try:
            assert len(A_list) == self.D
        except AssertionError as e:
            print(self.D, len(A_list))
            raise e
        return EfficientGraphConvFn(A_list)(X, self.W.data(X.context))


class Policy(nn.Block):

    def __init__(self, F_in, F_h, N_A, N_B, k=1):
        super(Policy, self).__init__()
        self.F_in = F_in # number of input features for each atom
        self.F_h = F_h # number of context variables
        self.N_A = N_A # number of atom types
        self.N_B = N_B # number of bond types
        self.k = k # number of softmax used in the mixture


        with self.name_scope():
            self.linear_h = Linear_BN(F_in * 2, self.F_h * k)
            self.linear_h_t = Linear_BN(F_in, self.F_h * k)

            self.linear_x = nn.Dense(self.N_B + self.N_B*self.N_A, in_units=self.F_h)
            self.linear_x_t = nn.Dense(1, in_units=self.F_h)

            if self.k > 1:
                self.linear_pi = nn.Dense(self.k, in_units=self.F_in)
            else:
                self.linear_pi = None

    def forward(self, X, NX, NX_rep, X_end=None):
        # segment mean for X
        if X_end is None:
            X_end = SegmentSumFn(NX_rep, NX.shape[0])(X)/nd.cast(unsqueeze(NX, 1), 'float32')
        X = nd.concat(X, X_end[NX_rep, :], dim=1)

        X_h = nd.relu(self.linear_h(X)).reshape([-1, self.F_h])
        X_h_end = nd.relu(self.linear_h_t(X_end)).reshape([-1, self.F_h])

        X_x = nd.exp(self.linear_x(X_h)).reshape([-1, self.k, self.N_B + self.N_B*self.N_A])
        X_x_end = nd.exp(self.linear_x_t(X_h_end)).reshape([-1, self.k, 1])

        X_sum = nd.sum(SegmentSumFn(NX_rep, NX.shape[0])(X_x), -1, keepdims=True) + X_x_end
        X_sum_gathered = X_sum[NX_rep, :, :]

        X_softmax = X_x / X_sum_gathered
        X_softmax_end = X_x_end/ X_sum

        if self.k > 1:
            pi = unsqueeze(nd.softmax(self.linear_pi(X_end), axis=1), -1)
            pi_gathered = pi[NX_rep, :, :]

            X_softmax = nd.sum(X_softmax * pi_gathered, axis=1)
            X_softmax_end = nd.sum(X_softmax_end * pi, axis=1)
        else:
            X_softmax = squeeze(X_softmax, 1)
            X_softmax_end = squeeze(X_softmax_end, 1)

        # generate output probabilities
        connect, append = X_softmax[:, :self.N_B], X_softmax[:, self.N_B:]
        append = append.reshape([-1, self.N_A, self.N_B])
        end = squeeze(X_softmax_end, -1)
        #print(f"connect probability shape is: {connect.shape}")
        #print(f"connect probabilities are: {connect}")
        return append, connect, end


class BatchNorm(nn.Block):

    def __init__(self, in_channels, momentum=0.9, eps=1e-5):
        super(BatchNorm, self).__init__()
        self.F = in_channels

        self.bn_weight = self.params.get('bn_weight', shape=(self.F,), init=mx.init.One(),
                                         allow_deferred_init=False)
        self.bn_bias = self.params.get('bn_bias', shape=(self.F,), init=mx.init.Zero(),
                                       allow_deferred_init=False)

        self.running_mean = self.params.get('running_mean', grad_req='null',
                                            shape=(self.F,),
                                            init=mx.init.Zero(),
                                            allow_deferred_init=False,
                                            differentiable=False)
        self.running_var = self.params.get('running_var', grad_req='null',
                                           shape=(self.F,),
                                           init=mx.init.One(),
                                           allow_deferred_init=False,
                                           differentiable=False)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        if autograd.is_training():
            return nd.BatchNorm(x,
                                gamma=self.bn_weight.data(x.context),
                                beta=self.bn_bias.data(x.context),
                                moving_mean=self.running_mean.data(x.context),
                                moving_var=self.running_var.data(x.context),
                                eps=self.eps, momentum=self.momentum,
                                use_global_stats=False)
        else:
            return nd.BatchNorm(x,
                                gamma=self.bn_weight.data(x.context),
                                beta=self.bn_bias.data(x.context),
                                moving_mean=self.running_mean.data(x.context),
                                moving_var=self.running_var.data(x.context),
                                eps=self.eps, momentum=self.momentum,
                                use_global_stats=True)

"""# Building generative network"""

class MoleculeGenerator(nn.Block):

    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation,
                 *args, **kwargs):
        super(MoleculeGenerator, self).__init__()
        self.N_A = N_A                   # no. of unique atom types
        self.N_B = N_B                   # no. of unique bond types
        self.D = D                       # receptive field
        self.F_e = F_e                   # initial embedding layer
        self.F_skip = F_skip             # skip connection layer
        self.F_c = list(F_c) if isinstance(F_c, tuple) else F_c                   # size of grpah convolution layers
        self.Fh_policy = Fh_policy
        self.activation = get_activation(activation)

        with self.name_scope():
            # embeddings
            self.embedding_atom = nn.Embedding(self.N_A, self.F_e)
            self.embedding_mask = nn.Embedding(3, self.F_e)

            # graph conv
            self._build_graph_conv(*args, **kwargs)

            # fully connected
            self.dense = nn.Sequential()
            for i, (f_in, f_out) in enumerate(zip([self.F_skip, ] + self.F_c[:-1], self.F_c)):
                self.dense.add(Linear_BN(f_in, f_out))

            # policy
            self.policy_0 = self.params.get('policy_0', shape=[self.N_A, ],
                                            init=mx.init.Zero(),
                                            allow_deferred_init=False)
            self.policy_h = Policy(self.F_c[-1], self.Fh_policy, self.N_A, self.N_B)

        self.mode = 'loss'

    @abstractmethod
    def _build_graph_conv(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _graph_conv_forward(self, X, A):
        raise NotImplementedError

    # for prob score of starting atom
    def _policy_0(self, ctx):
        policy_0 = nd.exp(self.policy_0.data(ctx))
        policy_0 = policy_0/policy_0.sum()
        return policy_0

    def _policy(self, X, A, NX, NX_rep, last_append_mask):
        # get initial embedding
        print(f"We are in policy now")
        print(f"shape of X is {X.shape} and last_append_mask is {last_append_mask.shape}")
        print(f"X is {X[0]}")
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        print(f"shape of X after operation is {X.shape}")

        # convolution
        X = self._graph_conv_forward(X, A)

        # linear
        X = self.dense(X)

        # policy
        append, connect, end = self.policy_h(X, NX, NX_rep)

       

        return append, connect, end

    def _likelihood(self, init, append, connect, end,
                    action_0, actions, iw_ids, log_p_sigma,
                    batch_size, iw_size, tox_class_batch):




        # decompose action:
        action_type, node_type, edge_type, append_pos, connect_pos = \
            actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3], actions[:, 4]
        _log_mask = lambda _x, _mask: _mask * nd.log(_x + 1e-10) + (1- _mask) * nd.zeros_like(_x)
       
        init = init.reshape([batch_size * iw_size, self.N_A])
      
        index = nd.stack(nd.arange(action_0.shape[0], ctx=action_0.context, dtype='int32'), action_0, axis=0)
       
        loss_init = nd.log(nd.gather_nd(init, index) + 1e-10)

        
        loss_end = _log_mask(end, nd.cast(action_type == 2, 'float32'))
        


        # append
        index = nd.stack(append_pos, node_type, edge_type, axis=0)
        loss_append = _log_mask(nd.gather_nd(append, index), nd.cast(action_type == 0, 'float32'))
        

        # connect
        index = nd.stack(connect_pos, edge_type, axis=0)
        loss_connect = _log_mask(nd.gather_nd(connect, index), nd.cast(action_type == 1, 'float32'))
        


        # sum up results
        log_p_x = loss_end + loss_append + loss_connect
        log_p_x = squeeze(SegmentSumFn(iw_ids, batch_size*iw_size)(unsqueeze(log_p_x, -1)), -1)
        #print(f"log_p_x shape before transformation is {log_p_x.shape}")
        #log_p_x = unsqueeze(log_p_x, -1)
       
        log_p_x = log_p_x + loss_init

        # reshape
        log_p_x = log_p_x.reshape([batch_size, iw_size])
        log_p_sigma = log_p_sigma.reshape([batch_size, iw_size])
        l = log_p_x - log_p_sigma
        l = logsumexp(l, axis=1) - math.log(float(iw_size))

        



        #print(f"Loss is {l}")
        return l

    def forward(self, *input):
        if self.mode=='loss' or self.mode=='likelihood':
            X, A, iw_ids, last_append_mask, \
            NX, NX_rep, action_0, actions, log_p,tox_class_batch, \
            batch_size, iw_size = input

            init = self._policy_0(X.context).tile([batch_size * iw_size, 1])
            append, connect, end = self._policy(X, A, NX, NX_rep, last_append_mask)
            l = self._likelihood(init, append, connect, end, action_0, actions, iw_ids, log_p, batch_size, iw_size, tox_class_batch)
            if self.mode=='likelihood':
                #print(f"We are in likelihood mode")
                return l
            else:
                return -l.mean()
        elif self.mode == 'decode_0':
            #print(f"We are in decode_0 mode")
            return self._policy_0(input[0])
        elif self.mode == 'decode_step':
            #print(f"We are in decode_step mode")
            X, A, NX, NX_rep, last_append_mask = input
            return self._policy(X, A, NX, NX_rep, last_append_mask)


class MoleculeGenerator_RNN(MoleculeGenerator):

    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation,
                 N_rnn, *args, **kwargs):
        super(MoleculeGenerator_RNN, self).__init__(N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation,
                                                    *args, **kwargs)
        self.N_rnn = N_rnn

        with self.name_scope():
            self.rnn = gluon.rnn.GRU(hidden_size=self.F_c[-1],
                                     num_layers=self.N_rnn,
                                     layout='NTC', input_size=self.F_c[-1] * 2)

    def _rnn_train(self, X, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum):
        X_avg = SegmentSumFn(NX_rep, NX.shape[0])(X) / nd.cast(unsqueeze(NX, 1), 'float32')
        X_curr = nd.take(X, indices=NX_cum-1)
        X = nd.concat(X_avg, X_curr, dim=1)

        # rnn
        X = nd.take(X, indices=graph_to_rnn) # batch_size, iw_size, length, num_features
        batch_size, iw_size, length, num_features = X.shape
        X = X.reshape([batch_size*iw_size, length, num_features])
        X = self.rnn(X)

        X = X.reshape([batch_size, iw_size, length, -1])
        X = nd.gather_nd(X, indices=rnn_to_graph)

        return X

    def _rnn_test(self, X, NX, NX_rep, NX_cum, h):
        # note: one partition for one molecule
        X_avg = SegmentSumFn(NX_rep, NX.shape[0])(X) / nd.cast(unsqueeze(NX, 1), 'float32')
        X_curr = nd.take(X, indices=NX_cum - 1)
        X = nd.concat(X_avg, X_curr, dim=1) # size: [NX, F_in * 2]

        # rnn
        X = unsqueeze(X, axis=1)
        X, h = self.rnn(X, h)

        X = squeeze(X, axis=1)
        return X, h

    def _policy(self, X, A, NX, NX_rep, last_append_mask, graph_to_rnn, rnn_to_graph, NX_cum):
        # get initial embedding
        # X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)

        
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)

        # convolution
        X = self._graph_conv_forward(X, A)

        # linear
        X = self.dense(X)

        # rnn
        X_mol = self._rnn_train(X, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum)

        # policy
        append, connect, end = self.policy_h(X, NX, NX_rep, X_mol)
        #print(f"end probability is: {end}")
       # print(f"Length of end_probability is {len(end)}")
        return append, connect, end

        

    def _decode_step(self, X, A, NX, NX_rep, last_append_mask, NX_cum, h):
        # get initial embedding
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)

        # convolution
        X = self._graph_conv_forward(X, A)

        # linear
        X = self.dense(X)

        # rnn
        X_mol, h = self._rnn_test(X, NX, NX_rep, NX_cum, h)

        # policy
        append, connect, end = self.policy_h(X, NX, NX_rep, X_mol)
        

        return append, connect, end, h

    def forward(self, *input):
        if self.mode=='loss' or self.mode=='likelihood':
            X, A, iw_ids, last_append_mask, \
            NX, NX_rep, action_0, actions, log_p, \
            batch_size, iw_size, tox_class, \
            graph_to_rnn, rnn_to_graph, NX_cum = input

            init = self._policy_0(X.context).tile([batch_size * iw_size, 1])
            append, connect, end = self._policy(X, A, NX, NX_rep, last_append_mask, graph_to_rnn, rnn_to_graph, NX_cum)
            l = self._likelihood(init, append, connect, end, action_0, actions, iw_ids, log_p, batch_size, iw_size, tox_class)
            if self.mode=='likelihood':
               # print(f"We are in likelihood mode")
                return l
            else:
                return -l.mean()
        elif self.mode == 'decode_0':
           # print(f"We are in decode_0 mode")
            return self._policy_0(input[0])
        elif self.mode == 'decode_step':
           # print(f"We are in decode_step mode")
            X, A, NX, NX_rep, last_append_mask, NX_cum, h = input
            return self._decode_step(X, A, NX, NX_rep, last_append_mask, NX_cum, h)
        else:
            raise ValueError

class VanillaMolGen_RNN(MoleculeGenerator_RNN):

    def __init__(self, N_A, N_B, D, F_e, F_h, F_skip, F_c, Fh_policy, activation, N_rnn):
        super(VanillaMolGen_RNN, self).__init__(N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn,
                                                F_h)

    def _build_graph_conv(self, F_h):
        self.F_h = list(F_h) if isinstance(F_h, tuple) else F_h
        self.conv, self.bn = [], []
        for i, (f_in, f_out) in enumerate(zip([self.F_e] + self.F_h[:-1], self.F_h)):
           # print(f"We are in graph_conv forward yes")
           # print(f"f_in is {f_in}, f_out is {f_out},N_B is {self.N_B}, D is {self.D}")
            conv = GraphConv(f_in, f_out, self.N_B + self.D)
            self.conv.append(conv)
            self.register_child(conv)

            if i != 0:
                bn = BatchNorm(in_channels=f_in)
                self.register_child(bn)
            else:
                bn = None
            self.bn.append(bn)

        self.bn_skip = BatchNorm(in_channels=sum(self.F_h))
        self.linear_skip = Linear_BN(sum(self.F_h), self.F_skip)

    def _graph_conv_forward(self, X, A):
       
        X_out = [X]
        for conv, bn in zip(self.conv, self.bn):
            X = X_out[-1]
            if bn is not None:
                
                X_out.append(conv(self.activation(bn(X)), A))
                
            else:
                X_out.append(conv(X, A))
                
        X_out = nd.concat(*X_out[1:], dim=1)
        return self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))

"""# Loading and preprocessing the dataset(Necessary to load previous checkpoints)"""

if all([os.path.isfile(os.path.join(ckpt_dir, _n)) for _n in ['log.out', 'ckpt.params', 'trainer.status']]):
    is_continuous = True
else:
    is_continuous = False

db_train = dataset
num_workers=0
# get sampler and loader for training set
sampler_train = BalancedSampler(cost=[len(l) for l in db_train], batch_size=batch_size)
loader_train = MolRNNLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers, k=k, p=p)

it_train = iter(loader_train)


"""### Below code is just for initialization,skip if you want to continue from previous checkpoints

# Construcing the model and initializing the parameters
"""

if not is_continuous:
    configs = {'F_e': F_e,
               'F_h': F_h,
               'F_skip': F_skip,
               'F_c': F_c,
               'Fh_policy': Fh_policy,
               'activation': activation,
               'N_rnn': N_rnn}
    with open(os.path.join(ckpt_dir, 'configs.json'), 'w') as f:
        json.dump(configs, f)
else:
    with open(os.path.join(ckpt_dir, 'configs.json')) as f:
        configs = json.load(f)

model = VanillaMolGen_RNN(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)

ctx = mx.gpu()
if not is_continuous:
    model.collect_params().initialize(mx.init.Xavier(), force_reinit=True, ctx=ctx)
else:
    model.load_parameters(os.path.join(ckpt_dir, 'ckpt.params'), ctx=ctx)

# construct optimizer
opt = mx.optimizer.Adam(learning_rate=lr, clip_gradient=clip_grad)
trainer = gluon.Trainer(model.collect_params(), opt)
if is_continuous:
    trainer.load_states(os.path.join(ckpt_dir, 'trainer.status'))

"""# Training the model"""
print("We are in training part....")
iterations = 480000


if not is_continuous:
    t0 = time.time()
    global_counter = 0
    print(f"Training is starting from scratch....")
else:
    with open(os.path.join(ckpt_dir, 'log.out')) as f:
        records = f.readlines()
        if records[-1] != 'Training finished\n':
            final_record = records[-1]
        else:
            final_record = records[-2]
    count, t_final = int(final_record.split('\t')[0]), float(final_record.split('\t')[1])
    t0 = time.time() - t_final * 60
    global_counter = count
    print(f"Training is continuing from golabal_counter : {global_counter}")

with open(os.path.join(ckpt_dir, 'log.out'),
          mode='w' if not is_continuous else 'a') as f:
    if not is_continuous:
        f.write('step\ttime(h)\tloss\tlr\n')
    while True:
        global_counter += 1
        

        try:
            inputs = next(it_train)
        except StopIteration:
            it_train = iter(loader_train)
            inputs = next(it_train)

        # move to gpu
        inputs = MolRNNLoader.from_numpy_to_tensor(inputs)

        with autograd.record():
            loss = [(model(*inputs)).as_in_context(mx.gpu())]
            loss = sum(loss)
            loss.backward()

        nd.waitall()
        gc.collect()

        trainer.step(batch_size=1)
        if global_counter % decay_step == 0:
            trainer.set_learning_rate(trainer.learning_rate * (1.0 - decay))

        if global_counter % summary_step == 0:
            loss = (sum(loss)).asnumpy().item()

            model.save_parameters(os.path.join(ckpt_dir, 'ckpt.params'))
            trainer.save_states(os.path.join(ckpt_dir, 'trainer.status'))

            print('Iteration', global_counter, 'done!')
            f.write('{}\t{}\t{}\t{}\n'.format(global_counter, float(time.time() - t0)/60, loss, trainer.learning_rate))
            try:
                    f.flush()
            except Exception as flush_error:
                    print(f"Error flushing to file: {flush_error}")

            del loss, inputs
            gc.collect()

        if global_counter >= iterations:
            break

    # save before exit
    model.save_parameters(os.path.join(ckpt_dir, 'ckpt.params'))
    trainer.save_states(os.path.join(ckpt_dir, 'trainer.status'))

    f.write('Training finished\n')


num_params = sum(p.data().size for p in model.collect_params().values())
print(f"Number of parameters in the model: {num_params}")


'''
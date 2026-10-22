"""Exec the real traverse_graph + get_graph_from_smiles out of the notebook."""
import random, numpy as np, networkx as nx, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from _nbsource import load_defs, source_path

SRC = "pretraining_torch.py"
ns = {"np": np, "nx": nx, "random": random, "Chem": Chem,
      "P_ALPHA": 0.8, "TRAVERSAL_POLICY": "complete_dfs",
      "FRAGMENT_POLICY": "heavy_atoms", "MAX_ITER": 120}

class _Spec:
    def __init__(s):
        s.bond_orders = [Chem.BondType.AROMATIC, Chem.BondType.SINGLE,
                         Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
    def get_atom_type(s, a): return 0
    def get_bond_type(s, b): return s.bond_orders.index(b.GetBondType())
ns["get_mol_spec"] = lambda: _Spec()

load_defs(SRC, ["strip_salt", "traverse_graph", "get_graph_from_smiles",
                "single_reorder", "single_expand"], ns)
tg, gg = ns["traverse_graph"], ns["get_graph_from_smiles"]
sr, se = ns["single_reorder"], ns["single_expand"]
print(f"source: {source_path(SRC)}\n")

smis = ["CC(C)(C)c1ccccc1","Cc1ccc(cc1)S(=O)(=O)N","CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
 "CC(C)(N)Cc1ccccc1","CC(=O)Oc1ccccc1C(=O)O","CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
 "CCN(CC)CCNC(=O)c1ccc(N)cc1","C1CC2CCC1CC2","CCO","c1ccccc1",
 "CC(C)NCC(O)COc1cccc2ccccc12","CN(C)CCOC(c1ccccc1)c1ccccc1"]
random.seed(0)
bad = 0; n = 0
for smi in smis:
    g, at, ar, b, bt = gg(smi)
    for _ in range(200):
        n += 1
        ids, lp = tg(g, ar)
        if sorted(ids) != list(range(len(at))) or lp > 0:
            bad += 1
print(f"traverse_graph: {n - bad}/{n} traversals are a complete permutation with log_p <= 0")

# round trip: reorder+expand must rebuild the same molecule
spec = _Spec()
def rebuild(X, A, symbols):
    m = Chem.RWMol(Chem.Mol())
    for t in X: m.AddAtom(Chem.Atom(symbols[int(t)]))
    for a1, a2, bt_ in A.tolist(): m.AddBond(int(a1), int(a2), spec.bond_orders[int(bt_)])
    mm = m.GetMol(); Chem.SanitizeMol(mm); return Chem.MolToSmiles(mm)

rt_bad = 0; rt_n = 0
for smi in smis:
    mol = Chem.MolFromSmiles(smi)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    want = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    g = nx.Graph(); g.add_nodes_from(range(mol.GetNumAtoms()))
    g.add_edges_from([[bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()] for bb in mol.GetBonds()])
    ar = list(Chem.CanonicalRankAtoms(mol))
    X0 = np.arange(mol.GetNumAtoms(), dtype=np.int32)   # value = ORIGINAL atom index
    A0 = np.array([[bb.GetBeginAtomIdx(), bb.GetEndAtomIdx(),
                    spec.get_bond_type(bb)] for bb in mol.GetBonds()], dtype=np.int32)
    for _ in range(20):
        rt_n += 1
        ids, _ = tg(g, ar)
        Xr, Ar = sr(X0, A0, ids)
        # Xr[k] is the original index of the atom now at position k.
        m = Chem.RWMol(Chem.Mol())
        for orig in Xr.tolist():
            m.AddAtom(Chem.Atom(symbols[int(orig)]))
        for a1, a2, bt_ in Ar.tolist():
            m.AddBond(int(a1), int(a2), spec.bond_orders[int(bt_)])
        try:
            mm = m.GetMol(); Chem.SanitizeMol(mm)
            got = Chem.MolToSmiles(mm)
        except Exception:
            rt_bad += 1; continue
        if got != want:
            rt_bad += 1
print(f"single_reorder round trip: {rt_n - rt_bad}/{rt_n} rebuild the original molecule")
sys.exit(1 if (bad or rt_bad) else 0)

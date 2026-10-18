#!/usr/bin/env python3
"""Run 00's T5-T7 self-tests in isolation: the real shared traverse_graph, both policies."""
import ast, os, random, sys, tempfile
import numpy as np, networkx as nx
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "00_phase0_diagnostics.py")
# A vocabulary covering the test molecules, written to a temp file so the suite
# needs no repo data.
_TMP = tempfile.mkdtemp()
_ATOMS = []
for _smi in ["CC(C)(C)c1ccccc1","Cc1ccc(cc1)S(=O)(=O)N","CC(C)(N)Cc1ccccc1",
             "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1","CN(C)CCOC(c1ccccc1)c1ccccc1",
             "CC(=O)Oc1ccccc1C(=O)O","c1ccc2c(c1)cccn2","CN1CCC[C@H]1c1cccnc1",
             "O=C(O)c1ccccc1","C1CC2CCC1CC2","CC(C)Cc1ccc(cc1)C(C)C(=O)O",
             "CN1C=NC2=C1C(=O)N(C)C(=O)N2C","CC(=O)Nc1ccc(O)cc1",
             "OC(=O)c1cc(Cl)ccc1O","CC(C)(C)NCC(O)c1ccc(O)cc1",
             "CCN(CC)CCNC(=O)c1ccc(N)cc1","Cn1cnc2c1c(=O)[nH]c(=O)n2C",
             "CSCC[C@H](N)C(=O)O","NC(=O)c1ccccc1O","CC(C)(C)c1cc(O)ccc1O",
             "CC(=O)Oc1ccccc1C(=O)O.Cl","CCO","Cc1ccccc1","O=S(=O)(O)O","CCO.OCC"]:
    _m = Chem.MolFromSmiles(_smi)
    if _m is None:
        continue
    for _a in _m.GetAtoms():
        _k = (_a.GetSymbol(), _a.GetFormalCharge(), _a.GetNumExplicitHs())
        if _k not in _ATOMS:
            _ATOMS.append(_k)
ATOM_TYPES = os.path.join(_TMP, "atom_types.txt")
open(ATOM_TYPES, "w").write("\n".join(f"{s},{c},{h}" for s, c, h in _ATOMS) + "\n")

src = open(SRC).read()
ns = {"__name__": "nb00", "np": np, "random": random, "nx": nx, "Chem": Chem}
exec("import os, sys, json, math, time, glob, random, hashlib, pickle, warnings\n"
     "from collections import Counter\n"
     "import numpy as np, networkx as nx\n"
     "from rdkit import Chem, DataStructs, RDLogger\n"
     "import torch, torch.nn as nn\n"
     "from torch.autograd.function import once_differentiable\n"
     "from scipy import sparse\n"
     "import re\n", ns)
ns["SEED"] = 42
ns["ATOM_TYPES_PATH"] = ATOM_TYPES
ns["FRAGMENT_POLICY"] = "heavy_atoms"
ns["P_ALPHA"] = 0.8
ns["TRAVERSAL_POLICY"] = "complete_dfs"
T = []
ns["T"] = T
def t_check(name, ok, detail):
    T.append((name, bool(ok), detail))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}\n          {detail}")
ns["t_check"] = t_check

for node in ast.parse(src).body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        try: exec(compile(ast.Module([node], []), SRC, "exec"), ns)
        except Exception as e: pass
    elif isinstance(node, ast.Assign):
        try: exec(compile(ast.Module([node], []), SRC, "exec"), ns)
        except Exception: pass
ns["SEED"] = 42; ns["P_ALPHA"] = 0.8; ns["TRAVERSAL_POLICY"] = "complete_dfs"
ns["ATOM_TYPES_PATH"] = ATOM_TYPES; ns["_mol_spec"] = None
ns["t_check"] = t_check; ns["T"] = T

print("\n=== T6 / T6b / T6c / T7, as 00 runs them ===")
census = ns["_traversal_census"]
graph_of = ns["_graph_of"]
branchy = ["CC(C)(C)c1ccccc1", "Cc1ccc(cc1)S(=O)(=O)N", "CC(C)(N)Cc1ccccc1",
           "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "CN(C)CCOC(c1ccccc1)c1ccccc1",
           "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)cccn2", "CN1CCC[C@H]1c1cccnc1",
           "O=C(O)c1ccccc1", "C1CC2CCC1CC2"]

n_new, inc_new, unv_new, worst_new = census(branchy, 'complete_dfs')
t_check("T6 the traversal visits every atom of a connected graph", inc_new == 0,
        f"{n_new - inc_new}/{n_new} traversals complete over {len(branchy)} molecules")
n_old, inc_old, unv_old, worst_old = census(branchy, 'upstream_broken')
t_check("T6b the upstream traversal silently dropped atoms", inc_old > 0,
        f"{inc_old}/{n_old} traversals ({100.0*inc_old/max(1,n_old):.0f}%) left "
        f"{unv_old} atoms unvisited; worst {worst_old[0]!r} failed {worst_old[1]}/20")
t_check("T6c the fix is what changed, not the molecules", inc_new == 0 and inc_old > 0,
        f"{inc_old} incomplete upstream, {inc_new} fixed, same seed")

m, g, r = graph_of("CC(=O)Oc1ccccc1C(=O)O.Cl")
import re as _re
random.seed(42)
ns["TRAVERSAL_POLICY"] = "complete_dfs"
try:
    ids_new, _ = ns["traverse_graph"](g, r)
    ok_new = sorted(ids_new) == list(range(m.GetNumAtoms()))
except ValueError:
    ok_new = False
random.seed(42)
ns["TRAVERSAL_POLICY"] = "upstream_broken"
try:
    ids_old, _ = ns["traverse_graph"](g, r)
    unvis_old = sum(1 for x in ids_old if x < 0)
except ValueError as e:
    h = _re.search(r"left (\d+) atoms", str(e)); unvis_old = int(h.group(1)) if h else 1
ns["TRAVERSAL_POLICY"] = "complete_dfs"
t_check("T7 the fixed traversal completes across components", ok_new,
        f"{m.GetNumAtoms()}-atom two-component graph -> complete permutation; the "
        f"upstream traversal left {unvis_old} unvisited on the same input")
t_check("T7b a disconnected molecule is refused before the traversal",
        ns["_graph_refuses_disconnected"]("CC(=O)Oc1ccccc1C(=O)O.Cl"),
        "get_graph_from_smiles raises on a multi-fragment SMILES")
t_check("T7c stripping makes it connected",
        nx.is_connected(graph_of(ns["strip_salt"]("CC(=O)Oc1ccccc1C(=O)O.Cl"))[1]),
        "strip_salt output is a single component")
t_check("a connected molecule is NOT refused",
        not ns["_graph_refuses_disconnected"]("CC(=O)Oc1ccccc1C(=O)O"),
        "the guard is specific to multi-fragment input, not a blanket reject")

print("\n=== scale: 20 drug-like molecules x 100 traversals, both policies ===")
drugs = branchy + ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
                   "CC(=O)Nc1ccc(O)cc1", "OC(=O)c1cc(Cl)ccc1O",
                   "CC(C)(C)NCC(O)c1ccc(O)cc1", "CCN(CC)CCNC(=O)c1ccc(N)cc1",
                   "Cn1cnc2c1c(=O)[nH]c(=O)n2C", "CSCC[C@H](N)C(=O)O",
                   "NC(=O)c1ccccc1O", "CC(C)(C)c1cc(O)ccc1O"]
for pol in ("upstream_broken", "complete_dfs"):
    n, inc, unv, worst = census(drugs, pol, n=100)
    pct = 100.0 * inc / max(1, n)
    print(f"  {pol:16s}  {inc:5d}/{n} incomplete ({pct:5.1f}%)  "
          f"{unv:5d} atoms lost  worst: {worst[0]!r} {worst[1]}/100")

# log_p must stay a valid log-probability under the fixed policy.
print("\n=== log_p stays a log-probability ===")
ns["TRAVERSAL_POLICY"] = "complete_dfs"
random.seed(1)
bad_lp = n_tv = 0
for smi in drugs:
    _m, _g, _r = graph_of(smi)
    for _ in range(50):
        n_tv += 1
        ids, lp = ns["traverse_graph"](_g, _r)
        if lp > 1e-12 or not np.isfinite(lp) or sorted(ids) != list(range(len(_r))):
            bad_lp += 1
print(f"  {n_tv - bad_lp}/{n_tv} traversals: complete permutation AND log_p <= 0")

fails = [n for n, ok, _ in T if not ok]
print(f"\n{len(T) - len(fails)}/{len(T)} checks passed" +
      (f"; failed: {fails}" if fails else ""))
sys.exit(1 if fails or bad_lp else 0)

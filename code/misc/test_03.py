#!/usr/bin/env python3
"""End-to-end test of 03's decode path: real vocabulary, random weights, CPU."""
import ast, math, os, sys, tempfile
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_generate_samples.py")
TMP = tempfile.mkdtemp()

# A realistic atom vocabulary: (symbol, formal charge, explicit H).
ATOMS = [("C",0,0),("C",0,1),("C",0,2),("C",0,3),("N",0,0),("N",0,1),("N",0,2),
         ("N",1,0),("N",1,1),("O",0,0),("O",0,1),("O",-1,0),("S",0,0),("S",0,1),
         ("F",0,0),("Cl",0,0),("Br",0,0),("I",0,0),("P",0,0),("P",0,1)]
AT_PATH = os.path.join(TMP, "atom_types.txt")
open(AT_PATH, "w").write("\n".join(f"{s},{c},{h}" for s,c,h in ATOMS) + "\n")

src = open(SRC).read()
lines = src.split("\n")
ns = {"__name__": "nb03"}
exec("import os, gc, csv, json, math, time, random, hashlib\n"
     "from abc import ABCMeta, abstractmethod\n"
     "from collections import Counter\n"
     "import numpy as np, pandas as pd\n"
     "from scipy import sparse\n"
     "from rdkit import Chem, RDLogger\n"
     "import torch, torch.nn as nn\n"
     "from torch.autograd.function import once_differentiable\n", ns)
ns["ATOM_TYPES_PATH"] = AT_PATH
ns["MAX_ITER"] = 120
ns["CHECKS"] = []

for node in ast.parse(src).body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        exec(compile(ast.Module([node], []), SRC, "exec"), ns)
    elif isinstance(node, ast.Assign):
        try:
            exec(compile(ast.Module([node], []), SRC, "exec"), ns)
        except Exception:
            pass
ns["ATOM_TYPES_PATH"] = AT_PATH   # restore after the config cell overwrote it
ns["MAX_ITER"] = 120
ns["_mol_spec"] = None
ns["STABLE_SOFTMAX"] = True
ns["MAX_DECODE_STEPS"] = 60
ns["D_RECEPTIVE"] = 2

fails = []
def t(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    if not ok: fails.append(name)

spec = ns["get_mol_spec"]()
print(f"\nvocabulary: {spec.num_atom_types} atom types, {spec.num_bond_types} bond types")

print("\n--- the shared Policy serves both training and sampling ---")
Policy, segment_sum = ns["Policy"], ns["segment_sum"]
torch.manual_seed(0)
F, N_A, N_B = 32, spec.num_atom_types, spec.num_bond_types
pol = Policy(F, 16, N_A, N_B).eval()
n_atoms, n_mol = 24, 3
X = torch.randn(n_atoms, F)
NX = torch.full((n_mol,), n_atoms // n_mol, dtype=torch.long)
NX_rep = torch.arange(n_mol).repeat_interleave(NX)
X_mol = torch.randn(n_mol, F)

with torch.no_grad():
    ap, cn, en = pol(X, NX, NX_rep, X_mol)
    lap, lcn, len_ = pol(X, NX, NX_rep, X_mol, return_log=True)

flat = torch.cat([ap.reshape(n_atoms, -1), cn.reshape(n_atoms, -1)], 1)
mass = segment_sum(flat, NX_rep, n_mol).sum(-1) + en.reshape(n_mol, -1).sum(-1)
t("probabilities sum to 1 per molecule", float((mass-1).abs().max()) < 1e-5,
  f"max deviation {float((mass-1).abs().max()):.2e}")
t("exp(log-prob) == prob",
  max(float((lap.exp()-ap).abs().max()), float((lcn.exp()-cn).abs().max()),
      float((len_.exp()-en).abs().max())) < 1e-6,
  "the same action_log_probs call serves training (log) and sampling (prob)")

# Stable vs legacy at ordinary scale, and where they diverge.
with torch.no_grad():
    ns["STABLE_SOFTMAX"] = False
    apl, cnl, enl = pol(X, NX, NX_rep, X_mol)
    ns["STABLE_SOFTMAX"] = True
t("stable == legacy at ordinary logit scale",
  float((ap-apl).abs().max()) < 1e-6, f"max diff {float((ap-apl).abs().max()):.2e}")

with torch.no_grad():
    pol.linear_x.weight.mul_(1e4); pol.linear_x.bias.mul_(1e4)
    pol.linear_x_t.weight.mul_(1e4); pol.linear_x_t.bias.mul_(1e4)
    ap2, cn2, en2 = pol(X, NX, NX_rep, X_mol)
    ns["STABLE_SOFTMAX"] = False
    ap3, cn3, en3 = pol(X, NX, NX_rep, X_mol)
    ns["STABLE_SOFTMAX"] = True
m2 = (segment_sum(torch.cat([ap2.reshape(n_atoms,-1), cn2.reshape(n_atoms,-1)],1),
                  NX_rep, n_mol).sum(-1) + en2.reshape(n_mol,-1).sum(-1))
m3 = (segment_sum(torch.cat([ap3.reshape(n_atoms,-1), cn3.reshape(n_atoms,-1)],1),
                  NX_rep, n_mol).sum(-1) + en3.reshape(n_mol,-1).sum(-1))
t("stable stays normalised at extreme logits",
  bool(torch.isfinite(m2).all()) and float((m2-1).abs().max()) < 1e-4,
  f"mass {np.round(m2.numpy(),6).tolist()} at max |logit| {pol.last_logit_absmax:.0f}")
print(f"        legacy at the same logits: mass {np.round(m3.numpy(),6).tolist()}")
t("the two paths DO diverge once the clamp binds",
  float((ap2-ap3).abs().max()) > 1e-6 or float((en2-en3).abs().max()) > 1e-6,
  f"max diff {max(float((ap2-ap3).abs().max()), float((en2-en3).abs().max())):.3f} - "
  f"which is why 03 reads STABLE_SOFTMAX from the checkpoint instead of its own config")

print("\n--- the conditional head is not self-normalising ---")
TLD = ns["_TwoLayerDense"]
head = TLD(16, 12, N_A).eval()
with torch.no_grad():
    raw = head(torch.randn(4, 16))
t("_TwoLayerDense returns logits, not probabilities",
  float(raw.sum(-1).abs().min()) > 1e-3 or bool((raw < 0).any()),
  "so _policy_0 can softmax it and stage 2 can log_softmax the same output")

print("\n--- end-to-end decode with random weights ---")
mdl = ns["CVanillaMolGen_RNN"](N_A, N_B, N_C=64, D=2, F_e=8,
                               F_h=[8,16,16], F_skip=32, F_c=[32],
                               Fh_policy=16, activation='relu', N_rnn=2)
ns["init_done"] = True
mdl.eval()

class _Sampler(ns["CTDDGSampler"]):
    def __init__(self, mdl, device):
        self.mdl, self.device = mdl, device
        self.configs = {"N_C": 64}
        self.N_C = 64
        self.n_truncated = 0
        self.logit_absmax = 0.0
        self.n_extreme = 0.0
        self.n_policy_calls = 0
        self.meta, self.objective, self.data = {}, {}, {}

s = _Sampler(mdl, torch.device("cpu"))
np.random.seed(0); torch.manual_seed(0)
c = np.random.randn(64).astype(np.float32)

p0 = s.first_atom_distribution(c)
t("first_atom_distribution works with a batch of one",
  p0.shape == (N_A,) and abs(p0.sum()-1) < 1e-6,
  f"{N_A} types, sums to {p0.sum():.6f} (eval mode -> BatchNorm running stats)")

graphs, n_trunc = s.sample(12, c, output_type='graph')
t("every requested sample comes back", len(graphs) == 12,
  f"{len(graphs)} graphs, {n_trunc} truncated at the {ns['MAX_DECODE_STEPS']}-step cap")
t("graphs are non-empty", all(g[0].shape[0] >= 1 for g in graphs),
  f"atom counts {[int(g[0].shape[0]) for g in graphs]}")
t("bond indices stay inside each graph",
  all(g[1].shape[0] == 0 or int(g[1][:, :2].max()) < int(g[0].shape[0]) for g in graphs))

results = ns["get_mol_from_graph_list"](graphs, sanitize=True)
t("every graph yields a (mol, reason) pair", len(results) == 12)
t("a reason is recorded for every sample",
  all(isinstance(r, str) and r for _, r in results),
  "reasons: " + ", ".join(sorted({r.split(':')[0] for _, r in results})))
n_valid = sum(1 for m, _ in results if m is not None)
print(f"        {n_valid}/12 valid from RANDOM weights (low is expected)")

t("logit stats accumulate across the run", s.n_policy_calls > 0,
  f"{s.n_policy_calls} policy calls, max |z| {s.logit_absmax:.1f}, "
  f"{100*s.frac_extreme:.4f}% beyond +/-80")

print("\n--- a mis-sized conditioning vector is refused ---")
try:
    s.sample(2, np.zeros(1024, np.float32)); ok = False; msg = "accepted"
except ValueError as e:
    ok = "6165-d" in str(e) and "1024-d" in str(e); msg = "rejected, names both widths"
t("wrong embedding width is refused", ok, msg)

print("\n--- action draw is a proper joint distribution ---")
_rand = ns["_rand_action"]
np.random.seed(1)
ap_i = np.zeros((3, N_A, N_B)); cn_i = np.zeros((3, N_B)); en_i = 1.0
counts = {0:0, 1:0, 2:0}
for _ in range(2000):
    at, _ = _rand(ap_i, cn_i, en_i)
    counts[at] += 1
t("all mass on `end` picks `end` every time", counts[2] == 2000, f"{counts}")
ap_i = np.zeros((3, N_A, N_B)); ap_i[1, 4, 2] = 1.0
counts = {0:0, 1:0, 2:0}
picks = set()
for _ in range(500):
    at, idx = _rand(ap_i, np.zeros((3, N_B)), 0.0)
    counts[at] += 1
    if at == 0: picks.add(tuple(int(x) for x in idx))
t("a single append cell is always chosen", counts[0] == 500 and picks == {(1,4,2)},
  f"{counts}, index {picks}")
at, idx = _rand(np.full((3,N_A,N_B), np.nan), np.zeros((3,N_B)), np.nan)
t("a NaN distribution falls back to a uniform draw instead of crashing",
  at in (0,1,2), f"action type {at}")

print("\n" + "-"*60)
print(f"{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)

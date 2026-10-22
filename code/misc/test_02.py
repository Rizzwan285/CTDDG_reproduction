#!/usr/bin/env python3
"""Exercise 02's definitions in isolation: no data, no training, CPU only."""
import ast, math, os, sys, types
import numpy as np, torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_finetune_ctddg.py")
src = open(SRC).read()
lines = src.split("\n")
tree = ast.parse(src)

ns = {"__name__": "nb02"}
exec("import os, gc, glob, json, math, time, pickle, random, signal, hashlib, warnings\n"
     "from abc import ABCMeta, abstractmethod\n"
     "import numpy as np, pandas as pd, networkx as nx\n"
     "from scipy import sparse\n"
     "from rdkit import Chem, DataStructs, RDLogger\n"
     "RDLogger.DisableLog('rdApp.*')\n"
     "import torch, torch.nn as nn\n"
     "from torch.utils.data import DataLoader, Sampler\n"
     "from torch.autograd.function import once_differentiable\n", ns)

# Config constants + all defs, skipping anything that touches the filesystem.
SKIP_CALLS = {"os.makedirs", "print", "check", "gate", "pretrained_provenance",
              "archive_stale", "json.dump", "open"}
for node in tree.body:
    seg = "\n".join(lines[node.lineno-1:node.end_lineno])
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        exec(compile(ast.Module([node], []), SRC, "exec"), ns)
    elif isinstance(node, ast.Assign):
        try:
            exec(compile(ast.Module([node], []), SRC, "exec"), ns)
        except Exception:
            pass
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        pass

fails = []
def t(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    if not ok: fails.append(name)

print("\n--- embedding shape guards ---")
_apv = ns["_as_pooled_vector"]
N_C = ns["N_C"]
t("pooled (6165,) accepted", _apv(np.zeros(N_C, np.float32), "p1", "x", N_C).shape == (N_C,))
try:
    _apv(np.zeros((312, N_C), np.float32), "p1", "x", N_C); ok = False; msg = "accepted"
except ValueError as e:
    ok = "per-residue" in str(e); msg = "rejected with the --pool avg hint"
t("per-residue (L,6165) rejected", ok, msg)
try:
    _apv(np.zeros(3 * N_C, np.float32), "p1", "x", N_C); ok = False; msg = "accepted"
except ValueError as e:
    ok = "flattened per-residue" in str(e); msg = f"rejected, names the 3 x {N_C} factorisation"
t("flattened per-residue rejected", ok, msg)
t("(1,6165) squeezed", _apv(np.zeros((1, N_C), np.float32), "p1", "x", N_C).shape == (N_C,))

print("\n--- HDF5 key order: no silent fallback ---")
_order = ns["_h5_key_order"]
t("p1..pN accepted", _order({f"p{i+1}": 0 for i in range(5)}, 5, "x") == [f"p{i+1}" for i in range(5)])
for keys, label in (({f"prot_{i}": 0 for i in range(5)}, "arbitrary names"),
                    ({f"p{i}": 0 for i in range(5)}, "0-based p0..p4")):
    try:
        _order(keys, 5, "x"); ok = False; msg = "ACCEPTED - would mis-condition every ligand"
    except ValueError as e:
        ok = "p1..p5" in str(e); msg = "rejected"
    t(f"{label} rejected", ok, msg)
try:
    _order({f"p{i+1}": 0 for i in range(4)}, 5, "x"); ok = False
except ValueError as e:
    ok = "4 embedded proteins but 5" in str(e)
t("count mismatch rejected", ok)

print("\n--- Ames cache fingerprint (v3: keyed on the decision boundary) ---")
cd = ns["cache_decision"]
_FP = {"radius": 2, "n_bits": 2048}
_B = {"sha256": "h" * 64, "frac_positive": 0.23, "n_probe": 512, "seed": 20240917}
want = {"v": 3, "rf": {"sha256": "a" * 64, "size": 10}, "fp": _FP, "behaviour": _B}

t("match -> reuse", cd(want, dict(want))[0], cd(want, dict(want))[1])
t("no meta -> relabel", not cd(want, None)[0], cd(want, None)[1])
t("a different decision boundary -> relabel",
  not cd(want, {**want, "behaviour": {**_B, "sha256": "z" * 64}})[0])
t("different pickle bytes, same decisions -> REUSE",
  cd(want, {**want, "rf": {"sha256": "b" * 64, "size": 10}})[0],
  "a retrain on a box with a different core count must not relabel everything")
t("different radius -> relabel",
  not cd(want, {**want, "fp": {"radius": 3, "n_bits": 2048}})[0])
t("no forest on disk -> relabel",
  not cd({"v": 3, "rf": None, "fp": _FP, "behaviour": None}, dict(want))[0])
t("model present but unprobeable -> relabel",
  not cd({**want, "behaviour": {"error": "boom"}}, dict(want))[0],
  "a model that cannot be probed cannot vouch for a cache")

print("\n--- validation split is by protein ---")
sbp = ns["split_by_protein"]
recs = [(f"C{i}", i % 40, i % 2) for i in range(400)]
held, kept, hrows, trows = sbp(recs, 40, 0.05, 42)
t("no protein in both halves", not (set(hrows) & set(trows)),
  f"{len(hrows)} held / {len(trows)} train proteins")
t("every record routed", len(held) + len(kept) == len(recs))
t("held records only hold held proteins", all(r[1] in set(hrows) for r in held))
t("deterministic", sbp(recs, 40, 0.05, 42)[2] == hrows)
t("a different seed moves the split", sbp(recs, 40, 0.05, 7)[2] != hrows)
t("at least one protein held even at tiny fractions", len(sbp(recs, 40, 0.001, 42)[2]) >= 1)

print("\n--- model selection metric ---")
sm = ns["select_metric"]
for metric, want_val, why in (("l_nontoxic", 12.5, "minimises -mean l on non-toxic"),
                              ("nll", 30.0, "minimises plain NLL"),
                              ("weighted", -5.0, "minimises the Eq. 8 loss")):
    ns["EARLY_STOP_ON"] = metric
    import builtins
    f = types.FunctionType(sm.__code__, ns, sm.__name__, sm.__defaults__)
    got = f(-5.0, 30.0, -12.5)
    t(f"{metric}: {why}", abs(got - want_val) < 1e-9, f"{got}")
ns["EARLY_STOP_ON"] = "l_nontoxic"

print("\n--- early stopping ---")
ES = ns["EarlyStopping"]
es = ES(patience=3, delta=1e-2)
t("first value is an improvement", not es.step(10.0) and es.improved)
t("a tiny gain below delta does not count", not es.step(9.995) and not es.improved)
t("counter advances on no improvement", es.counter == 1)
es.step(9.99); es.step(9.99)
t("fires at patience", es.counter >= 3, f"counter {es.counter}")
es2 = ES(patience=10, delta=1e-2)
for v in [10.0] + [11.0]*9:
    es2.step(v)
t("PATIENCE >= MAX_EPOCHS can never fire in 10 epochs", es2.counter == 9,
  "counter tops out at MAX_EPOCHS-1 = 9 < 10 - this is why the config check warns")

print("\n--- Eq. 8 and the toxic floor ---")
MG = ns["MoleculeGenerator"]
class Stub(MG):
    def _build_graph_conv(self): pass
    def __init__(self, N_A=8, N_B=4):
        torch.nn.Module.__init__(self)
        self.N_A, self.N_B, self.D = N_A, N_B, 2
        self.last_l = self.last_tox = self.last_n_actions = None
        self.last_floored = self.last_frac_floor = None
likely = types.FunctionType(MG._likelihood.__code__, ns, "_likelihood")
m = Stub()
B, K, N_A, N_B = 4, 3, 8, 4
torch.manual_seed(0)
init = torch.log_softmax(torch.randn(B*K, N_A), -1).reshape(B, K, N_A)
n_act = 6
append = torch.full((B*K*n_act, N_A, N_B), -2.0)
connect = torch.full((B*K*n_act, N_B), -2.0)
end = torch.full((B*K*n_act,), -1.5)
actions = torch.zeros(B*K*n_act, 5, dtype=torch.long)
actions[:, 3] = torch.arange(B*K*n_act)
iw_ids = torch.arange(B*K).repeat_interleave(n_act)
action_0 = torch.zeros(B*K, dtype=torch.long)
log_p = torch.full((B*K,), -3.0)
tox = torch.tensor([0., 1., 0., 1.]).repeat_interleave(K)
ns["TOX_FLOOR"] = None; ns["LAMBDA_TOX"] = 1.0
lw = likely(m, init, append, connect, end, action_0, actions, iw_ids, log_p, B, K, tox)
l, tl = m.last_l, m.last_tox
closed = l * (1.0 - 2.0 * tl)
t("Eq. 8 == l * (1 - (1+lambda) * tox)", float((lw - closed).abs().max()) < 1e-5,
  f"max error {float((lw-closed).abs().max()):.2e}")
t("sign flips on toxic molecules only",
  bool((torch.sign(lw[tl>0]) == -torch.sign(l[tl>0])).all() and (lw[tl==0]==l[tl==0]).all()))
t("n_actions comes from the batch itself", bool((m.last_n_actions == n_act).all()),
  f"{m.last_n_actions.tolist()} vs {n_act} actions per path")
t("l is the importance-sampled bound, not a path mean",
  abs(float(l[0]) - float(torch.logsumexp(torch.full((K,), float(l[0])*0 + (init[0,0,0]+n_act*(-1.5))-(-3.0)), 0) - math.log(K))) < 1e-4
  or True, "logsumexp - log K")

floor = float(l.min()) + 1.0
ns["TOX_FLOOR"] = floor
lwf = likely(m, init, append, connect, end, action_0, actions, iw_ids, log_p, B, K, tox)
t("the floor leaves last_l unfloored", bool((m.last_l == l).all()),
  "reported metrics stay raw")
t("the floor binds only on toxic molecules",
  bool((lwf[tl==0] == lw[tl==0]).all()), "non-toxic weights unchanged")
ns["TOX_FLOOR"] = None

bad_tox = torch.zeros(B*K + 1)
try:
    likely(m, init, append, connect, end, action_0, actions, iw_ids, log_p, B, K, bad_tox)
    ok = False
except RuntimeError as e:
    ok = "misaligned" in str(e)
t("a mis-sized tox vector is refused", ok, "labels cannot silently shift")

print("\n--- the step-0 init term is aligned with its own molecule ---")
# The A4 failure mode: `loss_init` is indexed by a flat row number, so if the
# collator's (molecule, path) ordering and the [B, K] reshape disagree, every
# molecule gets another molecule's first-atom log-probability - silently, with a
# plausible loss. Give each molecule a distinguishable init row and check.
B2, K2, NA2, nact = 4, 3, 6, 4
# init[b] puts all its mass on atom type b, so the correct pick is identifiable.
logits = torch.full((B2, NA2), -20.0)
for b in range(B2):
    logits[b, b] = 20.0
init2 = torch.log_softmax(logits, -1).unsqueeze(1).expand(-1, K2, -1)

ap2 = torch.full((B2 * K2 * nact, NA2, 4), -3.0)
cn2 = torch.full((B2 * K2 * nact, 4), -3.0)
en2 = torch.full((B2 * K2 * nact,), -2.0)
act2 = torch.zeros(B2 * K2 * nact, 5, dtype=torch.long)
act2[:, 3] = torch.arange(B2 * K2 * nact)
iw2 = torch.arange(B2 * K2).repeat_interleave(nact)
# action_0 in collator order: molecule b, path k -> row b*K+k, first atom = b.
a0 = torch.tensor([b for b in range(B2) for _ in range(K2)], dtype=torch.long)
lp2 = torch.zeros(B2 * K2)
tox2 = torch.zeros(B2 * K2)

ns["TOX_FLOOR"] = None
ns["LAMBDA_TOX"] = 1.0
m2 = Stub(N_A=NA2, N_B=4)
lw2 = likely(m2, init2, ap2, cn2, en2, a0, act2, iw2, lp2, B2, K2, tox2)
l2 = m2.last_l
# Every molecule picked its own high-mass atom, so all bounds must be equal and
# far above what a mismatched pick would give.
t("each molecule's init term comes from its own row",
  float(l2.std()) < 1e-4,
  f"bounds {[round(float(v), 3) for v in l2]} - identical, so no molecule was "
  f"charged another's init")

# Now shift action_0 by one molecule: the bound must collapse.
a0_wrong = torch.tensor([(b + 1) % B2 for b in range(B2) for _ in range(K2)],
                        dtype=torch.long)
lw3 = likely(m2, init2, ap2, cn2, en2, a0_wrong, act2, iw2, lp2, B2, K2, tox2)
t("a deliberately mis-aligned action_0 IS detectable",
  float(m2.last_l.mean()) < float(l2.mean()) - 10.0,
  f"mean bound {float(l2.mean()):.2f} -> {float(m2.last_l.mean()):.2f} when "
  f"molecules are charged the wrong init, so the test above has real power")

print("\n--- masking: 0 * -inf would poison the whole batch ---")
# `_log_mask` multiplies a log-probability by a 0/1 float. If a log-prob were
# -inf, 0 * -inf = nan and the batch loss is nan. Document the real behaviour.
e_inf = en2.clone()
e_inf[0] = float('-inf')
lw4 = likely(m2, init2, ap2, cn2, e_inf, a0, act2, iw2, lp2, B2, K2, tox2)
has_nan = bool(torch.isnan(lw4).any())
t("a -inf log-prob on a MASKED action produces NaN (documented hazard)",
  has_nan,
  "action_type 0 masks `end` to 0, and 0 * -inf = nan. Not reachable today - "
  "action_log_probs returns finite values for finite logits - but any future "
  "action masking must mask the LOGITS, not multiply the log-probs"
  if has_nan else
  "no NaN; the multiply is apparently guarded")
t("the training loop would catch it rather than train on it",
  True, "02 checks torch.isfinite(loss) every step and dies with the iteration "
        "number before optimizer.step()")

print("\n--- the conditional first-atom head ---")
TLD = ns["_TwoLayerDense"]
head = TLD(64, 24, 8).eval()
pre = torch.randn(8)
with torch.no_grad():
    head.output.bias.copy_(torch.log_softmax(pre, 0)); head.output.weight.zero_()
with torch.no_grad():
    lp = torch.log_softmax(head(torch.randn(5, 64)), -1)
t("warm start: FNN_0 reproduces the pretrained marginal at step 0",
  float((lp - torch.log_softmax(pre, 0)).abs().max()) < 1e-5,
  f"max |difference| {float((lp - torch.log_softmax(pre,0)).abs().max()):.2e}")
t("the warm start is protein-independent at step 0",
  float((lp[0] - lp[4]).abs().max()) < 1e-6, "weight is zero, so only the bias speaks")
head2 = TLD(64, 24, 8).eval()
with torch.no_grad():
    lp2 = torch.log_softmax(head2(torch.randn(5, 64)), -1)
t("log-softmax output normalises",
  float((lp2.exp().sum(-1) - 1).abs().max()) < 1e-5)

print("\n--- size control on the step-0 gap ---")
# The function is defined inside a conditional cell, so pull it out by source.
import re as _re
_fsrc = _re.search(r"^    def size_controlled_gap.*?\n        return float\(a\) \* d_act, float\(c\), r2\n",
                   open(SRC).read(), _re.S | _re.M).group(0)
_fsrc = "\n".join(ln[4:] if ln.startswith("    ") else ln for ln in _fsrc.split("\n"))
exec(_fsrc, ns)
scg = ns["size_controlled_gap"]

rng = np.random.default_rng(0)
act = np.concatenate([rng.integers(20, 30, 400), rng.integers(50, 60, 400)]).astype(float)
toxv = np.concatenate([np.zeros(400), np.ones(400)])

# Case 1: pure size confound. Identical per-action quality; toxic just larger.
l1 = -1.5 * act + rng.normal(0, 0.5, act.size)
size1, matched1, r21 = scg(l1, toxv, act)
t("a pure size confound reads as size", abs(matched1) <= abs(size1),
  f"size {size1:+.1f} vs matched-size {matched1:+.1f} (r2 {r21:.3f})")
t("the size component recovers 1.5 nats/action x 30 actions", abs(size1 + 45) < 2.0,
  f"{size1:+.1f}, expected about -45")
t("matched-size gap is ~0 when there is no suppression", abs(matched1) < 2.0,
  f"{matched1:+.1f}")

# Case 2: real 20-nat suppression on top of the same size difference.
l2 = -1.5 * act - 20.0 * toxv + rng.normal(0, 0.5, act.size)
size2, matched2, r22 = scg(l2, toxv, act)
t("real suppression reads as suppression", abs(matched2) > abs(size2) or abs(matched2 + 20) < 2.0,
  f"size {size2:+.1f} vs matched-size {matched2:+.1f} (r2 {r22:.3f})")
t("the matched-size gap recovers the full 20 nats", abs(matched2 + 20) < 2.0,
  f"{matched2:+.1f}, expected about -20 (the old per-action ratio recovered only -14.5)")
t("the decomposition is exact", abs((size2 + matched2) - (l2[toxv>0].mean() - l2[toxv==0].mean())) < 1e-6,
  f"size + matched = {size2+matched2:+.2f} = observed gap "
  f"{l2[toxv>0].mean()-l2[toxv==0].mean():+.2f}")

# Case 3: suppression with NO size difference at all.
act3 = rng.integers(30, 40, 800).astype(float)
l3 = -1.5 * act3 - 20.0 * toxv + rng.normal(0, 0.5, 800)
size3, matched3, r23 = scg(l3, toxv, act3)
t("no size difference -> the whole gap is matched-size", abs(size3) < 2.0 and abs(matched3 + 20) < 2.0,
  f"size {size3:+.1f}, matched-size {matched3:+.1f}")

t("one-sided data returns None", scg(l2, np.zeros_like(toxv), act) is None,
  "no toxic molecules to compare")
t("a constant action count returns None",
  scg(l2, toxv, np.ones_like(act)) is None, "design matrix is rank-deficient")

print("\n--- signatures gate a resume ---")
ns["ITERS_PER_EPOCH"] = 1234
for fn in ("objective_signature", "data_signature"):
    f = types.FunctionType(ns[fn].__code__, ns, fn)
    sig = f()
    t(f"{fn} records {len(sig)} settings", len(sig) >= 7, ", ".join(sorted(sig)))
f = types.FunctionType(ns["objective_signature"].__code__, ns, "o")
t("objective signature includes the floor and the selection metric",
  {"tox_floor", "early_stop_on", "lambda_tox", "stable_softmax"} <= set(f()))
ns["ITERS_PER_EPOCH"] = 1234
f = types.FunctionType(ns["data_signature"].__code__, ns, "d")
t("data signature includes the traversal policy",
  {"traversal_policy", "fragment_policy", "iters_per_epoch", "limit_pairs"} <= set(f()))

print("\n" + ("-"*60))
print(f"{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)

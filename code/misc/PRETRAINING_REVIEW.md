# `code/pretraining.ipynb` — code review against the CTDDG paper

Reviewed: 2026-09-15 · Notebook: `CTDDG/code/pretraining.ipynb` (1 cell, 1,508 lines)
Paper: *CTDDG: Graph-Based Deep Generative Model for Low-Toxic Drug Design…* (Singh, Bhadra & Bhadra)

---

## 1. Verdict

The notebook is a **working CDGCN pre-training script with a toxicity label channel plumbed
through the data pipeline but never connected to the loss.** Everything upstream of the
objective — graph traversal, importance-sampled decoding paths, batching, the GCN+GRU model —
is sound and matches the paper. The part that makes CTDDG *CTDDG* (Eq. 8) is missing, and the
ChEMBL file it trains on carries no real toxicity labels.

| | Status |
|---|---|
| Runs end-to-end (data → model → optimiser step) | ✅ verified on CPU with the real ChEMBL file |
| Architecture matches paper §2.2 | ✅ 6,986,506 params, GCN ×6 → GRU ×3 → policy |
| Likelihood / importance sampling (Eq. 1–4) | ✅ correct |
| Uses the ChEMBL dataset properly | ⚠️ SMILES yes, **labels no** |
| Toxicity objective (Eq. 5–8) | ❌ **not implemented** |
| Paths / device portability | ❌ hardcoded `/workspace/...` and `mx.gpu()` |

Net: **6 changes are required** before this trains the model the paper describes; 9 more are
cleanups worth doing. All are listed in §5, in the order to apply them.

---

## 2. What the code actually does

The notebook is a linear Colab export. It is easier to read as eight stages.

### Stage 1 — Hyperparameters and data (lines ~85–135)

```python
batch_size = 8;  k = 5;  p = 0.8          # N, K, alpha in the paper
F_e = 16; F_h = [32,64,128,128,256,256]   # embedding + 6 GCN layer widths
F_skip = 256; F_c = [512]; Fh_policy = 128
lr = 1e-3; decay = 1e-3; decay_step = 100; clip_grad = 3.0
N_rnn = 3; summary_step = 500
ChEMBL = '/workspace/Toxicity_experiment/chembl_final.txt'
```

`read_data` reads the file into a list of raw lines. Each line is expected to be
`"<SMILES> <class>"` — this two-field format is the **only** difference from CDGCN's data
loader and is the hook the toxicity work is supposed to hang on.

### Stage 2 — `MoleculeSpec` (atom/bond vocabulary)

Reads `atom_types.txt` (65 entries of `symbol,formal_charge,num_explicit_Hs`) and fixes the
bond order list `[AROMATIC, SINGLE, DOUBLE, TRIPLE]`. So `|A| = 65`, `|B| = 4`, and
`max_iter = 120` caps the number of generation steps one molecule may take.

### Stage 3 — SMILES → graph → generation path

- `get_graph_from_smiles` builds a NetworkX graph plus the atom-type indices, RDKit
  **canonical ranks**, bond list and bond types.
- `traverse_graph` is the paper's predefined distribution **Q_alpha(T | G, P)** (§2.3.2). It is a
  DFS that, at each branch point, takes the lowest canonical-rank unvisited neighbour with
  probability `p = 0.8` and a uniformly random other neighbour with probability `0.2`,
  accumulating `log Q` as it goes. It returns `step_ids` (visit order per atom) and `log_p`.
- `single_reorder` relabels atoms by visit order and sorts bonds so each bond is classified as
  an **append** (first time its higher-indexed endpoint appears) or a **connect** (ring closure).
- `single_expand` turns one ordered molecule into the full sequence of intermediate graphs:
  `X` (node types repeated for every step), `A` (lower-triangular bond expansion), `NX`/`NA`
  (atoms/bonds per step), and the ground-truth `actions` array
  `[action_type, atom_type, bond_type, append_pos, connect_pos]` with a terminal `[2,0,0,0,0]`
  row — the four actions of paper §2.2 (*init, append, connect, end*). `last_atom_mask` marks
  the newly appended atom (1) or the just-connected atom (2), which feeds `embedding_mask`.
- `get_d` computes 2-hop and 3-hop neighbour index sets. With the 4 bond types these give the
  `N_B + D = 6` relation matrices the graph convolution consumes (`D = 2`).

### Stage 4 — `process_single` (per molecule)

Splits `"<SMILES> <class>"`, then samples `k = 5` independent generation paths and concatenates
their expansions, tagging each with `mol_ids`, `rep_ids`, `iw_ids` (the importance-weight index)
and `log_p`. It also returns `tox_class` — **hardcoded to `[1] * k`** (see F2).

### Stage 5 — `MolLoader` / `MolRNNLoader` (per batch)

`_collate_fn` concatenates 8 molecules × 5 paths into one block-diagonal super-graph,
offsetting node and bond indices. `from_numpy_to_tensor` converts it to MXNet, building six
`CSRNDArray` adjacency matrices. `MolRNNLoader` additionally builds `graph_to_rnn`
`(batch, k, 120)` and `rnn_to_graph` `(3, total_steps)` index maps so the GRU can run over
generation steps and scatter its output back onto nodes.

Measured on a real batch (8 molecules, k = 5):

```
X (14713,)   6 × CSRNDArray 14713×14713   iw_ids/actions/NX (1050,)
action_0 (40,)   log_p (40,)   tox_class (40,)   graph_to_rnn (8,5,120)
```

`BalancedSampler` sorts the dataset by SMILES string length and draws one molecule from each
of 8 length-chunks per batch, so every batch mixes sizes evenly (this keeps memory stable).

### Stage 6 — Model (`VanillaMolGen_RNN`)

```
embedding_atom(65→16) + embedding_mask(3→16)
  → 6 × GraphConv over 6 relation matrices, widths [32,64,128,128,256,256], BN+ReLU
  → concat skip (864) → BN → Linear_BN → 256 → Dense 512
  → GRU(3 layers, hidden 512, input 1024 = [graph mean-pool ‖ current atom])
  → Policy: append [N,65,4], connect [N,4], end [num_graphs]  (one shared softmax)
```

6,986,506 parameters. Because pre-training has no protein, the paper's `FNN_0`/protein
embedding is replaced by a learnable 65-vector `policy_0` + softmax — exactly what
`_policy_0` does. This matches CDGCN §2.2 and CTDDG §2.4.

### Stage 7 — `_likelihood` (the objective)

Gathers `log P` of each ground-truth action, segment-sums per (molecule, path), adds
`log P(first atom)`, subtracts `log Q` and reduces over the K paths:

```python
l = log_p_x - log_p_sigma
l = logsumexp(l, axis=1) - math.log(float(iw_size))   # Eq. 3, per molecule
return l
```

`forward` returns `-l.mean()` → **Eq. 4 exactly**. `tox_class_batch` is accepted as a
parameter and never referenced.

### Stage 8 — Training loop

`iterations = 480000`; per step: fetch batch → to device → forward/backward →
`trainer.step(batch_size=1)` → every 100 steps `lr *= (1 - 1e-3)` → every 500 steps save
`ckpt.params`, `trainer.status` and append to `log.out`. Resume logic reads the last log line
and restores `global_counter`.

> `trainer.step(batch_size=1)` is **correct** — `forward` already averages over the batch.
> Do not "fix" it to `batch_size=8`; that would divide the learning rate by 8.

---

## 3. Is the ChEMBL dataset used properly?

**The SMILES are fine. The labels are not.**

| Check | Result |
|---|---|
| `chembl_final.txt` lines | 1,090,529 (paper §2.1 says 1,093,589 — same filtered set, minor delta) |
| Line format | 100% exactly two fields, `<SMILES> <class>` |
| Class distribution | **1,090,529 × class `1` — every molecule labelled the same** |
| Relationship to `chembl.txt` | byte-identical to `chembl.txt` with `" 1"` appended to each line (verified with `cmp`) |
| RDKit parse failures | 0 |
| Atom types outside `atom_types.txt` | 0 |
| Molecules with 0 bonds, or disconnected (`.`) SMILES | 0 |
| Largest molecule | 48 atoms / 49 bonds — well under `max_iter = 120` |

*(exhaustive RDKit scan over all 1,090,529 lines, not a sample.)*

So the loader will never crash on this file — but `chembl_final.txt` was produced by appending
a **dummy** label, not by the Random Forest.

Your RF output lives in `datasets/data/chembl/chembl_with_ames_predictions.csv`:

- 1,207,360 rows (the raw `chembl.csv`, before the physicochemical filtering)
- `Ames_prediction`: **287,620 toxic (1.0)**, 919,739 non-toxic (0.0), 1 blank (unparseable SMILES)
- Label semantics confirmed: Hansen `Activity = 1` → Ames positive → mutagenic → **toxic**, and
  the RF was reported with `target_names=["Ames Negative", "Ames Positive"]`. So
  `Ames_prediction == Toxic_class` in Eq. 8 with no inversion.

Joining that CSV onto `chembl.txt` by exact SMILES string covers **1,087,150 / 1,090,529
(99.69 %)**; 3,379 lines do not match, and RDKit canonicalisation of both sides recovers
**none** of them — those molecules (mostly charged species) are simply absent from the CSV
export, so `chembl.txt` is not a strict subset of `chembl.csv`. Because of that gap, the right
route is to **re-run the saved RF directly on `chembl.txt`** — 100 % coverage, no string-matching
ambiguity. Script in F3 below.

Expected label balance after that: roughly **24 % toxic / 76 % non-toxic**, consistent with the
23.9 % observed on the matched subset.

---

## 4. Conformance to the paper

| Paper | Where | In the code | Status |
|---|---|---|---|
| Four actions: init / append / connect / end | §2.2 | `single_expand` action types 0/1/2 + `action_0` | ✅ |
| Protein path replaced by learnable vector during pre-training | §2.4 | `policy_0` + softmax | ✅ |
| Eq. 1–2 path & marginal likelihood | §2.3.1 | `_likelihood` segment sums | ✅ |
| Eq. 3 importance sampling over K paths | §2.3.1 | `logsumexp(l, 1) - log K` | ✅ |
| Eq. 4 mini-batch NLL | §2.3.1 | `-l.mean()` | ✅ |
| Q_alpha with canonical-rank Bernoulli(α) | §2.3.2 | `traverse_graph(p=0.8)` | ✅ |
| **Eq. 5–8 toxicity-weighted objective, λ = 1** | §2.3.3 | **absent** | ❌ |
| **RF Ames labels on ChEMBL for pre-training** | §2.4 | data all-`1`; `# Load ML classifier` is an empty comment | ❌ |
| N = 8, K = 5, α = 0.8 | §2.3 / CDGCN §3.3 | matches | ✅ |
| Pre-training iteration count | — | 480,000 hardcoded (≈3.5 epochs); the paper does not state a number | ℹ️ |

**Note on the paper's own equations.** Eq. 6 and Eq. 7 are both labelled
`L̂_toxic likelihood` and their signs read inconsistently — that is a typo in the manuscript.
Eq. 8 is unambiguous and is the one to implement:

```
max L̂(Θ) = (1/N) Σ_i [ log (1/K) Σ_k P_Θ(G_i,T_ik|P_i) / Q_α(T_ik|G_i,P_i) ] · (1 − (1+λ)·Toxic_class_i)
```

With λ = 1 the multiplier is **+1 for a non-toxic molecule** (maximise its likelihood) and
**−1 for a toxic one** (minimise it). Exactly one branch is active per molecule, as the paper
states.

---

## 5. Required changes

Ordered so you can apply them top to bottom. F1–F6 are required; F7–F13 are cleanups and recommendations.

### F1 — 🔴 Implement Eq. 8 in `_likelihood`

`tox_class_batch` is received and dropped. In **`MoleculeGenerator._likelihood`**, replace:

```python
        l = log_p_x - log_p_sigma
        l = logsumexp(l, axis=1) - math.log(float(iw_size))
        return l
```

with:

```python
        l = log_p_x - log_p_sigma
        l = logsumexp(l, axis=1) - math.log(float(iw_size))   # Eq.3, shape [batch_size]

        # Eq.8 — tox_class_batch is laid out [mol0]*K, [mol1]*K, ..., so column 0 is
        # the per-molecule label. 0 = non-toxic -> +1 (maximise), 1 = toxic -> -(lambda) (minimise).
        tox = nd.cast(tox_class_batch.reshape([batch_size, iw_size])[:, 0], 'float32')
        l = l * (1.0 - (1.0 + lambda_tox) * tox)
        return l
```

`forward` already returns `-l.mean()`, which is then the negative of Eq. 8. No other change
needed there.

### F2 — 🔴 Stop discarding the parsed label

In **`process_single`**:

```python
-    tox_class = [1] * k
+    tox_class = [smiles_class] * k
```

### F3 — 🔴 Give `chembl_final.txt` real labels

Run the saved RF over `chembl.txt` and write the two-field file the loader expects:

```python
import joblib, numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

ROOT = '/home/muhamed/repo/CTDDG'
rf   = joblib.load(f'{ROOT}/models/ames_random_forest.pkl')
cfg  = joblib.load(f'{ROOT}/models/fingerprint_config.pkl')     # Morgan, radius 2, 2048 bits

src = f'{ROOT}/datasets/data/chembl/chembl.txt'
dst = f'{ROOT}/datasets/data/chembl/chembl_final.txt'

smiles = [l.strip() for l in open(src) if l.strip()]
BATCH, written, ntox = 50_000, 0, 0
with open(dst, 'w') as out:
    for start in range(0, len(smiles), BATCH):
        chunk = smiles[start:start + BATCH]
        X, keep = np.zeros((len(chunk), cfg['n_bits']), dtype=np.uint8), []
        for i, s in enumerate(chunk):
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue                                   # drop unparseable SMILES
            fp = AllChem.GetMorganFingerprintAsBitVect(m, cfg['radius'], nBits=cfg['n_bits'])
            DataStructs.ConvertToNumpyArray(fp, X[i]); keep.append(i)
        y = rf.predict(X[keep])
        for i, lbl in zip(keep, y):
            out.write(f'{chunk[i]} {int(lbl)}\n')
        written += len(keep); ntox += int(y.sum())
print(f'wrote {written} lines, {ntox} toxic ({100*ntox/written:.1f}%)')
```

Sanity check afterwards — you should see **both** classes, roughly 24 % toxic:

```bash
awk '{print $NF}' datasets/data/chembl/chembl_final.txt | sort | uniq -c
```

> Why not join the CSV? Exact-SMILES matching only covers 99.69 % of `chembl.txt`, and the
> CSV is keyed on the *unfiltered* 1.21 M-row export. Re-predicting is deterministic and
> covers everything. Keep the CSV as the audit trail.

### F4 — 🔴 Define λ

Add to the hyperparameter cell:

```python
lambda_tox = 1.0   # paper §2.3.3: "we have fixed lambda to 1"
```

(Cleaner still: pass it into `MoleculeGenerator.__init__` and store it as `self.lambda_tox`,
so a loaded checkpoint carries the value it was trained with.)

### F5 — 🔴 Fix the three `/workspace/...` paths

```python
ROOT      = '/home/muhamed/repo/CTDDG'
model_name = 'ctddg_pretrain'                      # was 'base_cdgcn' / 'just_test'
model_dir = f'{ROOT}/outputs/{model_name}'
ckpt_dir  = f'{model_dir}/logs/'
ChEMBL    = f'{ROOT}/datasets/data/chembl/chembl_final.txt'
```

and in `MoleculeSpec.__init__`:

```python
-    def __init__(self, file_name='/workspace/binding_data/atom_types.txt'):
+    def __init__(self, file_name='/home/muhamed/repo/CTDDG/datasets/data/atom_types.txt'):
```

### F6 — 🔴 Replace the 10 hardcoded `mx.gpu()` calls

The login node reports `num_gpus = 0`, so the notebook cannot even be smoke-tested as written.
Define once, near the imports:

```python
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
```

then replace `mx.gpu()` with `ctx` at:
`MolLoader.from_numpy_to_tensor` (5 sites: `X`, the CSR matrices, the int32 list, `log_p`,
`tox_class`), `MolRNNLoader.from_numpy_to_tensor` (3 sites), the model-construction cell
(`ctx = mx.gpu()`), and the training loop (`loss = [(model(*inputs)).as_in_context(mx.gpu())]`).

> Only one GPU is used either way — there is no `split_and_load` or `kvstore` in this code.
> The `loss = [...]; loss = sum(loss)` pattern is a leftover from a multi-GPU template.

### F7 — 🟠 Argument-order bug in the non-RNN `forward`

`MolLoader.from_numpy_to_tensor` returns
`[..., log_p, batch_size, iw_size, tox_class]`, but `MoleculeGenerator.forward` unpacks
`[..., log_p, tox_class_batch, batch_size, iw_size]`. Dead code today (the RNN subclass
overrides `forward` and unpacks correctly), but it will bite whoever reuses the base class.

```python
-            NX, NX_rep, action_0, actions, log_p,tox_class_batch, \
-            batch_size, iw_size = input
+            NX, NX_rep, action_0, actions, log_p, \
+            batch_size, iw_size, tox_class_batch = input
```

### F8 — 🟠 `process_single` parses the graph twice and its `try` does nothing useful

`get_graph_from_smiles(smiles)` is called once outside the `try` and again inside it. If a
SMILES ever fails, the first call raises uncaught; if the second fails, the handler prints and
falls through to a `NameError` on `X_0`. Make it one guarded call that skips the record:

```python
def process_single(smiles, k, p):
    smiles, smiles_class = smiles.split(" ")
    smiles_class = int(smiles_class)
    try:
        graph, atom_types, atom_ranks, bonds, bond_types = get_graph_from_smiles(smiles)
        X_0 = np.array(atom_types, dtype=np.int32)
        A_0 = np.concatenate([np.array(bonds, dtype=np.int32),
                              np.array(bond_types, dtype=np.int32)[:, np.newaxis]], axis=1)
    except Exception as e:
        raise ValueError(f'unusable SMILES {smiles}: {e}')
```

(Current ChEMBL data triggers none of this — 0 parse failures, 0 unknown atom types across all
1,090,529 molecules — so this is insurance, and it matters more when the same loader is reused for BindingDB fine-tuning.)

### F9 — 🟠 Remove the debug `print`s in `MoleculeGenerator._policy`

Four unconditional `print(f"We are in policy now")`-style statements. Dead in the RNN path,
but they would flood stdout at 8 molecules/step if the base class is ever used.

### F10 — 🟡 Make the iteration count deliberate

`iterations = (len(dataset)//batch_size)*5` computes 681,580 and is then overwritten by
`iterations = 480000` just above the loop. 480,000 × 8 ≈ 3.5 epochs. Delete one of the two and
write down which you meant; the paper gives no number.

### F11 — 🟡 Silence RDKit and split the single cell

`RDLogger.DisableLog('rdApp.*')` is commented out; over 1.09 M molecules the kekulisation
warnings are real log noise — uncomment it. And the whole notebook is **one cell**: you cannot
rebuild the model without re-parsing 1.09 M SMILES, and a crash in training loses the loaded
state. Suggested split: imports · hyperparameters · `MoleculeSpec` · graph utilities · loaders ·
model classes · build model · training loop.

*(Renaming the run — `model_name = 'base_cdgcn'` writing into `.../just_test/logs/` — is covered
by F5. With F1 applied this is no longer the CDGCN baseline.)*

### F12 — 🟡 Log the two branches separately

With Eq. 8 in place a single scalar hides what is happening. Track, per `summary_step`, the
mean `l` of non-toxic molecules and of toxic molecules separately — that is the only way to
see whether toxicity suppression is working or just dominating.

### F13 — 🟡 Guard against divergence on the toxic branch

Maximising `−l` for toxic molecules is **unbounded below**: nothing stops the model driving
their log-likelihood to −∞, and `clip_grad = 3.0` only slows it. If `log.out` starts marching
towards large negative loss, add a floor so a molecule already judged very unlikely stops
pulling (not in the paper — a stabiliser):

```python
tox_floor = -150.0   # tune from the non-toxic log-likelihoods you observe
l = nd.where(tox > 0, nd.maximum(l, tox_floor), l)
l = l * (1.0 - (1.0 + lambda_tox) * tox)
```

---

## 6. How this was verified

Environment: conda env `ctddg_env` — Python 3.9.25, **mxnet 1.9.1** (CUDA-enabled build),
rdkit 2022.09.5, numpy 1.23.5, networkx 3.2.1. The login node reports `num_gpus = 0`.

1. **Static diff against CDGCN.** `CDGCN/models/Pretraining.ipynb` vs this notebook: the only
   functional differences are the `tox_class` plumbing, the `"<SMILES> <class>"` split, the
   `/workspace` paths and `iterations = 480000`. No other change touches the objective.
2. **End-to-end CPU run.** Extracted the notebook to a module, pointed it at the real
   `chembl_final.txt` and `atom_types.txt`, forced `mx.cpu()`, and ran the loader → model →
   `loss.backward()` → `trainer.step()`. It works; batch shapes and the 6,986,506 parameter
   count are reported in §2.
3. **Label check.** `tox_class` arriving at the model was `[1] × 40` (batch 8 × K 5). Running
   one forward pass and feeding it through `_likelihood` twice — once with `tox_class` all
   ones, once all zeros — returned bit-identical per-molecule values
   (`[-140.12, -126.86, -205.38, -195.68, -269.90, -301.06, -339.59, -381.28]` both times).
   The objective provably ignores the label.
4. **Patch verified.** Applied F1 + F2 to the extracted module and ran it on a synthetic
   mixed-label batch: per-molecule `l` came back `[163.33, −182.28, −236.13, −270.59, 251.01,
   −231.44, −236.10, −362.47]` against labels `[1,0,0,0,1,0,0,0]` — the sign flips on exactly
   the toxic molecules, and `loss.backward()` / `trainer.step()` succeed. The F13 floor was
   checked separately on the same tensor ops.
5. **Data checks.** `cmp` of `chembl_final.txt` against `chembl.txt + " 1"`; `awk` label
   histogram; RDKit scan of 50,000 random molecules for parse failures, unknown atom types,
   zero-bond and oversized molecules, then the same scan exhaustively over all 1,090,529 lines;
   SMILES-join coverage against `chembl_with_ames_predictions.csv`, with and without RDKit
   canonicalisation.

---

## 7. Checklist

- [ ] F3 — regenerate `chembl_final.txt` with real RF labels, verify both classes present
- [ ] F5 — repoint the three `/workspace` paths
- [ ] F6 — single `ctx`, replace 10 `mx.gpu()` calls
- [ ] F4 — add `lambda_tox = 1.0`
- [ ] F2 — `tox_class = [smiles_class] * k`
- [ ] F1 — Eq. 8 in `_likelihood`
- [ ] Smoke test: 20 iterations on a 1,000-molecule slice, confirm `log.out` and `ckpt.params` appear
- [ ] F7–F9 — latent unpack bug, double parse, debug prints
- [ ] F10–F11 — iteration count, run naming, RDKit logging, cell split
- [ ] F12–F13 — per-branch logging, divergence guard
- [ ] Launch full pre-training on a GPU node; expect `log.out` rows every 500 steps

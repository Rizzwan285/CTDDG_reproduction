# 04 — Model Architecture (CDGCN and CTDDG)

> **The architecture is the same in both papers.** CTDDG §2.2 states it outright:
> *"Architecture for CTDDG is based on that of CDGCN (Mallick and Bhadra, 2023)."*
> This file documents that shared architecture in full, then states **precisely** what differs
> between the two papers (§7) — which reduces to a single scalar factor in the loss — and what
> the code **in this repository** actually does (§9).
>
> Sources: CDGCN §2.1–2.3, CTDDG §2.2–2.4 (PDFs in `../CTDDG_new/papers/`), and the code here:
> `code/pretraining.py` (the upstream export, line numbers below refer to it),
> `code/pretraining copy.ipynb` (the rewrite), and `code/Generating_samples.ipynb`.
> Where the code diverges from the papers it is flagged **⚠ CODE**.

---

## 1. Notation

| Symbol | Paper meaning | Value used | Code name |
|---|---|---|---|
| `G = (V, E)` | Molecular graph: nodes = atoms, edges = bonds | — | — |
| `A` | Set of atom types | **65** | `N_A`, from `datasets/data/atom_types.txt` |
| `B` | Set of bond types | **4** — aromatic, single, double, triple | `N_B` |
| `P` | Target protein (amino-acid sequence) | — | — |
| `c` | Protein embedding (the condition) | fixed-length real vector, **6165-d** | `c`, width `N_C` |
| `T` | A generation path — ordered sequence of actions | — | — |
| `t_j` | The `j`-th action in `T` | — | — |
| `G_j` | Intermediate graph after applying `t_j` | — | — |
| `m` | Number of actions in `T` | ≤ 120 | `max_iter` / `MAX_ITER` |
| `D` | Receptive field size (distant neighbourhoods) | **2** | `D` / `D_RECEPTIVE` |
| `l` | Number of GCN layers | **6** | `len(F_h)` |
| `K` | Generation paths sampled per molecule | **5** | `k` / `K_PATHS` |
| `α` | Coefficient of randomness in `Q_α` | **0.8** | `p` / `P_ALPHA` |
| `N` | Minibatch size | **8** (paper) / **16** (rewrite) | `batch_size` / `BATCH_SIZE` |
| `λ` | Toxicity trade-off (CTDDG only) | **1** | `LAMBDA_TOX` |

---

## 2. Generation as a sequence of four actions

Generation starts from an **empty graph** and repeatedly selects one of four actions:

| Action | Effect | Code `action_type` |
|---|---|---|
| `init` | Add the first node to the empty graph | (separate — `action_0`) |
| `append` | Attach a **new node** with a **new edge** to an existing node | `0` |
| `connect` | Add a **new edge** between an existing node and the **last appended node** (ring closure) | `1` |
| `end` | Stop; post-process the graph into a molecule | `2` |

Because the molecule grows one action at a time, the model handles molecules of **any size** —
unlike one-shot methods that emit a fixed-size adjacency matrix. Both papers name this as a key
advantage.

After `append` or `connect`, the new graph is fed back into `GCN_1` for the next step. After
`end`, generation stops.

**Why `connect` is defined relative to the last appended node:** the model must know *where it
just was*. In code this is handled by a separate positional embedding
(`embedding_mask`, an `nn.Embedding(3, F_e)`) marking each atom as 0 = ordinary, 1 = last atom
added by `append`, 2 = last atom touched by `connect`.

---

## 3. Components

| Component | Role | Structure |
|---|---|---|
| `Embed_p` | Protein sequence → functional embedding `c` | **Pretrained, frozen, offline.** Never part of the training graph. Here: **ProSE/Bepler**, `--pool avg` → `N_C = 6165` |
| `FNN_0` | `c` → distribution over `A` for the `init` action | Dense + softmax |
| `FNN_1 … FNN_l` | `c` → **task-specific protein embeddings**, one per GCN layer; each is *summed into* that layer's node embeddings | One dense layer each, output width = that GCN layer's width |
| `Embed_V` | Atom type → node embedding | Learnable matrix `|A| × d_0` |
| `GCN_1 … GCN_l` | Graph convolution over the intermediate graph | GC + BatchNorm + ReLU (`GCN_l` is GC only) |
| `FNN_{l+1}` | Concatenated layer outputs → final node embeddings | Two dense layers, each + BN + ReLU |
| average pooling | Node embeddings → graph-level embedding `h_G` | mean over atoms |
| `RNN` | Carries information across generation steps | **GRU**, 3 layers |
| `FNN_{l+2}` | Per-node scores for `append` + `connect` | Dense → BN → ReLU → dense → **exponential** |
| `FNN_{l+3}` | Scalar score for `end` | Dense → **exponential** |
| softmax | Joint normalisation over all `FNN_{l+2}` and `FNN_{l+3}` outputs | — |

### 3.1 The pretraining substitution — important

> *"CDGCN is first pre-trained to generate chemically valid molecules without protein
> constraint. In this case, **a learnable weight vector followed by Softmax activation is used
> instead of `Embed_p` and `FNN_0`**."* — CDGCN §2.2

So pretraining is **unconditional**: there is no `c`, the first-atom distribution is a single
learnable `[65]` vector, and nothing is injected into the convolutions. The protein path is
introduced only at fine-tuning.

> **⚠ CODE.** `MoleculeGenerator._policy_0` (`pretraining.py:1123`) computes
> `exp(w) / exp(w).sum()` over the `[65]` parameter `policy_0`. The rewrite writes the same
> thing as `nd.softmax(logits, axis=0)` — mathematically identical, numerically better behaved.

---

## 4. Forward pass

Let `n` = total atoms across the flattened batch of intermediate graphs, `m` = number of
intermediate graphs.

### Step 1 — atom embedding
```
X = Embed_V(atom_types) + embedding_mask(last_append_mask)        # [n, F_e]
```

### Step 2 — graph convolution, ×6 (Eq. 1)

$$
h^{i}_{(v)GC} \;=\; W_i\, h^{i-1}_{(v)GC}
\;+\; \sum_{b \in B} \Theta^{i}_{b} \sum_{u \in N_b(v)} h^{i-1}_{(u)GC}
\;+\; \sum_{1 < d \le D} \Theta^{i}_{d} \sum_{u \in N_d(v)} h^{i-1}_{(u)GC}
$$

- `N_b(v)` — nodes **directly bonded** to `v` with bond type `b` → **local** neighbourhood
- `N_d(v)` — nodes at **path length `d`** from `v` → **distant** neighbourhood
- `W_i`, `Θ^i_b`, `Θ^i_d` — learnable parameters of layer `i`

With `|B| = 4` and `D = 2` this is `1 + 4 + 2 = 7` terms per layer: the self term, four
bond-type neighbourhoods, and the distance-2 and distance-3 neighbourhoods.

> **⚠ CODE.** The implementation realises this as a single matmul: `EfficientGraphConvFn`
> concatenates `[X, A₁X, A₂X, A₃X, A₄X, D₂X, D₃X]` into `[n, F_in·7]` and multiplies by one
> weight matrix `[F_in·7, F_out]`. So `W_i`, the four `Θ^i_b` and the two `Θ^i_d` are **folded
> into a single weight matrix** — mathematically equivalent, and the reason
> `len(A_list) == |B| + D == 6`. The six adjacency matrices are built as `CSRNDArray`s in the
> loader's `from_numpy_to_tensor`.

**Conditional injection.** The task-specific protein embedding from `FNN_i` is **summed into
the output of `GCN_i`**, for every layer. In code: `+ linear_c[i](c)[ids, :]`, where `ids` maps
each atom to its parent molecule — essential because molecules in a batch have different sizes.
This path lives in `CVanillaMolGen_RNN` (`code/Generating_samples.ipynb`), **not** in the
pretraining notebooks.

### Step 3 — concatenate and project

The outputs of the `l` summations are concatenated and passed through `FNN_{l+1}` to obtain
final node embeddings `h_v`.

### Step 4 — pool and recur (Eq. 2)

$$
h^{GRU}_{m} \;=\; \mathrm{GRU}_{\Theta}\!\left(h^{GRU}_{m-1},\; h_{v^*} \,\|\, h_G\right)
$$

- `h_G` = average pooling over the node embeddings → graph-level embedding
- `h_{v*}` = embedding of the **last appended node** `v*`
- `‖` = concatenation

The GRU is what makes the model remember *the story of what it has been building*, rather than
judging only the current snapshot. In code the GRU input is width **1024** = `[h_G ‖ h_{v*}]`
with `F_c[-1] = 512` each (`pretraining.py:1245`), and `MolRNNLoader` builds `graph_to_rnn`
`(batch, k, 120)` and `rnn_to_graph` `(3, total_steps)` index maps to run the GRU over
generation steps and scatter its output back onto nodes.

### Step 5 — the policy head

The RNN output is concatenated with each node embedding from `FNN_{l+1}` and passed to
`FNN_{l+2}`, producing a tensor of size

$$
|V| \times \left(\,|A|\times|B| \;+\; |B|\,\right)
$$

per intermediate graph — for **every node**: every `(new atom type, bond type)` `append`
option (`|A|×|B|`), plus every bond type `connect` option (`|B|`). The RNN output separately
goes through `FNN_{l+3}` to give **one scalar** for `end`.

With `|A| = 65`, `|B| = 4`: `65×4 + 4 = 264` scores per atom, plus 1 for `end`.

> **⚠ CODE — the softmax is joint, not per-atom.** Both `FNN_{l+2}` and `FNN_{l+3}` use
> **exponential** activation, and the code then divides by a single per-molecule sum of *all*
> atom scores *plus* the `end` score. So it is one softmax **over every possible action of
> every atom plus `end`**. Per molecule, `append + connect + end` probabilities sum to 1.

| Output | Shape | Meaning |
|---|---|---|
| `append` | `[n, 65, 4]` | P(attach a new atom of type `a` with bond `b` at this atom) |
| `connect` | `[n, 4]` | P(bond of type `b` from the last appended atom to this atom) |
| `end` | `[m]` | P(stop) |

### Step 6 — the first atom (`init`)

| Mode | Source of the distribution |
|---|---|
| **Unconditional** (pretraining) | `softmax(exp(w))` over a learnable `[65]` parameter |
| **Conditional** (fine-tuning, generation) | `FNN_0(c)` → distribution over `A` |

An action is then **sampled** and applied. Generation is stochastic, so repeated sampling gives
different molecules.

---

## 5. Where the protein condition enters — exactly two points

This is the whole conditioning mechanism, and the extension point for **multi-target**:

1. **`FNN_0(c)` → the first-atom distribution.** The protein decides how the molecule starts.
2. **`FNN_i(c)` added to the output of every GCN layer** (all 6). The protein colours the
   representation of every atom at every generation step.

Everything else — `Embed_V`, the GCN stack, `FNN_{l+1}`, the GRU, the policy head, the
likelihood — is identical between the conditional and unconditional models.

**Key structural facts:**
- The condition is **a single fixed-length real vector** derived from one protein's amino-acid
  sequence. Not properties, not toxicity, not multiple targets, not a 3D pocket.
- The protein encoder is **frozen and offline**. Embeddings are computed during preprocessing
  and read back from a text file. `Embed_p` is never in the MXNet graph, never receives
  gradients.

### 5.1 The condition's on-disk format — already pinned

Both the generator here and the fine-tuning loop in `../CDGCN` agree on one format, so match it:

```
<SMILES>\t<c_1>\t<c_2>\t … \t<c_6165>
```

one protein–ligand pair per line. CDGCN's `Delimited` conditional
(`../CDGCN/models/Finetuning.ipynb`, cell 18) parses exactly this; this repo's
`datasets/data/data_preprocessing_bindingdb.ipynb` writes it (cells 115–119); and
`code/Generating_samples.ipynb` reads the embedding side back with
`line.strip().split('\t')`. Nothing downstream needs changing if you produce this shape.

---

## 6. The loss

### 6.1 Shared derivation (CDGCN Eq. 3–5 = CTDDG Eq. 1–3)

Maximise the log-likelihood of generating graph `G` via path `T`, given protein `P`:

$$
\log P_\Theta(G, T \mid P) \;=\; \sum_{j=1}^{m} \log P_\Theta\!\left(t_{j+1} \mid G_j, \ldots, G_1, P\right)
$$

The next action depends on **all** previous intermediate graphs and on `P`.

Marginalising over paths is intractable, since `S(G)` — the set of all generation paths — is
combinatorial:

$$
\log P_\Theta(G \mid P) \;=\; \log \sum_{T \in S(G)} P_\Theta(G, T \mid P)
$$

So use **importance sampling** with the proposal `Q_α`, giving a lower bound:

$$
\log P_\Theta(G \mid P) \;=\; \log \mathbb{E}_{T \sim Q_\alpha(T\mid G,P)}
\left[\frac{P_\Theta(G, T \mid P)}{Q_\alpha(T \mid G, P)}\right]
\;\ge\; \log \frac{1}{K} \sum_{k=1}^{K} \frac{P_\Theta(G, T_k \mid P)}{Q_\alpha(T_k \mid G, P)}
$$

Write the per-molecule bound as

$$
\ell_i \;=\; \log \frac{1}{K}\sum_{k=1}^{K} \frac{P_\Theta(G_i, T_{ik}\mid P_i)}{Q_\alpha(T_{ik}\mid G_i, P_i)}
$$

— this single quantity is what both papers' losses are built from. In code it is the three
lines at the end of `_likelihood`:

```python
l = log_p_x - log_p_sigma
l = logsumexp(l, axis=1) - math.log(float(iw_size))   # Eq. 3, shape [batch_size]
```

### 6.2 `Q_α(T | G, P)` — the proposal over generation paths

A predetermined distribution: at each step the next atom is chosen by **canonical atom
ordering** with probability `α`, or by **random ordering** with probability `1 − α`. Realised
as a Bernoulli(`α`) draw at each branch point.

> **⚠ CODE.** `traverse_graph` is a recursive DFS. At each branch point: if only one unvisited
> neighbour remains it is forced (no log-prob contribution); otherwise with probability `p` it
> takes the **lowest canonical-rank** unvisited neighbour (`log_p += log(p)`), else a uniformly
> random one of the rest (`log_p += log((1-p)/(len-1))`). `log_p` accumulates
> `log Q_α(T | G)`.
>
> ⚠ **The `p = 0.9` trap.** In the upstream code the function's own default is `p = 0.9`
> (`pretraining.py:275`), as is `MolLoader.__init__`'s (`pretraining.py:560`, alongside
> `k = 10` and `batch_size = 10`). The paper's `0.8` and `K = 5` are supplied at construction.
> **Any reimplementation that relies on the defaults silently trains a different model.**
> The rewrite removes the trap: `code/pretraining copy.ipynb` declares
> `def traverse_graph(..., p=P_ALPHA, ...)` and `def process_single(record, k=K_PATHS,
> p=P_ALPHA)`, so the constants bind to the module-level `0.8` / `5`.

### 6.3 CDGCN's loss — Eq. 6 (= CTDDG Eq. 4)

$$
\hat{L}(\Theta) \;=\; -\frac{1}{N}\sum_{i=1}^{N} \ell_i
$$

Plain negative log-likelihood over the minibatch. **No property is optimised.** Drug-likeness
emerges from the training data distribution, not from the objective.

### 6.4 CTDDG's loss — Eq. 8

Two parts: minimise the likelihood of toxic graphs, maximise the likelihood of non-toxic ones.

$$
\max \hat{L}(\Theta) = \frac{1}{N}\left(\lambda\, \hat{L}_{\text{toxic}}(\Theta)
+ \hat{L}_{\text{non-toxic}}(\Theta)\right)
\tag{5}
$$

$$
\hat{L}_{\text{toxic}}(\Theta) = -\sum_{i=1}^{N} \ell_i \cdot \mathbb{1}\{G_i \text{ is toxic}\}
\tag{6}
$$

$$
\hat{L}_{\text{non-toxic}}(\Theta) = \sum_{i=1}^{N} \ell_i \cdot \mathbb{1}\{G_i \text{ is non-toxic}\}
\tag{7}
$$

Collapsed into the single operative form:

$$
\boxed{\;\max \hat{L}(\Theta) \;=\; \frac{1}{N}\sum_{i=1}^{N} \ell_i \cdot
\left(1 - (1+\lambda)\cdot \text{Toxic\_class}_i\right)\;}
\tag{8}
$$

where `Toxic_class_i ∈ {0, 1}` — **1 for toxic, 0 for non-toxic** — and `λ ≥ 0` is the
trade-off parameter, **fixed to 1** in the paper.

> ⚠ The published PDF labels **both** Eq. 6 and Eq. 7 `L_toxic likelihood`; Eq. 7 is the
> non-toxic component, as its indicator and Eq. 8 make clear. The `max`/`min` wording around
> Eqs. 5–7 is also loose. **Eq. 8 is unambiguous — implement that.**

### 6.5 What the weight does

| `Toxic_class_i` | Weight `1 − (1+λ)·Toxic_class_i` | At `λ = 1` | Effect |
|---|---|---|---|
| 0 (non-toxic) | `1` | **+1** | Maximise likelihood — ordinary training |
| 1 (toxic) | `−λ` | **−1** | Minimise likelihood — gradient reversal |

Only one branch is active per molecule. At `λ = 1` the weight is simply **±1**.

Setting `λ = 0` — or `tox` all zeros — recovers CDGCN's loss exactly. **CDGCN is the `λ = 0`
special case of CTDDG.**

### 6.6 How it is implemented here

`code/pretraining copy.ipynb`, at the end of `MoleculeGenerator._likelihood`:

```python
# Original CDGCN log-likelihood calculation
l = log_p_x - log_p_sigma
l = logsumexp(l, axis=1) - math.log(float(iw_size))

# CTDDG Eq. 8
tox_labels = tox_class_batch.reshape((batch_size, iw_size))[:, 0]
tox_weight = 1.0 - (1.0 + LAMBDA_TOX) * tox_labels
l_weighted = l * tox_weight
return l_weighted
```

and `forward` returns `-l_weighted.mean()` when `self.mode == 'loss'`.

> **⚠ Shape trap — this is why the `[:, 0]` is there.** `l` has shape `[batch_size]` (after
> `logsumexp` over the `K` paths) but `tox_class` arrives with `K` entries per molecule, shape
> `[batch_size * K]`, because `process_single` returns `np.full((k,), tox_label)`. All `K`
> paths of a molecule share its label, so reshaping to `[batch_size, K]` and taking column 0 is
> correct. Verify on a tiny mixed-label batch before any long run: the sign of `l` must flip on
> exactly the toxic molecules.

Two review items on this objective are still **open** (see `code/PRETRAINING_REVIEW.md`):

- **F12** — log the toxic and non-toxic branch means separately. A single scalar loss hides
  whether toxicity suppression is working or merely dominating.
- **F13** — guard the toxic branch. Maximising `−ℓ` for toxic molecules is **unbounded below**:
  nothing stops the model driving their log-likelihood to −∞, and `clip_grad = 3.0` only slows
  it. If the loss starts marching towards large negative values, add a floor:
  ```python
  tox_floor = -150.0
  l = nd.where(tox > 0, nd.maximum(l, tox_floor), l)
  ```
  (Not in the paper — a stabiliser.)

---

## 7. What differs between the two papers — the complete list

| Aspect | CDGCN | CTDDG |
|---|---|---|
| Graph representation | `G = (V,E)`, `\|A\|=65`, `\|B\|=4` | **identical** |
| Four actions | `init`/`append`/`connect`/`end` | **identical** |
| `Embed_p`, `FNN_0` | pretrained encoder → init distribution | **identical** |
| `Embed_V`, GCN stack, per-layer `FNN_i` injection | Eq. 1, `D=2`, 6 layers | **identical** |
| `FNN_{l+1}`, average pooling, GRU | Eq. 2 | **identical** |
| `FNN_{l+2}`, `FNN_{l+3}`, joint softmax | `\|V\| × (\|A\|×\|B\| + \|B\|)` + 1 | **identical** |
| `Q_α` proposal, `α = 0.8`, `K = 5` | Eq. 5 | **identical** |
| Datasets (ChEMBL, BindingDB), splits, <20% similarity | §3.1 | **identical** (§2.1 says so) |
| Batch size 8, two-phase training | §3.3 | **identical** |
| **Toxicity labels on the data** | **none** | **added** — RF (ChEMBL) + pkCSM (BindingDB) |
| **Loss** | Eq. 6: `−(1/N) Σ ℓ_i` | **Eq. 8: `−(1/N) Σ ℓ_i · (1 − (1+λ)·Toxic_class_i)`** |
| **λ** | n/a | **1** |
| Evaluation | + `R_c` random-forest activity metric | + **% toxic (pkCSM)**; drops `R_c` |
| Case study | Five novel proteins, docking only | + **CDK2**: docking → PLIP → MD → MM-PBSA → fingerprints |

**The entire architectural difference is: none.** The entire functional difference is the
scalar factor `(1 − (1+λ)·Toxic_class_i)` in the loss, plus the labelling needed to compute it.

### 7.1 Where the toxicity information enters — precisely

| Stage | CDGCN | CTDDG |
|---|---|---|
| Data file format | `<SMILES>` | **`<SMILES> <class>`** — one extra integer field |
| Preprocessing | — | **RF classifier** (Ames) labels ChEMBL; **pkCSM** labels BindingDB ligands |
| Data loader | — | `tox_class` carried alongside the graph, `K` copies per molecule |
| `Embed_V` / GCN / GRU / policy head | — | **unchanged — toxicity never reaches the architecture** |
| First-atom distribution | — | **unchanged** |
| Likelihood `ℓ_i` | computed | **unchanged** |
| **Final reduction over the batch** | `−mean(ℓ)` | **`−mean(ℓ · (1 − (1+λ)·tox))`** ← **the only functional insertion point** |
| Generation / sampling | — | **unchanged** — the model is never told a desired toxicity at generation time |

> **Read that last row carefully.** CTDDG's toxicity control is *implicit*: it is baked into the
> weights during training. You **cannot** ask the trained model for "a non-toxic molecule" at
> inference — there is no toxicity input. This is a meaningful limitation, and a place where
> multi-property work could diverge: making properties an explicit **conditioning input** (like
> `c`) rather than only a training-time weight would allow controllable generation.

---

## 8. Hyperparameters

The released configuration, as written to `checkpoints/configs.json` by a run here:

```json
{"F_e": 16, "F_h": [32, 64, 128, 128, 256, 256], "F_skip": 256,
 "F_c": [512], "Fh_policy": 128, "activation": "relu", "N_rnn": 3}
```

| Symbol | Value | Meaning |
|---|---|---|
| `N_A` | **65** | atom types (= lines in `datasets/data/atom_types.txt`) |
| `N_B` | **4** | bond types — aromatic, single, double, triple |
| `D` | **2** | receptive field; adds distance-2 and distance-3 neighbourhoods |
| `F_e` | 16 | atom embedding width |
| `F_h` | [32,64,128,128,256,256] | GCN layer widths (6 layers) |
| `F_skip` | 256 | skip-connection width |
| `F_c` | [512] | dense stack after graph convolution (`FNN_{l+1}`) |
| `Fh_policy` | 128 | hidden width in the policy head |
| `N_rnn` | 3 | GRU layers |
| `N_C` | **6165** | protein embedding width — **fine-tuning only**, absent from the pretraining configs |
| `K` / `K_PATHS` | **5** | generation paths per molecule |
| `α` / `P_ALPHA` | **0.8** | coefficient of randomness in `Q_α` |
| `λ` / `LAMBDA_TOX` | **1** | CTDDG only |
| `max_iter` | 120 | cap on generation steps per molecule |

### 8.1 Training schedules — they differ by phase

| | Pretraining (paper / upstream) | Pretraining (rewrite here) | Fine-tuning (`../CDGCN`) |
|---|---|---|---|
| Batch size | **8** | **16** ⚠ | 8 |
| Learning rate | 1e-3 | 1e-3 | **1e-4** |
| Decay | 1e-3 per 100 steps | 1e-3 per 100 steps | **0.002** per 100 steps |
| Grad clip | 3.0 | 3.0 | 3.0 |
| Length | 480,000 iterations | 480,000 iterations | **10 epochs**, early stopping `patience=10` |
| Checkpoint every | 500 steps | **5 steps** ⚠ | 200 steps |

> ⚠ Both flagged values in the rewrite need attention before a real run — the batch size breaks
> comparability with the published tables, and `SUMMARY_STEP = 5` writes ~84 MB of checkpoint
> per 5 iterations. See [01](01_project_background_and_motivation.md) §5.3.

**Total parameters: 6,986,506** — printed by the rewrite's model-construction cell, and
consistent with the 27,950,253-byte `ckpt.params` (≈ 4 bytes × 6.99 M). The GRU accounts for
~79% of them.

Framework: **MXNet** — 1.7.0 in CDGCN, 1.9.1 in the CTDDG environment (`ctddg_env`,
Python 3.9). Single GPU. `trainer.step(batch_size=1)` is **correct** in both phases, because
`forward` already averages over the batch; changing it to the real batch size would divide the
learning rate by that factor.

---

## 9. Implementation reality in *this* repository

Two pretraining implementations coexist here. Know which one you are reading.

| | `code/pretraining.ipynb` + `.py` | `code/pretraining copy.ipynb` |
|---|---|---|
| Provenance | Upstream Colab export, unmodified | The rewrite |
| Shape | 5 cells, the first **1,509 lines** | 36 cells, sectioned |
| Paths | `/workspace/Toxicity_experiment/…` ×5 | `/home/muhamed/repo/CTDDG/…` |
| Device | `mx.gpu()` hardcoded ×11 | `CTX = mx.gpu(0) if num_gpus() > 0 else mx.cpu()` |
| Data | `chembl_final.txt` — **all labels `1`** | `chembl_toxicity_labels.csv` — **real RF labels**, 23.8% toxic |
| `process_single` | `tox_class = [1] * k` | `tox_class = np.full((k,), tox_label)` |
| **Eq. 8** | **absent** — `_likelihood` takes `tox_class_batch` and never references it | **implemented** |
| `λ` | undefined | `LAMBDA_TOX = 1.0` |
| `traverse_graph` default `p` | **0.9** (paper says 0.8) | `P_ALPHA` = 0.8 |
| `forward` unpack order | ⚠ base-class bug (`log_p, tox_class_batch, batch_size, iw_size` vs the loader's `log_p, batch_size, iw_size, tox_class`) — dead today, live if the base class is reused | fixed |
| Run naming | `model_name = 'base_cdgcn'`, writing into `.../just_test/logs/` | writes to `checkpoints/` |

**Still missing from both, and from the repository as a whole:**

| Papers describe | This repository contains |
|---|---|
| Two-phase training (unconditional pretrain → conditional fine-tune) | Pretraining only. `CVanillaMolGen_RNN` is defined in `code/Generating_samples.ipynb`, but **no fine-tuning training loop exists here**. The reference implementation is `../CDGCN/models/Finetuning.ipynb` |
| pkCSM-labelled BindingDB ligands | Nothing — no pkCSM code, data or labels |
| `Embed_p` producing `c` for each protein | The **call** exists (`prose/embed_sequences.py --pool avg`, preprocessing cell 82) but has never been run: `*_bepler.fa` inputs are present for all ten splits, **no `*_embeddings_bepler.txt` outputs are** |
| CDK2 case study, MACCS/Tanimoto/PLIP/MM-PBSA analysis | Only the authors' *result* archives (`Molecular_dynamics_files.zip`, `Moleculsr_docking_top10GeneratedMol.zip`, `supplimentaryAPIN.pdf`) — not the code |

**Consequence:** any checkpoint produced by `code/pretraining.ipynb` is a **CDGCN unconditional
generator**, regardless of what the directory is named. A checkpoint from
`code/pretraining copy.ipynb` would be a genuine CTDDG-objective pretrained model — but
**none has been trained yet**: `checkpoints/log.out` contains only its header row.

---

## 10. Extension points for this project

Mapping the multi-target / multi-property goals onto the architecture above.

### 10.1 Multi-target → generalise §5

The two injection points are the whole conditioning mechanism:

| Point | Current | Needs to become |
|---|---|---|
| `FNN_0(c)` → first-atom distribution | one vector → one distribution | a function of a **set** of protein embeddings |
| `FNN_i(c)` added to each GCN layer | one vector broadcast per molecule via `ids` | aggregation over multiple targets before (or after) the per-layer projection |

Candidate mechanisms (all **TBD**): concatenation of a fixed number of targets; permutation-
invariant pooling (mean/max/sum) over a variable-size set; attention over the target set;
separate conditioning streams per target with a learned combination. Selectivity objectives
(bind A, avoid B) may need *signed* conditioning, which is closer in spirit to CTDDG's ±λ
weighting than to `c`.

Note the `ids` tensor (atom → parent molecule) already exists to handle variable-size molecules
in a flat batch; a multi-target design needs the analogous machinery for variable-size target
sets. Note too that at `N_C = 6165` the conditioning vector is already **wider than any GCN
layer** — naive concatenation of several targets gets expensive fast.

### 10.2 Multi-property → generalise §6.4

CTDDG's `(1 − (1+λ)·Toxic_class_i)` is a **single binary property expressed as a scalar
multiplier**. Natural generalisations (all **TBD**):

- Several binary properties: `w_i = Π_p (1 − (1+λ_p)·class_{p,i})` or `1 − Σ_p (1+λ_p)·class_{p,i}`.
  Note the product form can flip sign unintuitively with an even number of violations —
  worth care.
- Continuous property values instead of binary labels, with per-property weights.
- Properties as **explicit conditioning inputs** alongside `c`, enabling controllable
  generation at inference — which CTDDG's training-time-only weighting cannot do (§7.1).

The plumbing is friendlier than it looks: `tox_class` is already carried end-to-end from
`process_single` through the collator to `_likelihood` as a `[batch_size * K]` array. Widening
it to `[batch_size * K, n_properties]` touches the loader, the `batch_to_tensor` transfer and
the one reshape in `_likelihood` — and nothing else.

### 10.3 Also worth knowing

- **`λ = 1` was chosen by argument, not by a sweep.** A λ sweep is cheap and is a standalone
  contribution — and with Eq. 8 now implemented here, it is one config constant away.
- **Both papers name the same future work**: improve the graph encoding scheme. CDGCN cites
  **GAT** and **GIN** explicitly; CTDDG adds *"maximizing binding affinity, optimizing drug
  like properties"* — i.e. this project's direction.
- **`datasets/data/atom_types.txt` is effectively immutable** once a checkpoint exists:
  atom-type identity is a *line number* in that file, so regenerating it invalidates every
  trained atom embedding. The rewrite's OOV shield silently drops molecules with atom types
  outside it — check how many are being dropped rather than assuming zero.
- **The GRU is ~79% of the parameters.** Any capacity change should probably start there.

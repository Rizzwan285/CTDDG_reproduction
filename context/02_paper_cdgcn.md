# 02 — Paper: CDGCN

**Full title:** *CDGCN: Conditional de novo Drug generative model using Graph Convolution
Networks*
**Authors:** Shikha Mallick¹, Sahely Bhadra²
¹ Dept. of Computer Science and Engineering, IIT Palakkad · ² Dept. of Data Science, IIT Palakkad
**Year:** 2023 · **Code:** `https://github.com/mshik/CDGCN`
**PDF:** `../CTDDG_new/papers/CDGCN.pdf` (16 pages) — not in this repository
**Reference implementation:** `/home/muhamed/repo/CDGCN` (clone, `Rizzwan285/CDGCN`)

> **Role in this project: CDGCN is THE ARCHITECTURE.** CTDDG reuses it verbatim. Everything
> about how the model is built, conditioned, decoded and trained comes from this paper. Read
> this before CTDDG.
>
> **Practical reason to care about the CDGCN clone specifically:** it contains
> `models/Finetuning.ipynb` — the **conditional fine-tuning training loop that this repository
> does not have**. See §10.

---

## 1. One-paragraph summary

CDGCN is a conditional graph generative model that takes a **protein amino-acid sequence** as
input and generates **novel molecular graphs** predicted to bind it. It builds each molecule
atom-by-atom through a sequential decoding scheme, and it injects protein information via a
**pretrained protein-sequence embedding** that is added into every graph convolution layer and
that determines the distribution over the first atom. Its headline claim is that it is the
first *graph* generative network able to design drugs for **novel** proteins — proteins with
no known drug at all — and that it beats the previous state of the art (a SMILES Transformer)
on chemical validity, novelty, drug-likeness, speed, and binding energy.

---

## 2. The gap it identifies

| Prior method | What it does | Why it is insufficient |
|---|---|---|
| **Li et al. (2018)** — conditional graph generative model | Sequential graph decoding; generates for a *predetermined set of known* proteins | Cannot learn structural/functional information of proteins → **fails on novel proteins** |
| **Grechishnikova (2021)** — Transformer, protein seq → SMILES | The only prior method that handles **novel** proteins | SMILES-based: learns string syntax rather than structure → **fewer chemically valid molecules**; also slower and more memory-hungry |

CDGCN's thesis: keep Li et al.'s *graph* decoding (which gives high validity and flexible
molecule size) and add Grechishnikova's *generalisation to novel proteins* (via a protein
sequence encoder). The sequential decoder is specifically praised over one-shot
fixed-size-adjacency-matrix methods because it can produce molecules of arbitrary size.

Why sequences and not 3D structures: **novel proteins are far more widely available as
amino-acid sequences than as solved 3D structures**, so conditioning on sequence is what makes
the method usable for genuinely new targets.

---

## 3. Stated contributions (paper §1)

1. The **first graph generative network for protein-specific de novo drug generation for both
   known and novel proteins**.
2. A **non-trivial way to include protein function information into the graph generation
   process**, which is what enables high-affinity generation for novel proteins.
3. Superior performance vs Grechishnikova (2021): best binding energy **≤ −7.3 kcal/mol** for
   each of five novel proteins, vs **−6.2 kcal/mol** for the baseline.

---

## 4. Method

### 4.1 Molecular representation

A molecule is a graph `G = (V, E)`. Each node carries an atom type from a fixed set `A`; each
edge carries a bond type from a fixed set `B`. In this work `|A| = 65` and `|B| = 4`
(single, double, triple, aromatic).

### 4.2 The four actions

Generation starts from an **empty graph** and repeatedly picks one of four actions:

| Action | Effect |
|---|---|
| `init` | Add the first node to the empty graph |
| `append` | Attach a **new node** with a **new edge** to an existing node |
| `connect` | Add a **new edge** between an existing node and the **last appended node** (ring closure) |
| `end` | Stop generation; post-process the graph into a molecule |

### 4.3 Components (paper §2.1–2.2, Figure 1)

| Component | What it does |
|---|---|
| `Embed_p` | Protein sequence → embedding containing functional information. **Pretrained, from Dallago et al. (2021)**, trained to predict protein function from sequence |
| `FNN_0` | Protein embedding → probability scores over `A` for the `init` action (dense + softmax) |
| `FNN_1 … FNN_l` | Protein embedding → **task-specific protein embeddings**, one per GCN layer; each is *summed into* that layer's output node embeddings |
| `Embed_V` | Atom type → node embedding. A learnable `|A| × d_0` matrix |
| `GCN_1 … GCN_l` | Graph convolution layers: GC + BatchNorm + ReLU (the last is GC only) |
| `FNN_{l+1}` | Takes the concatenation of all `l` summed outputs → final node embeddings → average pooling → graph-level embedding |
| `RNN` | GRU carrying information across generation steps |
| `FNN_{l+2}` | Per-node action scores: tensor of size `|V| × (|A|×|B| + |B|)` for all `append` and `connect` actions. Dense → BN → ReLU → dense → **exponential** |
| `FNN_{l+3}` | Scalar score for `end`. Dense → **exponential** |
| Softmax | Applied jointly across `FNN_{l+2}` and `FNN_{l+3}` outputs to yield final action probabilities |

**Pretraining note (important):** during unconditional pretraining, `Embed_p` and `FNN_0` are
replaced by a **single learnable weight vector followed by softmax**. There is no protein at
pretraining time.

Full tensor-level detail, shapes and equations are in
[04_model_architecture.md](04_model_architecture.md).

### 4.4 Equations

**Graph convolution (Eq. 1)** — following Wu et al. (MoleculeNet) for molecular graphs:

$$
h^{i}_{(v)GC} \;=\; W_i\, h^{i-1}_{(v)GC}
\;+\; \sum_{b \in B} \Theta^{i}_{b} \sum_{u \in N_b(v)} h^{i-1}_{(u)GC}
\;+\; \sum_{1 < d \le D} \Theta^{i}_{d} \sum_{u \in N_d(v)} h^{i-1}_{(u)GC}
$$

- `N_b(v)` — nodes directly bonded to `v` with bond type `b` → **local** neighbourhood
- `N_d(v)` — nodes at path length `d` from `v` → **distant** neighbourhood
- `D` — receptive field size
- `W_i`, `Θ^i_b`, `Θ^i_d` — learnable parameters of layer `i`

**Recurrence (Eq. 2)** — a GRU over generation steps:

$$
h^{GRU}_{m} \;=\; \mathrm{GRU}_{\Theta}\!\left(h^{GRU}_{m-1},\; h_{v^*} \,\|\, h_G\right)
$$

where `h_{v*}` is the embedding of the **last appended node** `v*`, `h_G` is the graph-level
embedding, and `‖` is concatenation.

### 4.5 Loss (paper §2.3)

Objective: maximise the log-likelihood of generating graph `G` via generation path `T`, given
protein `P`.

**Eq. 3 — path likelihood:**

$$
\log P_\Theta(G, T \mid P) \;=\; \sum_{j=1}^{m} \log P_\Theta\!\left(t_{j+1} \mid G_j, \ldots, G_1, P\right)
$$

**Eq. 4 — marginal likelihood (intractable):**

$$
\log P_\Theta(G \mid P) \;=\; \log \sum_{T \in S(G)} P_\Theta(G, T \mid P)
$$

`S(G)` is the set of *all* possible generation paths — intractable for realistic molecules.

**Eq. 5 — importance-sampled lower bound:**

$$
\log P_\Theta(G \mid P) \;=\; \log \mathbb{E}_{T \sim Q_\alpha(T \mid G, P)}
\left[\frac{P_\Theta(G, T \mid P)}{Q_\alpha(T \mid G, P)}\right]
\;\ge\; \log \frac{1}{K} \sum_{k=1}^{K} \frac{P_\Theta(G, T_k \mid P)}{Q_\alpha(T_k \mid G, P)}
$$

**Eq. 6 — minibatch negative log-likelihood loss (the training objective):**

$$
\hat{L}(\Theta) \;=\; -\frac{1}{N} \sum_{i=1}^{N}
\log \frac{1}{K} \sum_{k=1}^{K}
\frac{P_\Theta(G_i, T_{ik} \mid P_i)}{Q_\alpha(T_{ik} \mid G_i, P_i)}
$$

**`Q_α(T | G, P)`** is a *predetermined* proposal distribution over generation paths: at each
step it picks the next atom by **canonical atom ordering** with probability `α`, and by
**random ordering** with probability `1 − α`. `K` is the number of paths sampled.

> **This is the equation CTDDG modifies.** CTDDG's Eq. 4 is identical to this Eq. 6; its Eq. 8
> is this expression with a per-molecule toxicity weight inserted.

---

## 5. Data

| Dataset | Role | Details |
|---|---|---|
| **ChEMBL** | Pretraining | ~2.3 M curated molecules; **~10^6** satisfying the desired physicochemical properties were extracted |
| **BindingDB** (Grechishnikova's version) | Fine-tuning | Experimentally measured protein–ligand binding energies. Proteins as **FASTA**, ligands as **SMILES** |

- BindingDB is split into **five cross-validation folds**; each validation fold is halved into
  validation (hyperparameter tuning) and test (evaluation).
- **Train/test protein similarity is kept < 20%** (Needleman–Wunsch, EMBOSS) so test proteins
  are truly novel.
- Most protein pairs *within* train and *within* test share **< 40%** similarity, for diversity.
- **BindingDB ligands were removed from the modified ChEMBL set** to prevent data leakage.
- Combined vocabulary: **65 unique atom types**, **4 bond types**.

---

## 6. Baselines, implementation and metrics

**Baselines:** Li et al. (2018) — graph-based, known proteins only; Grechishnikova (2021) —
SMILES Transformer, the only novel-protein-capable prior method.

**Implementation:** MXNet 1.7.0, single NVIDIA GeForce RTX 2080 Ti (8 GB).

| Hyperparameter | Value | Note |
|---|---|---|
| Batch size `N` | **8** | Chosen by 5-fold CV; larger was impossible due to GPU memory |
| `α` (coefficient of randomness) | **0.8** | "A suitable choice for α and K is crucial" |
| `K` (generation paths) | **5** | |
| Fine-tuning length | ~10 epochs | With early stopping, per fold |

All other hyperparameters and the optimiser follow Li et al. (2018).

**Metrics:**
- Time to generate `S_10` / `S_100` / `S_1000`
- **Validity** (RDKit) and **Novelty** (exact string match against the dataset)
- Lipinski's rule of five: logP < 5, MW < 500 Da, H-bond donors < 5, H-bond acceptors < 10
- Veber criteria: rotatable bonds < 10, TPSA < 140 Å²
- **QED** (0–1, higher = more drug-like) and **SAS** (< 6 = synthesizable)
- **`R_c`** — % "binding/active" predictions from per-protein binary random-forest classifiers
  on ECFP6 fingerprints (100 known ligands as positives, 100 non-interacting as negatives;
  only classifiers with ≥95% accuracy used)
- **Binding energy** via molecular docking: conformers from OpenBabel, docking with **SMINA**
  (default settings), orchestrated through **Jupyter Dock**. Threshold: −6.0 kcal/mol is the
  minimum for drug development consideration

Each experiment repeated **10 times**; mean ± std reported; paired t-tests for significance.

---

## 7. Results

### 7.1 Quality and speed (Table 1)

Both graph-based methods beat Grechishnikova by a large margin on validity, novelty and
drug-like properties. Representative CDGCN numbers for **novel** proteins at `S_1000`:

| Metric | CDGCN (novel proteins) |
|---|---|
| Valid | 94.0 ± 0.51 % |
| Novel | 99.8 ± 0.05 % |
| logP < 5 | 77.0 ± 2.60 % |
| MW < 500 | 71.0 ± 2.10 % |
| SAS < 6 | 87.8 ± 0.96 % |
| QED | 0.54 ± 0.02 |

- **>85% of CDGCN molecules satisfy SAS** for both known and novel proteins — they are easy to
  synthesize.
- On **known** proteins, Li et al. is slightly better on some metrics, but CDGCN is comparable.
- On `R_c` for known proteins, Grechishnikova is highest — but the paper attributes this to
  **overfitting**, evidenced by its very low novelty (76.8–84.1% vs CDGCN's 99.7–99.9%).

### 7.2 Binding energy (Table 2) — the headline result

Five novel proteins with available binding sites:

| PDB | Protein | Known ligand best | CDGCN `S_1000` best |
|---|---|---|---|
| 4nos | Nitric oxide synthase, inducible | −7.9 | **−8.6 ± 0.12** |
| 2nru | Interleukin-1 receptor-associated kinase 4 | −8.8 | **−10.1 ± 0.45** |
| 1lqf | Tyrosine-protein phosphatase non-receptor type 1 | −11.0 | **−11.1 ± 0.21** |
| 1k2r | Nitric oxide synthase, brain | −9.0 | **−10.2 ± 0.17** |
| 1cqp | Integrin alpha-L | −9.6 | **−10.1 ± 0.39** |

(kcal/mol; lower is better.) CDGCN beats Grechishnikova in **all** cases, and for **all five**
novel proteins it produced molecules with **lower binding energy than the proteins' known
ligands**.

### 7.3 Diversity and ablation

- Tanimoto similarity: ~13% of generated molecules (all three models) are structurally similar
  to dataset molecules; ~42% differ significantly.
- **`Embed_p` ablation:** comparing (i) Li et al. on known proteins, (ii) pretrained CDGCN
  *without* `Embed_p`, (iii) finetuned CDGCN *with* `Embed_p` — the finetuned model with
  `Embed_p` gave the **best binding energies for all five novel proteins**. This is the
  paper's direct evidence that the protein conditioning is doing real work.

---

## 8. Conclusion and stated future work

CDGCN is presented as the first deep conditional graph generative network for novel
protein-specific de novo drug generation, superior to the SMILES Transformer baseline on
binding energy, validity, novelty, diversity, drug-likeness and synthetic accessibility.

**Stated future work:** improve the **graph encoding scheme** with more informative molecular
graph representations — the paper explicitly cites **Graph Attention Networks** (Veličković et
al.) and **GIN** ("How powerful are graph neural networks?", Xu et al.).

---

## 9. What matters for this project

1. **This is the architecture to build on.** Read [04_model_architecture.md](04_model_architecture.md)
   for the implementation-level version.
2. **The conditioning mechanism is the extension point for multi-target.** `Embed_p` produces
   one vector for one protein; `FNN_0` turns it into the first-atom distribution and
   `FNN_1…FNN_l` inject it into every GCN layer. Generalising to *multiple* targets means
   generalising exactly this path.
3. **The loss (Eq. 6) is the extension point for multi-property.** CTDDG already demonstrated
   that a per-molecule scalar weight can be attached to it.
4. **CDGCN optimises no properties at all.** It maximises likelihood. Drug-likeness emerges
   from the training data, not from the objective. This is precisely the limitation CTDDG
   names and partially fixes.
5. **The evaluation protocol is the benchmark to match.** Same five PDB targets, same metrics,
   same 10-run mean±std, or comparisons will not be meaningful.
6. **`α = 0.8`, `K = 5`, batch = 8** — carried over unchanged into CTDDG. ⚠ In the code the
   *defaults* are `p = 0.9`, `k = 10`, `batch_size = 10`; the paper's values are passed at
   construction. See [04_model_architecture.md](04_model_architecture.md) §6.2.

---

## 10. What the CDGCN clone gives you that this repository lacks

`/home/muhamed/repo/CDGCN/models/Finetuning.ipynb` (38 cells) is the missing second training
stage, in working form:

| Piece | Where |
|---|---|
| `CVanillaMolGen_RNN` construction with `N_C` in `configs` | cell 35 |
| Conditional data reader — `Delimited`, parsing `"<SMILES>\t<c_1>\t…"` | cell 18 |
| `EarlyStopping(patience=10)` | cell 31 |
| Resume-from-checkpoint logic | cell 33 |
| The training loop itself (105 lines) | cell 37 |

Its hyperparameters differ from pretraining and are the ones to copy: `lr = 1e-4`,
`decay = 0.002`, `max_epochs = 10`, `summary_step = 200`, `batch_size = 8`, **`N_C = 6165`**.

Porting this loop and multiplying its per-molecule likelihood by CTDDG's Eq. 8 weight is the
shortest credible path to a conditional CTDDG model. The alternative — writing the loop from
scratch — is strictly more work for the same result.

> Note the data file it expects, `d3_tr_cdgcn.txt`, is the tab-delimited SMILES + embedding
> file described in [04_model_architecture.md](04_model_architecture.md) §5.1. This
> repository's preprocessing notebook already writes that format; what it has not produced is
> the embeddings that go in it.

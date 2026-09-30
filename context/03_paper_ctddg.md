# 03 — Paper: CTDDG

**Full title:** *CTDDG: Graph-Based Deep Generative Model for Low-Toxic Drug Design for a
Target Protein with Optimal Binding Affinity*
**Authors:** Nitesh Singh¹, Sahely Bhadra¹˒², Pratiti Bhadra³
¹ Mehta Family School of Data Science and AI / Dept. of Data Science, IIT Palakkad ·
² Dept. of CSE, IIT Palakkad · ³ Center for Computational Engineering and Networking, Amrita
Vishwa Vidyapeetham, Coimbatore
**Code:** `https://github.com/sahelybhadra/CTDDG` — **this repository's `upstream` remote**
**PDF:** `../CTDDG_new/papers/CTDDG.pdf` (24 pages) — not in this repository
**Supplementary:** `supplimentaryAPIN.pdf` (here) — Table S1 (top-10 CDK2 SMILES), Table S2 (MD interactions)

> **This repository is the reproduction of this paper.** §11 lists what the published code
> does and does not do; §12 records which of those gaps have since been closed here.

> **CTDDG = Conditional Target-based Detoxed Drug Generation.**
> **Role in this project: CTDDG is the PROPERTY-AWARE EXTENSION of CDGCN.** Its architecture
> and datasets are CDGCN's, unchanged. Its entire contribution lives in **§2.3 (the loss)** and
> **§2.4 (the labelling)**. If you are short on time, read only those two sections plus the
> CDK2 case study.

---

## 1. One-paragraph summary

CTDDG takes CDGCN and adds a single idea: **train the model to simultaneously maximise the
likelihood of non-toxic molecules and minimise the likelihood of toxic ones**, by multiplying
each molecule's log-likelihood term by a signed weight derived from its toxicity label. Toxicity
is operationalised as **Ames mutagenicity** (the ability to induce DNA mutations). Labels are
supplied by a Random Forest classifier for the ChEMBL pretraining set and by the **pkCSM** web
server for the BindingDB fine-tuning set. The result: ~95% of generated molecules are valid and
unique, >80% adhere to Lipinski's rules, >95% show favourable binding affinity, and the
percentage of predicted-toxic molecules drops from CDGCN's 23.30% to **19.32%** — without
degrading drug-likeness or target awareness. A CDK2 case study, taken all the way through
docking, MD simulation and MM-PBSA, identifies a candidate inhibitor (`mol3`).

---

## 2. The gap it identifies

Reviewing VAE (CharVAE, JT-VAE, HierVAE), GAN (ORGAN, MolGAN, Mol-CycleGAN), flow-based
(GraphNVP, MolFlow, MolGrow) and 3D-structure (Pocket2Mol, 3D generative model) approaches, the
paper concludes that **none of them handle protein-target-based generation for proteins with
few or no known drugs**, and that 3D methods additionally require binding-pocket information,
restricting them to known proteins.

Toxicity-specific prior work and its limitations:

| Prior work | Does | Does not |
|---|---|---|
| **Yang et al. (2022)** — semi-supervised VAE, 5 toxicity factors (cardio-, muta-, hepato-, pulmonary, skin) | Produces ~25% more non-toxic molecules | **No target specificity**; toxicity not integrated into the generation process |
| **Kang & Cho (2019)** — SSVAE conditional molecular design | Conditions on desired properties incl. toxicity | **No target protein conditioning** |
| **DrugEx (Liu et al., 2021)** — RNN + RL multi-objective | Incorporates targets *and* optimises multiple properties including toxicity | Specifically designed for **adenosine receptors** |
| **CDGCN (Mallick & Bhadra, 2023)** — the predecessor | Generates for known *and* novel proteins | Its loss **maximises graph likelihood only** — no explicit optimisation of toxicity, binding affinity, QED or SAS |

The paper's framing of the core tension is worth quoting, because it motivates this project's
multi-property direction directly:

> *"Optimizing individual properties, such as toxicity, can negatively impact other crucial
> attributes like binding affinity and adherence to Lipinski's Rule of Five."*

**Claim:** CTDDG is *"the first method designed to generate less toxic de novo drugs for novel
protein targets."*

---

## 3. Why Ames mutagenicity

The paper focuses on preventing generation of molecules that can **induce DNA mutations**.
Mutagenic agents damage genetic material, which can cause cancer, genetic abnormalities and
cell death. Toxicity is therefore *conceptualised within the framework of Ames mutagenicity*
(citing Wu et al., 2021), assessed by the **Ames test** (Stead et al., 1981).

Note this is a **single binary endpoint**. Extending to several endpoints — and to continuous
rather than binary property values — is exactly the "multi-property" axis of this project.

---

## 4. Architecture — identical to CDGCN

> *"Architecture for CTDDG is based on that of CDGCN (Mallick and Bhadra, 2023)."* — §2.2

Same graph representation `G = (V, E)`, same fixed atom set `A` and bond set `B`, same four
actions (`init`, `append`, `connect`, `end`), same `Embed_p → FNN_0` init path, same
`Embed_V → GCN_1…GCN_l` stack with `FNN_1…FNN_l` protein embeddings summed in, same
concatenation → `FNN_{l+1}` → average pooling → RNN, same `FNN_{l+2}` / `FNN_{l+3}` policy
heads with a joint softmax.

CTDDG's §2.2 spells out the action tensor size slightly more explicitly than CDGCN's:
`|V| × ((|A| × |B|) + |B|)` — for each node, every (new atom type × bond type) `append`
option, plus every bond type `connect` option.

**There is no architectural change whatsoever.** See
[04_model_architecture.md](04_model_architecture.md) §7 for the precise statement of what does
and does not differ between the two papers.

---

## 5. The contribution: the toxicity-weighted loss (§2.3)

### 5.1 Starting point — CDGCN's loss, restated

CTDDG's Eq. 1–4 are identical to CDGCN's Eq. 3–6:

$$
\log P_\Theta(G, T \mid P) = \sum_{j=1}^{m} \log P_\Theta\!\left(t_{j+1} \mid G_j, \ldots, G_1, P\right)
\tag{1}
$$

$$
\log P_\Theta(G \mid P) = \log \sum_{T \in S(G)} P_\Theta(G, T \mid P)
\tag{2}
$$

$$
\log P_\Theta(G \mid P) = \log \mathbb{E}_{T \sim Q_\alpha}\!\left[\frac{P_\Theta(G, T \mid P)}{Q_\alpha(T \mid G, P)}\right]
\;\ge\; \log \frac{1}{K}\sum_{k=1}^{K} \frac{P_\Theta(G, T_k \mid P)}{Q_\alpha(T_k \mid G, P)}
\tag{3}
$$

$$
\hat{L}(\Theta) = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{1}{K}\sum_{k=1}^{K}
\frac{P_\Theta(G_i, T_{ik} \mid P_i)}{Q_\alpha(T_{ik} \mid G_i, P_i)}
\tag{4}
$$

### 5.2 `Q_α(T | G, P)` — the proposal distribution (§2.3.2)

Taken from Li et al. (2018). `α` is the **coefficient of randomness**: the probability of
choosing the next atom according to the **canonical rank** among the neighbours of the last
visited atom while traversing the generation path. It is realised as a **Bernoulli(α)** draw at
each branch point, giving "a kind of stochastic option to choose deterministic or a random path
at each step of the traversal."

The paper notes something useful: *"this distribution gives you an edge on what is being
learned"* — `Q_α` is not just a variance-reduction trick, it shapes which decompositions of a
molecule the model is trained on.

### 5.3 The new objective (§2.3.3)

**Motivation.** Eq. 4 maximises the likelihood of *every* training graph. We want to avoid
generation paths that produce toxic molecules while still maximising the likelihood of
non-toxic ones. So the loss is split into a minimisation part and a maximisation part.

**Eq. 5 — the two-part form:**

$$
\max \hat{L}(\Theta) = \frac{1}{N}\left(\lambda\, \hat{L}_{\text{toxic likelihood}}(\Theta)
\;+\; \hat{L}_{\text{non-toxic likelihood}}(\Theta)\right)
\tag{5}
$$

`λ ≥ 0` is the **trade-off parameter**. The paper's reasoning on its value:

> *"Higher lambda indicates excessive suppression of toxic molecules which might limit the
> model's exploration of certain important molecular scaffolds. Hence for our experiment, we
> have fixed λ to 1."*

**Eq. 6 — the toxic component:**

$$
\hat{L}_{\text{toxic likelihood}}(\Theta) = -\sum_{i=1}^{N}
\left[\log \frac{1}{K}\sum_{k=1}^{K} \frac{P_\Theta(G_i, T_{ik} \mid P_i)}{Q_\alpha(T_{ik} \mid G_i, P_i)}\right]
\mathbb{1}\{G_i \text{ is toxic}\}
\tag{6}
$$

**Eq. 7 — the non-toxic component:**

$$
\hat{L}_{\text{non-toxic likelihood}}(\Theta) = \sum_{i=1}^{N}
\left[\log \frac{1}{K}\sum_{k=1}^{K} \frac{P_\Theta(G_i, T_{ik} \mid P_i)}{Q_\alpha(T_{ik} \mid G_i, P_i)}\right]
\mathbb{1}\{G_i \text{ is non-toxic}\}
\tag{7}
$$

> ⚠ **Typo in the published PDF:** Eq. 7 is printed with the label
> `L_toxic likelihood` — the same label as Eq. 6. From the indicator function and from Eq. 8,
> it is unambiguously the **non-toxic** component. The `max`/`min` wording around Eqs. 5–7 is
> also loose; **Eq. 8 is the operative, unambiguous statement** and is what should be
> implemented.

**Eq. 8 — the combined objective (implement this one):**

$$
\max \hat{L}(\Theta) = \frac{1}{N}\sum_{i=1}^{N}
\left[\log \frac{1}{K}\sum_{k=1}^{K} \frac{P_\Theta(G_i, T_{ik} \mid P_i)}{Q_\alpha(T_{ik} \mid G_i, P_i)}\right]
\cdot \left(1 - (1 + \lambda)\cdot \text{Toxic\_class}_i\right)
\tag{8}
$$

where `Toxic_class_i ∈ {0, 1}` is the toxicity class of the `i`-th ligand — **1 for toxic,
0 for non-toxic**.

### 5.4 How the weight behaves

Everything about CTDDG reduces to this one factor:

| `Toxic_class_i` | Weight `1 − (1+λ)·Toxic_class_i` | With `λ = 1` | Effect on that molecule |
|---|---|---|---|
| 0 (non-toxic) | `1` | **+1** | **Maximise** its likelihood — ordinary training |
| 1 (toxic) | `−λ` | **−1** | **Minimise** its likelihood — gradient reversal |

The paper states the consequence explicitly:

> *"for an individual data point, represented here by the ligand, only one aspect of the loss
> function is activated at a time, either the toxic or non-toxic component."*

**As a minimisation loss** (what you actually write in code), Eq. 8 becomes:

```
loss = -mean( l_i * (1.0 - (1.0 + lam) * tox_i) )
```

where `l_i` is the per-molecule importance-sampled log-likelihood bound (the bracketed term
above), `tox_i ∈ {0,1}`, and `lam = 1.0`.

> **Implementation warning.** In the published code `l` has shape `[batch_size]` (after
> `logsumexp` over the `K` paths) while `tox_class` arrives with `K` entries per molecule,
> i.e. shape `[batch_size * K]`. It must be reduced to one label per molecule (all `K` paths of
> a molecule share its label) before the multiplication. Verify this alignment on a tiny batch
> before any long run.

---

## 6. Training and labelling (§2.4)

BindingDB alone is too small to train the architecture, so CTDDG uses **two phases**, following
Grechishnikova (2021):

| Phase | Data | Model | Toxicity labels from |
|---|---|---|---|
| **1. Pretraining** | ChEMBL | Unconditional — learns to generate valid **non-toxic** molecules | **Random Forest classifier**, trained on the Ames mutagenicity data of **Wu et al. (2021)**, applied to ChEMBL |
| **2. Fine-tuning** | BindingDB protein–ligand pairs | Conditional — learns target specificity | **pkCSM** web server, used to identify ligands with Ames mutagenicity |

> Note the asymmetry: **two different labellers** are used for the two phases. The ChEMBL labels
> are RF predictions; the BindingDB labels are pkCSM predictions. Neither is experimental
> ground truth.

---

## 7. Data (§2.1)

Follows CDGCN's setup exactly.

| Dataset | Role | Numbers as stated |
|---|---|---|
| **ChEMBL** | Pretraining | ~2.3 M curated; **1,093,589** extracted after physicochemical filtering. **No toxicity information in the source data** — hence the RF labeller |
| **BindingDB** (Grechishnikova's version) | Fine-tuning | **166,400** protein–ligand pairs after CDGCN's preprocessing; **1042** unique train proteins, **104** test proteins |

- Proteins as FASTA, ligands as SMILES.
- **65** unique atom types, **4** bond types.
- Train-internal pairwise similarity mostly **< 40%**; train↔test similarity kept **< 20%**
  (Needleman–Wunsch / EMBOSS).
- BindingDB ligands removed from the modified ChEMBL set to avoid leakage.

> ⚠ **Internal inconsistency in the paper:** §2.1 says **166,400** pairs; §2.4 says
> **66,400** pairs for **1146** unique proteins (which is consistent with 1042 + 104). Neither
> figure matches the per-fold counts in the released data. Treat the pair count as unresolved.

---

## 8. Evaluation

**Benchmarks (§3.1):**

| Model | Target-aware? | Toxicity-aware? |
|---|---|---|
| **CDGCN** | Yes | No |
| **SSVAE** (Kang & Cho, 2019) | **No** | Yes — conditioned on logP, QED and predicted toxicity probability; sampled at toxicity probability 0.50 |
| **Pocket2Mol** (Peng et al., 2022) | Yes, via 3D pocket | No |
| **CTDDG** | Yes | Yes |

**Protocol:** all models trained on 1042 proteins, tested on 104. Results averaged over
**10 runs** of **100 generated molecules** per protein. Docking evaluation restricted to the
same **five novel proteins** CDGCN used, with SMINA at default settings.

### 8.1 Drug-like properties (Table 1) — 104 test proteins

| Property | CDGCN | SSVAE | **CTDDG** |
|---|---|---|---|
| Validity (%) ↑ | 91.7 ± 0.22 | 88.4 ± 3.29 | **94.4 ± 0.32** |
| Uniqueness (%) ↑ | 91.7 ± 0.22 | 58.1 ± 4.13 | **94.4 ± 0.32** |
| logP < 5 ↑ | 68.6 ± 0.38 | 51.4 ± 4.31 | **74.8 ± 0.33** |
| MW < 500 ↑ | **73.5 ± 0.47** | 51.4 ± 4.31 | 71.1 ± 0.32 |
| H-bond donors < 5 ↑ | 84.6 ± 0.25 | 51.4 ± 4.31 | **87.9 ± 0.4** |
| H-bond acceptors < 10 ↑ | 84.5 ± 0.28 | 51.4 ± 4.31 | **87.2 ± 0.43** |
| Rotatable bonds < 10 ↑ | **80.3 ± 0.40** | 51.4 ± 4.31 | 79.9 ± 0.38 |
| TPSA < 140 ↑ | 83.5 ± 0.23 | 51.4 ± 4.31 | **84.1 ± 0.36** |
| QED (absolute) ↑ | **0.55 ± 0.00** | 0.44 ± 0.02 | 0.51 ± 0.0 |
| SAS < 6 ↑ | 84.7 ± 0.23 | 51.4 ± 4.31 | **88.6 ± 0.37** |

**Inference:** toxicity optimisation **did not degrade** drug-likeness — CTDDG matches or beats
CDGCN on 7 of 10 metrics.

### 8.2 Target awareness (Table 2) — % of samples docking < −7.0 kcal/mol

| PDB | CDGCN | Pocket2Mol | SSVAE | **CTDDG** |
|---|---|---|---|---|
| 4nos | **99.10 ± 1.08** | 72.49 ± 6.34 | 11.97 ± 3.04 | 95.55 ± 2.56 |
| 2nru | **96.28 ± 2.15** | 95.25 ± 2.91 | 2.49 ± 1.19 | 93.33 ± 3.58 |
| 1lqf | 96.02 ± 1.51 | — | 4.23 ± 1.38 | **98.03 ± 2.09** |
| 1k2r | **98.57 ± 0.67** | 61.22 ± 11.1 | 19.04 ± 3.08 | 95.71 ± 2.92 |
| 1cqp | 85.00 ± 4.07 | **97.74 ± 1.67** | 2.49 ± 1.19 | 86.35 ± 3.43 |
| **Avg** | **95.00 ± 1.90** | 81.68 ± 5.51 | 8.04 ± 1.98 | 93.80 ± 2.92 |

**Inference:** CDGCN is best on average; CTDDG is **comparable** — target awareness is
*preserved* under toxicity optimisation. SSVAE's collapse (8.04%) confirms it has no target
awareness. Pocket2Mol underperforms despite using 3D pocket information, and fails entirely on
1lqf. (Note the small but real trade-off: CTDDG is ~1.2 points below CDGCN on average.)

### 8.3 Toxicity (Table 3) — % of generated samples predicted toxic by pkCSM

| PDB | CDGCN | Pocket2Mol | SSVAE | **CTDDG** |
|---|---|---|---|---|
| 4nos | 24.13 ± 4.52 | 17.5 ± 3.77 | **11.7 ± 4.13** | 19.84 ± 4.38 |
| 2nru | 26.89 ± 5.31 | 30.1 ± 11.17 | **11.7 ± 4.13** | 23.32 ± 4.52 |
| 1lqf | 23.12 ± 4.73 | — | **11.7 ± 4.13** | 20.26 ± 3.88 |
| 1k2r | 22.49 ± 4.00 | 17.9 ± 7.06 | **11.7 ± 4.13** | 20.02 ± 7.29 |
| 1cqp | 19.88 ± 4.11 | 36.2 ± 8.36 | **11.7 ± 4.13** | 13.17 ± 5.31 |
| **Avg** | 23.30 ± 4.53 | 25.43 ± 6.4 | **11.7 ± 4.13** | **19.32 ± 5.08** |

**Inference:** a **consistent decline** in toxic samples vs CDGCN on every one of the five
proteins. Pocket2Mol is highly variable, indicating no toxicity optimisation. SSVAE produces
the fewest toxic ligands — but has no target awareness (Table 2), so the comparison is not
meaningful. **Among methods that can generate for novel proteins, CTDDG is best.**

> This is the paper's central empirical result, and it is modest: **23.30% → 19.32%**, roughly
> a 4-point absolute reduction, with overlapping error bars. Worth keeping in mind as a
> baseline to beat.

---

## 9. Case study: de novo CDK2 inhibitors (§4)

A full downstream validation pipeline, and a template for how to evaluate this project's output.

**Target:** CDK2 (Cyclin-Dependent Kinase 2), a serine/threonine kinase regulating the G1→S
cell-cycle transition. Overexpressed in colorectal, ovarian, breast and prostate cancers.
Input: FASTA of human CDK2 (**UniProt P24941**).

**Pipeline:**

1. **Generate** 100 inhibitors with CTDDG.
2. **Dock** with SMINA against the CDK2 binding pocket (**PDB 1AQ1**), docking box 5 Å larger
   than the ligand. Reference inhibitors: **Staurosporine, Dinaciclib, Roscovitine**.
3. **Screen 1 — binding affinity:** take the top 10 (best: `mol1` at −13.09 kcal/mol).
4. **Screen 2 — Lipinski violations:** `mol3`, `mol4`, `mol6`, `mol8`, `mol10` have zero
   violations, matching the known inhibitors.
5. **Screen 3 — interaction fingerprints (PLIP):** all three known inhibitors form a hydrogen
   bond with **LEU83** — documented in 66 CDK2 crystal structures. `mol3` and `mol7` also form
   it. Selected for MD.
6. **MD simulation:** CHARMM-GUI setup, 150 mM NaCl, **GROMACS 2023.1**, 303.15 K, CHARMM36m
   force field, TIP3P water.
7. **MM-PBSA** binding free energy over the last 5 ns (50 frames), via gmxMMPBSA:

   | Complex | ΔG (kcal/mol) |
   |---|---|
   | CDK2 – Roscovitine (reference) | **−13.42** |
   | CDK2 – **mol3** | **−12.05** |
   | CDK2 – mol7 | −6.28 |

8. **Per-residue decomposition:** ILE10, PHE82, LEU83 contribute strongly for both `mol3` and
   Roscovitine; ASP145 contributes *positively* (i.e. unfavourably). `mol7`'s residue profile
   differs completely, and its LEU83 hydrogen bond is disrupted during the trajectory.
9. **Diversity check:** Tanimoto between `mol3` and Roscovitine — **0.46** on MACCS keys,
   **0.21** on 2D pharmacophore fingerprints. QED differs by ~0.2.

**Conclusion:** `mol3` is a plausible novel CDK2 inhibitor — *structurally diverse from
Roscovitine yet reproducing its interaction pattern*, which the authors argue may confer better
selectivity and efficacy.

**Toxicity of the top 10 (Table 4):** Rat LD50 and Flathead Minnow LC50 from pkCSM. `mol7` has
the lowest Rat LD50 toxicity; only `mol3` matches the known inhibitors on LC50. All except
`mol1` have SAS < 4.5.

---

## 10. Conclusion and stated limitations

CTDDG is presented as the first deep conditional graph generative network for **non-toxic**,
protein-specific de novo drug generation, together with a screening pipeline that goes from
generative output to a lead using biophysical insight.

**Acknowledged limitations / future work:**
- **No wet-lab validation** — explicitly out of scope.
- Future work: improve the **graph encoding scheme** with more informative molecular graph
  representations — and, notably, *"such as maximizing binding affinity, optimizing drug like
  properties etc."* — i.e. **the paper itself points at multi-property optimisation**.

---

## 11. Known discrepancies between the paper and the released code

Important when reproducing. All verified against the published repository.

| # | Paper says | Released code does |
|---|---|---|
| 1 | Eq. 8 toxicity-weighted loss | `_likelihood()` accepts `tox_class_batch` and **never references it**; `forward()` returns plain `-l.mean()` — i.e. CDGCN's Eq. 4 |
| 2 | ChEMBL labelled toxic/non-toxic by a Random Forest | `chembl_final.txt` labels **all 1,090,529 molecules class `1`**; `process_single` parses the label then hardcodes `tox_class = [1] * k` |
| 3 | BindingDB labelled with pkCSM | No pkCSM code, data or labels anywhere in the repository |
| 4 | Two-phase training (pretrain + conditional fine-tune) | The conditional model class exists (`CVanillaMolGen_RNN`) but **no fine-tuning training loop exists anywhere** |
| 5 | Model name CTDDG | `pretraining.ipynb` sets `model_name = 'base_cdgcn'`; `Evaluation_metrics.ipynb` prints results labelled `CDGCN` |
| 6 | MACCS / 2D pharmacophore / Tanimoto / pkCSM / RF / CDK2 analysis | None of `MACCS`, `pharmacophore`, `Tanimoto`, `pkCSM`, `RandomForest`, `Ames`, `CDK2`, `1AQ1`, `Roscovitine`, `PLIP`, `MM-PBSA`, `GROMACS`, `LD50`, `LC50`, `P24941` appear in any notebook |
| 7 | 166,400 BindingDB pairs (§2.1) | §2.4 of the same paper says 66,400; neither matches the released per-fold counts |

**Net:** the published repository is **CDGCN with CTDDG-shaped scaffolding** — the `tox_class`
channel is threaded end-to-end through the data pipeline but never connected to the loss. The
repository *does* contain CTDDG's **results** (`Moleculsr_docking_top10GeneratedMol/`,
`Molecular_dynamics_files/`, `supplimentaryAPIN.pdf`) — just not the code that produced them.

Because CTDDG §2.1–2.2 state that the architecture and datasets are the same as CDGCN's, **the
missing delta is small and precise**: the Eq. 8 loss weighting, the two labelling steps, and
the case-study analysis.

### 11.1 Which of those gaps this reproduction has closed

Verified against the working tree, 2026-09-26. Full detail in
[01_project_background_and_motivation.md](01_project_background_and_motivation.md) §5.

| # (from the table above) | Status here |
|---|---|
| 1 — Eq. 8 not implemented | ✅ **Closed.** Implemented in `code/pretraining copy.ipynb` with `LAMBDA_TOX = 1.0` |
| 2 — ChEMBL labels all `1` | ✅ **Closed.** `code/random_forest_classifier.ipynb` trains the Ames RF (ROC-AUC 0.890) and writes `datasets/data/chembl/chembl_toxicity_labels.csv` — 23.8% toxic. ⚠ The upstream `chembl_final.txt` is still the all-`1` dummy; do not train on it |
| 3 — no pkCSM labels for BindingDB | ❌ **Open.** Nothing yet. Labeller for the fine-tuning set is still undecided |
| 4 — no fine-tuning loop | ❌ **Open** here — but the CDGCN clone has one (`../CDGCN/models/Finetuning.ipynb`), and porting it is the planned route |
| 5 — run named `base_cdgcn` | ✅ **Closed.** The rewrite checkpoints to `checkpoints/` with no misleading name |
| 6 — no CDK2 / MACCS / PLIP / MM-PBSA code | ❌ **Open.** Still only the authors' result archives |
| 7 — pair-count inconsistency | ℹ️ **Unresolved, and now measurable:** the five folds on disk hold 1,042 / 1,000 / 1,004 / 1,002 / 1,036 train proteins and 104 / 112 / 122 / 103 / 124 test proteins, with 143,999 ligand–protein pair lines in the d2 train split. Neither published figure matches |

Two defects **introduced** by the reproduction, not present upstream, both worth fixing before
a real training run:

- The rewrite reads the **unfiltered** `chembl_toxicity_labels.csv` (1,207,359 molecules)
  rather than the filtered `chembl.txt` (1,090,529). That re-admits molecules over 100 atoms
  **and the BindingDB ligands**, reintroducing exactly the leakage §2.1 of this paper says was
  removed.
- `BATCH_SIZE = 16` instead of the paper's 8, so results will not be directly comparable to
  Tables 1–3.

---

## 12. What matters for this project

1. **CTDDG's mechanism is the template for multi-property.** It shows that a property can be
   injected as a **per-molecule scalar multiplier on the log-likelihood**, with no
   architectural change at all. Generalising `(1 − (1+λ)·Toxic_class)` from one binary label to
   several properties — possibly continuous-valued — is the natural next step.
2. **The architecture is untouched by CTDDG.** So the multi-target axis has to be attacked at
   CDGCN's conditioning path, not here.
3. **`λ = 1` was fixed by argument, not by a sweep.** A λ sweep is cheap and would be a real
   contribution on its own.
4. **The toxicity gain is modest** (23.30% → 19.32%, overlapping error bars) and comes with a
   small target-awareness cost (95.00% → 93.80%). That trade-off curve is exactly what
   multi-property optimisation should improve.
5. **Labels are model predictions, not measurements** — RF for ChEMBL, pkCSM for BindingDB.
   Label quality is a genuine lever.
6. **The CDK2 case study is the evaluation template** for demonstrating that generated
   molecules are real candidates rather than just good metric scores.

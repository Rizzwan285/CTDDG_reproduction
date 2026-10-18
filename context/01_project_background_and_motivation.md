# 01 — Project Background and Motivation

> **Read this file first.** It gives the scientific problem, the lineage of prior work this
> project sits on, what we are building, and what the state of *this repository* is right now.
> For the papers themselves see [02_paper_cdgcn.md](02_paper_cdgcn.md) and
> [03_paper_ctddg.md](03_paper_ctddg.md); for the model see
> [04_model_architecture.md](04_model_architecture.md).
>
> **This repository is `Rizzwan285/CTDDG_reproduction`** — a fork of the published
> `sahelybhadra/CTDDG`, in which the CTDDG paper is being reproduced and repaired. Everything
> in §5–§6 describes what is actually on disk here, verified 2026-09-26 and updated 2026-10-11.
>
> **For what has been run and what the output showed, see
> [05_training_runs_and_diagnostics.md](05_training_runs_and_diagnostics.md).** That file is the
> one to read before launching anything expensive; this file is the standing description of the
> project and the tree.

---

## 1. The problem domain

### 1.1 De novo drug design

Behind most diseases there is a **target protein** whose function needs to be suppressed or
modulated. Behind most drugs there is a **lead** — a small molecule with drug-like properties
that binds that protein. Finding leads is the bottleneck of drug discovery.

There are two ways to find them:

| Approach | What it does | Limitation |
|---|---|---|
| **Drug repurposing** | Search known drugs that inhibit proteins similar to the target | Requires detailed knowledge of the target; restricted to known chemical space |
| **De novo drug design** | Invent new molecules for the target | Enormous search space |

This project is entirely about the second.

### 1.2 Why it is hard

- The drug-like chemical space is estimated at **~10^60** molecules; only ~10^8 are
  synthesizable.
- The full pipeline from target identification to market approval takes **10–15 years**.
- It costs **>$1.8 billion** on average, with a high failure rate.
- **~30% of drug candidate attrition is caused by safety/toxicity issues**, and toxicity also
  causes costly withdrawal of already-marketed drugs.

A generative model that proposes a *small* set of plausible candidates for a target shrinks
that search enormously — this is the core value proposition of every method discussed here.

### 1.3 Known vs novel proteins — a distinction used throughout

Both papers use this terminology consistently, and so should you:

- **Known protein** — a protein already known to interact with some known drug. These are in
  the training set.
- **Novel protein** — a protein *not* known to interact with any known drug. These are in the
  test set, and they are the hard, interesting case.

The train/test split is constructed so that test proteins share **<20% sequence similarity**
with training proteins (Needleman–Wunsch global alignment, EMBOSS). That is what makes the
test proteins genuinely "novel" rather than near-duplicates. In this repository that split is
not a claim on paper — the pairwise alignments are materialised on disk under
`datasets/data/bindingdb/needle_outputs/` (§5.4).

---

## 2. Lineage of the work

This project sits at the end of a specific chain. Each link adds one thing.

```
Li et al. (2018) — "Multi-objective de novo drug design with conditional graph generative model"
  │   Sequential graph decoding: build a molecule atom-by-atom with init/append/connect/end.
  │   Conditions on a PREDETERMINED SET of known proteins. Cannot handle unseen proteins.
  ▼
Grechishnikova (2021) — Transformer, protein sequence → SMILES as machine translation
  │   FIRST method that handles NOVEL proteins. But SMILES-based: learns string grammar,
  │   not molecular structure → lower chemical validity, slower, memory-hungry.
  │   ➜ THE DATA. The BindingDB folds in this repo are hers.
  ▼
CDGCN (Mallick & Bhadra, 2023) — "Conditional de novo Drug generative model using GCNs"
  │   Takes Li et al.'s graph decoder and makes it work for NOVEL proteins by injecting a
  │   pretrained protein-sequence embedding into the generator.
  │   ➜ THE ARCHITECTURE. Everything downstream inherits it.
  ▼
CTDDG (Singh, Bhadra & Bhadra) — "Graph-Based Deep Generative Model for Low-Toxic Drug Design"
  │   Reuses CDGCN's architecture verbatim. Adds ONE thing: a toxicity-weighted loss
  │   (Eq. 8) plus toxicity labelling of both datasets.
  │   ➜ THE PROPERTY-AWARE EXTENSION. Single property: Ames mutagenicity.
  │   ➜ THIS REPOSITORY is the reproduction of this paper.
  ▼
THIS PROJECT — Conditional Multi-Target, Multi-Property Drug Generation
      Extends both: more than one target protein at a time, more than one property at a time.
```

**Practical rule for reading:** to understand *how the model works*, read CDGCN. To understand
*what CTDDG adds*, read only CTDDG §2.3 and §2.4. CTDDG §2.1–2.2 explicitly state that its
datasets and architecture are the same as CDGCN's.

---

## 3. What this project is

**Conditional Multi-Target Multi-Property Drug Generation.**

The two predecessor models are both *singular* on two axes, and this project generalises both
axes:

| Axis | CDGCN | CTDDG | This project |
|---|---|---|---|
| **Targets** | One protein per generation, conditioned on a single sequence embedding `c` | Same — one protein | **Multiple targets** conditioned jointly |
| **Properties** | None optimised; likelihood only | **One** — Ames mutagenicity, as a binary label in the loss weight | **Multiple properties** optimised jointly |

### 3.1 Why each axis matters

**Multi-target.** Many therapeutic strategies require a molecule that engages more than one
protein — polypharmacology, or a drug that must hit a target while *avoiding* an anti-target
(a selectivity constraint). CDGCN and CTDDG structurally cannot express this: the condition
`c` is a single fixed-length vector for a single protein.

**Multi-property.** CTDDG's own paper acknowledges the core tension: *"Optimizing individual
properties, such as toxicity, can negatively impact other crucial attributes like binding
affinity and adherence to Lipinski's Rule of Five."* CTDDG handles exactly one property with a
binary ±λ weight. Real lead optimisation needs several properties (toxicity, binding affinity,
QED, SAS, logP, …) balanced simultaneously, and ideally continuous-valued rather than binary.

### 3.2 What "improving upon" means concretely

Three kinds of work, in rough order of how well-grounded they are:

1. **Reproduce and repair.** Get CTDDG actually running end to end. The published repository
   does not run as-is (see §5.1). **This is the current phase, and it is what this repository
   is for.**
2. **Implement what the paper describes but the code does not.** Most notably CTDDG's Eq. 8
   toxicity loss, which is present in the published code only as inert plumbing. *(Done in
   `code/pretraining copy.ipynb` — §5.3.)*
3. **Extend.** Generalise the conditioning mechanism to multiple targets and the loss to
   multiple properties, and evaluate against the CDGCN/CTDDG baselines on the same metrics.

### 3.3 Design constraints inherited from the predecessors

Anything proposed for the extension has to fit within — or deliberately break — these:

- The condition enters at exactly **two points** (first-atom distribution, and an additive bias
  on every GCN layer). See [04_model_architecture.md](04_model_architecture.md) §5.
- The property signal enters at exactly **one point**: a scalar multiplier on the per-molecule
  log-likelihood. It never touches the architecture.
- Protein embeddings are computed **offline and frozen** — the protein encoder is never part
  of the training graph. In this repo the encoder is **ProSE/Bepler**, `--pool avg`, giving
  `N_C = 6165` (§5.4).
- The atom-type vocabulary is **65 types**, and atom-type identity is a *line number* in
  `datasets/data/atom_types.txt`, so that file is effectively immutable once a checkpoint
  exists.

---

## 4. What we are going to do

> This section states direction. Items marked **TBD** are genuinely undecided and should not
> be treated as settled facts by anything reading this file.

### 4.1 Confirmed direction

- Build a **working, properly structured, runnable** implementation of the CDGCN/CTDDG model,
  replacing the published single-giant-cell notebooks with real, sectioned code.
- Implement the **CTDDG toxicity loss (Eq. 8)** for real, with real toxicity labels on both
  ChEMBL (pretraining) and BindingDB (fine-tuning). *Both sides now done — ChEMBL by the Ames
  RF, BindingDB by the same RF rather than pkCSM (§4.2 Q6).*
- Extend conditioning to **multiple target proteins**.
- Extend the objective to **multiple molecular properties**.
- Benchmark against CDGCN and CTDDG on their own metrics: validity, uniqueness, Lipinski
  adherence, QED, SAS, docking-based target awareness, predicted toxicity percentage.

### 4.2 Open design questions — TBD

| # | Question |
|---|---|
| 1 | How are multiple targets combined into one condition? (concatenation / pooling / attention over a set of protein embeddings / separate conditioning streams) |
| 2 | Are multi-target objectives *conjunctive* (bind all of them) or *selective* (bind A, avoid B)? |
| 3 | Which properties beyond Ames mutagenicity? (binding affinity, QED, SAS, logP, hERG, hepatotoxicity, …) |
| 4 | Binary labels with ±λ weights as in CTDDG, or continuous property values with a learned/conditioned head? |
| 5 | Which dataset supplies multi-target supervision? The BindingDB folds here are one-protein-per-ligand; pairing a ligand with several proteins needs a scheme that does not yet exist. |
| ~~6~~ | ~~Which labeller for the BindingDB ligands?~~ **Resolved in code, 2026-10:** the repo's own **Ames Random Forest** labels them, cached in `datasets/data/bindingdb/ligand_ames_labels.csv`. Consistent across both phases; **diverges from the paper**, which used pkCSM. Record it as an experimental deviation, not a reproduction. |
| ~~7~~ | ~~MXNet or PyTorch?~~ **Resolved, 2026-10: ported to PyTorch.** Pretraining, ProSE embedding and fine-tuning are all PyTorch notebooks now. The MXNet notebooks remain as the audit baseline. |
| 8 | `BATCH_SIZE`: **partly resolved.** Fine-tuning uses the paper's **8**; the stage-1 PyTorch pretraining run used **16**. Since the published numbers were obtained at 8 throughout, stage 1 would have to be re-run at 8 for a comparable result. See [05](05_training_runs_and_diagnostics.md) §4 B5. |
| 9 | **New.** How should the fine-tuning model be selected? The paper specifies no rule. `val_nll` penalises successful toxicity suppression by construction and `val_weighted` is unbounded below; `val_l_nontoxic` is the sensible target but is not selectable. See [05](05_training_runs_and_diagnostics.md) §3 D6. |
| 10 | **New.** Does Eq. 8 as published converge at all? The first reproduction run says no — the toxic branch keeps trading non-toxic likelihood for toxic suppression indefinitely. Whether this is fixable within the paper's formulation, or needs a stabiliser the paper does not have, is open. See [05](05_training_runs_and_diagnostics.md) §3. |

> **Resolved since the earlier draft of this file:** the protein encoder question. This
> repository's preprocessing calls **ProSE/Bepler** (`prose/embed_sequences.py --pool avg`),
> matching upstream CDGCN's `N_C = 6165`. There is **no ProtBert code anywhere in this
> repository** — grep confirms zero hits for `protbert` / `reduce_per_protein`.

---

## 5. State of this repository right now

**Current phase (2026-10-11): diagnosing the first end-to-end run.** Both training stages exist
and have been executed once in PyTorch; the fine-tuning result does not converge, and the
stage-1 checkpoint's provenance is unverified. Full account in
[05_training_runs_and_diagnostics.md](05_training_runs_and_diagnostics.md).

### 5.1 Why the upstream code needed a rebuild

The published CTDDG repository (`github.com/sahelybhadra/CTDDG`, this repo's `upstream` remote)
cannot be run end to end:

- The code is **Colab exports**. `code/pretraining.ipynb` is 5 cells, of which the first is
  **1,509 lines** and the rest are trivial. Nothing is importable; model definitions are
  duplicated across notebooks rather than shared. (`code/PRETRAINING_REVIEW.md` describes it as
  "1 cell, 1,508 lines" — accurate when the review was written.)
- Paths are hardcoded to the authors' machine (`/workspace/Toxicity_experiment/...`, 5
  occurrences) and the device is hardcoded to `mx.gpu()` (11 occurrences).
- **An entire training stage is missing.** The conditional fine-tuning stage — the stage that
  makes the model protein-aware — has a model class defined (`CVanillaMolGen_RNN`, in
  `code/Generating_samples.ipynb`) but **no training loop anywhere in this repository**.
- **The toxicity mechanism is inert.** `tox_class` is threaded through the whole data pipeline
  and then never used: `process_single` parses the label and hardcodes `tox_class = [1] * k`,
  and `_likelihood()` accepts `tox_class_batch` and never references it.

So the upstream code is, functionally, **CDGCN with CTDDG-shaped scaffolding**. Its own
internals agree: `code/pretraining.ipynb` sets `model_name = 'base_cdgcn'`, and
`code/Evaluation_metrics.ipynb` prints its results under the label `CDGCN`.

The full line-by-line audit is in **[`code/PRETRAINING_REVIEW.md`](../code/PRETRAINING_REVIEW.md)**,
which lists 13 numbered changes (F1–F13). Read it before touching the pretraining code.

### 5.2 The Ames mutagenicity classifier — done

`code/random_forest_classifier.ipynb` builds the toxicity labeller the paper describes:

| Step | Detail |
|---|---|
| Source data | The two Hansen mutagenicity sets in `datasets/` — `Mutagenicity_N7090.csv` (tab-separated, 7,090 rows) and `Mutagenicity_N6512.csv` (comma-separated, 6,512 rows), columns `CAS_NO, Source, Activity, Steroid, WDI, Canonical_Smiles, REFERENCE` |
| Merge + dedup | Merged, exact duplicates and duplicate molecules removed, rows without SMILES or labels dropped → **7,083** usable molecules |
| Features | **Morgan fingerprints, radius 2, 2048 bits** (config saved to `models/fingerprint_config.pkl`) |
| Split | 80/20 stratified, `random_state=42` → 5,666 train / 1,417 test |
| Model | `RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)` |
| Test metrics | accuracy **0.811**, precision **0.832**, recall **0.807**, F1 **0.819**, ROC-AUC **0.890** |
| Confusion matrix | `[[542, 123], [145, 607]]` |
| Saved to | `models/ames_random_forest.pkl` |

**Label semantics** (important, and easy to get backwards): Hansen `Activity = 1` → Ames
positive → mutagenic → **toxic**. The classification report uses
`target_names=["Ames Negative", "Ames Positive"]`. So the RF's `1` **is** `Toxic_class = 1` in
Eq. 8, with no inversion.

> ⚠ **`models/ames_random_forest.pkl` was not on disk as of 2026-09-26.** `.gitignore` has a
> blanket `*.pkl`, and only `models/fingerprint_config.pkl` survives (it was committed before
> that rule). Re-running the training cells of `code/random_forest_classifier.ipynb` regenerates
> it deterministically (`random_state=42`).
>
> `02_finetune_ctddg.ipynb` has `RF_AUTO_TRAIN = True`, so it rebuilds the RF in place if the
> file is missing or unreadable by the current sklearn. **Whether the `.pkl` now on disk is the
> original or an auto-rebuild is unverified**, and the two are not guaranteed identical: the
> notebook's rebuild adds an RDKit-parseability filter the original did not have, which can shift
> the dedup count below 7,083 and move the split. The label cache records a SHA-256 of whichever
> model wrote it, so a swap is detected — it just is not recorded here which one it was.

### 5.3 ChEMBL labelling and the pretraining notebooks

> **Lineage.** Upstream MXNet export → MXNet rewrite (`code/pretraining copy.ipynb`) → **PyTorch
> port, which is what actually produced `checkpoints/ckpt.pt`** (§5.3.2). The MXNet rewrite is
> superseded; it is described below because the F1–F13 audit trail runs through it.

**Labelled ChEMBL.** The same notebook applies the RF to the full ChEMBL export and writes
`datasets/data/chembl/chembl_toxicity_labels.csv` (377 MB; the notebook calls it
`chembl_with_ames_predictions.csv` — it was renamed on disk afterwards):

- **1,207,359 rows** (the one unparseable SMILES was dropped), 34 columns: the 31 ChEMBL export
  columns plus `Mol`, `Fingerprint` and **`Ames_prediction`**.
- `Ames_prediction`: **287,620 toxic (1.0) / 919,739 non-toxic (0.0)** → **23.8% toxic**.

**The reworked notebook.** `code/pretraining copy.ipynb` is the real work item: 36 cells,
sectioned (imports · config · vocab & traversal · loaders · model & loss · init/resume ·
training loop), replacing the upstream single cell. Against `PRETRAINING_REVIEW.md` it applies:

| Review item | Status in `pretraining copy.ipynb` |
|---|---|
| **F1** — Eq. 8 in `_likelihood` | ✅ `tox_labels = tox_class_batch.reshape((batch_size, iw_size))[:, 0]`; `tox_weight = 1.0 - (1.0 + LAMBDA_TOX) * tox_labels`; returns `l * tox_weight` |
| **F2** — stop discarding the parsed label | ✅ `tox_class = np.full((k,), tox_label)` from the record |
| **F3** — real labels on the pretraining data | ✅ but **differently** — see the warning below |
| **F4** — define λ | ✅ `LAMBDA_TOX = 1.0` |
| **F5** — kill the `/workspace` paths | ✅ `DATA_PATH` / `ATOM_TYPES_PATH` / `CHECKPOINT_DIR` under `/home/muhamed/repo/CTDDG` |
| **F6** — one device context | ✅ `CTX = mx.gpu(GPU_ID) if mx.context.num_gpus() > 0 else mx.cpu()` |
| **F7** — `forward` unpack order | ✅ fixed, with a comment |
| **F8** — guarded SMILES parsing | ✅ replaced by an "OOV shield" that pre-filters molecules with out-of-vocabulary atoms in the collator |
| **F9** — debug prints | ✅ gone |
| **F11** — RDKit logging, split cells | ✅ `RDLogger.DisableLog('rdApp.*')`, 36 cells |
| **F12** — log the toxic/non-toxic branches separately | ❌ not done *in the MXNet rewrite* — ✅ **done in the PyTorch port**, which writes `l_nontoxic` / `l_toxic` columns |
| **F13** — divergence guard on the toxic branch | ❌ not done *in the MXNet rewrite* — ⚙️ **present but disabled in the PyTorch port** (`TOX_FLOOR = None`, the paper-faithful default) |

Hyperparameters in the reworked notebook: `BATCH_SIZE = 16` (**paper: 8**), `K_PATHS = 5`,
`P_ALPHA = 0.8`, `LAMBDA_TOX = 1.0`, `LR = 1e-3`, `DECAY = 1e-3` every 100 steps,
`CLIP_GRAD = 3.0`, `N_RNN = 3`, `TOTAL_ITERATIONS = 480000`, `SUMMARY_STEP = 5`. The comment in
the cell says the batch size was raised for an **L40 (48 GB)** GPU.

> ⚠ **Three things to fix before a long run.** Status as carried into the PyTorch port:
> (1) fixed — `SUMMARY_STEP = 500`; (2) **not fixed — the port trains on the unfiltered CSV, and
> this is now known to have actually happened**, see [05](05_training_runs_and_diagnostics.md)
> §4 B2; (3) not fixed — stage 1 ran at `BATCH_SIZE = 16`.
>
> 1. **`SUMMARY_STEP = 5` is a debug value.** It saves `ckpt.params` (27.9 MB) *and*
>    `trainer.status` (55.9 MB) every 5 iterations — ~84 MB of I/O per 5 steps, for 480,000
>    steps. Put it back to 500 before launching.
> 2. **The notebook trains on the *unfiltered* export.** `DATA_PATH` points at
>    `chembl_toxicity_labels.csv` (1,207,359 molecules), not at the filtered
>    `chembl_final.txt` / `chembl.txt` (1,090,529). The filtered file was produced by
>    `datasets/data/data_preprocessing_bindingdb.ipynb` cell 70 as
>    `set(chembl_mols) - set(bindingdb_ligs)` **after** dropping molecules with >100 atoms.
>    Training on the raw CSV therefore (a) re-admits oversized molecules, which `max_iter = 120`
>    only partly contains, and (b) **re-admits the BindingDB ligands — i.e. reintroduces
>    leakage from the fine-tuning/test set into pretraining**, which both papers explicitly
>    avoid. The OOV shield filters unknown *atom types*, not either of these. This is the most
>    consequential open defect in the current pipeline.
> 3. **`BATCH_SIZE = 16` departs from the published 8**, so results are not directly comparable
>    to Tables 1–3 of either paper. (`trainer.step(batch_size=1)` remains correct — `forward`
>    already averages over the batch. Do not "fix" it.)

**What the MXNet rewrite actually ran.** Its stored outputs show execution on a GPU node
(`Active Compute Context: gpu(0)`) through data loading (`Total molecules loaded: 1207359`,
`Non-Toxic (0) = 919739, Toxic (1) = 287620`), a batch sanity check (`Batch size (N): 16,
Paths per molecule (K): 5`, `X shape: (28694,)`) and model construction
(`Total trainable parameters: 6,986,506`). **The training-loop cell has no output**, and
`checkpoints/log.out` was truncated to its header on 2026-09-17 by a re-run with
`FORCE_RESTART = True`. The MXNet artifacts `ckpt.params` / `trainer.status` (2026-09-16) are a
**smoke-test artifact, not a trained model.**

> For reference, a completed **480,000-iteration unconditional run** (the **CDGCN** objective,
> batch 8, final loss 15.99 after 2,799 h of accumulated wall-clock) exists in the *other*
> repository, at `../CTDDG_new/outputs/pretrain/`. Same architecture, same parameter count,
> different weights. It predates the Eq. 8 work and is **not** a CTDDG model.
>
> ⚠ **Do not use 15.99 as a yardstick for any PyTorch number.** Different framework, different
> objective (unweighted, so that column *is* an NLL; the PyTorch `loss` column is the Eq. 8
> weighted loss and goes negative), different repository. See
> [05](05_training_runs_and_diagnostics.md) §6.

#### 5.3.1 The MXNet rewrite — superseded

`code/pretraining copy.ipynb` remains on disk as the record of the F1–F13 work. It is not what
produced the current checkpoint and should not be run.

#### 5.3.2 The PyTorch pretraining port — what produced `checkpoints/ckpt.pt`

A line-for-line PyTorch port of the MXNet pretraining notebook. Architecture, loss and
hyperparameters unchanged; only framework mechanics differ. (Its exact filename in the tree is
not recorded here — identify it by its header, *"CTDDG Pretraining — PyTorch port"*.)

| | Value |
|---|---|
| Objective | **CTDDG Eq. 8**, `LAMBDA_TOX = 1.0`, `TOX_FLOOR = None` |
| Data | `chembl_with_ames_predictions.csv` — **unfiltered** (see the warning above) |
| Batch / K / α | **16** / 5 / 0.8 |
| LR / decay / clip | 1e-3 / 1e-3 per 100 steps / 3.0 |
| Length / checkpoint | 480,000 iterations / every **500** steps |
| Artifacts | `checkpoints/ckpt.pt`, `trainer.pt`, `configs.json`, `log.out` (`step, time(h), loss, lr, l_nontoxic, l_toxic`) |
| Parameters | **6,981,514** trainable **+ 4,992** BatchNorm running stats **= 6,986,506**, matching MXNet (which counted the buffers as parameters) |

Deliberate deviations from the MXNet original, all flagged inline with `# PORT:`:
`optimizer.zero_grad()` added; `self.dense(X)` computed once instead of twice; CSV read with
`usecols`; stricter toxicity-column detection (the original's `'label' in c.lower()` would have
matched ChEMBL's **`BAO Label`** column); per-branch logging (F12); optional toxic floor (F13,
off); `.pt` filenames so a torch checkpoint cannot be mistaken for the MXNet one beside it.

> ⚠ **How much of the 480,000 iterations actually ran is unverified.** `max(num_batches_tracked)`
> in `ckpt.pt` is the exact count — see [05](05_training_runs_and_diagnostics.md) §4 B1 for why,
> and §5 for how to read it. The port also has **no bond-count cap** in its collator, which
> predicts a crash on any molecule with ≥120 bonds (§4 B6).

### 5.4 BindingDB — preprocessing done, embeddings done for fold d1

`datasets/data/bindingdb/` holds Grechishnikova's five folds, already expanded:

| Fold (train/test dirs) | Train proteins | Test proteins |
|---|---|---|
| `*_4_org_1042_104` (d1) | 1,042 | 104 |
| `*_4_org_1000_112` (d2) | 1,000 | 112 |
| `*_4_org_1004_122` (d3) | 1,004 | 122 |
| `*_4_org_1002_103` (d4) | 1,002 | 103 |
| `*_4_org_1036_124` (d5) | 1,036 | 124 |

Per split directory: one `proteinN.fasta` per unique protein, `d{n}_{tr,te}_unique_proteins.txt`
(space-separated residues, one protein per line), `d{n}_{tr,te}_unique_proteins_bepler.fa`
(standard FASTA, `>p1…>pN`), and the raw aligned pair files `ligands_*_corrected` /
`proteins_*_corrected` — **143,999 lines each** in the d2 train split, one ligand–protein pair
per line, SMILES and residues both space-separated.

Pairwise Needleman–Wunsch alignments are **computed and on disk** under `needle_outputs/`:
`train-test` 574,526 files, `within_train` 5,171,080 files, `within_test` 64,229 files. (That
is ~5.8 M small files — be careful with any recursive tool over this directory.)

`datasets/data/data_preprocessing_bindingdb.ipynb` (130 cells) is the repathed and extended
descendant of the upstream `code/data_preprocessing_1.ipynb` (108 cells): `data_path` now
points at `/home/muhamed/repo/CTDDG/datasets/data/bindingdb/`. It carries the whole chain —
fold repair, FASTA export, needle alignments, ligand filtering (>100 atoms dropped), atom-type
vocabulary construction, BindingDB-ligand removal from ChEMBL, ProSE embedding, and assembly of
the final per-model training files.

**Embeddings — resolved for d1 (2026-10).** `01_generate_prose_embeddings.ipynb` (conda env
**`bioemb`**) now drives ProSE from a vendored `prose/` checkout with
`saved_models/prose_mt_3x1024.sav`, embedding each split in resumable shards of 128 and merging
them atomically into `<fold>_<tr|te>_unique_embeddings_bepler.h5` — one average-pooled
`(6165,)` vector per protein, keyed `p1 … pN`, validated for shape and finiteness.

- **Fold d1 (train + test) is done** — [CONFIRMED] because `02_finetune_ctddg.ipynb` loaded
  those files and passed `verify_fasta_order`, which compares the bepler FASTA against the
  protein list record for record and refuses to start on a mismatch.
- The notebook ships with `FOLDS = ["d1"]`. **Whether d2–d5 have been embedded is unverified.**
- `DEVICE = -1`, i.e. **CPU**. The `resolve_device` helper only falls back to GPU when `DEVICE`
  is `None`, so set it explicitly if you want the GPU.
- The notebook's own header promises a Slurm job as step 6; the notebook ends at verification.

> Note the format change: the file is **`.h5`**, not the `.txt` the older preprocessing chain
> and `04` §5.1 describe. The fine-tuning notebook reads `.h5` directly and assembles pairs from
> the raw fold files, so the tab-delimited `<SMILES>\t<c_1>…` intermediate is no longer needed.

### 5.5 The blocking chain — cleared as far as a conditional checkpoint

The old blocker (no ProSE embeddings, no fine-tuning loop) is gone. `02_finetune_ctddg.ipynb`
is a self-contained PyTorch fine-tuning notebook: it loads `checkpoints/ckpt.pt`, adds the
conditioning path, assembles pairs straight from the raw fold files, labels ligands with the
Ames RF, trains under Eq. 8, and writes `outputs/finetune/<fold>/`.

What now blocks progress is **validity, not plumbing**:

```
stage-1 provenance unverified  ──┐
BindingDB ligands leaked into    ├─> the fine-tuning result cannot be interpreted
  stage-1 pretraining data     ──┤    or compared to the paper
conditioning scale unmeasured  ──┘
      └─> no trustworthy conditional checkpoint
            └─> generation / evaluation / docking still not started
```

See [05_training_runs_and_diagnostics.md](05_training_runs_and_diagnostics.md) §4 for each
blocker and §5 for the evidence that settles them.

Still true and still useful: `../CDGCN/models/Finetuning.ipynb` (38 cells) is the reference the
PyTorch loop was built against — `EarlyStopping(patience=10)`, resume logic, `N_C = 6165`,
`lr = 1e-4`, `decay = 0.002`, `max_epochs = 10`, `summary_step = 200`, `batch_size = 8`.

### 5.6 Summary table

| Item | State (2026-10-11) |
|---|---|
| Framework | ✅ **PyTorch** for all three live notebooks. `ctddg_env` (MXNet) remains for the superseded notebooks; `bioemb` for ProSE |
| Ames RF classifier trained and evaluated | ✅ Done — 0.811 acc / 0.890 ROC-AUC |
| `models/ames_random_forest.pkl` present | ⚠️ Unverified which copy is on disk — may be an auto-rebuild (§5.2) |
| ChEMBL labelled with real Ames predictions | ✅ 23.8% toxic. ⚠️ Two filenames in circulation (§5.3.2) |
| Upstream `chembl_final.txt` | ⚠️ Still the dummy file — all 1,090,529 lines labelled `1`. Do not train on it |
| Eq. 8 implemented | ✅ In **both** stages, `λ = 1.0`, `TOX_FLOOR = None`. Verified correct by reconciliation — [05](05_training_runs_and_diagnostics.md) §3 D1 |
| Stage-1 pretraining run | ⚠️ A run happened and produced `checkpoints/ckpt.pt`; **how many iterations is unverified** — [05](05_training_runs_and_diagnostics.md) §4 B1 |
| Stage-1 data provenance | ❌ **Unfiltered CSV — BindingDB ligands leaked into pretraining.** The most serious open issue — [05](05_training_runs_and_diagnostics.md) §4 B2 |
| BindingDB folds, FASTA, needle alignments | ✅ On disk |
| ProSE/Bepler protein embeddings | ✅ Fold **d1** (train + test). d2–d5 unverified |
| Conditional fine-tuning loop | ✅ `02_finetune_ctddg.ipynb`, self-contained PyTorch |
| Fine-tuning run | ⚠️ One completed on d1 (6 epochs). **Does not converge** — best epoch is 1 and non-toxic validation NLL degrades after it. [05](05_training_runs_and_diagnostics.md) §2–3 |
| Generation / evaluation / docking | ❌ Not started; notebooks still carry `/workspace/...` paths |
| Comparable to the paper's Tables 1–3 | ❌ **No** — stage-1 leakage (B2), `BATCH_SIZE = 16` in stage 1 (B5), and RF instead of pkCSM for BindingDB labels |

---

## 6. Where things live

### 6.1 Inside this repository

| Path | What it is |
|---|---|
| `code/pretraining.ipynb`, `code/pretraining.py` | The **upstream** Colab export and its flat extraction (1,509 lines / 1,508 lines). Unmodified — `/workspace` paths, `mx.gpu()`, `tox_class = [1] * k`. Kept as the audit baseline |
| `code/pretraining copy.ipynb` | The **MXNet** rewrite — 36 cells, Eq. 8 connected, local paths, real labels. **Superseded by the PyTorch port**; kept for the F1–F13 trail |
| PyTorch pretraining port (filename not recorded) | **Stage 1 as actually run.** Eq. 8, `λ=1`, batch 16, 480k target, writes `checkpoints/*.pt`. Header: *"CTDDG Pretraining — PyTorch port"*. See §5.3.2 |
| `01_generate_prose_embeddings.ipynb` | ProSE/Bepler embedding driver, env `bioemb`, resumable shards → `*_unique_embeddings_bepler.h5`. Fold d1 done |
| `02_finetune_ctddg.ipynb` | **Stage 2.** Self-contained PyTorch conditional fine-tuning with Eq. 8, Ames-RF ligand labelling, FASTA-order verification, atomic checkpoints, provenance reporting. 81 cells |
| `code/PRETRAINING_REVIEW.md` | Line-by-line review of the upstream notebook against the paper; F1–F13 change list |
| `code/random_forest_classifier.ipynb` | Ames RF: Hansen → merge → Morgan → RF → metrics → ChEMBL annotation |
| `code/data_preprocessing_1.ipynb` | Upstream BindingDB preprocessing (still `/workspace/mtp_data/`) |
| `code/Generating_samples.ipynb` | Conditional sampler + `CVanillaMolGen_RNN`. `/workspace/CTDGD/...` paths |
| `code/Evaluation_metrics.ipynb` | Validity / uniqueness / Lipinski / QED / SAS. Labels its output `CTDGD` |
| `code/Molecular_docking.ipynb` | Jupyter Dock / SMINA / Vina pipeline |
| `datasets/Mutagenicity_N7090.csv`, `Mutagenicity_N6512.csv` | Hansen Ames sets (tab- and comma-separated respectively — they differ) |
| `datasets/hansen_dataset_exploration.ipynb` | Quick shape/column inspection of the two |
| `datasets/data/atom_types.txt` | **65** lines of `symbol,formal_charge,num_explicit_Hs`. Line number = atom-type id |
| `datasets/data/chembl/` | `chembl.csv` (raw export, 1,207,360), `chembl.txt` (filtered, 1,090,529), `chembl_final.txt` (same + dummy `" 1"`), `chembl_toxicity_labels.csv` (RF-labelled, 1,207,359) |
| `datasets/data/bindingdb/` | Five folds, per-protein FASTA, `needle_outputs/`, vocab file. See §5.4 |
| `datasets/data/data_preprocessing_bindingdb.ipynb` | The repathed, extended preprocessing chain — the one to use |
| `models/fingerprint_config.pkl` | `{'type': 'Morgan', 'radius': 2, 'n_bits': 2048}` |
| `models/checking.ipynb` | Loads the RF + config and predicts on aspirin / 2-nitrofluorene. Needs the missing `.pkl` |
| `checkpoints/` | **MXNet:** `ckpt.params`, `trainer.status` (2026-09-16 smoke test), header-only `log.out`. **PyTorch:** `ckpt.pt`, `trainer.pt`, `configs.json`, `log.out`. Gitignored. ⚠ Note `log.out` is shared by both lineages — the PyTorch run rewrites it |
| `outputs/finetune/<fold>/` | Stage-2 artifacts: `ckpt_best.pt` (evaluate this), `ckpt_last.pt` (resume), `trainer.pt`, `configs.json`, `log.out`, `val.out`, `test_set.npz`, `split_manifest.json`. A `LIMIT_PAIRS` run writes `.smoke` variants of the last two |
| `datasets/data/bindingdb/ligand_ames_labels.csv` + `.meta.json` | Ames labels for BindingDB ligands, with a SHA-256 fingerprint of the RF that produced them. Rejected and rebuilt if the model cannot be verified |
| `prose/` | Vendored ProSE checkout + `saved_models/prose_mt_3x1024.sav` |
| `supplimentaryAPIN.pdf` | CTDDG supplementary — Table S1 (top-10 CDK2 SMILES), Table S2 (MD interactions) |
| `Molecular_dynamics_files.zip`, `Moleculsr_docking_top10GeneratedMol.zip` | The paper's *results*, shipped by the authors — not the code that produced them |
| `context/` | **This folder** |

**Not tracked by git:** `datasets/`, `checkpoints/`, `*.pkl` (except the already-committed
`fingerprint_config.pkl`), `outputs/`, `results/`. Only 16 files are tracked. The large data is
local-only — see the upstream `README.md` for the authors' Google Drive link.

### 6.2 Sibling repositories

| Path | What it is |
|---|---|
| `/home/muhamed/repo/CTDDG` | **This repo.** `origin` = `Rizzwan285/CTDDG_reproduction`, `upstream` = `sahelybhadra/CTDDG` |
| `/home/muhamed/repo/CDGCN` | `Rizzwan285/CDGCN` — the predecessor's reference implementation. **Has the fine-tuning loop this repo lacks** (`models/Finetuning.ipynb`) |
| `/home/muhamed/repo/CTDDG_new` | `Rizzwan285/CTDDG` — the earlier working copy: `PROJECT_DOCS/` (13 long-form files), `.claude/`, `papers/CDGCN.pdf` + `papers/CTDDG.pdf`, `scripts/`, `cluster/`, and the completed 480k pretraining run under `outputs/pretrain/` |

**The two PDFs live in `../CTDDG_new/papers/`, not here.** Files 02 and 03 of this folder
summarise them closely enough that you should rarely need the originals.

**Compute:** Bhavani cluster, IIT Palakkad (`bhavani.iitpkd.ac.in`), Slurm. The reworked
notebook's comments target an **L40 (48 GB)**; the earlier 480k run was on an A30 24 GB.
`conda activate` alone is **not sufficient** on compute nodes — `PATH` and `LD_LIBRARY_PATH`
must also be exported, or the cluster's system Anaconda shadows `ctddg_env`.

---

## 7. Terminology to preserve

| Term | Meaning |
|---|---|
| **known protein** / **novel protein** | In train set (has known drugs) / in test set (has none) |
| **lead** | A candidate molecule with drug-like properties |
| **generation path `T`** | The ordered sequence of actions that builds one molecule |
| **`Q_α`** | Predefined proposal distribution over generation paths |
| **`α`** (code: `p` / `P_ALPHA`) = 0.8 | Probability of following canonical atom rank rather than a random neighbour |
| **`K`** (code: `k` / `K_PATHS`) = 5 | Number of generation paths sampled per molecule |
| **`λ`** (code: `LAMBDA_TOX`) = 1 | Toxicity trade-off parameter in CTDDG Eq. 5/8 |
| **`N_C`** = 6165 | Width of the ProSE/Bepler protein embedding — the conditioning vector |
| **`S_N`** | A set of `N` generated molecules |
| **target awareness** | % of generated molecules docking below −7.0 kcal/mol |
| **Ames mutagenicity** | The toxicity endpoint CTDDG uses — ability to induce DNA mutations |
| **actions** | `init`, `append`, `connect`, `end` |

Four traps:

- `beam_size` in the evaluation/docking notebooks means **number of stochastic samples**.
  There is no beam search anywhere in this codebase.
- `CTDGD` in output paths is a typo for CTDDG, but it is load-bearing across three notebooks.
- **`chembl_final.txt` is not the labelled file.** It is `chembl.txt` with a dummy `" 1"`
  appended to every line. The real labels are in `chembl_toxicity_labels.csv`.
- **`code/pretraining copy.ipynb` is no longer the current one.** It was, in the MXNet era.
  Stage 1 as actually run is the **PyTorch port**; `pretraining.ipynb` is the untouched upstream
  export. Three implementations now coexist — check which one you are reading
  ([04_model_architecture.md](04_model_architecture.md) §9).
- **`checkpoints/log.out` is written by both the MXNet and the PyTorch pretraining notebooks.**
  Its header row tells you which: the PyTorch one adds `l_nontoxic` and `l_toxic` columns.
- **`λ = 0` does not recover CDGCN**, despite [04](04_model_architecture.md) §6.5 saying the
  weight reduces to CDGCN's loss. The weight `1 − (1+0)·tox` is **1 for non-toxic and 0 for
  toxic**, so toxic ligands are *dropped from the batch average*, not trained on normally. It
  removes the gradient reversal, which makes it a useful control, but it is not CDGCN.

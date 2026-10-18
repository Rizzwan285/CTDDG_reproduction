# `context/` — Project context for LLMs

Self-contained background on **this repository** (`Rizzwan285/CTDDG_reproduction`, forked from
`sahelybhadra/CTDDG`), written to be pasted or loaded into an LLM before asking it to work on
the code. Each file stands alone; together they give the full picture without needing the
papers, the sibling repositories, or a tour of the codebase.

## Files

| File | Covers | Load when… |
|---|---|---|
| [01_project_background_and_motivation.md](01_project_background_and_motivation.md) | The scientific problem, the lineage (Li et al. → CDGCN → CTDDG → this project), what we are building, **what is actually on disk here and what state it is in**, where everything lives, terminology | **Always. Start here.** |
| [02_paper_cdgcn.md](02_paper_cdgcn.md) | The CDGCN paper — the architecture, the conditioning mechanism, datasets, metrics, results; plus what the CDGCN clone next door provides | Working on the model, conditioning, fine-tuning, or the evaluation protocol |
| [03_paper_ctddg.md](03_paper_ctddg.md) | The CTDDG paper — the toxicity-weighted loss, labelling, results, CDK2 case study, where the paper and the released code disagree, and **which of those gaps this reproduction has closed** | Working on the loss, toxicity, properties, or reproducing CTDDG |
| [04_model_architecture.md](04_model_architecture.md) | The shared architecture in full, all equations, the complete CDGCN↔CTDDG diff, hyperparameters per phase, **the three coexisting pretraining implementations here**, extension points | Writing or modifying model/training code |
| [05_training_runs_and_diagnostics.md](05_training_runs_and_diagnostics.md) | **What has actually been run, what the logs showed, what that does and does not establish, the open blockers, and the evidence to collect next** | **Before launching any training run, or interpreting any result** |

**Minimum useful context:** `01` + `04`.
**Before running or interpreting training:** add `05`. It is the only file with measured numbers.
**For anything touching the objective:** add `03`.
**For anything touching fine-tuning or conditioning:** add `02`.

## Conventions

- **TBD** marks a genuinely open design decision. Do not treat it as settled.
- **⚠ CODE** marks a place where an implementation diverges from the papers.
- **⚠** on its own marks a trap or a defect to fix.
- Equations use the papers' own numbering. Where the two papers number the same equation
  differently, both numbers are given.
- **[CONFIRMED]** / **[UNVERIFIED]** / **[MISSING]** tag claims in `05`. Treat an
  **[UNVERIFIED]** claim as a hypothesis, not a fact, and do not repeat it without the tag.
- Repository facts (file counts, row counts, metrics, checkpoint state) were verified against
  the working tree on **2026-09-26**, with the training stack and run state updated
  **2026-10-11**. Both dates appear inline where things are likely to move.

## Six things that most often trip up a fresh session

1. **The stack is PyTorch now.** Stage 1 (pretraining), ProSE embedding and stage 2
   (fine-tuning) are PyTorch notebooks. The MXNet notebooks — including
   `code/pretraining copy.ipynb`, which *used* to be the live one — are superseded and kept as
   the audit baseline. Three pretraining implementations coexist; check which you are reading
   ([04](04_model_architecture.md) §9).
2. **`chembl_final.txt` is not the labelled dataset.** It is `chembl.txt` with a dummy `" 1"`
   appended to every line — all 1,090,529 molecules carry the same label. The real Ames labels
   are in the RF-annotated CSV. ⚠ That CSV is referred to by **two names** in the tree
   (`chembl_toxicity_labels.csv` and `chembl_with_ames_predictions.csv`); which one stage 1
   trained on matters and is unverified ([01](01_project_background_and_motivation.md) §5.3.2).
3. **Something *has* now been trained under the CTDDG objective — but how much is unverified.**
   `checkpoints/ckpt.pt` came from the PyTorch pretraining port, which applies Eq. 8 at
   `λ = 1.0`. The iteration count is an open question
   ([05](05_training_runs_and_diagnostics.md) §4 B1). Do not assume 480,000.
4. **Stage 1 trained on the unfiltered ChEMBL export, which contains the BindingDB ligands.**
   This leaks the fine-tuning and test ligands into pretraining and is the single most serious
   open issue ([05](05_training_runs_and_diagnostics.md) §4 B2).
5. **`λ = 0` is not CDGCN.** The weight `1 − (1+0)·tox` is 0 on toxic ligands, so they are
   dropped from the batch average rather than trained on normally. Earlier drafts of `04` §6.5
   said otherwise; that has been corrected.
6. **`datasets/data/bindingdb/needle_outputs/` holds ~5.8 million small files.** Exclude it
   from any recursive search, listing or archive.

## Relationship to the other documentation

| Location | Audience | Content |
|---|---|---|
| `context/` (here) | **LLMs** — background, scientific grounding, current state, run history | These six files |
| `code/PRETRAINING_REVIEW.md` | Anyone touching the pretraining code | Line-by-line audit of the upstream notebook against the paper; the numbered change list **F1–F13**, with which are applied tracked in `01` §5.3 |
| `../CTDDG_new/PROJECT_DOCS/` | Humans — long-form, audit-grade | 13 files written against the *earlier* working copy: file-by-file codebase map, data statistics, checkpoint semantics, cluster/Slurm instructions, modification guide |
| `../CTDDG_new/.claude/` | Coding sessions in that repo | Compact operational state for `CTDDG_new` |

`context/` explains **what the project is, why, and where it currently stands in this repo**.
`code/PRETRAINING_REVIEW.md` explains **what is wrong with the pretraining code and in what
order to fix it** — note F12 is now done and F13 implemented-but-disabled in the PyTorch port,
which that file predates.

> ⚠ `PROJECT_DOCS/` and `.claude/` live in the **sibling** repository `../CTDDG_new` and
> describe *that* tree — a different layout (`scripts/`, `cluster/`, `notebooks/`,
> `outputs/pretrain/`) and an earlier state of the work. Where they disagree with these files
> about paths, filenames or progress, **these files are the ones describing this repository.**
> Notably, `PROJECT_DOCS` and the older context refer to a **ProtBert-BFD** embedding path —
> that does not exist here; this repo uses **ProSE/Bepler** at `N_C = 6165`.
>
> ⚠ The same note used to say `chembl_with_ames_predictions.csv` does not exist here. **That is
> no longer safe to assert**: the PyTorch pretraining port points its `DATA_PATH` at exactly
> that filename. Whether both files exist, or one was renamed back, is unverified — check before
> assuming which one stage 1 trained on.

## Keeping this current

Files that go stale, in order of how fast:

| Where | Why it moves |
|---|---|
| **`05` (all of it)** | **Fastest-moving file.** Every run, every diagnostic read, every blocker closed |
| `01` §5 (state of the repository) and §4.2 (open questions) | Every time a pipeline stage completes or a design decision is made |
| `03` §11.1–11.2 (which gaps are closed, what runs showed) | Whenever an F-item or a paper gap is addressed |
| `04` §8.1 and §9 (schedules, the three implementations) | Whenever hyperparameters change or an implementation is added or retired |

When a blocker in `05` §4 is settled, move the conclusion into `01` or `04` and leave `05` with
the evidence and the date. `05` is the record of *how we know*; `01`–`04` are the record of
*what is true*.

The paper summaries in `02` and `03` §1–§10 and the architecture description in `04` §1–§7
describe fixed published work and should only change if an error is found.

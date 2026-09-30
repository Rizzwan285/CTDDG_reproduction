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
| [04_model_architecture.md](04_model_architecture.md) | The shared architecture in full, all equations, the complete CDGCN↔CTDDG diff, hyperparameters per phase, **the two coexisting pretraining implementations here**, extension points | Writing or modifying model/training code |

**Minimum useful context:** `01` + `04`.
**For anything touching the objective:** add `03`.
**For anything touching fine-tuning or conditioning:** add `02`.

## Conventions

- **TBD** marks a genuinely open design decision. Do not treat it as settled.
- **⚠ CODE** marks a place where an implementation diverges from the papers.
- **⚠** on its own marks a trap or a defect to fix.
- Equations use the papers' own numbering. Where the two papers number the same equation
  differently, both numbers are given.
- Repository facts (file counts, row counts, metrics, checkpoint state) were verified against
  the working tree on **2026-09-26** and are dated where they are likely to move.

## Four things that most often trip up a fresh session

1. **`code/pretraining copy.ipynb` is the current pretraining code**, despite the name.
   `code/pretraining.ipynb` and `code/pretraining.py` are the untouched upstream Colab export,
   kept as the audit baseline.
2. **`chembl_final.txt` is not the labelled dataset.** It is `chembl.txt` with a dummy `" 1"`
   appended to every line — all 1,090,529 molecules carry the same label. The real Ames labels
   are in `datasets/data/chembl/chembl_toxicity_labels.csv`.
3. **Nothing has been trained yet under the CTDDG objective.** `checkpoints/log.out` contains
   only its header row.
4. **`datasets/data/bindingdb/needle_outputs/` holds ~5.8 million small files.** Exclude it
   from any recursive search, listing or archive.

## Relationship to the other documentation

| Location | Audience | Content |
|---|---|---|
| `context/` (here) | **LLMs** — background, scientific grounding, current state | These five files |
| `code/PRETRAINING_REVIEW.md` | Anyone touching the pretraining code | Line-by-line audit of the upstream notebook against the paper; the numbered change list **F1–F13**, with which are applied tracked in `01` §5.3 |
| `../CTDDG_new/PROJECT_DOCS/` | Humans — long-form, audit-grade | 13 files written against the *earlier* working copy: file-by-file codebase map, data statistics, checkpoint semantics, cluster/Slurm instructions, modification guide |
| `../CTDDG_new/.claude/` | Coding sessions in that repo | Compact operational state for `CTDDG_new` |

`context/` explains **what the project is, why, and where it currently stands in this repo**.
`code/PRETRAINING_REVIEW.md` explains **what is wrong with the pretraining code and in what
order to fix it**.

> ⚠ `PROJECT_DOCS/` and `.claude/` live in the **sibling** repository `../CTDDG_new` and
> describe *that* tree — a different layout (`scripts/`, `cluster/`, `notebooks/`,
> `outputs/pretrain/`) and an earlier state of the work. Where they disagree with these files
> about paths, filenames or progress, **these files are the ones describing this repository.**
> Notably, `PROJECT_DOCS` and the older context refer to a ProtBert-BFD embedding path and to
> `chembl_with_ames_predictions.csv`; neither exists here — this repo uses ProSE/Bepler and
> `chembl_toxicity_labels.csv`.

## Keeping this current

Files that go stale, in order of how fast:

| Where | Why it moves |
|---|---|
| `01` §5 (state of the repository) and §4.2 (open questions) | Every time a pipeline stage completes or a design decision is made |
| `03` §11.1 (which gaps are closed) | Whenever an F-item or a paper gap is addressed |
| `04` §8.1 and §9 (schedules, the two implementations) | Whenever hyperparameters change or the rewrite is restructured |

The paper summaries in `02` and `03` §1–§10 and the architecture description in `04` §1–§7
describe fixed published work and should only change if an error is found.

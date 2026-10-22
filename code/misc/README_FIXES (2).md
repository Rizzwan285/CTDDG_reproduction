# CTDDG reproduction — repaired notebooks

Six notebooks, an offline test suite and a cross-notebook consistency audit.
`AUDIT.md` is the finding-by-finding record; this file is how to run things.

## Order

| # | Notebook | Arm switch | Cost when armed |
|---|---|---|---|
| 0b | `00_phase0_diagnostics.ipynb` | — (read-only) | seconds to minutes |
| 0 | `random_forest_classifier.ipynb` | `WRITE_ARTIFACTS`, `BUILD_PRETRAIN_DATASET`, `RUN_INDEPENDENT_EVALUATOR` | ~20 min + 1.09M predictions |
| 1 | `pretraining_torch.ipynb` | `RUN_TRAINING = False` | days on one GPU |
| 2 | `02_finetune_ctddg.ipynb` | `RUN_TRAINING = False` | hours |
| 3 | `03_generate_samples.ipynb` | `SMOKE_FIRST = True` | hours |
| 4 | `04_evaluate_metrics.ipynb` | — | minutes |

**Every expensive notebook ships disarmed.** Run it top to bottom as it is: it
does the preflight, prints a check summary and stops. Only flip the arm switch
once the checks read the way you expect.

**Nothing on disk is replaced unless you say so.** `random_forest_classifier`
needs `WRITE_ARTIFACTS = True` to write anything at all, and
`OVERWRITE_ARTIFACTS = True` on top of that to replace a file that already
exists. Two flags rather than one because the failure modes differ: replacing
`ames_random_forest.pkl` silently redefines `Toxic_class` for Eq. 8 in *both*
training phases and invalidates every cached label downstream, so it deserves its
own deliberate act. A refused write says what it would have invalidated, and each
run ends with a ledger of what it wrote, skipped or refused.

Its default run is a preflight, not a no-op: it loads the Hansen sets, reports
structural duplicates, fits the two split diagnostics (~7,000 molecules, writes
nothing), and verifies the artifacts already on disk — the forest against the
hash in its own metadata, the pretraining CSV against its manifest. That last
check is the one stage 1 gates on, so it is cheaper to fail here.

## Checks

Each notebook records checks as it goes and calls `gate(...)` before anything
expensive. A `PASS`/`WARN` prints and continues; a `FAIL` raises at the next gate
with the reason. Warnings are for things that make a number *bad*; failures are
for things that make it *wrong*.

Set `LIMIT_PAIRS` (notebook 2) or keep `SMOKE_FIRST` (notebook 3) for a pipeline
rehearsal. Artifacts from those runs are tagged `smoke_test` and every downstream
notebook refuses to treat them as results.

## Before any long job

    ./run_tests.sh

About a minute, CPU only, no repo data and no GPU. It runs:

* `99_consistency_audit.py` — the definitions duplicated across notebooks must be
  byte-identical, and the config values must agree. 25 definitions, 7 values.
  Two differences are exempted in `KNOWN_BENIGN`, each with a reason, and a stale
  exemption that no longer matches anything is reported so the list cannot
  accumulate excuses.
* `test_00_selftests.py` — the traversal, both policies, against the molecules the
  old one corrupted.
* `test_02.py` — embedding shape guards, HDF5 key order, the Ames cache
  fingerprint, the protein-disjoint validation split, Eq. 8 against its closed
  form, the FNN_0 warm start, the step-0 size control.
* `test_03.py` — `Policy` serving training and sampling from one code path, where
  the two normalisations diverge, and an end-to-end decode with random weights.
* `test_04.py` — the markdown fallback, `nanmean`, the label-cache fingerprint,
  the shared featurisers.
* `test_05_review_fixes.py` — `single_reorder`'s input validation, the
  behavioural label-cache key against pickle-byte noise, the RF notebook's write
  guard, and all three evidence scripts executed from an unrelated directory.

Run it after editing any notebook. A notebook that loads a checkpoint cleanly and
then computes something different is the hardest class of bug to find afterwards,
and the audit exists to catch exactly that.

## Editing

The `.py` files are the source; the `.ipynb` files are generated.

    python3 _recell.py <file>.py        # normalise cell structure (idempotent)
    python3 _to_ipynb.py <file>.py      # -> .ipynb, refuses to lose a line

`_repair_markers.py` restores a `# %%` wherever a markdown block runs straight
into code — without it the markdown cell absorbs that code and the notebook
silently loses whole functions while still parsing.

## Evidence scripts

Standalone, CPU, re-runnable from any directory; each prints the source file it
read, so the evidence cannot silently describe a different file.

    python3 measure_traversal.py          > traversal_evidence.txt
    python3 measure_softmax_mechanism.py  > softmax_mechanism_evidence.txt
    python3 verify_graph_pipeline.py      > graph_pipeline_evidence.txt

They load the real functions out of the notebook sources through `_nbsource.py`
rather than keeping copies. A copy can pass its own test while the shipped code
differs, which is the whole failure mode the consistency audit exists to catch.

## Paths

`CTDDG_REPO_ROOT` (default `/home/muhamed/repo/CTDDG`), `CTDDG_FOLD` (`d1`),
`CTDDG_GPU`. Notebook 3 additionally reads `CTDDG_N_SAMPLES`, `CTDDG_N_RUNS`,
`CTDDG_LIMIT_PROTEINS`, `CTDDG_MAX_DECODE_STEPS`.

## Before the next GPU run

In order, cheapest first:

1. `./run_tests.sh` — about a minute. Must be clean.
2. `random_forest_classifier.ipynb` with the defaults. Confirm the forest matches
   its metadata hash and the pretraining CSV matches its manifest. **Do not** set
   `WRITE_ARTIFACTS` unless you intend to rebuild the labels.
3. `00_phase0_diagnostics.ipynb`. Read check A and check F, and note that the
   existing checkpoint is void regardless of what they say (A21).
4. `pretraining_torch.ipynb` with `RUN_TRAINING = False`. Read the preflight,
   especially the dataset hash gate and the held-out slice size.
5. Only then set `RUN_TRAINING = True`.

Stage 1 must be re-run: the existing checkpoint was trained under the broken
traversal. Budget for it rather than trying to fine-tune out of it.

## Start here

`AUDIT.md` §3, finding **A21**. The existing checkpoint was trained under a
traversal that left atoms unvisited on 21% of traversals and mis-wired every bond
of the molecules affected. It cannot be repaired by fine-tuning, so stage 1 has to
be re-run before any number from stages 2–4 means anything.

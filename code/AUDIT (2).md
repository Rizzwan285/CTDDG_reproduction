# CTDDG reproduction — independent audit

Scope and date: a read of eight source files supplied in full
(`00_phase0_diagnostics.py`, `random_forest_classifier.py`, `pretraining_torch.py`,
`02_finetune_ctddg.py`, `03_generate_samples.py`, `04_evaluate_metrics.py`,
`99_consistency_audit.py`, plus the five `context/` documents).

## 0. What this audit could and could not check

**Not checked, because the repository was not available to me:** git status, file
hashes, file timestamps, row counts of any ChEMBL file, the 20,000-row RF agreement
figure, the checkpoint's iteration counters, the reported logit range
(−35288.7 … 1496.6, 7.94% at the clamp), the reported validity figures (63.2%
unconditional / 23.3% conditional), and the reported multi-fragment counts.

Every one of those is a *measurement*, and none of them is restated as fact below.
Where a finding depends on one, it is marked **[needs measurement]** and the exact
command that settles it is given.

**Checked:** the source of all eight files, read end to end. Everything marked
*confirmed* below is a defect visible in the code itself and demonstrable without a
GPU; the accompanying test in `test_ctddg_fixes.py` demonstrates it.

A note on the brief's framing: it asks for a code-first investigation and says not
to accept the prior analysis. Several items in that analysis turned out to be
**already fixed** in the files as supplied, and one — the `rebuilt` check — is a
real bug in a *diagnostic*, which means it was reporting a failure that may never
have happened. Those matter as much as the new bugs, because they are what sends
you to re-run a 480,000-iteration job you may not need.

---

## 1. Audit table

Severity: **S1** scientific validity / leakage · **S2** objective and labels ·
**S3** graph and data correctness · **S4** numerical stability · **S5** generation
and evaluation · **S6** reproducibility and provenance.

| # | Issue | Status | Location | Evidence | Impact | Fix |
|---|---|---|---|---|---|---|
| **A1** | `load_bindingdb_ligands()` locates ligand files by a hardcoded name `ligands_{split}_4_org_corrected` and **silently skips any fold directory where that name does not exist** (`if not os.path.isfile(p): continue`). `04_evaluate_metrics.py` and `02_finetune_ctddg.py` glob `ligands_*_corrected` instead, and `context/05` quotes a real filename of `ligands_4_org_1042_104_corrected`. At least two naming conventions are in play. | **Confirmed bug** (S1) | `random_forest_classifier.py`, `load_bindingdb_ligands`, the `continue` on a missing path | Three files, three different ways of naming the same file. The only guard is `check("BindingDB ligands loaded", len(bdb_canon) > 0)`, which passes if **one** fold out of five is found. | **This is the single most serious finding.** The de-leakage filter can run on a partial ligand set and still write `"bindingdb_ligands_removed": true` into the manifest. Stage 1 then refuses nothing, and every downstream check — including `04`'s `check("stage-1 leakage", ...)` — reports clean. A silent partial filter is worse than no filter, because it manufactures evidence of its own success. | Glob `ligands_*_corrected` per fold directory; require every *existing* fold directory to yield at least one file; record per-directory file paths and line counts in the manifest; fail loudly, not silently. | 
| **A2** | `scaffold_split()` assigns the **largest** Murcko scaffold groups to the **test** side. `ordered` is sorted by descending group size and the first groups encountered fill `test_idx` (because `len(idxs) <= n_test_target // 4` ≈ 354 is true for nearly every group). | **Confirmed bug** (S1) | `random_forest_classifier.py`, `scaffold_split`, the `for _, idxs in ordered:` loop | Standard practice (DeepChem `ScaffoldSplitter`) is the opposite: largest groups fill **train** first, remainder to test. | The "honest number" you intend to quote in the thesis is computed under a non-standard and unusually adversarial split — test is a handful of large analogue series entirely absent from train, train is mostly singletons. It is not comparable to any published scaffold-split figure, in either direction. | Fill train first from the largest groups, remainder to test. Return the scaffold map so the disjointness check reuses it instead of recomputing (which can raise outside the `try`). |
| **A3** | The joint action softmax exponentiates raw logits after a ±80 clamp, then takes `log(p + 1e-10)` in the likelihood. | **Confirmed design defect, mechanism measured** (S4, and the probable S2 root cause) | `Policy.forward` and `MoleculeGenerator._likelihood` in `pretraining_torch.py`, `02_finetune_ctddg.py`, `03_generate_samples.py`, `00_phase0_diagnostics.py` (four copies) | **Measured** (`measure_softmax_mechanism.py`, CPU, seconds): `torch.clamp` passes **exactly zero** gradient outside ±80 — 1267 of 1267 saturated logits had `grad == 0`. `log(p + 1e-10)` floors each action's log-probability at −23.026 and drives `d log(p+ε)/d logit = p(1−p)/(p+ε)` to **zero** as `p → 0`: at logit scale 20 and above the corrective signal for a suppressed action is `0`, while the exact log-softmax holds it at `1`. | **Both halves are gradient *sinks*, not gradient bombs — this corrects the explanation in `pretraining_torch.py`'s own header, which attributes the collapse to ~1e10 gradients and clip saturation. That is not what the arithmetic does.** Once the toxic branch pushes an action below `p ≈ 1e-10`, the model receives no signal to move it in *either* direction and the reported `l` stops responding. That is why stage-1 `l_tox` plateaued near −637: 28 × −23.026 = −644.7, i.e. the floor was binding on ~28 of ~34 actions. The non-toxic branch could not recover the mass it had lost. | Exact max-shifted joint log-softmax (`action_log_probs`), feeding **log-probabilities** into `_likelihood` so the `+1e-10` disappears. Verified: agrees with the legacy path to 4.8e-06 on ordinary logits, stays finite and normalised to 0.0 error across the full reported range [−35288, 1496], and keeps `|d log p / d logit| = 1` on exactly the actions the legacy path freezes. Legacy retained behind `STABLE_SOFTMAX = False`. **This improves fidelity to Eq. 1–3; it is not a deviation.** |
| **A4** | Step-0 per-action normalisation pairs unrelated molecules. `s0_l` is produced by iterating `loader_val`, whose `BalancedSampler` **sorts by cost and drops the tail**; `act` is built in `records_val` order. `per_action = s0_l[:n_v] / act[:n_v]` therefore divides molecule *i*'s likelihood by a different molecule's action count. | **Confirmed bug** (S2) | `02_finetune_ctddg.py`, the `MEASURE_STEP0` cell, `act = np.array([...for r in records_val])` | `BalancedSampler.__init__` sorts `index = np.argsort(cost)` and chunks; `_g()` shuffles within chunks. Order is unrelated to `records_val`. | The check that decides whether the toxic/non-toxic gap is *suppression* or a *molecule-size confound* — blocker B4's entire basis, and the only thing that can justify a `TOX_FLOOR` — is computing a meaningless ratio. Its verdict is currently worthless in both directions. | Compute the action count inside `_likelihood` from `torch.bincount(iw_ids)`, which is exactly `GetNumBonds() + 1` per molecule by construction, expose it as `last_n_actions`, and collect it alongside `l` in `evaluate(collect=True)`. Alignment then cannot drift. |
| **A5** | Check H infers "the forest was rebuilt" from the mere **existence** of `models/ames_random_forest.meta.json`. | **Confirmed bug, and the brief was right to flag it** (S6) | `00_phase0_diagnostics.py`, check H, `rebuilt = os.path.isfile(RF_META_PATH)` | `random_forest_classifier.py` §4 **always** writes that file (`with open(RF_META_PATH, "w")`). Its presence is the normal state. Separately, `02_finetune_ctddg.py` has `RF_AUTO_TRAIN = False` *and* never references the flag anywhere — `ames_labels()` raises `FileNotFoundError` instead. There is no auto-rebuild path in the code as supplied. | A hard `FAIL` ("two definitions of toxic in one objective") that fires on a healthy repository, pointing at a filename that does not exist (`02_random_forest_classifier.py`). This is how a diagnostic sends you to redo work that was fine. | Compare hashes, not file existence: `rf_meta["sha256"]` recorded at training time vs. the file on disk now, vs. the BindingDB cache's recorded hash, vs. the pretraining manifest's `labeller.sha256`. Three agreeing hashes is the real test; anything else is `UNRESOLVED`, not `FAIL`. |
| **A6** | Check F inspects `chembl_with_ames_predictions.csv` as "the file the pretraining port points at by default". `pretraining_torch.py` now defaults `DATA_PATH` to `chembl_pretrain_labelled.csv`. | **Confirmed bug** (S6) | `00_phase0_diagnostics.py`, `chembl_candidates`, `_trained_on` | The two files disagree about which file stage 1 reads. Check F also compares against hardcoded `N_CHEMBL_FILTERED = 1_090_529` / `N_CHEMBL_RAW = 1_207_359`, neither of which can match the new deduplicated-on-canonical-SMILES set. | Check F will report `SKIP` or a spurious `WARN` on a correctly built dataset, and it answers the wrong question anyway: the default path tells you what the *next* run will read, not what the *existing* checkpoint read. | Read the `data` signature **recorded inside `ckpt.pt`** (`data_path`, `data_sha256`, `n_molecules`) — `pretraining_torch.save_checkpoint` now writes it — and compare against the manifest. If the checkpoint predates that field, report `UNRESOLVED` explicitly. That is the honest answer, and it is the one the brief asks for. |
| **A7** | `strip_salt` picks the longest fragment **by string length**. | **Confirmed, and the brief's concern is correct** (S3) | `pretraining_torch.py`, `02_finetune_ctddg.py`, `00_phase0_diagnostics.py`, and inline in `random_forest_classifier.py` / `04_evaluate_metrics.py` (five copies, two spellings) | Counterexample, no measurement needed: `Cc1ccccc1.O=S(=O)(O)O` → string lengths 9 vs 12, so the rule returns **sulfuric acid** and discards the toluene. Ties also resolve to whichever fragment RDKit happened to write first. | Wrong parent fragment becomes the training target, and non-determinism on ties means the same input can yield different targets across runs. Small in count, but it is a correctness bug in the data, not a style issue. | One shared `strip_salt` with `FRAGMENT_POLICY = 'heavy_atoms'`: most heavy atoms, ties broken by string length then lexicographically. `'longest_string'` retained to reproduce `data_preprocessing_1`. The dataset builder now **counts and reports** how many molecules the two rules disagree on — the brief's "quantify how many examples would be affected". |
| **A8** | The check that salts were stripped is a tautology: `stats_train["salt_stripped"] + stats_test["salt_stripped"] >= 0`. | **Confirmed bug** (S3) | `02_finetune_ctddg.py`, `check("salts were stripped", ...)` | A sum of two non-negative counters is always `>= 0`. It is a print statement wearing a check's clothes, and it will report `PASS` with stripping disabled. | One of the checks you would rely on to confirm the fix landed cannot fail. | Replace with the real invariant: no kept SMILES contains `.` after assembly. |
| **A9** | `UNMATCHED_TOLERANCE = 1.0`, compared with `if frac > UNMATCHED_TOLERANCE`. | **Confirmed dead guard** (S3) | `02_finetune_ctddg.py`, `assemble_pairs` | A fraction cannot exceed 1.0, so the `raise` is unreachable. | A guard that cannot fire is worse than none: it reads as protection during review. | Per-split tolerances (pooling means the train split drops few rows and the test split drops most), both reachable. |
| **A10** | Gradient finiteness is checked **only on summary steps** in stage 1, every step in stage 2. | **Confirmed inconsistency** (S4) | `pretraining_torch.py`, `if is_summary:` → `if DIE_ON_NONFINITE and not math.isfinite(gn)` | `02_finetune_ctddg.py` does the same check unconditionally and its own comment explains why: "Clamping a NaN returns NaN, so it would have passed straight into `optimizer.step()`". | 499 out of every 500 stage-1 iterations can admit a NaN gradient into the optimizer with a finite loss. On a 480,000-iteration run that is the difference between a 20-minute loss and a silently poisoned checkpoint. | Check every step, as stage 2 does. |
| **A11** | Stage 1 writes checkpoints with a plain `torch.save`. | **Confirmed robustness bug** (S6) | `pretraining_torch.py`, `save_checkpoint` | `02_finetune_ctddg.py` has `atomic_save` (tmp → fsync → rotate → rename) and explains that "a plain `torch.save` leaves a truncated file if the job is killed mid-write". Stage 1 writes a 28 MB file every 500 steps for days on a SLURM node. | A SLURM `SIGTERM` landing inside one of those writes destroys the only artifact of a multi-day run. | Reuse stage 2's `atomic_save` plus a `.prev` rotation, and add the `SIGTERM`/`SIGINT` handler stage 2 already has. |
| **A12** | `_resolve_columns` falls back to `cols[0]` when no SMILES-like column is found. | **Confirmed bug** (S6) | `pretraining_torch.py`, `_resolve_columns` | The toxicity branch raises on failure; the SMILES branch silently takes column 0. | Training on an ID column, with no error, for as long as it takes you to notice. The file's own header documents the sibling bug this replaced (`'label' in c.lower()` matching ChEMBL's `BAO Label`) — same class, opposite branch. | Raise, naming the columns it saw. |
| **A13** | `report.md` is built with `TABLE1[_cols].to_markdown(...)`, which requires `tabulate`. | **Confirmed fragility** (S5) | `04_evaluate_metrics.py`, the `report` list | `to_markdown` raises `ImportError` if `tabulate` is absent. It runs at the very end, after descriptors, two labelling passes and novelty canonicalisation. | Losing the whole report to a missing optional dependency after the expensive part has run. | `try/except` → `to_string`. |
| **A14** | `np.mean(per_run_tox)` over a run whose proteins all scored `NaN` propagates `NaN` into the headline toxicity figure. | **Confirmed** (S5) | `04_evaluate_metrics.py`, `toxicity_table` | `pct_toxic` is `float('nan')` when no molecule in a cell was labelled; `groupby.mean()` skips NaN per group but a fully-NaN group yields NaN, and `np.mean` then propagates it. | One empty run turns the entire Table 3 column into `nan`. | `np.nanmean` / `np.nanstd`, and report how many runs contributed. |
| **A15** | `_train_frac` is assigned `None`, then conditionally reassigned `None`. | **Confirmed dead code** (S5) | `04_evaluate_metrics.py`, the toxicity confound block | Literally `_train_frac = None; if SPLIT_MAN: _train_frac = None`. | Harmless, but it marks a missing comparison: the generated toxic fraction has no baseline to sit beside. | `02` now records train/val/test toxic fractions in `split_manifest.json`; `04` prints the generated fraction against the training fraction, which is the number that says whether suppression did anything at all. |
| **A16** | `99_consistency_audit.py`'s two `KNOWN_BENIGN` entries are unreachable: `get_d` and `independent_features` are not in `MUST_MATCH`. | **Confirmed dead config** (S6) | `99_consistency_audit.py` | `benign = [k for k in KNOWN_BENIGN if k[0] == name]` only runs for names in `MUST_MATCH`. | The audit documents two differences it never actually inspects, and its scope misses the functions that matter most — the ones this patch set changes. | Add `get_d`, `independent_features`, `morgan_matrix`, `process_single`, `get_graph_from_smiles` as live benign entries, and add the new shared numerics (`segment_max`, `action_log_probs`, `morgan_bits`, `independent_feature_row`, `_TwoLayerDense`) plus the constants `STABLE_SOFTMAX` and `FRAGMENT_POLICY` to `MUST_MATCH`. A constant that drifts between files is exactly as dangerous as a function that does. |
| **A17** | `dense_policy_0` is created fresh at fine-tuning and the pretrained `policy_0` is multiplied by zero. | **Confirmed as designed; improvable** (S5) | `02_finetune_ctddg.py`, `CMoleculeGenerator_RNN._policy_0` | CDGCN §2.2 does specify replacing the learnable vector with `FNN_0(c)`, so the substitution is faithful. The paper does not specify FNN_0's **initialisation**. | The first-atom distribution restarts near-uniform over 65 atom types while `loss_init` is one term in ~34 per molecule, so molecules begin on exotic or charged atoms for a long time. | Initialise `dense_policy_0.output.bias` to `log_softmax(pretrained policy_0)` and zero that layer's weight, so at step 0 the conditional first-atom distribution **is** the pretrained marginal and protein-dependence is learned from there. Behind `INIT_POLICY0_BIAS_FROM_PRETRAINED`, documented as a deliberate departure from Xavier. |
| **A18** | `sanity_check`'s conditioning test uses a single random permutation of eight embeddings. | **Confirmed weak test** (S5) | `02_finetune_ctddg.py`, `sanity_check` | With batch 8, a permutation can be near-identity, and a single draw gives a one-bit verdict on the check that decides whether Table 2 means anything. | A pass or fail here is mostly noise. | Average over several permutations and report the fraction of shuffles the true pairing beats. Same cost, a real signal. |
| **A19** | Check E1 reimplements the old traversal and records `FAIL` when it reproduces the old bug. | **Confirmed mislabelled** (S6) | `00_phase0_diagnostics.py`, E1 | Both training files now reject disconnected graphs in `get_graph_from_smiles`, so E1 is demonstrating history, not current state — and it reports that demonstration as a failure of the current code. It also calls `random` unseeded, so it is non-deterministic. | A permanent red `FAIL` that cannot be cleared trains you to ignore the summary table. | Keep the demonstration as `INFO`, seed it, and add a real assertion: the **current** `get_graph_from_smiles` must raise on the same input. That one can pass, and it is the thing you actually care about. |
| **A20** | Checks E2 and F2 locate ligand files by the same hardcoded name as A1. | **Confirmed** (S1) | `00_phase0_diagnostics.py`, E2 and F2 | Same `ligands_{split}_4_org_corrected` literal. | The leakage check — the one result the brief calls decisive — can silently examine nothing and report `SKIP`, or examine one fold and report `PASS`. | Glob, and report which files were read. |
| **A21** | `traverse_graph` abandons a branch point whose first branch is a dead end. After recursing into a neighbour it recomputes the candidate set from **that neighbour**, not from `current_node`, so the branch point's remaining neighbours are never visited. | **Confirmed bug, measured. The most serious finding in this audit.** (S3) | `traverse_graph` in `pretraining_torch.py` and `02_finetune_ctddg.py`; the demo copy in `00_phase0_diagnostics.py` | **Measured** (`measure_traversal.py`): on 20 drug-like molecules x 100 traversals, **419/2000 traversals (21%) leave atoms unvisited, and 13 of the 20 molecules are affected.** `CC(C)(C)c1ccccc1` (tert-butylbenzene), `Cc1ccc(cc1)S(=O)(=O)N`, `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` and `CC(C)(N)Cc1ccccc1` fail on **100/100**. Minimal reproducer: a 4-atom star graph fails 388/500. | **This is not a salt problem — it hits ordinary connected molecules**, any time a branch point's first branch is a dead end: *gem*-dimethyl, *tert*-butyl, N,N-dimethyl, sulfonamide. Upstream, those molecules reached `single_reorder` with `-1` entries in `step_ids`, which argsorts them to the front and remaps every bond endpoint through the raw ids — so **every bond in the training target is mis-wired, silently and uncounted**. A large, structured share of ChEMBL and BindingDB was trained on corrupted targets. This is a more plausible primary cause of the validity collapse than the Eq. 8 story, and it is independent of it. | Recompute the candidate set from `current_node` after each branch returns. Verified: 2400/2400 traversals now yield a complete permutation with `log_p <= 0`, and `single_reorder` round-trips 240/240 back to the original molecule (`verify_graph_pipeline.py`). `TRAVERSAL_POLICY = 'upstream_broken'` retains the old behaviour for A/B. **This changes Q_alpha**, so it is recorded in the checkpoint signature and a resume across the change is refused. |
| **B1** | "ChEMBL pretraining used plain NLL because `tox_class` was discarded." | **Already fixed / superseded** | `pretraining_torch.py`, `_likelihood` | `tox_class` is parsed, carried end to end, and consumed. Stage 1 now uses plain NLL **by deliberate choice** (`TOX_WEIGHTING = False`), with the reasoning in the file header, not by dropping a variable. `eff_tox` vs `real_tox` keeps the diagnostics honest under either setting. | None. | None. Do not "fix" this — re-enabling Eq. 8 in stage 1 is what the header says collapsed validity. |
| **B2** | "`RF_AUTO_TRAIN = True` can silently rebuild the forest." | **Already fixed** | `02_finetune_ctddg.py` | `RF_AUTO_TRAIN = False`, and the flag is referenced **nowhere** else in the file. `ames_labels()` raises with a pointer to `random_forest_classifier.py`. | None — but a flag that does nothing is a trap for the next reader. | Delete the flag, keep the comment. |
| **B3** | "`λ = 0` recovers CDGCN." | **False, and correctly handled everywhere** | all files | `1 − (1+0)·tox` = 0 on toxic ligands → dropped from the batch average. Every file that mentions it says so. | None. | None. |
| **B4** | Eq. 8 implementation, the `[:, 0]` reshape, the `batch_size × K` guard. | **Verified correct by reading** | `_likelihood` in both training files | `tox_class` is `np.full((k,), label)` per molecule, concatenated in molecule order; `iw_ids` runs `mol*K + path`; the reshape and column-0 take are right, and the `numel()` guard raises on misalignment. | None. | None. |
| **B5** | "Stage-1 checkpoint has 480,000 iterations; counters agree." | **[needs measurement]** | — | Cannot be checked without the file. `max(num_batches_tracked)` is the exact count and the reasoning in `context/05` §4 B1 for *why* it is exact (one train-mode forward per step, `dense` computed once) is sound. | — | Run patched check G. |
| **B6** | Logit range −35288 … 1496, 7.94% clamped. | **Range [needs measurement]; consequence now measured** | — | The range itself still needs the repo. The *consequence* is measured (A3): at that range the legacy path freezes every saturated logit at zero gradient and floors the log-probabilities. The brief's §E suspicion of "flat or zero gradients" was right, and my first reading of it as exploding gradients was wrong. | — | Run patched check A/D. `STABLE_SOFTMAX` makes the range harmless: verified normalised with 0.0 error at [−35288, 1496]. |
| **B7** | ChEMBL CSV timestamp predates the RF metadata timestamp. | **Unresolvable from code** | — | Nothing in the source settles it, and 100% prediction agreement cannot: a retrained forest with the same seed, data and hyperparameters reproduces its predecessor's predictions exactly *by design* (`random_state=42`, unchanged merge/dedup order — the file says so itself). | Provenance of the existing ChEMBL labels is **unresolved** and should be reported that way in the thesis. | Not fixable retrospectively. The manifest + SHA-256 the current code writes makes it resolvable *going forward*; that is the most that can be claimed. |
| **B8** | "99_consistency_audit passed with 11 identical, 0 drift." | **Plausible but out of scope** | `99_consistency_audit.py` | The 11 names do appear to match across the files as supplied. But the audit does not cover `Policy`, `_likelihood`, `get_graph_from_smiles`, `process_single`, `morgan_matrix` or any constant — i.e. nearly everything this patch set touches. | A green audit was never evidence about the code that was broken. | A16. |

### Correction to this audit

An earlier draft of A3 said the `log(p + 1e-10)` term produces ~1e10 gradients.
**It does not**, and neither does the clamp. `measure_softmax_mechanism.py` shows
both are gradient *sinks*: the clamp zeroes the gradient outside ±80, and
`p(1−p)/(p+ε) → 0` as `p → 0`. The practical damage is the same but the mechanism
is the opposite, so any claim about "clip saturation letting the toxic branch own
the gradient sign" is unsupported and has been removed from the notebooks too. The
measurement is in `softmax_mechanism_evidence.txt`; re-run it with
`python3 measure_softmax_mechanism.py`.

### Findings from building the repair (post-audit)

Four things turned up while writing and testing the final notebooks, after the
audit table above was written. They are listed here rather than folded into the
table so the record of what was found by reading stays separate from what was
found by running.

**A22 — `MAX_DECODE_STEPS = 100` is below the training action budget.**
`03`'s decode loop ran at most 100 actions while `MAX_ITER = 120` admitted
molecules needing up to 120. A molecule inside the training distribution could
therefore not be finished by the sampler: it hit the cap, never emitted `end`, and
was counted as invalid. That is a manufactured invalidity with nothing to do with
the model. The default is now 120 and a check refuses a cap below the training
budget. *How much of the validity collapse this explains is unmeasured* — the
smoke pass reports what fraction of invalid molecules sit near the cap, which is
the number that settles it.

**A23 — the sampler normalised the action distribution differently from training.**
`03` carried its own `Policy` using the legacy `exp`-and-divide path while `02`
used the max-shifted log-space one. These agree in exact arithmetic and diverge
once the ±80 clamp binds — measured at up to **0.997 absolute difference in a
single action probability** on the same logits (`test_03.py`). Both still sum to
1.0 per molecule, so nothing downstream could have noticed. `03` now shares
`Policy` byte-identically and reads `STABLE_SOFTMAX` *from the checkpoint* rather
than its own config, with a parity cell that proves `exp(log p) == p` and compares
the two paths before anything is generated.

**A24 — `_TwoLayerDense` applied a softmax inside `forward` in `03` but not `02`.**
Stage 2 takes a `log_softmax` of FNN_0's output; the sampler took a `softmax` of a
module that had already softmaxed. The weights load cleanly either way. Both files
now return logits and normalise at the call site, and the consistency audit
enforces it.

**A25 — the step-0 "size confound" check understated real suppression.**
My first version divided each molecule's bound by its own action count and
multiplied the group difference back by the *pooled* mean count. On synthetic data
with a known 20-nat suppression and toxic molecules 30 actions larger, that
recovers only 14.5 of the 20 nats — a genuine effect reads as "mostly a size
confound". It is replaced by a least-squares control, `l ~ a·actions + b + c·toxic`,
whose two components sum exactly to the observed gap; it recovers 19.9 of the
20 nats and reports 0.1 on a pure confound (`test_02.py`).

**One correction to the audit's own reasoning, not to the code:** check T7 in `00`
previously asserted that the traversal cannot walk a disconnected graph. With the
A21 fix that is **false** — the fixed traversal's top-level candidate pool is every
atom, so when a component is exhausted it starts a new root and returns a complete
permutation. The guard against salts therefore has to live in
`get_graph_from_smiles`, which is where it is; T7 now states the real behaviour and
T7b checks the guard fires *for disconnection specifically* rather than for any
`ValueError`.

### Findings from the independent review round

A second review raised five leads. Each was checked against the code rather than
accepted; two were confirmed as reported, one was confirmed and turned out worse,
and two were already addressed. The review also surfaced two defects of my own.

**A26 — `random_forest_classifier` replaced the Ames forest with no guard.**
*Reported as "may perform expensive work by default"; the expense was the smaller
half.* Section 4 called `joblib.dump(rf_model, RF_MODEL_PATH)` **unconditionally**
— no flag, no existence check — and then printed a warning that doing so
invalidates every cached label. Running the notebook top to bottom replaced six
artifacts: the labelling forest, the fingerprint config, the forest metadata, the
independent evaluator, the stage-1 pretraining CSV and its manifest. The forest is
the one that writes `Toxic_class` for Eq. 8 in *both* phases, so replacing it
silently redefines the objective; replacing the CSV breaks the hash gate stage 1
resumes from. Fixed: `RUN_RANDOM_SPLIT_FIT` and `RUN_SCAFFOLD_SPLIT` stay on (both
cheap, both write nothing), `RUN_INDEPENDENT_EVALUATOR` and
`BUILD_PRETRAIN_DATASET` are now off, and every artifact write goes through
`may_write()`, which needs `WRITE_ARTIFACTS` to write at all and
`OVERWRITE_ARTIFACTS` to replace a file that exists. A refusal names what the
write would have invalidated, and the run ends with a ledger of what it wrote,
skipped or refused. The default run is now a real preflight: it verifies the forest
against its own recorded hash and the pretraining CSV against its manifest.

**A27 — the label-cache key was machine-dependent.** *My own defect, introduced by
the A-series cache fix.* `rf_fingerprint()` keyed the cache on the SHA-256 of the
pickle. `n_jobs` is an estimator attribute, so it is serialised: the same forest,
with the same `random_state`, trained on boxes with different core counts gives
**3 distinct pickle hashes and identical predictions** (measured —
`test_05_review_fixes.py`). Keying on the bytes therefore relabels every BindingDB
ligand, and in `04` every generated molecule, for a model that has not changed.
Replaced with `model_behaviour_hash()`: a SHA-256 over the labeller's predictions
on a fixed deterministic probe set. One hash for the same boundary under any
`n_jobs`, a different hash when the boundary moves. The pickle hash is still
recorded, as provenance only. `cache_decision` now says *"differs byte for byte but
predicts identically"* and reuses; a pre-v3 cache falls back to byte equality
rather than guessing. The function is enforced byte-identical across `02` and `04`
— if the probe sets differed, each file would reject the other's cache and they
would relabel in a loop.

**A28 — `single_reorder` trusted its caller.** *Reported under "step_ids and bond
endpoints valid before remapping"; confirmed.* `traverse_graph` refuses to return a
bad permutation, but the guard sat in a different function from the one that does
the damage: `single_reorder` argsorts `step_ids`, and a `-1` lands at position 0
and remaps every bond endpoint through a wrong table. It now validates that
`step_ids` is a permutation of `0..n-1` and that no bond endpoint is negative
(which would wrap round the lookup), reusing the argsort it already computes.
Measured cost: 31% of this function's own runtime, about 83 s per 1.09M-molecule
epoch at K=5, inside the dataloader workers.

**A29 — the traversal evidence script measured a copy, not the shipped code.** *My
own defect.* `measure_traversal.py` carried its own `traverse_NEW`, so it could
have reported a clean result while the shipped `traverse_graph` differed — exactly
the drift `99_consistency_audit.py` exists to prevent. It now loads the shipped
function and runs it under both `TRAVERSAL_POLICY` values, and additionally checks
that `'upstream_broken'` reproduces the historical `traverse_OLD` **exactly**:
1000/1000 draws match. Without that, the A21 A/B compared the fix against a story
about the old code rather than the old code itself.

**A30 — evidence scripts hardcoded an absolute path.** *Reported; confirmed.*
`measure_softmax_mechanism.py` and `verify_graph_pipeline.py` opened
`/home/claude/ctddg/pretraining_torch.py`, so they failed anywhere else. Both now
resolve through `_nbsource.py`, which locates sources relative to itself, raises if
a requested definition is absent, and prints the path it read. Tested from an
unrelated working directory.

**Checked and found already correct** (no change made): the step-0 `loss_init`
term is aligned with its own molecule — pinned now by a test that mis-aligns
`action_0` deliberately and confirms the bound collapses from −12.0 to −52.0, so
the test has real power; and no checkpoint write in either training notebook is
reachable at module level, so a preflight run with `RUN_TRAINING = False` cannot
write one.

**One hazard documented rather than changed.** `_likelihood` masks by multiplying a
log-probability by a 0/1 float, and `0 * -inf = nan`. It is not reachable today —
`action_log_probs` returns finite values for finite logits — and the training loop
checks `torch.isfinite(loss)` before every `optimizer.step()`. A test records the
behaviour so that anyone adding real action masking masks the *logits* instead.
Changing the masking now would alter the objective for no measured benefit.

### Findings not in the brief

A1, A2, A4, A8, A9, A10, A11, A12, A13, A14, A15, A16, A18 and **A21** are mine.
**A21 is the one to act on first** — it was found by running the self-tests I wrote
for A7, not by reading, and it invalidates the training targets themselves. A1 and
A4 come next.

---

## 2. Pipeline map

| Stage | Entry point | Key inputs | Key outputs | Config source |
|---|---|---|---|---|
| 0 Labeller + dataset | `random_forest_classifier.py` | `Mutagenicity_N{7090,6512}.csv`, `chembl.csv`, all five folds' `ligands_*_corrected` | `models/ames_random_forest.pkl` + `.meta.json`, `models/ames_independent_gbm.pkl` + `.meta.json`, `models/fingerprint_config.pkl`, `chembl_pretrain_labelled.csv` + `.manifest.json` | module constants, `CTDDG_REPO_ROOT` |
| 0b Diagnostics | `00_phase0_diagnostics.py` | `checkpoints/ckpt.pt`, `trainer.pt`, `log.out`, fold ligand files, RF artifacts | `outputs/diagnostics/phase0_<fold>.json` | `CTDDG_REPO_ROOT`, `CTDDG_FOLD`, `CTDDG_GPU` |
| 1 Pretraining | `pretraining_torch.py` | `chembl_pretrain_labelled.csv` (+ manifest, hash-verified), `atom_types.txt` | `checkpoints/ckpt.pt`, `trainer.pt`, `configs.json`, `log.out` | module constants, `CTDDG_DATA_PATH`, `CTDDG_CHECKPOINT_DIR` |
| 1b Embeddings | `01_generate_prose_embeddings.ipynb` *(not supplied)* | fold FASTA | `<fold>_<tr\|te>_unique_embeddings_bepler.h5` | — |
| 2 Fine-tuning | `02_finetune_ctddg.py` | `ckpt.pt`, fold pair files, `*_bepler.h5`, RF | `outputs/finetune/<fold>/{ckpt_best,ckpt_last,trainer}.pt`, `configs.json`, `log.out`, `val.out`, `test_set.npz`, `split_manifest.json`, `step0_likelihoods.npz` | module constants, `CTDDG_FOLD` |
| 3 Generation | `03_generate_samples.py` | `ckpt_best.pt`, `test_set.npz` | `outputs/samples/<fold>/run*/protein_*.csv`, `manifest.json`, `timing.csv`, `failure_census.csv` | env overrides for every knob |
| 4 Evaluation | `04_evaluate_metrics.py` | samples, both RF/GBM models, manifests, fold ligands | `table1_*.csv`, `table3_toxicity.csv`, `pkcsm_upload_*.smi`, `report.md` | `CTDDG_REPO_ROOT`, `CTDDG_FOLD` |
| 5 Docking | `05_docking_table2.py` | **does not exist in the files supplied** | `pdb_protein_map.json` (consumed by `04`) | — |

**Stale or inconsistent paths found:** `00`'s `chembl_candidates` (A6); the three
ligand-file naming conventions (A1, A20); `05_docking_table2.py` referenced by both
`03` and `04` but absent; the brief's `fixed_scripts/` prefix, which none of the
files assume.

---

## 3. Prioritised repair plan

Ordered as the brief asks.

**0 — Training targets.** **A21 (traversal).** Nothing else matters until this is
fixed: 21% of traversals were building mis-wired molecules, so every other number
in the project was measured on a model trained partly on garbage. It is also the
cheapest to verify — `verify_graph_pipeline.py` runs on CPU in seconds.

**1 — Scientific validity / leakage.** A1 (ligand glob), A2 (scaffold split), A20
(diagnostic globs), A6 (what the checkpoint actually trained on). A1 first: until
the de-leakage filter is known to have seen all five folds, no leakage verdict —
clean or dirty — means anything, and A6 determines whether the *existing*
checkpoint is salvageable at all.

**2 — Objective and labels.** A3 (exact log-softmax and log-probability
likelihood) and A4 (step-0 alignment). A3 changes numerics, not the objective:
Eq. 8's weighting, λ, the floor and the importance-sampling reduction are all
untouched. What changes is that `log P(G|P)` is now computed as the paper defines
it rather than through a clamp-and-epsilon approximation. A4 restores a diagnostic
that is currently computing noise.

**3 — Graph and data correctness.** A7 (fragment policy), A8 (real salt
invariant), A9 (reachable tolerance), A19 (E1 as a real assertion).

**4 — Numerical stability.** A10, A11, A12 — all stage 1, all cheap, all about
not losing a multi-day run.

**5 — Generation and evaluation.** A13, A14, A15, A17, A18.

**6 — Reproducibility and provenance.** A5, A16, plus the manifest additions
(fragment policy, per-fold ligand sources, policy-disagreement counts).

### What changes scientifically, stated plainly

Four changes alter behaviour rather than fixing an outright defect. Each is
toggleable and each is recorded in the checkpoint's signature so a resume across
the change is refused:

1. **`STABLE_SOFTMAX = True`** — exact log-softmax replaces clamp-and-epsilon.
   Closer to the paper, not further. Set `False` to reproduce prior runs bit for
   bit (up to float ordering).
2. **`FRAGMENT_POLICY = 'heavy_atoms'`** — deviates from `data_preprocessing_1`'s
   longest-string rule. The dataset builder reports how many molecules the two
   rules disagree on, so the deviation is quantified rather than assumed small.
3. **`INIT_POLICY0_BIAS_FROM_PRETRAINED = True`** — warm-starts `FNN_0`'s output
   bias from the pretrained first-atom marginal. The paper specifies the
   substitution, not the initialisation.
4. **`TRAVERSAL_POLICY = 'complete_dfs'`** — changes `Q_alpha`, because the old
   traversal did not sample complete paths at all for branched molecules. There is
   no correct prior distribution being preserved here; the fix restores the DFS
   that Li et al. describe. `'upstream_broken'` reproduces the old behaviour.

**Consequence for the existing checkpoint:** it was trained under the broken
traversal, so it cannot be repaired by fine-tuning and should be treated as void
for reproduction purposes, whatever `00`'s check A reports.

Nothing else touches preprocessing semantics, the objective, sampling, the toxicity
labels, or the train/test split.

### Explicitly not done

- No training started, no checkpoint written, no dataset regenerated, no cached
  labels invalidated. Every number in this audit comes from a CPU measurement on
  synthetic or public-SMILES inputs; `./run_tests.sh` reproduces all of them in
  about a minute.
- **No claim that any of this restores the paper's numbers.** The repairs remove
  specific, measured defects. Whether validity recovers is unknown until stage 1
  is re-run under the fixed traversal, and that run has not happened.
- **Now done, and it changes the training set:** stage 1 holds out
  `VAL_MOLECULES = 2000` molecules (seed `VAL_SEED = 1234`) and logs NLL to
  `val.out` every `VAL_EVERY = 5000` iterations. An earlier draft of this audit
  listed it as deliberately deferred; it is in `pretraining_torch.py`, the count
  and seed are recorded in the checkpoint, and `VAL_MOLECULES = 0` restores the
  old behaviour. It is flagged here because a 480,000-iteration run with no
  held-out signal is how the last one went wrong, and because *it removes 2,000
  molecules from training* — a real change to the experiment, not a pure
  instrumentation addition.
- The RF's external validation question (brief §B) is **not solved**. Scaffold
  split with A2 fixed, plus the structural-duplicate leakage count, bounds how
  optimistic the random-split number is. It does not make the RF an independent
  measure of toxicity. The defensible claim remains: *the method reduces
  RF-predicted Ames mutagenicity*, and pkCSM on the five Table-3 PDBs (~1,000
  molecules, which `04` already writes out) is the only route to the paper's own
  metric.

# 05 — Training Runs, Diagnostics and Open Blockers

> **Read this before launching anything expensive.** `01`–`04` describe the project, the papers
> and the architecture. This file records **what has actually been run, what the output showed,
> what that does and does not establish, and what has to be checked before the next run.**
>
> Everything here is dated. Claims are tagged:
> **[CONFIRMED]** — established from code read in full, or from arithmetic on a log we have.
> **[UNVERIFIED]** — plausible but not measured. Do not repeat it as fact.
> **[MISSING]** — the evidence that would settle it, and where to get it.

---

## 1. Pipeline as it stands (2026-10-11)

The stack was ported from MXNet to **PyTorch**. Three notebooks now form the chain:

| Stage | Notebook | Writes | Status |
|---|---|---|---|
| 1 — unconditional pretraining | PyTorch port of the MXNet pretraining notebook (exact filename not recorded here) | `checkpoints/ckpt.pt`, `trainer.pt`, `configs.json`, `log.out` | Ran; **iteration count unverified** (§4, B1) |
| 1b — protein embeddings | `01_generate_prose_embeddings.ipynb` (env `bioemb`) | `<fold>_<tr\|te>_unique_embeddings_bepler.h5` | **Fold d1 done** [CONFIRMED — fine-tuning loaded them and passed `verify_fasta_order`]. Other folds not confirmed; the notebook ships with `FOLDS = ["d1"]` |
| 2 — conditional fine-tuning | `02_finetune_ctddg.ipynb` | `outputs/finetune/<fold>/ckpt_best.pt`, `ckpt_last.pt`, `log.out`, `val.out`, `test_set.npz`, `split_manifest.json` | One run completed on d1 (§2) |

Both stages implement **CTDDG Eq. 8 with `λ = 1.0` and `TOX_FLOOR = None`**. [CONFIRMED by reading
both sources.] So `ckpt.pt` is a **CTDDG-objective** checkpoint, not a CDGCN one — which
`pretrained_provenance()` correctly says it cannot determine *from the file*, but which the
source settles.

Stage 2 labels BindingDB ligands with **the repo's own Ames Random Forest**, not pkCSM. This is
a deliberate divergence from the paper (`01` §4.2 Q6), cached in
`datasets/data/bindingdb/ligand_ames_labels.csv` with a SHA-256 fingerprint of the RF beside it.

---

## 2. Run FT-d1-01 — fine-tuning, fold d1

Log reviewed 2026-10-11. Settings at the time: `BATCH_SIZE = 8`, `K = 5`, `α = 0.8`,
`LAMBDA_TOX = 1.0`, `TOX_FLOOR = None`, `LR = 1e-4`, `DECAY = 0.002/100`, `MAX_EPOCHS = 10`,
`PATIENCE = 10`, `EARLY_STOP_ON = 'nll'`, `VAL_PROTEIN_FRACTION = 0.05`, `LIMIT_PAIRS = None`.

Epoch boundaries fall between logged iterations 14,800/15,000 · 29,600/29,800 ·
44,600/44,800 · 59,400/…, so **≈14,850 iterations per epoch** (the log granularity is 200),
i.e. ≈119,000 training pairs after the protein-level validation carve-out and the sampler's
drop of the final partial batch. Validation toxic fraction **21.6%**, constant across epochs.

### 2.1 Validation, by epoch (from `val.out`)

| epoch | val_weighted | val_nll | **val_l_nontoxic** | val_l_toxic | separation |
|---|---|---|---|---|---|
| 1 | 19.1252 | **108.6283** | **−81.485** | −207.093 | 125.6 |
| 2 | 25.6976 | 135.1947 | −102.622 | −253.356 | 150.7 |
| 3 | 24.8823 | 128.4532 | −97.802 | −239.643 | 141.8 |
| 4 | 20.7826 | 117.4059 | −88.141 | −223.568 | 135.4 |
| 5 | 32.3552 | 146.2422 | −113.915 | −263.513 | 149.6 |
| 6 | 28.9271 | 134.6966 | −104.364 | −244.731 | 140.4 |

`separation = l_nontoxic − l_toxic`. Best epoch on every metric is **epoch 1**.

### 2.2 Training branches (from `log.out`)

| iteration | loss | l_nontoxic | l_toxic | separation | tox% |
|---|---|---|---|---|---|
| 200 | +2.489 | −85.50 | −287.58 | 202.1 | 22.2 |
| 14,800 | −9.313 | −82.57 | −331.57 | 249.0 | 22.2 |
| 29,600 | −6.158 | −76.13 | −298.98 | 222.9 | 21.9 |
| 44,600 | −11.911 | −78.71 | −330.87 | 252.2 | 22.1 |
| 59,400 | −5.905 | −80.38 | −343.12 | 262.7 | 20.4 |

---

## 3. What the run establishes

### D1 — Eq. 8 and the metrics are implemented correctly [CONFIRMED]

Two independent reconciliations against the logged numbers:

- `loss = −[(1−f)·l_nontoxic + f·(−l_toxic)]` reproduces the logged loss at iterations 200,
  14,800 and 59,400 to within the rounding of the printed `tox%` (2.489 vs 2.676; −9.313 vs
  −9.369; −5.905 vs −6.014).
- `val_nll = −mean(l)` over all molecules back-solves to a toxic fraction of **21.6% at every
  one of the four epochs checked** — the same validation set, so it must be constant, and it is.

Sign, label semantics (`1 = toxic`), the `[:, 0]` reshape over the `K` paths, and the
`batch_size × K` guard are all correct. `last_l` is the **unfloored** bound, so reported metrics
are never a function of `TOX_FLOOR`.

**The numbers are a real result, not a bug.**

### D2 — the non-toxic branch degrades on held-out data [CONFIRMED]

`val_l_nontoxic` goes 81.5 → 104.4 over six epochs, best at epoch 1. This is the quantity the
toxicity objective is supposed to leave alone, and it is measured on non-toxic molecules only,
so it is not contamination from the toxic branch. The generative model is getting worse at the
molecules it is meant to be getting better at.

### D3 — toxicity suppression stops generalising after epoch 2 [CONFIRMED]

Validation separation rises 125.6 → 150.7 (epoch 1→2) then plateaus at 135–150. Training
separation keeps widening (202 → 263). The toxicity signal is being fitted to the training
ligands and is not transferring to held-out proteins. **After epoch 2 the objective buys nothing
on held-out data and costs non-toxic likelihood.**

### D4 — the gap is inherited from stage 1, not created by fine-tuning [CONFIRMED]

Stage 1 runs the **same unbounded gradient reversal**, so the checkpoint already encodes a
driven-down toxic likelihood. `l_toxic = −287.58` at fine-tuning **iteration 200** is
approximately the stage-1 exit state, not something 200 steps produced.

Consequence: **a `TOX_FLOOR` applied in stage 2 only would be inert.** Toxic `l` is already at
−287 when fine-tuning starts, so any floor above that binds from iteration 1 and reduces the
toxic branch to `λ = 0` while still reporting as Eq. 8. A floor has to go into **stage 1**,
which means re-running pretraining.

### D5 — why the toxic branch is unbounded [CONFIRMED, by construction]

For a toxic ligand the objective is `min l_i`, and `l_i = log P(G|P)` has no lower bound. The
gradient does not vanish as it succeeds — it grows, since `∂ log p / ∂p = 1/p`.
`clip_grad_value_(3.0)` bounds per-element step size, not direction or the number of steps.
`λ` scales the term; it does not bound it.

### D6 — `val_nll` is the wrong model-selection metric [CONFIRMED by inspection]

`evaluate()` returns `(−mean(l_weighted), −mean(l_raw), mean l | non-toxic, mean l | toxic)`.
Both aggregates are computed correctly, but:

- `val_nll` averages over toxic **and** non-toxic. Suppressing toxic likelihood is the method's
  intended behaviour, so a *working* CTDDG drives `val_nll` up by construction. Selecting on it
  penalises success.
- `val_weighted` is the Eq. 8 objective, which is unbounded below — "improvement" can mean
  divergence, and best-checkpoint selection on it can pick the most degenerate model.
- The quantity that answers *"is the generative model still good"* is **`val_l_nontoxic`**. It is
  already written to `val.out` (column 6) but **is not selectable via `EARLY_STOP_ON`**, which
  offers only `'nll'` and `'weighted'`. Known gap; not yet changed.

### D7 — a falling training loss is not progress here [CONFIRMED]

With ~22% toxic at `|l| ≈ 300`, `loss ≈ −[0.78·(−80) + 0.22·300]`, so the loss falls as `l_toxic`
falls, without bound. Loss going from +2 to −25 is the toxic term growing, nothing else. **Never
report this loss as evidence that training is working.**

---

## 4. Open blockers, in priority order

### B1 — stage-1 checkpoint provenance [UNVERIFIED] [MISSING]

Nothing on disk has been read to confirm how long stage 1 actually trained.

What the code *can* establish, now that the pretraining source is known:

- **`max(num_batches_tracked)` in `ckpt.pt` is the exact stage-1 iteration count**, not a lower
  bound. The only train-mode forward in the pretraining notebook is the one per training
  iteration: `verify_eq8` runs under `model.eval()`, the batch smoke test never calls the model,
  and every BatchNorm sees exactly one forward per step (`dense` is computed once). It also
  survives resumes, because it lives in the state dict.
- `pretrained_provenance()` in `02_finetune_ctddg.ipynb` reads it, plus `trainer.pt`'s
  `global_counter` and the row count in `checkpoints/log.out`. It calls the checkpoint "trained"
  if **any** is non-zero — so a 500-iteration smoke run passes. Read the number, not the verdict.
- Cross-check worth making and not currently made: `bn_steps` vs the last step in `log.out`.
  Equal ⇒ one clean run. `bn_steps` greater ⇒ resumed after a log truncation (the pretraining
  resume path parses the counter from the *log*, not from `trainer.pt`, so a truncated log
  resets it). `bn_steps` smaller ⇒ `ckpt.pt` is stale relative to the log.

Already ruled out [CONFIRMED]: the weights are **finite**. Stage 1 has no non-finite guard, so a
NaN would have propagated silently into every parameter — but `02_finetune_ctddg.ipynb`'s
loss/gradient checks passed at all ~59,400 iterations.

Not determinable from `ckpt.pt` alone: nothing about the objective is stored (no λ, no flag).
That is settled by the pretraining source instead (§1).

### B2 — pretraining data leakage [CONFIRMED to have happened] — the most serious issue

Stage 1's `DATA_PATH` is the **unfiltered** ChEMBL export (`chembl_with_ames_predictions.csv`,
1,207,359 rows), flagged as known defect #1 in the port's own header and knowingly carried over.
Per `01` §5.3 the filtered set was built as `set(chembl_mols) − set(bindingdb_ligs)` **after**
dropping >100-atom molecules, so the unfiltered CSV contains **the BindingDB ligands** —
including the fine-tuning validation and held-out test ligands.

Consequences:

1. Every number downstream of this checkpoint is contaminated for reproduction purposes. CDGCN
   §5 and CTDDG §2.1 both state the BindingDB ligands were removed from ChEMBL *specifically* to
   prevent this. **Tables 1–3 cannot be compared against**, independent of how the objective
   behaves.
2. The epoch-1 validation figures are not clean held-out numbers. The fold controls *protein*
   novelty, not ligand novelty; pretraining adds a second leak on top.
3. It strengthens the D3 reading: part of the separation present at fine-tuning start was fitted
   directly to these ligands during pretraining.

> ⚠ Naming: `01` §5.3 records the labelled CSV as renamed to `chembl_toxicity_labels.csv`, but
> the PyTorch pretraining port points at `chembl_with_ames_predictions.csv`. Whether both exist,
> or one was renamed back, is **[UNVERIFIED]** — check before assuming which file was trained on.

### B3 — conditioning scale [UNVERIFIED] [MISSING]

`sanity_check()` prints, per GCN layer, `std(conv(X, A))` against `std(linear_c[i](c)[ids])` and
warns above 10×. The mechanism is plausible on priors — `linear_c[i]` is `Linear(6165, F_h[i])`
under Xavier, so output variance scales with `6165·var(c)`, and ProSE embeddings are not
normalised — but **plausible is not measured**, and those lines sit above `iter 000200` in a log
we have not seen.

Limits of that test even once read: it is one batch, at initialisation only, and a variance
ratio does not prove the pretrained features are *destroyed*. The decisive version is the step-0
ablation in §5(B).

### B4 — no justified `TOX_FLOOR` value [MISSING]

A floor only binds while toxic `l` is above it, so it must be set from the start of a run, and
choosing a value requires the toxic-`l` distribution **at step 0**, which is unmeasured.

> ⚠ **`−150.0` is not justified** by anything currently on hand, and the suggestion of that
> value in `04` §6.6 predates this run. Validation `l_toxic` was already −207 at epoch 1 and
> training `l_toxic` −287 at iteration 200, so a −150 floor may bind on essentially every toxic
> molecule from iteration 1, making the toxic branch inert while still reporting as Eq. 8. The
> `frac_floored` column in `02_finetune_ctddg.ipynb`'s `log.out` would reveal that — but only
> after the GPU time is spent.

### B5 — stage-1 batch size [CONFIRMED]

Stage 1 ran at `BATCH_SIZE = 16`, not the paper's 8 (known defect #3 in the port's header).
Stage 2 uses 8. Independent of B2, this already breaks comparability with the published tables.

### B6 — pretraining collator has no bond-count cap [CONFIRMED by inspection; consequence UNVERIFIED]

`graph_to_rnn` is `(batch, K, 120)` and the pretraining collator does
`graph_to_rnn[i, k, :num_iter] = np.arange(num_iter) + cum_sum` with
`num_iter = GetNumBonds() + 1`. For a molecule with `num_iter > 120` the slice truncates to 120
while the right-hand side has `num_iter` elements ⇒ broadcast `ValueError` inside a DataLoader
worker, killing the run.

`02_finetune_ctddg.ipynb` filters these in `assemble_pairs` (the `n_long` counter); stage 1 does
not, and the unfiltered CSV (B2) re-admits >100-atom molecules. **Prediction: if stage 1 did not
reach 480,000 iterations, this is the first thing to check.** If it did reach 480,000, the CSV
happened to contain no molecule with ≥120 bonds, which is itself worth knowing.

---

## 5. What to do next

### Evidence to collect first — zero GPU cost

**(i) The leakage check.** This outranks everything else, because it is the one result that
cannot be fixed by tuning anything:

```python
import pandas as pd
chembl = set(pd.read_csv("datasets/data/chembl/chembl_with_ames_predictions.csv",
                         usecols=["Smiles"])["Smiles"].astype(str))
lig = set(l.replace(" ", "").strip() for l in
          open("datasets/data/bindingdb/test_dataset/test_4_org_1042_104/"
               "ligands_4_org_1042_104_corrected") if l.strip())
print(len(lig & chembl), "of", len(lig), "held-out ligands were in the pretraining set")
```

**(ii) `checkpoints/log.out`** — the highest-value single file, because it answers B1 *and* B3 at
once. Columns are `step, time(h), loss, lr, l_nontoxic, l_toxic`.

```bash
wc -l checkpoints/log.out ; head -3 checkpoints/log.out ; tail -5 checkpoints/log.out
```

- **column 0, last row** — where stage 1 actually stopped.
- **column 5, last ~20 rows** — stage-1 `l_nontoxic` on ChEMBL. Compare with fine-tuning's
  epoch-1 `val_l_nontoxic` of −81.5. Close ⇒ the conditioning initialisation cost nothing and B3
  resolves negative. Materially better ⇒ the conditioning path is eating likelihood and B3 is
  live. *Caveat: different ligand populations, so normalise per action count before calling a
  small gap meaningful.*
- **column 6, trajectory** — whether the toxic branch was still falling at the end of stage 1 and
  how far it got. This is the only defensible basis for a floor value (B4). These branch columns
  are single-batch estimates at batch 16 (~12 non-toxic, ~4 toxic), so read the trend over many
  rows, not the last row.

**(iii) Checkpoint counters:**

```python
ck = torch.load("checkpoints/ckpt.pt",    map_location="cpu")["model"]
tr = torch.load("checkpoints/trainer.pt", map_location="cpu")
print("bn_steps:", max(int(v) for k, v in ck.items() if k.endswith("num_batches_tracked")))
print("trainer counter:", tr["global_counter"])
print("adam steps:", next(iter(tr["optimizer"]["state"].values()))["step"])
print("policy_0 absmax:", ck["policy_0"].abs().max().item())   # 0.0 ⇒ never trained
```

All four should agree with the log's last step. Disagreement is itself informative (B1).

**(iv) The top of the fine-tuning log**, above `iter 000200`: the five provenance lines, the
`Eq. 8 |`, `cond |` and `scale | layer 0..5` lines, the transferred-tensor count, and the
`[train] kept … | dropped: …` line.

### Experiments, in order

**A. Step-0 evaluation — minutes, no training.** Build the model as `02_finetune_ctddg.ipynb`
does and call `evaluate(model, loader_val)` **before entering the training loop**, in two
configurations: (i) as-initialised conditional, (ii) with `linear_c[*].weight` zeroed and
`dense_policy_0` bypassed, which reduces it to the loaded unconditional model. Record per-molecule
`l` alongside each molecule's action count (`GetNumBonds() + 1`) and toxicity label.

*Purpose:* one measurement answers four questions — baseline checkpoint quality on *this* data
(no domain-shift confound), the cost of the conditioning initialisation (B3, decisively, where a
variance ratio cannot), whether the toxic/non-toxic gap is a **molecule-size confound** rather
than suppression, and the only defensible basis for a floor (B4).

*Limit:* (ii) is a diagnostic ablation, not a proposal. Zero-initialising `linear_c` permanently
would deviate from upstream CDGCN, where it is Xavier.

**B. λ = 0 control — ~35 min, 2 epochs, same fold and `SPLIT_SEED`.** Compare `val_l_nontoxic`
at epochs 1–2 against FT-d1-01's −81.5 / −102.6.

*Interpretation:* if non-toxic NLL holds or improves at λ = 0, the toxic branch causes the
degradation and the problem is the objective. If it degrades similarly, the cause is upstream —
checkpoint or conditioning — and no amount of λ tuning or flooring will fix it.

*Limit:* **λ = 0 does not recover CDGCN.** The weight `1 − (1+0)·tox` is 0 on toxic ligands, so
they are *dropped* rather than trained on normally; the control trains on ~78% of the pairs. It
isolates the gradient reversal, not the data. A true CDGCN control needs weight 1 everywhere,
which is a code change.

**C. A floor experiment — not before A.** Without the step-0 distribution there is no principled
value, and an inert floor is indistinguishable from a working one in the aggregate metrics. If
run at all, it has to be set in **stage 1** (D4), i.e. bundled into a pretraining re-run.

### Recommendation as of 2026-10-11

**No full fine-tuning run is justified.** Of the candidate causes, none is confirmed and the
cheapest is free to settle. Run the leakage check, then the log reads, then A.

If the leak is material, this checkpoint cannot support a reproduction claim and **stage 1 has to
be re-run on the filtered `chembl.txt` (1,090,529 molecules)** regardless of what the other
diagnostics say — at which point B4 (floor in stage 1?), B5 (`BATCH_SIZE = 8`) and B6 (bond cap)
should all be settled *before* that re-run starts, since it is the expensive one.

---

## 6. Things we are not entitled to conclude

Recorded because each was asserted at some point in working on this and then withdrawn.

- **Do not compare the stage-1 loss to fine-tuning NLL.** They differ in dataset (filtered ChEMBL
  vs BindingDB ligands, different action-count distributions, and `l` scales with action count),
  in objective (the stage-1 `loss` column is `−mean(l_weighted)`, which is *not* an NLL and goes
  negative), in model (unconditional, no `linear_c`), and in population (fine-tuning's `val_nll`
  averages over toxic molecules that are being actively suppressed). In particular the
  "**final loss 15.99**" figure quoted in `01` §5.3 belongs to the **MXNet CDGCN-objective** run
  in `../CTDDG_new/outputs/pretrain/` — a different framework, objective and repository — and is
  not comparable to anything in the PyTorch lineage. The valid comparison is same-model-same-data:
  evaluate the loaded checkpoint on these BindingDB molecules *before* any optimizer step (§5 A).
- **NLL does not measure the paper's target.** Table 3 is the pkCSM-predicted toxic fraction of
  *generated* molecules. That suppressed likelihood of training toxic ligands produces fewer toxic
  *generated* molecules is an assumption of the method; nothing measured so far tests it. Testing
  it requires generation + scoring, which needs the sampler and the evaluation notebooks
  (`01` §5.6 — not started).
- **A clean `load_state_dict` is not evidence of training.** It proves the shapes match.
- **A falling training loss is not evidence of training working.** See D7.

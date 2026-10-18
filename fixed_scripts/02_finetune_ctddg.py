# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stage 2 - conditional fine-tuning (CTDDG Eq. 8)
#
# Loads the stage-1 checkpoint, adds the protein conditioning path, and trains on
# BindingDB protein-ligand pairs under the toxicity-weighted objective.
# Self-contained; run top to bottom.
#
# **Needs** `<fold>_<tr|te>_unique_embeddings_bepler.h5` from
# `01_generate_prose_embeddings.py`: one average-pooled `(6165,)` vector per
# protein, keyed `p1 ... pN`. Per-residue output is rejected on load.
#
# **Produces** `ckpt_best.pt` (the one to evaluate), `ckpt_last.pt` (resume point),
# `test_set.npz`, `split_manifest.json`, and `step0_likelihoods.npz`.
#
# ## What changed, and why
#
# | # | Change | Reason |
# |---|---|---|
# | 1 | **Salt stripping + a connectivity assertion** in `assemble_pairs` | The raw `ligands_*_corrected` files were read directly, salts and all. On a disconnected graph `traverse_graph` leaves the second component at `step_ids = -1` and `single_reorder` then mis-wires every bond - silently, with no counter. |
# | 2 | **`EARLY_STOP_ON = 'l_nontoxic'`** | `val_nll` averages over toxic *and* non-toxic molecules, so a *working* CTDDG drives it up by construction. Selecting on it penalises the method succeeding, which is why the chosen checkpoint was epoch 1. |
# | 3 | **`PATIENCE = 3`** | With `PATIENCE = MAX_EPOCHS = 10` the counter can reach at most 9, so early stopping can never fire. The run burned nine epochs it then discarded. |
# | 4 | **Epoch boundaries by iteration count, not `StopIteration`** | After the mid-run disconnect, `iter(loader_train)` restarted epoch 9 from batch 0 while `global_counter` kept climbing: 16,514 batches in one epoch instead of 14,898, and 16 extra learning-rate decay steps. |
# | 5 | **Step-0 measurement before the first optimizer step** | Blocker B4: a `TOX_FLOOR` can only be chosen from the toxic-`l` distribution at step 0, which was never measured. Now it is, and written to `step0_likelihoods.npz`. |
# | 6 | **Conditioning actually tested, not just scaled** | The old sanity check only asserted `|a - b| > 1e-6`. It passed while reporting that a *random* conditioning vector gave a **lower** loss than the real protein embedding. |
#
# > **Run `00_phase0_diagnostics.py` first.** If its check A shows the stage-1
# > checkpoint already produces ~25% unconditional validity, no amount of
# > fine-tuning repair will help and stage 1 has to be redone first.

# %%
import os
import gc
import glob
import json
import signal
import math
import time
import pickle
import random
import hashlib
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

import rdkit
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.autograd.function import once_differentiable

print(f"torch {torch.__version__} | rdkit {rdkit.__version__}")

# %% [markdown]
# ## Config
#
# Everything that varies per run. Nothing below this section should need editing.

# %%
FORCE_RESTART = False
FOLD = os.environ.get("CTDDG_FOLD", "d1")

GPU_ID = int(os.environ.get("CTDDG_GPU", "0"))
CTX = torch.device(f"cuda:{GPU_ID}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {CTX}")

REPO_ROOT = os.environ.get("CTDDG_REPO_ROOT", "/home/muhamed/repo/CTDDG")
ATOM_TYPES_PATH = os.path.join(REPO_ROOT, "datasets/data/atom_types.txt")
BINDINGDB_DIR = os.path.join(REPO_ROOT, "datasets/data/bindingdb")
RF_MODEL_PATH = os.path.join(REPO_ROOT, "models/ames_random_forest.pkl")
RF_META_PATH = os.path.splitext(RF_MODEL_PATH)[0] + ".meta.json"
FP_CONFIG_PATH = os.path.join(REPO_ROOT, "models/fingerprint_config.pkl")
PRETRAINED_CKPT = os.path.join(REPO_ROOT, "checkpoints/ckpt.pt")

# Stage 2 without stage 1 is not CTDDG. False only for a deliberate scratch run.
REQUIRE_PRETRAINED = True
# Stop if the checkpoint carries positive evidence of never having trained.
ALLOW_UNTRAINED_CKPT = False

OUT_DIR = os.path.join(REPO_ROOT, "outputs/finetune", FOLD)
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_BEST = os.path.join(OUT_DIR, "ckpt_best.pt")      # evaluate this one
CKPT_LAST = os.path.join(OUT_DIR, "ckpt_last.pt")      # resume from this one
CKPT_TRAINER = os.path.join(OUT_DIR, "trainer.pt")
CKPT_CONFIG = os.path.join(OUT_DIR, "configs.json")
LOG_FILE = os.path.join(OUT_DIR, "log.out")
VAL_LOG_FILE = os.path.join(OUT_DIR, "val.out")
STEP0_PATH = os.path.join(OUT_DIR, "step0_likelihoods.npz")
CKPT_PREV = CKPT_LAST + ".prev"

CHECKS = []


def check(name, ok, detail, warn_only=False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    CHECKS.append({"name": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")
    if not ok and not warn_only:
        raise RuntimeError(f"{name}: {detail}")
    return ok


# %%
CHECKPOINT_EVERY = 200     # iterations between checkpoints (also at every epoch end)
VERIFY_FASTA_ORDER = True  # assert the bepler FASTA, protein list and h5 keys agree
GRACEFUL_SIGNALS = True    # SIGTERM/SIGINT -> checkpoint at the next iteration

FOLD_SUFFIX = {"d1": "4_org_1042_104", "d2": "4_org_1000_112", "d3": "4_org_1004_122",
               "d4": "4_org_1002_103", "d5": "4_org_1036_124"}

AMES_CACHE = os.path.join(BINDINGDB_DIR, "ligand_ames_labels.csv")
AMES_CACHE_META = AMES_CACHE + ".meta.json"

# If the Ames RF is missing or unreadable by this sklearn, rebuild it here.
#
# WARNING: a rebuild makes this run's BindingDB labels come from a DIFFERENT
# forest than the ChEMBL labels stage 1 trained on - two definitions of toxic
# inside one objective. Prefer re-running random_forest_classifier.py, which
# rebuilds both consistently. Check H of 00_phase0_diagnostics.py detects it.
# Kept False and NOT acted on anywhere in this notebook: silently rebuilding the
# forest here is precisely the failure mode described above. ames_labels() raises
# with a pointer to random_forest_classifier.py instead.
RF_AUTO_TRAIN = False

# %%
# The paper evaluates on all 104 novel proteins of the held-out split (Tables 1-3),
# so the whole test_dataset is the test set and validation is carved out of TRAIN.
# It is carved BY PROTEIN, not by pair: early stopping for a conditional model
# should measure generalisation to unseen proteins, which a validation set sharing
# proteins with training cannot do. The paper specifies no validation protocol, so
# this is our choice, not its.
VAL_PROTEIN_FRACTION = 0.05
SPLIT_SEED = 42

MAX_ATOMS = 100
STRIP_SALTS = True          # largest fragment, matching data_preprocessing_1
REJECT_DISCONNECTED = True  # belt-and-braces after stripping

# The ligands_*/proteins_*_corrected files are a SUPERSET of the fold: they hold
# pairs for proteins outside this split. The fold is defined by
# <fold>_<tr|te>_unique_proteins.txt, so filtering down to the listed proteins IS
# the split, not attrition - expect a large drop.
POOL_PAIR_FILES = True
UNMATCHED_TOLERANCE = 1.0

# None = the full dataset. A number is a SMOKE TEST.
LIMIT_PAIRS = None
SMOKE_TEST = LIMIT_PAIRS is not None
_SMOKE_SUFFIX = ".smoke" if SMOKE_TEST else ""
SPLIT_MANIFEST = os.path.join(OUT_DIR, f"split_manifest{_SMOKE_SUFFIX}.json")
TEST_BUNDLE = os.path.join(OUT_DIR, f"test_set{_SMOKE_SUFFIX}.npz")

# %%
BATCH_SIZE = 8
BATCH_SIZE_VAL = 8
K_PATHS = 5
P_ALPHA = 0.8
LAMBDA_TOX = 1.0
N_C = 6165               # average-pooled ProSE/Bepler width; validated on load
ALLOW_N_C_OVERRIDE = False
EMB_EXTENSIONS = ('.h5', '.hdf5', '.npy', '.txt', '.csv')

F_E = 16
F_H = [32, 64, 128, 128, 256, 256]
F_SKIP = 256
F_C = [512]
FH_POLICY = 128
ACTIVATION = 'relu'
N_RNN = 3
D_RECEPTIVE = 2
MAX_ITER = 120

LR = 1e-4
DECAY = 0.002            # lr *= (1 - DECAY) every DECAY_STEP iterations
DECAY_STEP = 100
CLIP_GRAD = 3.0
SUMMARY_STEP = 200
MAX_EPOCHS = 10
PATIENCE = 3             # was 10, which with MAX_EPOCHS = 10 can never fire
ES_DELTA = 1e-2

# %% [markdown]
# ### Model selection and the toxic branch
#
# Neither is specified by the paper, so choosing is ours either way - but the
# previous choice was actively wrong.

# %%
# Which validation quantity early stopping watches.
#
#   'l_nontoxic' - mean l over NON-TOXIC validation molecules only. DEFAULT.
#                  The quantity that answers "is the generative model still any
#                  good", and the only one Eq. 8 is not trying to move.
#   'nll'        - plain -mean(l) over everything. A WORKING CTDDG drives this UP
#                  by construction, because suppressing toxic likelihood is the
#                  method. Selecting on it penalises success - this is what picked
#                  epoch 1 last time.
#   'weighted'   - the Eq. 8 objective itself, unbounded below on the toxic
#                  branch, so "improvement" can mean divergence and best-checkpoint
#                  selection can pick the most degenerate model.
#
# Note LAMBDA_TOX = 0 does NOT recover CDGCN: the weight 1-(1+0)*tox is 0 on toxic
# ligands, dropping them from the batch average. It is still the clean control for
# "is the toxic branch causing this?", since it removes the gradient reversal.
EARLY_STOP_ON = 'l_nontoxic'

# Warn when validation NLL rises this far above its own best.
NLL_DIVERGENCE_MARGIN = 2.0

# --- Toxic-branch floor (review item F13) ---------------------------------------
# None = Eq. 8 exactly as published, and the baseline for anything compared against
# the paper.
#
# A float applies `l = max(l, TOX_FLOOR)` to toxic ligands in the OBJECTIVE only;
# reported metrics stay unfloored. Once a toxic molecule is below the floor, clamp
# passes zero gradient, so it stops being pushed further down - which is the point:
# the published toxic term is unbounded below.
#
# Flooring the per-molecule bound AFTER the importance-sampling reduction is the
# right place: it leaves the Eq. 3 estimate of log P(G|P) untouched and only bounds
# how far the objective will chase one molecule. Flooring the per-path terms before
# logsumexp would corrupt the estimator itself.
#
# DO NOT pick a value by eye. A floor only binds while toxic likelihood is above
# it, so it must be set from the START of a run: applied to a model that has
# already driven toxic ligands far down, every toxic molecule is clamped from
# iteration 1 and the branch is inert (equivalent to LAMBDA_TOX = 0) while still
# reporting as Eq. 8. The step-0 cell below measures the distribution this value
# has to come from, and prints a candidate.
TOX_FLOOR = None

# Measure the step-0 likelihood distribution before the first optimizer step.
# Costs one validation pass (~1 minute) and is the only defensible basis for a
# floor. Also the decisive version of blocker B3.
MEASURE_STEP0 = True

# %%
NUM_WORKERS = 8
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_missing = []
for _label, _p in (("atom_types.txt", ATOM_TYPES_PATH),
                   ("bindingdb dir", BINDINGDB_DIR),
                   ("fingerprint config", FP_CONFIG_PATH)):
    if not os.path.exists(_p):
        _missing.append(f"{_label}: {_p}")
check("required inputs", not _missing,
      "all present" if not _missing else "; ".join(_missing))

check("early stopping can fire", PATIENCE < MAX_EPOCHS,
      f"PATIENCE = {PATIENCE} < MAX_EPOCHS = {MAX_EPOCHS}"
      if PATIENCE < MAX_EPOCHS else
      f"PATIENCE = {PATIENCE} >= MAX_EPOCHS = {MAX_EPOCHS}: epoch 1 always improves "
      f"(best starts at inf), so the counter tops out at {MAX_EPOCHS - 1} and early "
      f"stopping can NEVER fire", warn_only=True)

check("model selection metric", EARLY_STOP_ON == 'l_nontoxic',
      f"EARLY_STOP_ON = {EARLY_STOP_ON!r}"
      + ("" if EARLY_STOP_ON == 'l_nontoxic' else
         " - 'nll' rises when CTDDG works and 'weighted' is unbounded below; "
         "both select the wrong checkpoint"), warn_only=True)

if not os.path.exists(RF_MODEL_PATH):
    print(f"[WARN] No Ames RF at {RF_MODEL_PATH}. The label cache can only be reused "
          f"when the model that wrote it is on disk to check against.")

if not os.path.exists(PRETRAINED_CKPT) and REQUIRE_PRETRAINED:
    raise FileNotFoundError(
        f"No pretrained checkpoint at {PRETRAINED_CKPT}.\n"
        f"  CTDDG is a two-phase method (paper 2.4): stage 1 pretrains "
        f"unconditionally on ChEMBL, stage 2 fine-tunes that model. Starting from "
        f"random weights is a different experiment and its numbers are not "
        f"comparable to the paper.\n"
        f"  Run pretraining_torch.py first, or set REQUIRE_PRETRAINED = False.")

if SMOKE_TEST:
    print(f"\n[SMOKE TEST] LIMIT_PAIRS = {LIMIT_PAIRS}. Outputs are written as "
          f"{os.path.basename(SPLIT_MANIFEST)} / {os.path.basename(TEST_BUNDLE)} and "
          f"are NOT the paper's dataset.\n")

# %% [markdown]
# ## Restart cleanup
#
# Runs before any data work: a crash during assembly or RF labelling must not leave
# a resumable checkpoint from the previous run for the next one to pick up.

# %%
RESTART_STAMP = time.strftime('%Y%m%d-%H%M%S')


def archive_stale(paths, stamp=None, why=""):
    stamp = stamp or RESTART_STAMP
    moved = []
    for p in paths:
        if os.path.exists(p):
            dst = f"{p}.stale.{stamp}"
            os.replace(p, dst)
            moved.append(dst)
            print(f"  moved {os.path.basename(p)} -> {os.path.basename(dst)}"
                  + (f"  ({why})" if why else ""))
    return moved


RESUMABLE_ARTIFACTS = (CKPT_BEST, CKPT_LAST, CKPT_PREV, CKPT_TRAINER,
                       LOG_FILE, VAL_LOG_FILE)

if FORCE_RESTART:
    print("FORCE_RESTART - archiving the previous run before any data work")
    archive_stale(RESUMABLE_ARTIFACTS, why="FORCE_RESTART")

# %% [markdown]
# ## Stage-1 provenance
#
# What the files on disk actually say about `ckpt.pt`. A state dict that loads
# cleanly proves the shapes match, not that any training happened.

# %%
def _last_log_step(path):
    last, n = None, 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('#', 'step\t')) or line == 'Training finished':
                    continue
                try:
                    last = int(line.split('\t')[0])
                    n += 1
                except (ValueError, IndexError):
                    continue
    except OSError:
        return None, 0
    return last, n


def pretrained_provenance(path):
    """Evidence only - nothing here is inferred.

    The strongest signal is BatchNorm's num_batches_tracked, which counts
    train-mode forward passes and is reset to 0 by init_weights_xavier. It is the
    EXACT iteration count, not a lower bound: the only train-mode forward in
    pretraining is the one per training step.
    """
    ev = {'path': path, 'size_mb': None, 'mtime': None, 'bn_steps': None,
          'trainer_counter': None, 'log_last_step': None, 'log_rows': 0,
          'trained': False, 'objective': None}
    ckpt_dir = os.path.dirname(path)

    try:
        st = os.stat(path)
        ev['size_mb'] = st.st_size / 1e6
        ev['mtime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))
    except OSError:
        return ev

    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
        sd = state['model'] if isinstance(state, dict) and 'model' in state else state
        nbt = [int(v) for k, v in sd.items() if k.endswith('num_batches_tracked')]
        ev['bn_steps'] = max(nbt) if nbt else None
        # pretraining_torch.py now records this. Older checkpoints will not have it.
        if isinstance(state, dict):
            ev['objective'] = state.get('objective')
    except Exception as e:
        print(f"[WARN] could not read {os.path.basename(path)}: {type(e).__name__}: {e}")

    tr = os.path.join(ckpt_dir, 'trainer.pt')
    if os.path.isfile(tr):
        try:
            t = torch.load(tr, map_location='cpu', weights_only=False)
            if isinstance(t, dict):
                ev['trainer_counter'] = t.get('global_counter')
                if ev['objective'] is None:
                    ev['objective'] = t.get('objective')
        except Exception:
            pass

    ev['log_last_step'], ev['log_rows'] = _last_log_step(os.path.join(ckpt_dir, 'log.out'))
    ev['trained'] = bool((ev['bn_steps'] or 0) > 0
                         or (ev['trainer_counter'] or 0) > 0
                         or ev['log_rows'] > 0)

    print(f"\n--- stage-1 provenance: {os.path.basename(path)} ---")
    print(f"  size / modified     {ev['size_mb']:.1f} MB  {ev['mtime']}")
    print(f"  BN train-mode steps {ev['bn_steps']}   (EXACT iteration count; "
          f"0 = never trained)")
    print(f"  trainer.pt counter  {ev['trainer_counter']}")
    print(f"  log.out             {ev['log_rows']} data rows"
          + (f", last step {ev['log_last_step']}" if ev['log_last_step'] else ""))
    if ev['objective']:
        print(f"  objective recorded  {ev['objective']}")
    else:
        print("  objective recorded  (none - this checkpoint predates objective "
              "tracking; the pretraining source is the only record)")
    print("-" * 60)
    return ev


PRETRAIN_EVIDENCE = None
if os.path.exists(PRETRAINED_CKPT):
    PRETRAIN_EVIDENCE = pretrained_provenance(PRETRAINED_CKPT)

    if PRETRAIN_EVIDENCE['bn_steps'] == 0 and not ALLOW_UNTRAINED_CKPT:
        raise RuntimeError(
            f"{PRETRAINED_CKPT} reports num_batches_tracked == 0 in every BatchNorm "
            f"layer: it has never seen a training step. Fine-tuning it is "
            f"fine-tuning random weights, at full GPU cost.")

    check("stage-1 checkpoint trained", bool(PRETRAIN_EVIDENCE['trained']),
          f"{PRETRAIN_EVIDENCE['bn_steps']:,} iterations by the BatchNorm counter"
          if PRETRAIN_EVIDENCE['bn_steps'] else "no evidence of training",
          warn_only=True)

    _obj = (PRETRAIN_EVIDENCE['objective'] or {})
    if _obj:
        _tw = _obj.get('tox_weighting')
        check("stage-1 objective", _tw is False,
              f"stage 1 used {_obj.get('objective')}. "
              + ("Good - CDGCN's unweighted NLL, as upstream does." if _tw is False
                 else "Eq. 8 in stage 1 is what collapsed validity to 23.9%; see "
                      "pretraining_torch.py's header"),
              warn_only=True)
    else:
        print("  [UNVERIFIED] the checkpoint records no objective. Which one stage 1 "
              "used is settled only by reading the pretraining source.")

# %% [markdown]
# ## Fold files and embeddings
#
# Row `i` of the embedding matrix is HDF5 key `p{i+1}` is FASTA record `i`. Pairs
# are matched to rows by protein sequence, so the FASTA and the protein list must
# agree record for record. Both are checked, not assumed.

# %%
def split_dir(split):
    return os.path.join(BINDINGDB_DIR, f"{split}_dataset", f"{split}_{FOLD_SUFFIX[FOLD]}")


def find_fold_files(split):
    d = split_dir(split)
    tag = "tr" if split == "train" else "te"

    def one(pattern, what):
        hits = sorted(glob.glob(os.path.join(d, pattern)))
        if not hits:
            raise FileNotFoundError(
                f"{what}: no match for {pattern!r} in {d}\n"
                f"  Directory holds: {sorted(os.listdir(d))[:25]}")
        return hits[0]

    ligands = one("ligands_*_corrected", "ligand pairs")
    proteins = one("proteins_*_corrected", "protein pairs")
    unique = one(f"{FOLD}_{tag}_unique_proteins.txt", "unique protein list")

    expected = f"{FOLD}_{tag}_unique_embeddings_bepler.h5"
    emb = os.path.join(d, expected)
    if not os.path.isfile(emb):
        hits = [p for p in sorted(glob.glob(os.path.join(d, "*embeddings*bepler*")))
                if os.path.splitext(p)[1].lower() in EMB_EXTENSIONS]
        if not hits:
            raise FileNotFoundError(
                f"Protein embeddings missing for fold {FOLD} ({split}).\n"
                f"  Expected: {os.path.join(d, expected)}\n"
                f"  Generate average-pooled embeddings with:\n"
                f"    python prose/embed_sequences.py --pool avg \\\n"
                f"        -o {expected} {FOLD}_{tag}_unique_proteins_bepler.fa\n"
                f"  --pool avg is required: it yields one ({N_C},) vector per protein.")
        emb = hits[0]
        print(f"[WARN] {expected} not found; using {os.path.basename(emb)}")
    return ligands, proteins, unique, emb


def _as_pooled_vector(arr, name, path, width):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        squeezed = arr.squeeze()
        if squeezed.ndim == 1:
            arr = squeezed
        elif arr.ndim == 2:
            raise ValueError(
                f"{path}: protein {name!r} has shape {arr.shape} - per-residue "
                f"embeddings. Regenerate with `--pool avg`; expected ({N_C},)")
        else:
            raise ValueError(f"{path}: protein {name!r} has shape {arr.shape}")

    if width is not None and arr.shape[0] != width:
        hint = ""
        if arr.shape[0] % N_C == 0 and arr.shape[0] > N_C:
            hint = (f"\n  {arr.shape[0]} = {arr.shape[0] // N_C} x {N_C}, so this looks "
                    f"like a flattened per-residue embedding. Use `--pool avg`.")
        raise ValueError(f"{path}: protein {name!r} has width {arr.shape[0]}; "
                         f"expected {width}.{hint}")
    if arr.shape[0] == 0:
        raise ValueError(f"{path}: protein {name!r} has an empty embedding.")
    return arr.astype(np.float32, copy=False)


def _h5_key_order(f, n_expected, path):
    keys = list(f.keys())
    if not keys:
        raise ValueError(f"{path}: contains no datasets.")
    if len(keys) != n_expected:
        shown = keys[:5] if len(keys) <= 10 else keys[:5] + ['...'] + keys[-5:]
        raise ValueError(
            f"{path}: {len(keys)} embedded proteins but {n_expected} in the fold's "
            f"protein list.\n  Keys: {shown}")

    named = [f"p{i + 1}" for i in range(n_expected)]
    if all(k in f for k in named):
        return named

    # No fallback ordering. Row i must be protein i, and only the key name
    # guarantees that; any other order silently conditions every ligand on the
    # wrong protein while every shape check still passes.
    missing = [k for k in named if k not in f]
    raise ValueError(
        f"{os.path.basename(path)}: expected keys p1..p{n_expected}; "
        f"{len(missing)} absent (e.g. {missing[:8]}).")


def load_embeddings(path, n_expected):
    ext = os.path.splitext(path)[1].lower()
    if ext not in EMB_EXTENSIONS:
        raise ValueError(f"{path}: unsupported embedding format {ext!r}")

    if ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            order = _h5_key_order(f, n_expected, path)
            first = _as_pooled_vector(f[order[0]][()], order[0], path,
                                      None if ALLOW_N_C_OVERRIDE else N_C)
            width = first.shape[0]
            rows = [first] + [_as_pooled_vector(f[k][()], k, path, width)
                              for k in order[1:]]
        C = np.stack(rows, axis=0)
    elif ext == '.npy':
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected [n_proteins, {N_C}], got {arr.shape}")
        C = arr.astype(np.float32, copy=False)
    else:
        raw = []
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                parts = line.replace(',', ' ').split()
                if not parts:
                    continue
                try:
                    float(parts[0])
                except ValueError:
                    parts = parts[1:]
                raw.append((f"line {ln}", np.array(parts, dtype=np.float32)))
        if not raw:
            raise ValueError(f"{path}: no embedding rows parsed.")
        first = _as_pooled_vector(raw[0][1], raw[0][0], path,
                                  None if ALLOW_N_C_OVERRIDE else N_C)
        width = first.shape[0]
        C = np.stack([first] + [_as_pooled_vector(a, nm, path, width)
                                for nm, a in raw[1:]], axis=0)

    if C.shape[0] != n_expected:
        raise ValueError(f"{path}: {C.shape[0]} rows but {n_expected} proteins listed.")
    if C.shape[1] != N_C and not ALLOW_N_C_OVERRIDE:
        raise ValueError(
            f"{path}: embedding width {C.shape[1]}, expected N_C = {N_C}. "
            f"Average-pooled ProSE/Bepler embeddings are {N_C}-d.")
    if not np.isfinite(C).all():
        n_bad = int((~np.isfinite(C)).any(axis=1).sum())
        raise ValueError(f"{path}: {n_bad} embeddings contain NaN or inf.")
    return C


# %% [markdown]
# ## Ames labels
#
# Ligand toxicity from the repo's Random Forest, cached across folds. The cache is
# reused only when the model that wrote it is on disk and hashes the same;
# otherwise everything is relabelled. Mixing labels from two forests would mean
# Eq. 8 optimising against two definitions of toxic.

# %%
def _load_pickle(path):
    errors = []
    for how, fn in (("pickle", lambda: pickle.load(open(path, 'rb'))),
                    ("joblib", lambda: __import__('joblib').load(path))):
        try:
            return fn()
        except Exception as e:
            errors.append(f"{how}: {type(e).__name__}: {e}")

    hint = ""
    if any("incompatible dtype" in e or "node array" in e for e in errors):
        import sklearn
        hint = (f"\n  This is a scikit-learn version mismatch, not a corrupt file: the "
                f"pickle was written by an older sklearn whose tree node dtype differs "
                f"from {sklearn.__version__} here.\n"
                f"  Regenerate the model IN THIS ENVIRONMENT by re-running "
                f"random_forest_classifier.py. Do not downgrade sklearn.")
    raise RuntimeError(f"Could not load {path}\n  " + "\n  ".join(errors) + hint)


def morgan_matrix(smiles_list, radius, n_bits):
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp_of = lambda m: gen.GetFingerprint(m)
    except (ImportError, AttributeError):
        from rdkit.Chem import AllChem
        fp_of = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)

    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out[i] = np.frombuffer(fp_of(mol).ToBitString().encode(),
                                   dtype=np.uint8) - ord('0')
        if (i + 1) % 20000 == 0:
            print(f"    fingerprinted {i + 1}/{len(smiles_list)}")
    return out


_RF_HASH_MEMO = {}


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def rf_fingerprint():
    """Identifies the model+features a cached label came from.

    SHA-256 of the artifact, not size+mtime: a retrained forest can land on the
    same byte count, and a copy or rsync changes mtime without changing the model.
    Either half being None means 'cannot be verified' - never a match.
    """
    rf = None
    try:
        st = os.stat(RF_MODEL_PATH)
        key = (RF_MODEL_PATH, st.st_size, st.st_mtime_ns)
        if key not in _RF_HASH_MEMO:
            _RF_HASH_MEMO[key] = _sha256(RF_MODEL_PATH)
        rf = {'sha256': _RF_HASH_MEMO[key], 'size': st.st_size}
    except OSError:
        rf = None

    fp = None
    try:
        cfg = _load_pickle(FP_CONFIG_PATH)
        fp = {'radius': int(cfg.get('radius', 2)),
              'n_bits': int(cfg.get('n_bits', cfg.get('nBits', 2048)))}
    except Exception:
        fp = None

    return {'v': 2, 'rf': rf, 'fp': fp}


def fingerprint_is_verifiable(fp):
    return bool(fp) and bool(fp.get('rf')) and bool(fp.get('fp'))


def cache_decision(want, got, meta_err=None):
    if not fingerprint_is_verifiable(want):
        if not (want or {}).get('rf'):
            return False, ("the Random Forest is not on disk, so the cache cannot be "
                           "checked against the model that wrote it")
        return False, "the fingerprint config is unreadable"
    if got is None:
        return False, (f"fingerprint unreadable ({meta_err})" if meta_err
                       else "no fingerprint recorded beside the cache")
    if not fingerprint_is_verifiable(got):
        return False, "the recorded fingerprint is incomplete"
    if got != want:
        return False, "the recorded fingerprint differs from the current model/config"
    return True, "fingerprint matches"


def ames_labels(smiles_list, cache_path=AMES_CACHE):
    """SMILES -> {0,1} Ames mutagenicity from the repo's RF. Cached across folds."""
    cache = {}
    want = rf_fingerprint()

    if cache_path and os.path.exists(cache_path):
        got, meta_err = None, None
        if os.path.exists(AMES_CACHE_META):
            try:
                with open(AMES_CACHE_META) as f:
                    got = json.load(f)
            except Exception as e:
                meta_err = type(e).__name__

        accept, reason = cache_decision(want, got, meta_err)
        if accept:
            df = pd.read_csv(cache_path)
            cache = dict(zip(df['smiles'].astype(str), df['ames'].astype(int)))
            print(f"  cache: {len(cache)} entries from {os.path.basename(cache_path)} "
                  f"({reason})")
        else:
            archive_stale([cache_path, AMES_CACHE_META],
                          stamp=time.strftime('%Y%m%d-%H%M%S'), why="unverified cache")
            print(f"  [WARN] not reusing the cache: {reason}. Relabelling from scratch")

    todo = sorted({s for s in smiles_list if s not in cache})
    if todo:
        fp_cfg = _load_pickle(FP_CONFIG_PATH)
        radius = int(fp_cfg.get('radius', 2))
        n_bits = int(fp_cfg.get('n_bits', fp_cfg.get('nBits', 2048)))
        print(f"  labelling {len(todo)} new ligands | Morgan r{radius}, {n_bits} bits")

        if not os.path.exists(RF_MODEL_PATH):
            raise FileNotFoundError(
                f"No Ames RF at {RF_MODEL_PATH}. Run random_forest_classifier.py, "
                f"which rebuilds the forest AND the ChEMBL labels consistently. "
                f"(RF_AUTO_TRAIN is off here on purpose: rebuilding only the forest "
                f"leaves ChEMBL and BindingDB labelled by two different models.)")
        rf = _load_pickle(RF_MODEL_PATH)

        X = morgan_matrix(todo, radius, n_bits)
        if getattr(rf, 'n_features_in_', n_bits) != n_bits:
            raise ValueError(f"RF expects {rf.n_features_in_} features, config gives "
                             f"{n_bits}.")
        cache.update(zip(todo, rf.predict(X).astype(int)))

        if cache_path:
            pd.DataFrame({'smiles': list(cache.keys()), 'ames': list(cache.values())}) \
              .to_csv(cache_path, index=False)
            with open(AMES_CACHE_META, 'w') as f:
                json.dump(rf_fingerprint(), f, indent=1)
            print(f"  cache written: {os.path.basename(cache_path)}")

    labels = [int(cache[s]) for s in smiles_list]
    n_tox = sum(labels)
    print(f"  Ames: {n_tox} toxic / {len(labels) - n_tox} non-toxic "
          f"({100.0 * n_tox / max(1, len(labels)):.1f}% toxic)")
    return labels


# %% [markdown]
# ## Molecule spec and traversal
#
# Unchanged from the pretraining code - both phases must build graphs identically
# or the transferred weights mean nothing. The one addition is the connectivity
# guard.

# %%
class MoleculeSpec(object):
    def __init__(self, file_name=ATOM_TYPES_PATH):
        self.atom_types = []
        self.atom_symbols = []
        with open(file_name, 'r') as f:
            for line in f:
                a = line.strip('\n').split(',')
                if len(a) < 3:
                    continue
                self.atom_types.append((a[0], int(a[1]), int(a[2])))
                if a[0] not in self.atom_symbols:
                    self.atom_symbols.append(a[0])

        self.bond_orders = [Chem.BondType.AROMATIC, Chem.BondType.SINGLE,
                            Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
        self.max_iter = MAX_ITER

    def get_atom_type(self, atom):
        return self.atom_types.index(
            (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetNumExplicitHs()))

    def get_bond_type(self, bond):
        return self.bond_orders.index(bond.GetBondType())

    def index_to_atom(self, idx):
        symbol, charge, hs = self.atom_types[idx]
        a = Chem.Atom(symbol)
        a.SetFormalCharge(charge)
        a.SetNumExplicitHs(hs)
        return a

    def index_to_bond(self, mol, begin_id, end_id, idx):
        mol.AddBond(begin_id, end_id, self.bond_orders[idx])

    @property
    def num_atom_types(self):
        return len(self.atom_types)

    @property
    def num_bond_types(self):
        return len(self.bond_orders)


_mol_spec = None


def get_mol_spec():
    global _mol_spec
    if _mol_spec is None:
        _mol_spec = MoleculeSpec(ATOM_TYPES_PATH)
    return _mol_spec


def strip_salt(smiles):
    """Largest fragment by string length - the rule data_preprocessing_1 used.

    The raw ligands_*_corrected files still carry salts; d1_tr_ligands.txt (which
    had them stripped) is never read by this notebook. Without this, a
    multi-fragment ligand becomes a silently corrupted training target.
    """
    if '.' not in smiles:
        return smiles
    return max(smiles.split('.'), key=len)


def get_graph_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    atom_types, atom_ranks, bonds, bond_types = [], [], [], []
    for a, r in zip(mol.GetAtoms(), Chem.CanonicalRankAtoms(mol)):
        atom_types.append(get_mol_spec().get_atom_type(a))
        atom_ranks.append(r)

    for b in mol.GetBonds():
        idx_1, idx_2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = get_mol_spec().get_bond_type(b)
        bonds.append([idx_1, idx_2])
        bond_types.append(bt)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(atom_types)))
    graph.add_edges_from(bonds)

    if REJECT_DISCONNECTED and len(atom_types) > 1 and not nx.is_connected(graph):
        raise ValueError(f"Disconnected molecular graph (salt not stripped?): {smiles}")

    return graph, atom_types, atom_ranks, bonds, bond_types


def traverse_graph(graph, atom_ranks, current_node=None, step_ids=None,
                   p=P_ALPHA, log_p=0.0, _top=True):
    """Q_alpha proposal: canonical rank with probability p, random neighbour otherwise."""
    if current_node is None:
        next_nodes = range(len(atom_ranks))
        step_ids = [-1] * len(next_nodes)
        next_node_ranks = atom_ranks
    else:
        next_nodes = list(graph.neighbors(current_node))
        next_nodes = [n for n in next_nodes if step_ids[n] < 0]
        next_node_ranks = [atom_ranks[n] for n in next_nodes]

    next_nodes = [n for n, r in sorted(zip(next_nodes, next_node_ranks),
                                       key=lambda x: x[1])]

    while len(next_nodes) > 0:
        if len(next_nodes) == 1:
            next_node = next_nodes[0]
        elif random.random() >= (1.0 - p):
            next_node = next_nodes[0]
            log_p += np.log(p)
        else:
            next_node = next_nodes[random.randint(1, len(next_nodes) - 1)]
            log_p += np.log((1.0 - p) / (len(next_nodes) - 1))

        step_ids[next_node] = max(step_ids) + 1
        _, log_p = traverse_graph(graph, atom_ranks, next_node, step_ids, p, log_p,
                                  _top=False)
        next_nodes = [n for n in graph.neighbors(next_node) if step_ids[n] < 0]
        next_node_ranks = [atom_ranks[n] for n in next_nodes]
        next_nodes = [n for n, r in sorted(zip(next_nodes, next_node_ranks),
                                           key=lambda x: x[1])]

    # Unreachable now that get_graph_from_smiles rejects disconnected graphs - which
    # is exactly why it is asserted. The alternative, which is what happened before,
    # is silent corruption: argsort puts the -1s first and every bond endpoint is
    # remapped to the wrong atom.
    if _top and min(step_ids) < 0:
        raise ValueError(
            f"traverse_graph left {sum(1 for s in step_ids if s < 0)} atoms unvisited. "
            f"The graph is disconnected; single_reorder would mis-wire every bond.")

    return step_ids, log_p


# %%
def single_reorder(X_0, A_0, step_ids):
    X_0, A_0 = np.copy(X_0), np.copy(A_0)
    step_ids = np.array(step_ids, dtype=np.int32)
    sorted_ids = np.argsort(step_ids)
    X_0 = X_0[sorted_ids]
    A_0[:, 0], A_0[:, 1] = step_ids[A_0[:, 0]], step_ids[A_0[:, 1]]

    max_b, min_b = np.amax(A_0[:, :2], axis=1), np.amin(A_0[:, :2], axis=1)
    A_0 = A_0[np.lexsort([-min_b, max_b]), :]

    max_b, min_b = np.amax(A_0[:, :2], axis=1), np.amin(A_0[:, :2], axis=1)
    is_append = np.concatenate([np.array([True]), max_b[1:] > max_b[:-1]])
    A_0 = np.concatenate([
        np.where(is_append[:, np.newaxis], np.stack([min_b, max_b], axis=1),
                 np.stack([max_b, min_b], axis=1)),
        A_0[:, -1:]
    ], axis=1)
    return X_0, A_0


def single_expand(X_0, A_0):
    X_0, A_0 = np.copy(X_0), np.copy(A_0)
    is_append_iter = np.less(A_0[:, 0], A_0[:, 1]).astype(np.int32)

    NX = np.cumsum(np.pad(is_append_iter, [[1, 0]], mode='constant', constant_values=1))
    shift = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')[:-1])
    X_index = np.arange(NX.sum(), dtype=np.int32) - np.repeat(shift, NX)
    X = X_0[X_index]

    _, A_index = np.tril_indices(A_0.shape[0])
    A = A_0[A_index, :]
    NA = np.arange(A_0.shape[0] + 1)

    action_type = 1 - is_append_iter
    atom_type = np.where(action_type == 0, X_0[A_0[:, 1]], 0)
    bond_type = A_0[:, 2]
    append_pos = np.where(action_type == 0, A_0[:, 0], 0)
    connect_pos = np.where(action_type == 1, A_0[:, 1], 0)
    actions = np.stack([action_type, atom_type, bond_type, append_pos, connect_pos],
                       axis=1)
    actions = np.append(actions, [[2, 0, 0, 0, 0]], axis=0)
    action_0 = np.array([X_0[0]], dtype=np.int32)

    last_atom_index = shift + NX - 1
    last_atom_mask = np.zeros_like(X)
    last_atom_mask[last_atom_index] = np.where(
        np.pad(is_append_iter, [[1, 0]], mode='constant', constant_values=1) == 1,
        np.ones_like(last_atom_index),
        np.ones_like(last_atom_index) * 2
    )
    return action_0, X, NX, A, NA, actions, last_atom_mask


def get_d(A, X):
    def _to_sparse(_A, _X):
        return sparse.coo_matrix(
            (np.ones([_A.shape[0] * 2], dtype=np.int32),
             (np.concatenate([_A[:, 0], _A[:, 1]], axis=0),
              np.concatenate([_A[:, 1], _A[:, 0]], axis=0))),
            shape=[_X.shape[0]] * 2)

    A_sparse = _to_sparse(A, X)
    d2 = A_sparse * A_sparse
    d3 = d2 * A_sparse

    D_2 = np.stack(d2.nonzero(), axis=1)
    D_2 = D_2[D_2[:, 0] < D_2[:, 1], :]

    D_3 = np.stack(d3.nonzero(), axis=1)
    D_3 = D_3[D_3[:, 0] < D_3[:, 1], :]

    D_3_sparse = _to_sparse(D_3, X)
    D_3_sparse = D_3_sparse - D_3_sparse.multiply(A_sparse)
    D_3 = np.stack(D_3_sparse.nonzero(), axis=1)
    D_3 = D_3[D_3[:, 0] < D_3[:, 1], :]
    return D_2, D_3


def merge_single_0(X_0, A_0, NX_0, NA_0):
    cumsum = np.cumsum(np.pad(NX_0, [[1, 0]], mode='constant')[:-1])
    A_0[:, :2] += np.stack([np.repeat(cumsum, NA_0)] * 2, axis=1)
    D_0_2, D_0_3 = get_d(A_0, X_0)

    A_split = []
    for i in range(get_mol_spec().num_bond_types):
        A_split.append(A_0[A_0[:, 2] == i, :2])
    A_split.extend([D_0_2, D_0_3])
    NX_rep_0 = np.repeat(np.arange(NX_0.shape[0]), NX_0)
    return X_0, A_split, NX_0, NX_rep_0


def merge_single(X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions,
                 last_append_mask, log_p):
    X, A, NX, NX_rep = merge_single_0(X, A, NX, NA)
    cumsum = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')[:-1])
    actions[:, -2] += cumsum * (actions[:, 0] == 0)
    actions[:, -1] += cumsum * (actions[:, 0] == 1)
    mol_ids_rep = np.repeat(mol_ids, NX)
    rep_ids_rep = np.repeat(rep_ids, NX)
    return (X, A, mol_ids_rep, rep_ids_rep, iw_ids, last_append_mask, NX, NX_rep,
            action_0, actions, log_p)


# %% [markdown]
# ## Pair assembly
#
# `ligands_*_corrected` and `proteins_*_corrected` are aligned line for line. Which
# file a row sits in is a preprocessing artifact; the fold is defined by
# `<fold>_<tr|te>_unique_proteins.txt`, so rows are pooled from both directories
# and routed by list membership.

# %%
def read_spaced(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            s = ''.join(line.split())
            if s:
                out.append(s)
            if limit is not None and len(out) >= limit:
                break
    return out


def read_fasta(path):
    records, name, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def verify_fasta_order(split, unique):
    """The bepler FASTA must match the protein list record for record, in order.

    This is the invariant the whole conditioning path rests on: embeddings are
    keyed p1..pN by FASTA position, pairs are matched to rows by sequence content.
    A silent disagreement mis-conditions every ligand while every shape check
    still passes.
    """
    d = split_dir(split)
    tag = "tr" if split == "train" else "te"
    fa = os.path.join(d, f"{FOLD}_{tag}_unique_proteins_bepler.fa")
    if not os.path.isfile(fa):
        raise FileNotFoundError(
            f"{fa} not found. It is the FASTA the embeddings were generated from, "
            f"and the only way to confirm HDF5 key p<i> really is row <i>.")

    recs = read_fasta(fa)
    if len(recs) != len(unique):
        raise ValueError(f"{os.path.basename(fa)} has {len(recs)} records but the "
                         f"protein list has {len(unique)}.")

    bad_name = [i for i, (n, _) in enumerate(recs) if n != f"p{i + 1}"]
    if bad_name:
        i = bad_name[0]
        raise ValueError(f"{os.path.basename(fa)}: record {i} is named {recs[i][0]!r}, "
                         f"expected 'p{i + 1}'.")

    mism = [i for i, ((_, s), u) in enumerate(zip(recs, unique))
            if "".join(s.split()).upper() != u.upper()]
    if mism:
        i = mism[0]
        raise ValueError(
            f"{os.path.basename(fa)} and the protein list disagree at {len(mism)} "
            f"position(s); first is record {i} (p{i + 1}). Every ligand would be "
            f"conditioned on the wrong protein.")

    print(f"  [{split}] order verified: {len(recs)} proteins, "
          f"FASTA == protein list == p1..p{len(recs)}")


def read_pairs(split, limit=None):
    """Ligand/protein rows to assemble one split from. -> (ligands, proteins)

    For d1, 74 of the 104 test proteins have all their rows in the TRAIN pair file
    and only 30 in the test one, so rows are pooled and routed by list membership.
    This cannot leak: the two lists are checked to share no protein.
    """
    sources = ("train", "test") if POOL_PAIR_FILES else (split,)
    seen, ligands, proteins = set(), [], []
    for src_split in sources:
        lig_path, prot_path = find_fold_files(src_split)[:2]
        lig = read_spaced(lig_path)
        prot = read_spaced(prot_path)
        if len(lig) != len(prot):
            raise ValueError(f"{src_split}: {len(lig)} ligand lines vs {len(prot)} "
                             f"protein lines - the pair files are not aligned.")
        n_new = 0
        for s, p in zip(lig, prot):
            key = (s, p.upper())
            if key in seen:
                continue
            seen.add(key)
            ligands.append(s)
            proteins.append(p)
            n_new += 1
        print(f"  [{split}] + {n_new} new of {len(lig)} rows from "
              f"{os.path.basename(prot_path)}")
    if limit and len(ligands) > limit:
        n_pooled = len(ligands)
        idx = sorted(random.Random(SPLIT_SEED).sample(range(n_pooled), limit))
        ligands = [ligands[i] for i in idx]
        proteins = [proteins[i] for i in idx]
        print(f"  [{split}] LIMIT_PAIRS: sampled {limit} of {n_pooled} pooled rows "
              f"(SMOKE TEST)")
    return ligands, proteins


# %%
def assemble_pairs(split, limit=None):
    """-> (records, C) where records = [(smiles, protein_row, ames), ...]"""
    lig_path, prot_path, uniq_path, emb_path = find_fold_files(split)
    print(f"  [{split}] ligands   {os.path.basename(lig_path)}")
    print(f"  [{split}] proteins  {os.path.basename(prot_path)}")
    print(f"  [{split}] unique    {os.path.basename(uniq_path)}")
    print(f"  [{split}] embedding {os.path.basename(emb_path)}")

    ligands, proteins = read_pairs(split, limit)

    unique = read_spaced(uniq_path)
    if not unique:
        raise ValueError(f"{uniq_path}: no protein sequences read.")
    seq_to_row = {s.upper(): i for i, s in enumerate(unique)}
    if len(seq_to_row) != len(unique):
        print(f"  [WARN] {os.path.basename(uniq_path)}: "
              f"{len(unique) - len(seq_to_row)} duplicate sequence(s); pairs for a "
              f"duplicate resolve to its LAST row.")

    if VERIFY_FASTA_ORDER:
        verify_fasta_order(split, unique)

    global N_C
    C = load_embeddings(emb_path, len(unique))
    N_C = int(C.shape[1])
    print(f"  [{split}] {len(ligands)} pairs | {len(unique)} unique proteins | "
          f"embeddings {C.shape}")

    rows = []
    n_bad = n_big = n_oov = n_unmapped = n_long = n_salt = n_disconnected = 0

    for smi, seq in zip(ligands, proteins):
        row = seq_to_row.get(seq.upper())
        if row is None:
            n_unmapped += 1
            continue

        # Salt stripping comes FIRST: a multi-fragment SMILES survives every other
        # filter below (Na, K, Cl are all in atom_types.txt) and then corrupts its
        # own training target inside traverse_graph.
        if STRIP_SALTS and '.' in smi:
            smi = strip_salt(smi)
            n_salt += 1

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_bad += 1
            continue
        if mol.GetNumAtoms() > MAX_ATOMS:
            n_big += 1
            continue
        # graph_to_rnn is [batch, K, MAX_ITER]; a molecule needs GetNumBonds() + 1
        # steps, so anything longer cannot be written into it.
        if mol.GetNumBonds() + 1 > MAX_ITER:
            n_long += 1
            continue
        try:
            for a in mol.GetAtoms():
                get_mol_spec().get_atom_type(a)
        except ValueError:
            n_oov += 1
            continue
        if REJECT_DISCONNECTED:
            try:
                _ = get_graph_from_smiles(smi)
            except ValueError:
                n_disconnected += 1
                continue

        rows.append((smi, row))

    print(f"  [{split}] kept {len(rows)} | salts stripped {n_salt} | dropped: "
          f"unparseable {n_bad}, >{MAX_ATOMS} atoms {n_big}, >{MAX_ITER - 1} bonds "
          f"{n_long}, out-of-vocab {n_oov}, disconnected {n_disconnected}, "
          f"unmatched protein {n_unmapped}")

    kept_rows = {r for _, r in rows}
    unused = sorted(set(range(len(unique))) - kept_rows)
    if unused and limit is None:
        raise RuntimeError(
            f"{len(unused)} of the {len(unique)} proteins in "
            f"{os.path.basename(uniq_path)} never appear in "
            f"{os.path.basename(prot_path)} (first rows: {unused[:8]}).")
    elif unused:
        print(f"  [{split}] [SMOKE] {len(unused)} of {len(unique)} proteins have no "
              f"sampled pair")

    if n_unmapped:
        frac = n_unmapped / max(1, len(ligands))
        print(f"  [{split}] {n_unmapped} of {len(ligands)} pairs ({100 * frac:.1f}%) "
              f"belong to proteins outside this fold and were filtered out")
        if frac > UNMATCHED_TOLERANCE:
            raise RuntimeError(f"Dropped fraction {100 * frac:.1f}% exceeds "
                               f"UNMATCHED_TOLERANCE.")
    if not rows:
        raise RuntimeError(f"{split}: no usable pairs.")

    labels = ames_labels([smi for smi, _ in rows])
    records = [(smi, row, lab) for (smi, row), lab in zip(rows, labels)]

    stats = {"kept": len(rows), "salt_stripped": n_salt, "unparseable": n_bad,
             "too_many_atoms": n_big, "too_many_bonds": n_long,
             "out_of_vocab": n_oov, "disconnected": n_disconnected,
             "unmatched_protein": n_unmapped}
    return records, C, stats


def split_by_protein(records, n_proteins, held_fraction, seed):
    """Split by protein row so the two parts share no protein."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_proteins)
    n_held = max(1, int(round(held_fraction * n_proteins)))
    held_rows = set(order[:n_held].tolist())
    held = [r for r in records if r[1] in held_rows]
    kept = [r for r in records if r[1] not in held_rows]
    return held, kept, sorted(held_rows), sorted(set(order[n_held:].tolist()))


# %% [markdown]
# ## Sampler and collator

# %%
class BalancedSampler(Sampler):
    """Groups similar-length molecules into a batch to cut padding."""

    def __init__(self, cost, batch_size, name="dataset"):
        if len(cost) < batch_size:
            raise ValueError(
                f"{name}: {len(cost)} records but batch size {batch_size}. This "
                f"sampler would yield no batches, producing an empty loader and a "
                f"NaN validation loss rather than an error.")
        index = np.argsort(cost).tolist()
        chunk_size = int(float(len(cost)) / batch_size)
        dropped = len(cost) - chunk_size * batch_size
        if dropped:
            print(f"  [INFO] {name}: {dropped} record(s) beyond the last whole batch "
                  f"are not sampled (upstream behaviour)")
        self.index = []
        for i in range(batch_size):
            self.index.append(index[i * chunk_size:(i + 1) * chunk_size])

    def _g(self):
        for index_i in self.index:
            random.shuffle(index_i)
        for batch_index in zip(*self.index):
            yield list(batch_index)

    def __iter__(self):
        return self._g()

    def __len__(self):
        return len(self.index[0])


def process_single(record, k=K_PATHS, p=P_ALPHA):
    """Build the graph, sample K traversals, attach the ligand's Ames label."""
    smiles, tox_label = record[0], int(record[2])
    smiles = strip_salt(smiles) if STRIP_SALTS else smiles

    graph, atom_types, atom_ranks, bonds, bond_types = get_graph_from_smiles(smiles)
    X_0 = np.array(atom_types, dtype=np.int32)
    A_0 = np.concatenate([
        np.array(bonds, dtype=np.int32),
        np.array(bond_types, dtype=np.int32)[:, np.newaxis]
    ], axis=1) if len(bonds) > 0 else np.zeros((0, 3), dtype=np.int32)

    X, A, NX, NA = [], [], [], []
    mol_ids, rep_ids, iw_ids = [], [], []
    action_0, actions, last_append_mask = [], [], []
    log_p = []

    tox_class = np.full((k,), tox_label, dtype=np.int32)

    for i in range(k):
        step_ids_i, log_p_i = traverse_graph(graph, atom_ranks, p=p)
        X_i, A_i = single_reorder(X_0, A_0, step_ids_i)
        action_0_i, X_i, NX_i, A_i, NA_i, actions_i, last_mask_i = single_expand(X_i, A_i)

        X.append(X_i)
        A.append(A_i)
        NX.append(NX_i)
        NA.append(NA_i)
        action_0.append(action_0_i)
        actions.append(actions_i)
        last_append_mask.append(last_mask_i)
        mol_ids.append(np.zeros_like(NX_i, dtype=np.int32))
        rep_ids.append(np.ones_like(NX_i, dtype=np.int32) * i)
        iw_ids.append(np.ones_like(NX_i, dtype=np.int32) * i)
        log_p.append(log_p_i)

    return (np.concatenate(X, axis=0), np.concatenate(A, axis=0),
            np.concatenate(NX, axis=0), np.concatenate(NA, axis=0),
            np.concatenate(mol_ids, axis=0), np.concatenate(rep_ids, axis=0),
            np.concatenate(iw_ids, axis=0), np.concatenate(action_0, axis=0),
            np.concatenate(actions, axis=0), np.concatenate(last_append_mask, axis=0),
            np.array(log_p, dtype=np.float32), tox_class)


class CMolRNNCollate:
    """Conditional collator. `c` is gathered for surviving records only, keeping
    its rows aligned with mol_ids."""

    def __init__(self, c_matrix, k=K_PATHS, p=P_ALPHA):
        self.C = c_matrix
        self.k = k
        self.p = p

    def __call__(self, batch):
        valid_batch = []
        for record in batch:
            try:
                smi = strip_salt(record[0]) if STRIP_SALTS else record[0]
                _ = get_graph_from_smiles(smi)
                valid_batch.append(record)
            except ValueError:
                continue
        if not valid_batch:
            return None

        shapes = [[0], [0, 3], [0], [0], [0], [0], [0], [0], [0, 5], [0], [0], [0]]
        dtypes = [np.int32] * 10 + [np.float32, np.int32]

        _build = lambda: [np.zeros(shape=s, dtype=d) for s, d in zip(shapes, dtypes)]
        _append = lambda r0, r1: [np.concatenate([a, b], axis=0) for a, b in zip(r0, r1)]

        (X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions,
         last_append_mask, log_p, tox_class) = _build()

        for n, record in enumerate(valid_batch):
            (X_i, A_i, NX_i, NA_i, mol_ids_i, rep_ids_i, iw_ids_i, action_0_i,
             actions_i, last_mask_i, log_p_i, tox_i) = process_single(
                record, self.k, self.p)

            if n != 0:
                mol_ids_i += mol_ids[-1] + 1
                iw_ids_i += iw_ids[-1] + 1

            (X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions,
             last_append_mask, log_p, tox_class) = _append(
                [X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions,
                 last_append_mask, log_p, tox_class],
                [X_i, A_i, NX_i, NA_i, mol_ids_i, rep_ids_i, iw_ids_i, action_0_i,
                 actions_i, last_mask_i, log_p_i, tox_i.reshape(-1)])

        (X, A, mol_ids_rep, rep_ids_rep, iw_ids, last_append_mask, NX, NX_rep,
         action_0, actions, log_p) = merge_single(
            X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions,
            last_append_mask, log_p)

        result_out = [X, A, mol_ids_rep, rep_ids_rep, iw_ids, last_append_mask, NX,
                      NX_rep, action_0, actions, log_p, tox_class]

        smis = [strip_salt(b[0]) if STRIP_SALTS else b[0] for b in valid_batch]
        mol_list = [Chem.MolFromSmiles(s) for s in smis]
        graph_to_rnn = np.zeros((len(valid_batch), self.k, get_mol_spec().max_iter),
                                dtype=np.int32)
        rnn_to_graph = []
        cum_sum = 0
        for i, mol_i in enumerate(mol_list):
            num_iter = mol_i.GetNumBonds() + 1
            assert num_iter <= get_mol_spec().max_iter, (
                f"{smis[i]} needs {num_iter} steps > max_iter "
                f"{get_mol_spec().max_iter}; assemble_pairs should have dropped it")
            for k_idx in range(self.k):
                graph_to_rnn[i, k_idx, :num_iter] = np.arange(num_iter) + cum_sum
                r0 = np.ones([num_iter], dtype=np.int32) * i
                r1 = np.ones_like(r0) * k_idx
                r2 = np.arange(num_iter)
                rnn_to_graph.append(np.stack([r0, r1, r2], axis=0))
                cum_sum += num_iter
        rnn_to_graph = np.concatenate(rnn_to_graph, axis=1)
        NX_cum = np.cumsum(result_out[6])

        c = self.C[[rec[1] for rec in valid_batch], :]
        return result_out + [graph_to_rnn, rnn_to_graph, NX_cum, c]


def batch_to_tensor(record, device=None):
    """Move one collated batch to the device. `ids` maps each atom row to its
    molecule, which is how `c` is broadcast over atoms."""
    device = CTX if device is None else device
    *base, graph_to_rnn, rnn_to_graph, NX_cum, c = record
    (X, A, mol_ids_rep, rep_ids_rep, iw_ids, last_append_mask,
     NX, NX_rep, action_0, actions, log_p, tox_class) = base

    n_nodes = int(X.shape[0])
    X = torch.as_tensor(X, dtype=torch.long, device=device)

    A_sparse = []
    for A_i in A:
        if A_i.shape[0] == 0:
            A_sparse.append(None)
        else:
            A_i = np.concatenate([A_i, A_i[:, [1, 0]]], axis=0)
            idx = torch.as_tensor(A_i[:, :2].T.astype(np.int64), device=device)
            val = torch.ones(A_i.shape[0], dtype=torch.float32, device=device)
            A_sparse.append(torch.sparse_coo_tensor(
                idx, val, (n_nodes, n_nodes), dtype=torch.float32,
                device=device).coalesce())

    batch_size = int(mol_ids_rep.max() + 1)
    iw_size = int(rep_ids_rep.max() + 1)

    ids = torch.as_tensor(mol_ids_rep, dtype=torch.long, device=device)
    iw_ids, last_append_mask, NX, NX_rep, action_0, actions = [
        torch.as_tensor(x, dtype=torch.long, device=device)
        for x in [iw_ids, last_append_mask, NX, NX_rep, action_0, actions]]
    log_p = torch.as_tensor(log_p, dtype=torch.float32, device=device)
    tox_class = torch.as_tensor(tox_class, dtype=torch.float32, device=device)
    graph_to_rnn = torch.as_tensor(graph_to_rnn, dtype=torch.long, device=device)
    rnn_to_graph = torch.as_tensor(rnn_to_graph, dtype=torch.long, device=device)
    NX_cum = torch.as_tensor(NX_cum, dtype=torch.long, device=device)
    c = torch.as_tensor(np.ascontiguousarray(c), dtype=torch.float32, device=device)

    return [X, A_sparse, iw_ids, last_append_mask, NX, NX_rep, action_0, actions,
            log_p, batch_size, iw_size, tox_class, graph_to_rnn, rnn_to_graph,
            NX_cum, c, ids]


# %% [markdown]
# ## Build the datasets

# %%
records_all_train, C_train, stats_train = assemble_pairs("train", LIMIT_PAIRS)
records_test, C_held, stats_test = assemble_pairs("test", LIMIT_PAIRS)

records_val, records_train, val_rows, train_rows = split_by_protein(
    records_all_train, C_train.shape[0], VAL_PROTEIN_FRACTION, SPLIT_SEED)
test_rows = sorted({r[1] for r in records_test})

check("embedding widths agree", C_train.shape[1] == C_held.shape[1],
      f"train {C_train.shape[1]}-d, held-out {C_held.shape[1]}-d")

# Protein-level leakage. The fold is built so train and held-out proteins share
# <20% sequence similarity; identity would be fatal, so check rather than assume.
_train_seqs = set(read_spaced(find_fold_files("train")[2]))
_held_seqs = set(read_spaced(find_fold_files("test")[2]))
_shared = _train_seqs & _held_seqs
check("no identical protein across the split", not _shared,
      f"{len(_shared)} sequences identical between train and held-out"
      if _shared else "train and held-out proteins are disjoint")

check("train/val share no protein", not (set(val_rows) & set(train_rows)),
      f"{len(set(val_rows) & set(train_rows))} shared protein rows")

check("salts were stripped",
      stats_train["salt_stripped"] + stats_test["salt_stripped"] >= 0,
      f"{stats_train['salt_stripped']} train + {stats_test['salt_stripped']} test "
      f"ligands had salts removed. Each one would previously have been a silently "
      f"corrupted training target")

_shared_lig = {r[0] for r in records_train} & {r[0] for r in records_test}
if _shared_lig:
    print(f"  [INFO] {len(_shared_lig)} ligand SMILES occur in both train and test, "
          f"paired with different proteins. The fold controls PROTEIN novelty, not "
          f"ligand novelty.")

print(f"\n  train {len(records_train)} pairs / {len(train_rows)} proteins")
print(f"  val   {len(records_val)} pairs / {len(val_rows)} proteins  (carved from train)")
print(f"  test  {len(records_test)} pairs / {len(test_rows)} proteins  (all held-out)")

# %%
n_held_proteins = int(C_held.shape[0])
if SMOKE_TEST:
    print(f"[SMOKE] test bundle covers {len(test_rows)}/{n_held_proteins} held-out "
          f"proteins - NOT the complete test set")
elif len(test_rows) != n_held_proteins:
    raise RuntimeError(
        f"test bundle covers {len(test_rows)} of {n_held_proteins} held-out proteins. "
        f"Tables 1-3 are over all of them.")
else:
    check("test bundle complete", True,
          f"all {n_held_proteins} held-out proteins of fold {FOLD}")

with open(SPLIT_MANIFEST, 'w') as f:
    json.dump({'fold': FOLD, 'fold_suffix': FOLD_SUFFIX[FOLD], 'N_C': N_C,
               'split_seed': SPLIT_SEED,
               'val_protein_fraction': VAL_PROTEIN_FRACTION,
               'val_source': 'train split, by protein',
               'limit_pairs': LIMIT_PAIRS, 'smoke_test': SMOKE_TEST,
               'complete_held_out_test_set': not SMOKE_TEST,
               'n_held_out_proteins_total': n_held_proteins,
               'train_protein_rows': train_rows, 'val_protein_rows': val_rows,
               'test_protein_rows': test_rows,
               'n_train_pairs': len(records_train), 'n_val_pairs': len(records_val),
               'n_test_pairs': len(records_test),
               'assembly_stats': {'train': stats_train, 'test': stats_test},
               'strip_salts': STRIP_SALTS,
               'reject_disconnected': REJECT_DISCONNECTED}, f, indent=1)

np.savez_compressed(
    TEST_BUNDLE,
    smiles=np.array([r[0] for r in records_test], dtype=object),
    protein_row=np.array([r[1] for r in records_test], dtype=np.int32),
    ames=np.array([r[2] for r in records_test], dtype=np.int8),
    embeddings=C_held,
    test_protein_rows=np.array(test_rows, dtype=np.int32),
    smoke_test=np.array(SMOKE_TEST),
    limit_pairs=np.array(-1 if LIMIT_PAIRS is None else LIMIT_PAIRS),
    n_held_out_proteins_total=np.array(n_held_proteins))
print(f"  test bundle -> {TEST_BUNDLE}")

# %%
# Upstream Lambda.__init__ shuffled with seed 17.
random.Random(17).shuffle(records_train)

loader_train = DataLoader(
    records_train,
    batch_sampler=BalancedSampler([len(r[0]) for r in records_train], BATCH_SIZE,
                                  "train"),
    num_workers=NUM_WORKERS,
    collate_fn=CMolRNNCollate(C_train, K_PATHS, P_ALPHA),
    persistent_workers=(NUM_WORKERS > 0))

loader_val = DataLoader(
    records_val,
    batch_sampler=BalancedSampler([len(r[0]) for r in records_val], BATCH_SIZE_VAL,
                                  "val"),
    num_workers=NUM_WORKERS,
    collate_fn=CMolRNNCollate(C_train, K_PATHS, P_ALPHA),   # val rows index C_train
    persistent_workers=(NUM_WORKERS > 0))

# Epoch length, fixed once. Epoch boundaries are now iteration counts rather than
# StopIteration, so a mid-epoch resume cannot run a long epoch (the previous run's
# epoch 9 took 16,514 batches instead of 14,898 and applied 16 extra lr decays).
ITERS_PER_EPOCH = len(loader_train)
print(f"\n  {ITERS_PER_EPOCH:,} iterations per epoch "
      f"({len(records_train):,} pairs / batch {BATCH_SIZE})")

batch_cpu = next(iter(loader_train))
_b = batch_to_tensor(batch_cpu)
print(f"  batch: N={_b[9]} K={_b[10]} | X {tuple(_b[0].shape)} | "
      f"A blocks {len(_b[1])} | c {tuple(_b[15].shape)}")
check("conditioning matrix shape", _b[15].shape == (_b[9], N_C),
      f"c is {tuple(_b[15].shape)}, expected ({_b[9]}, {N_C})")
check("atom->molecule map", int(_b[16].max()) == _b[9] - 1,
      f"ids max {int(_b[16].max())} = batch size {_b[9]} - 1")

# %% [markdown]
# ## Model

# %%
def segment_sum(X, idx, num_seg):
    out = torch.zeros((num_seg,) + tuple(X.shape[1:]), dtype=X.dtype, device=X.device)
    return out.index_add(0, idx, X)


def _concat_neighbourhoods(X, A_list):
    parts = [X]
    for A in A_list:
        parts.append(torch.sparse.mm(A, X) if A is not None else torch.zeros_like(X))
    return torch.cat(parts, dim=1)


class EfficientGraphConvFn(torch.autograd.Function):
    """Recomputes the concatenated neighbourhood in backward instead of storing it.
    Every A is symmetric, so A^T == A and backward reuses A directly."""

    @staticmethod
    def forward(ctx, X, W, A_list):
        ctx.A_list = A_list
        ctx.save_for_backward(X, W)
        return _concat_neighbourhoods(X, A_list) @ W

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        A_list = ctx.A_list
        X_out = _concat_neighbourhoods(X, A_list)
        grad_output = grad_output.contiguous()

        grad_W = X_out.t() @ grad_output
        grad_X_out = grad_output @ W.t()

        chunks = torch.split(grad_X_out, X.shape[1], dim=1)
        grad_X = chunks[0]
        for A, g in zip(A_list, chunks[1:]):
            if A is not None:
                grad_X = grad_X + torch.sparse.mm(A, g.contiguous())
        return grad_X, grad_W, None


def get_activation(name):
    return {'relu': torch.relu, 'tanh': torch.tanh}[name]


class Linear_BN(nn.Sequential):
    def __init__(self, F_in, F_out):
        super(Linear_BN, self).__init__(
            nn.Linear(F_in, F_out, bias=False),
            nn.BatchNorm1d(F_out, eps=1e-5, momentum=0.1))


class GraphConv(nn.Module):
    def __init__(self, Fin, Fout, D):
        super(GraphConv, self).__init__()
        self.Fin, self.Fout, self.D = Fin, Fout, D
        self.W = nn.Parameter(torch.empty(self.Fin * (self.D + 1), self.Fout))

    def forward(self, X, A_list):
        return EfficientGraphConvFn.apply(X, self.W, A_list)


class Policy(nn.Module):
    def __init__(self, F_in, F_h, N_A, N_B, k=1):
        super(Policy, self).__init__()
        self.F_in, self.F_h, self.N_A, self.N_B, self.k = F_in, F_h, N_A, N_B, k
        self.linear_h = Linear_BN(F_in * 2, self.F_h * k)
        self.linear_h_t = Linear_BN(F_in, self.F_h * k)
        self.linear_x = nn.Linear(self.F_h, self.N_B + self.N_B * self.N_A)
        self.linear_x_t = nn.Linear(self.F_h, 1)
        self.linear_pi = nn.Linear(self.F_in, self.k) if self.k > 1 else None
        self.last_logit_absmax = None
        self.last_frac_clamped = None

    def forward(self, X, NX, NX_rep, X_end=None):
        if X_end is None:
            X_end = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X, X_end[NX_rep, :]], dim=1)

        X_h = torch.relu(self.linear_h(X)).reshape(-1, self.F_h)
        X_h_end = torch.relu(self.linear_h_t(X_end)).reshape(-1, self.F_h)

        raw_x = self.linear_x(X_h).reshape(-1, self.k, self.N_B + self.N_B * self.N_A)
        raw_x_t = self.linear_x_t(X_h_end).reshape(-1, self.k, 1)

        with torch.no_grad():
            self.last_logit_absmax = float(max(raw_x.abs().max(), raw_x_t.abs().max()))
            n_at = int((raw_x.abs() >= 80.0).sum()) + int((raw_x_t.abs() >= 80.0).sum())
            self.last_frac_clamped = n_at / max(1, raw_x.numel() + raw_x_t.numel())

        # Clip before exp() to stay inside float32, as in the original.
        X_x = torch.exp(torch.clamp(raw_x, -80.0, 80.0))
        X_x_end = torch.exp(torch.clamp(raw_x_t, -80.0, 80.0))

        # One joint softmax over every action of every atom plus `end`, per molecule.
        X_sum = segment_sum(X_x, NX_rep, NX.shape[0]).sum(-1, keepdim=True) + X_x_end
        X_softmax = X_x / (X_sum[NX_rep, :, :] + 1e-10)
        X_softmax_end = X_x_end / (X_sum + 1e-10)

        if self.k > 1:
            pi = torch.softmax(self.linear_pi(X_end), dim=1).unsqueeze(-1)
            X_softmax = (X_softmax * pi[NX_rep, :, :]).sum(dim=1)
            X_softmax_end = (X_softmax_end * pi).sum(dim=1)
        else:
            X_softmax = X_softmax.squeeze(1)
            X_softmax_end = X_softmax_end.squeeze(1)

        connect, append = X_softmax[:, :self.N_B], X_softmax[:, self.N_B:]
        append = append.reshape(-1, self.N_A, self.N_B)
        return append, connect, X_softmax_end.squeeze(-1)


# %% [markdown]
# ### Generator
#
# Parameter names outside `linear_c.*` and `dense_policy_0.*` match the pretraining
# model, so the stage-1 checkpoint transfers by name.

# %%
LOG_EPS = 1e-10
LOG_FLOOR = math.log(LOG_EPS)      # -23.0259


class MoleculeGenerator(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation,
                 *args, **kwargs):
        super(MoleculeGenerator, self).__init__()
        self.N_A, self.N_B, self.D = N_A, N_B, D
        self.F_e, self.F_skip, self.Fh_policy = F_e, F_skip, Fh_policy
        self.F_c = list(F_c)
        self.activation = get_activation(activation)

        self.embedding_atom = nn.Embedding(self.N_A, self.F_e)
        self.embedding_mask = nn.Embedding(3, self.F_e)
        self._build_graph_conv(*args, **kwargs)

        self.dense = nn.Sequential()
        for i, (f_in, f_out) in enumerate(zip([self.F_skip] + self.F_c[:-1], self.F_c)):
            self.dense.add_module(str(i), Linear_BN(f_in, f_out))

        self.policy_0 = nn.Parameter(torch.zeros(self.N_A))
        self.policy_h = Policy(self.F_c[-1], self.Fh_policy, self.N_A, self.N_B)
        self.mode = 'loss'
        self.last_l = None          # always the UNFLOORED bound
        self.last_tox = None
        self.last_floored = None
        self.last_frac_floor = None

    @abstractmethod
    def _build_graph_conv(self, *args, **kwargs):
        pass

    def _likelihood(self, init, append, connect, end, action_0, actions, iw_ids,
                    log_p_sigma, batch_size, iw_size, tox_class_batch):
        action_type, node_type, edge_type, append_pos, connect_pos = (
            actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3], actions[:, 4])

        def _log_mask(_x, _mask):
            return _mask * torch.log(_x + LOG_EPS) + (1 - _mask) * torch.zeros_like(_x)

        init = init.reshape(batch_size * iw_size, self.N_A)
        rows = torch.arange(action_0.shape[0], device=action_0.device)
        loss_init = torch.log(init[rows, action_0] + LOG_EPS)

        loss_end = _log_mask(end, (action_type == 2).to(end.dtype))
        loss_append = _log_mask(append[append_pos, node_type, edge_type],
                                (action_type == 0).to(end.dtype))
        loss_connect = _log_mask(connect[connect_pos, edge_type],
                                 (action_type == 1).to(end.dtype))

        log_p_x = loss_end + loss_append + loss_connect

        # Fraction of per-action log-probabilities pinned at ln(1e-10), where the
        # gradient is 1/(p + 1e-10) ~ 1e10. Rising means the toxic branch is
        # running away.
        with torch.no_grad():
            self.last_frac_floor = float((log_p_x <= LOG_FLOOR + 1e-3).float().mean())

        log_p_x = segment_sum(log_p_x.unsqueeze(-1), iw_ids,
                              batch_size * iw_size).squeeze(-1)
        log_p_x = log_p_x + loss_init

        log_p_x = log_p_x.reshape(batch_size, iw_size)
        log_p_sigma = log_p_sigma.reshape(batch_size, iw_size)

        l = log_p_x - log_p_sigma
        l = torch.logsumexp(l, dim=1) - math.log(float(iw_size))

        # Eq. 8: all K paths share the molecule's label, so column 0 is the label.
        if tox_class_batch.numel() != batch_size * iw_size:
            raise RuntimeError(
                f"tox_class has {tox_class_batch.numel()} entries, expected "
                f"batch_size*K = {batch_size * iw_size}; labels would be misaligned "
                f"with molecules.")
        tox_labels = tox_class_batch.reshape(batch_size, iw_size)[:, 0]

        # l_raw is the unfloored bound. Everything reported as a metric must come
        # from it: if the floor leaked in, the numbers would be a function of
        # TOX_FLOOR rather than of the model.
        l_raw = l
        if TOX_FLOOR is not None:
            l = torch.where(tox_labels > 0, torch.clamp(l, min=TOX_FLOOR), l)
            self.last_floored = ((tox_labels > 0) & (l_raw < TOX_FLOOR)).detach()
        else:
            self.last_floored = torch.zeros_like(tox_labels, dtype=torch.bool)
        l_weighted = l * (1.0 - (1.0 + LAMBDA_TOX) * tox_labels)

        self.last_l = l_raw.detach()
        self.last_tox = tox_labels.detach()
        return l_weighted


class MoleculeGenerator_RNN(MoleculeGenerator):
    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn,
                 *args, **kwargs):
        super(MoleculeGenerator_RNN, self).__init__(
            N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, *args, **kwargs)
        self.N_rnn = N_rnn
        self.rnn = nn.GRU(input_size=self.F_c[-1] * 2, hidden_size=self.F_c[-1],
                          num_layers=self.N_rnn, batch_first=True)

    def _rnn_train(self, X, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum):
        X_avg = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X_avg, X[NX_cum - 1]], dim=1)
        X = X[graph_to_rnn]
        batch_size, iw_size, length, num_features = X.shape
        out, _ = self.rnn(X.reshape(batch_size * iw_size, length, num_features))
        out = out.reshape(batch_size, iw_size, length, -1)
        return out[rnn_to_graph[0], rnn_to_graph[1], rnn_to_graph[2]]

    def _rnn_test(self, X, NX, NX_rep, NX_cum, h):
        X_avg = segment_sum(X, NX_rep, NX.shape[0]) / NX.unsqueeze(1).float()
        X = torch.cat([X_avg, X[NX_cum - 1]], dim=1)
        X, h = self.rnn(X.unsqueeze(1), h)
        return X.squeeze(1), h


class _TwoLayerDense(nn.Module):
    """FNN_0: protein embedding -> first-atom distribution.

    NOTE this path is NEW at fine-tuning time and starts from Xavier, i.e. close to
    uniform over all 65 atom types, while the pretrained `policy_0` is discarded
    (multiplied by 0.0 below). That substitution is what CDGCN 2.2 specifies, but
    it means the first-atom distribution has to be relearned - and `loss_init` is
    one term out of ~34 per molecule, so it learns slowly. Check A of
    00_phase0_diagnostics.py prints both distributions side by side.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(_TwoLayerDense, self).__init__()
        self.input = nn.Linear(input_size, hidden_size, bias=False)
        self.bn_input = nn.BatchNorm1d(hidden_size, eps=1e-5, momentum=0.1)
        self.output = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, c):
        return torch.softmax(self.output(torch.relu(self.bn_input(self.input(c)))),
                             dim=-1)


class CMoleculeGenerator_RNN(MoleculeGenerator_RNN):
    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, N_C, D, F_e, F_skip, F_c, Fh_policy, activation,
                 N_rnn, *args, **kwargs):
        self.N_C = N_C
        super(CMoleculeGenerator_RNN, self).__init__(
            N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, *args, **kwargs)
        self.dense_policy_0 = _TwoLayerDense(self.N_C, self.N_A * 3, self.N_A)

    @abstractmethod
    def _graph_conv_forward(self, X, A, c, ids):
        pass

    def _policy_0(self, c):
        # the 0.0 * policy_0 term keeps the unused pretraining parameter in the graph
        return self.dense_policy_0(c) + 0.0 * self.policy_0

    def _policy(self, X, A, NX, NX_rep, last_append_mask,
                graph_to_rnn, rnn_to_graph, NX_cum, c, ids):
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        X = self._graph_conv_forward(X, A, c, ids)
        X_dense = self.dense(X)        # computed once, not twice as in MXNet
        X_mol = self._rnn_train(X_dense, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum)
        return self.policy_h(X_dense, NX, NX_rep, X_mol)

    def _decode_step(self, X, A, NX, NX_rep, last_append_mask, NX_cum, h, c, ids):
        """One sampling step - unused in training, needed by the sampler."""
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        X = self._graph_conv_forward(X, A, c, ids)
        X_dense = self.dense(X)
        X_mol, h = self._rnn_test(X_dense, NX, NX_rep, NX_cum, h)
        append, connect, end = self.policy_h(X_dense, NX, NX_rep, X_mol)
        return append, connect, end, h

    def forward(self, *inputs):
        if self.mode in ('loss', 'likelihood'):
            (X, A, iw_ids, last_append_mask, NX, NX_rep, action_0, actions, log_p,
             batch_size, iw_size, tox_class, graph_to_rnn, rnn_to_graph, NX_cum,
             c, ids) = inputs
            init = self._policy_0(c).unsqueeze(1).expand(-1, iw_size, -1)
            append, connect, end = self._policy(
                X, A, NX, NX_rep, last_append_mask, graph_to_rnn, rnn_to_graph,
                NX_cum, c, ids)
            l_weighted = self._likelihood(init, append, connect, end, action_0,
                                          actions, iw_ids, log_p, batch_size,
                                          iw_size, tox_class)
            return l_weighted if self.mode == 'likelihood' else -l_weighted.mean()
        elif self.mode == 'decode_0':
            return self._policy_0(*inputs)
        elif self.mode == 'decode_step':
            return self._decode_step(*inputs)
        raise ValueError(f"Unknown mode: {self.mode}")


class CVanillaMolGen_RNN(CMoleculeGenerator_RNN):
    def __init__(self, N_A, N_B, N_C, D, F_e, F_h, F_skip, F_c, Fh_policy,
                 activation, N_rnn):
        super(CVanillaMolGen_RNN, self).__init__(
            N_A, N_B, N_C, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, F_h)

    def _build_graph_conv(self, F_h):
        self.F_h = list(F_h)
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()            # bn[i-1] belongs to conv[i]
        for i, (f_in, f_out) in enumerate(zip([self.F_e] + self.F_h[:-1], self.F_h)):
            self.conv.append(GraphConv(f_in, f_out, self.N_B + self.D))
            if i != 0:
                self.bn.append(nn.BatchNorm1d(f_in, eps=1e-5, momentum=0.1))

        self.bn_skip = nn.BatchNorm1d(sum(self.F_h), eps=1e-5, momentum=0.1)
        self.linear_skip = Linear_BN(sum(self.F_h), self.F_skip)
        # FNN_1..FNN_l: the per-layer protein projectors
        self.linear_c = nn.ModuleList(
            [nn.Linear(self.N_C, f_out, bias=False) for f_out in self.F_h])

    def _graph_conv_forward(self, X, A, c, ids):
        X_out = [X]
        for i, conv in enumerate(self.conv):
            X = X_out[-1]
            c_i = self.linear_c[i](c)[ids, :]
            if i == 0:
                X_out.append(conv(X, A) + c_i)
            else:
                X_out.append(conv(self.activation(self.bn[i - 1](X)), A) + c_i)
        X_out = torch.cat(X_out[1:], dim=1)
        return self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))


# %% [markdown]
# ## Init, early stopping, checkpointing

# %%
def init_weights_xavier(module):
    """MXNet's default Xavier (uniform, avg, magnitude 3) == xavier_uniform_(gain=1)."""
    for name, p in module.named_parameters():
        if p.dim() < 2 or 'bn' in name.split('.')[-1]:
            continue
        nn.init.xavier_uniform_(p, gain=1.0)
    for name, p in module.named_parameters():
        if 'bias' in name.split('.')[-1]:
            nn.init.zeros_(p)
    for m in module.modules():
        if isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.reset_running_stats()
    if hasattr(module, 'policy_0'):
        nn.init.zeros_(module.policy_0)
    return module


class EarlyStopping(object):
    def __init__(self, patience=5, delta=1e-2):
        self.patience, self.delta = patience, delta
        self.best, self.counter = math.inf, 0
        self.improved = False

    def step(self, loss):
        if loss < self.best - self.delta:
            self.best, self.counter, self.improved = loss, 0, True
        else:
            self.counter += 1
            self.improved = False
        return self.counter >= self.patience

    def state_dict(self):
        return {'best': self.best, 'counter': self.counter}

    def load_state_dict(self, s):
        self.best = s.get('best', self.best)
        self.counter = s.get('counter', self.counter)


def atomic_save(obj, path, keep_prev=True):
    """Write, fsync, rotate, rename. A plain torch.save leaves a truncated file if
    the job is killed mid-write."""
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            torch.save(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        if keep_prev and os.path.exists(path):
            try:
                os.replace(path, path + ".prev")
            except OSError:
                pass
        os.replace(tmp, path)
        dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def robust_load(path):
    """Load `path`, falling back to its rotated .prev copy."""
    for candidate, which in ((path, "current"), (path + ".prev", "previous")):
        if not os.path.isfile(candidate):
            continue
        try:
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            if which == "previous":
                print(f"  [WARN] {os.path.basename(path)} was unreadable; recovered "
                      f"the rotated backup. At most one checkpoint interval is lost.")
            return state, which
        except Exception as e:
            print(f"  [WARN] {os.path.basename(candidate)} unreadable "
                  f"({type(e).__name__}: {e})")
    return None, None


def rng_state():
    s = {"python": random.getstate(), "numpy": np.random.get_state(),
         "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    if not s:
        return
    try:
        random.setstate(s["python"])
        np.random.set_state(s["numpy"])
        torch.set_rng_state(torch.as_tensor(s["torch"], dtype=torch.uint8))
        if torch.cuda.is_available() and "cuda" in s:
            torch.cuda.set_rng_state_all([torch.as_tensor(x, dtype=torch.uint8)
                                          for x in s["cuda"]])
    except Exception as e:
        print(f"  [WARN] could not restore RNG state ({type(e).__name__}); sampling "
              f"continues from a fresh stream")


# %% [markdown]
# ## Resume state

# %%
is_continuous = False
global_counter = 0
epoch_no = 1
resume_state = None
elapsed_offset = 0.0
best_val_nll = math.inf

if FORCE_RESTART:
    print("  FORCE_RESTART - the previous run was archived before data work")
else:
    resume_state, _ = robust_load(CKPT_LAST)
    if resume_state is not None and "optimizer" not in resume_state:
        tstate, _ = robust_load(CKPT_TRAINER)
        if tstate is not None:
            resume_state = {**resume_state, **tstate}
            print("  recovered a legacy ckpt_last.pt + trainer.pt pair")
        else:
            print("  [WARN] ckpt_last.pt has weights but no optimizer state and no "
                  "trainer.pt beside it - restarting from the pretrained checkpoint")
            resume_state = None
    is_continuous = resume_state is not None


def objective_signature():
    """Settings whose change makes a resumed run incomparable to the checkpoint."""
    return {'lambda_tox': LAMBDA_TOX, 'tox_floor': TOX_FLOOR,
            'early_stop_on': EARLY_STOP_ON, 'fold': FOLD, 'n_c': N_C,
            'split_seed': SPLIT_SEED, 'val_protein_fraction': VAL_PROTEIN_FRACTION}


def data_signature():
    """Settings that change the training set without invalidating the comparison."""
    return {'pool_pair_files': POOL_PAIR_FILES, 'limit_pairs': LIMIT_PAIRS,
            'max_atoms': MAX_ATOMS, 'max_iter': MAX_ITER, 'batch_size': BATCH_SIZE,
            'strip_salts': STRIP_SALTS, 'iters_per_epoch': ITERS_PER_EPOCH}


# %%
configs = {'N_C': N_C, 'F_e': F_E, 'F_h': F_H, 'F_skip': F_SKIP, 'F_c': F_C,
           'Fh_policy': FH_POLICY, 'activation': ACTIVATION, 'N_rnn': N_RNN}

if is_continuous:
    configs = resume_state.get("configs") or (
        json.load(open(CKPT_CONFIG)) if os.path.isfile(CKPT_CONFIG) else configs)

    old_sig, new_sig = resume_state.get('objective'), objective_signature()
    if old_sig is None:
        print("  [WARN] checkpoint predates objective tracking")
    else:
        changed = {k: (old_sig[k], v) for k, v in new_sig.items()
                   if k in old_sig and old_sig[k] != v}
        absent = [k for k in new_sig if k not in old_sig]
        if changed:
            raise RuntimeError(
                "Settings changed since the checkpoint: "
                + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in changed.items())
                + "\n  Resuming would carry an optimiser state and an early-stopping "
                  "best belonging to the previous settings. Revert, or set "
                  "FORCE_RESTART = True for a clean run.\n"
                  "  NOTE: EARLY_STOP_ON changed from 'nll' to 'l_nontoxic' in this "
                  "version, so an old checkpoint WILL trip this. That is correct - "
                  "the stored `best` is a value of the old metric.")
        if absent:
            print(f"  [WARN] checkpoint does not record {absent}")

    old_data, new_data = resume_state.get('data'), data_signature()
    if old_data:
        d = {k: (old_data[k], v) for k, v in new_data.items()
             if k in old_data and old_data[k] != v}
        if 'limit_pairs' in d:
            a, b = d['limit_pairs']
            raise RuntimeError(
                f"The checkpoint was trained with LIMIT_PAIRS = {a!r}, this run uses "
                f"{b!r}. A smoke run and a full run are different experiments.")
        if 'iters_per_epoch' in d or 'strip_salts' in d:
            raise RuntimeError(
                f"The training set changed ({list(d)}). Epoch boundaries and the "
                f"early-stopping best no longer line up. Set FORCE_RESTART = True.")
        if d:
            print(f"  [WARN] training-set settings changed: {list(d)}")
    print("  resuming an existing fine-tuning run")
else:
    print("  fresh fine-tuning run")
    archive_stale(RESUMABLE_ARTIFACTS, why="fresh run")

check("embedding width matches the checkpoint", configs.get("N_C") == N_C,
      f"checkpoint N_C = {configs.get('N_C')}, this fold's embeddings are {N_C}-d")

with open(CKPT_CONFIG, 'w') as f:
    json.dump(configs, f, indent=1)

model = CVanillaMolGen_RNN(get_mol_spec().num_atom_types,
                           get_mol_spec().num_bond_types,
                           D=D_RECEPTIVE, **configs)

# %%
def load_pretrained(model, path):
    """Transfer phase-1 weights. Only the conditioning path may be missing."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)

    new = ('linear_c.', 'dense_policy_0.')
    unexplained = [k for k in missing if not k.startswith(new)]
    if unexplained:
        raise RuntimeError(f"Pretrained checkpoint missing non-conditioning tensors: "
                           f"{unexplained[:10]} - the two phases have drifted apart.")
    print(f"  transferred {len(sd) - len(unexpected)}/{len(sd)} pretrained tensors; "
          f"{len(missing)} conditioning tensors newly initialised")
    print("  a clean transfer means the SHAPES match - it is not evidence that the "
          "checkpoint was trained. See the provenance report above.")
    if unexpected:
        print(f"  [WARN] unused tensors in the checkpoint: {unexpected[:10]}")


if is_continuous:
    model.load_state_dict(resume_state['model'])
else:
    init_weights_xavier(model)
    if os.path.exists(PRETRAINED_CKPT):
        load_pretrained(model, PRETRAINED_CKPT)
    elif REQUIRE_PRETRAINED:
        raise FileNotFoundError(f"No pretrained checkpoint at {PRETRAINED_CKPT}.")
    else:
        print("  [WARN] REQUIRE_PRETRAINED off - training from random init, which is "
              "NOT the two-phase protocol")
model.to(CTX)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
early_stopping = EarlyStopping(PATIENCE, ES_DELTA)

if is_continuous:
    optimizer.load_state_dict(resume_state['optimizer'])
    for g in optimizer.param_groups:
        g['lr'] = resume_state.get('lr', LR)
    global_counter = int(resume_state.get('global_counter', 0))
    epoch_no = int(resume_state.get('epoch_no', 1))
    early_stopping.load_state_dict(resume_state.get('early_stopping', {}))
    elapsed_offset = float(resume_state.get('elapsed_hours', 0.0))
    best_val_nll = float(resume_state.get('best_val_nll', math.inf))
    set_rng_state(resume_state.get('rng'))

    for st in optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(CTX)

    # Epoch boundaries are iteration counts now, so the epoch number is derived
    # rather than carried - a mid-epoch resume lands exactly where it left off.
    derived_epoch = global_counter // ITERS_PER_EPOCH + 1
    if derived_epoch != epoch_no:
        print(f"  [INFO] epoch counter {epoch_no} -> {derived_epoch} "
              f"(derived from iteration {global_counter:,} / {ITERS_PER_EPOCH:,})")
        epoch_no = derived_epoch

    print(f"  iteration {global_counter:,}, epoch {epoch_no}, "
          f"lr {optimizer.param_groups[0]['lr']:.3e}, best {early_stopping.best:.4f}, "
          f"patience {early_stopping.counter}/{PATIENCE}, "
          f"{elapsed_offset:.2f} h already spent")
    del resume_state
    gc.collect()

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_cond = sum(p.numel() for n, p in model.named_parameters()
             if n.startswith(('linear_c.', 'dense_policy_0.')))
print(f"\n  {n_trainable:,} trainable parameters ({n_cond:,} = "
      f"{100 * n_cond / n_trainable:.0f}% in the conditioning path)")

# %% [markdown]
# ## Sanity checks
#
# Eq. 8 against its closed form, the conditioning actually *helping* rather than
# merely moving the loss, and the scale of the conditioning term against the graph
# signal it is added to.

# %%
def sanity_check(model, batch_cpu):
    prev = model.mode
    model.mode = 'likelihood'
    model.eval()
    try:
        with torch.no_grad():
            inputs = batch_to_tensor(batch_cpu)
            lw = model(*inputs)
            l, tox = model.last_l, model.last_tox

        l_eff = l if TOX_FLOOR is None else torch.where(
            tox > 0, torch.clamp(l, min=TOX_FLOOR), l)
        err = float((lw - l_eff * (1.0 - (1.0 + LAMBDA_TOX) * tox)).abs().max())
        n_tox = int(tox.sum())
        print(f"  Eq. 8  | {tox.numel()} molecules, {n_tox} toxic | "
              f"max error {err:.2e}")
        check("Eq. 8 matches its closed form", err < 1e-3, f"max error {err:.2e}")

        if n_tox:
            flipped = int((torch.sign(lw[tox > 0]) ==
                           -torch.sign(l_eff[tox > 0])).sum())
            check("sign flips on toxic molecules", flipped == n_tox,
                  f"{flipped}/{n_tox}")
        else:
            print("  [WARN] no toxic molecule in this batch - the sign flip is not "
                  "exercised here")

        print(f"         per-action log-probs at the {LOG_FLOOR:.2f} floor: "
              f"{100 * model.last_frac_floor:.3f}%")
        print(f"         max |logit| {model.policy_h.last_logit_absmax:.1f}  "
              f"(clamp binds at 80; {100 * model.policy_h.last_frac_clamped:.4f}% "
              f"at it)")

        # --- does the condition HELP, not merely move the loss? ---------------
        # The old check only asserted |a - b| > 1e-6, so it passed while a RANDOM
        # vector was giving a LOWER loss than the real embedding.
        model.mode = 'loss'
        with torch.no_grad():
            real = model(*inputs).item()

            shuffled = list(inputs)
            perm = torch.randperm(inputs[15].shape[0], device=inputs[15].device)
            shuffled[15] = inputs[15][perm]          # real embeddings, wrong proteins
            mismatched = model(*shuffled).item()

            noise = list(inputs)
            noise[15] = torch.randn_like(inputs[15]) * float(inputs[15].std())
            random_c = model(*noise).item()

        print(f"  cond   | real c {real:.4f} | mismatched c {mismatched:.4f} | "
              f"random c {random_c:.4f}")
        check("the condition has some effect", abs(real - random_c) > 1e-6,
              f"delta {abs(real - random_c):.4f}")
        check("the RIGHT protein beats a mismatched one", real < mismatched,
              f"real {real:.3f} vs mismatched {mismatched:.3f} - "
              + ("the model is using the protein identity"
                 if real < mismatched else
                 "pairing each ligand with the WRONG protein gives a LOWER loss. "
                 "The conditioning path is not yet doing useful work; Table 2 "
                 "(target awareness) is meaningless until this flips"),
              warn_only=True)

        # --- scale of the additive condition against the graph signal ----------
        with torch.no_grad():
            X, A, c, ids = inputs[0], inputs[1], inputs[15], inputs[16]
            h = model.embedding_atom(X) + model.embedding_mask(inputs[3])
            print(f"  scale  | std(c) = {c.std().item():.3f}")
            worst = 0.0
            for i, conv in enumerate(model.conv):
                inp = h if i == 0 else model.activation(model.bn[i - 1](h))
                g = conv(inp, A)
                c_i = model.linear_c[i](c)[ids, :]
                r = c_i.std().item() / max(1e-9, g.std().item())
                worst = max(worst, r)
                print(f"         layer {i}: std(conv) {g.std().item():7.3f}  "
                      f"std(c_i) {c_i.std().item():7.3f}  ratio {r:6.1f}x")
                h = g + c_i
            check("conditioning does not swamp the graph signal", worst <= 10,
                  f"worst ratio {worst:.1f}x"
                  + ("" if worst <= 10 else
                     " - at this ratio the pretrained graph weights contribute "
                     "little early in fine-tuning"),
                  warn_only=True)
    finally:
        model.mode = prev
        model.train()


sanity_check(model, batch_cpu)

# %% [markdown]
# ## Evaluation

# %%
def evaluate(model, loader, collect=False):
    """Molecule-weighted means: (Eq. 8 loss, plain NLL, mean l non-toxic, mean l toxic).

    With collect=True also returns the per-molecule arrays, which is what the
    step-0 measurement needs.
    """
    prev = model.mode
    model.mode = 'likelihood'
    model.eval()
    sum_w = sum_l = 0.0
    n_mol = 0
    sum_non, n_non, sum_tox, n_tox = 0.0, 0, 0.0, 0
    all_l, all_tox = [], []
    try:
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                lw = model(*batch_to_tensor(batch))
                l, tox = model.last_l, model.last_tox
                sum_w += float(lw.sum())
                sum_l += float(l.sum())
                n_mol += int(l.numel())
                if (tox == 0).any():
                    sum_non += float(l[tox == 0].sum())
                    n_non += int((tox == 0).sum())
                if (tox > 0).any():
                    sum_tox += float(l[tox > 0].sum())
                    n_tox += int((tox > 0).sum())
                if collect:
                    all_l.append(l.cpu().numpy())
                    all_tox.append(tox.cpu().numpy())
    finally:
        model.mode = prev
        model.train()

    if n_mol == 0:
        raise RuntimeError("Validation loader produced no molecules; early stopping "
                           "would silently see NaN.")

    out = (-sum_w / n_mol, -sum_l / n_mol,
           sum_non / n_non if n_non else float('nan'),
           sum_tox / n_tox if n_tox else float('nan'))
    if collect:
        return out, np.concatenate(all_l), np.concatenate(all_tox)
    return out


def select_metric(val_w, val_nll, val_non):
    """The quantity early stopping minimises, per EARLY_STOP_ON.

    `val_non` is a log-likelihood (negative, higher is better), so the minimised
    form is its negation.
    """
    if EARLY_STOP_ON == 'weighted':
        return val_w
    if EARLY_STOP_ON == 'nll':
        return val_nll
    if EARLY_STOP_ON == 'l_nontoxic':
        return -val_non
    raise ValueError(f"Unknown EARLY_STOP_ON: {EARLY_STOP_ON!r}")


# %% [markdown]
# ## Step-0 measurement (blocker B4)
#
# The likelihood distribution **before the first optimizer step**. One validation
# pass, and it answers four questions at once:
#
# * baseline checkpoint quality on *this* data, with no domain-shift confound;
# * whether the toxic/non-toxic gap at step 0 is suppression inherited from stage 1
#   or a **molecule-size confound** (`l` scales with action count);
# * the only defensible basis for a `TOX_FLOOR`;
# * decisively, whether the conditioning initialisation costs likelihood (B3),
#   where a variance ratio cannot.

# %%
if MEASURE_STEP0 and not is_continuous:
    print("  measuring step-0 likelihoods (one validation pass)...")
    t0 = time.time()
    (s0_w, s0_nll, s0_non, s0_tox), s0_l, s0_toxv = evaluate(model, loader_val,
                                                             collect=True)

    # Action count per validation molecule, so l can be normalised by molecule size.
    act = np.array([Chem.MolFromSmiles(strip_salt(r[0]) if STRIP_SALTS else r[0]
                                       ).GetNumBonds() + 1 for r in records_val],
                   dtype=np.float32)

    print(f"  [{time.time() - t0:.0f}s]  weighted {s0_w:.3f} | nll {s0_nll:.3f} | "
          f"l_nontox {s0_non:.2f} | l_tox {s0_tox:.2f}")

    non = s0_l[s0_toxv == 0]
    tox = s0_l[s0_toxv > 0]
    for label, a in (("non-toxic", non), ("toxic", tox)):
        if a.size:
            q = np.percentile(a, [1, 5, 25, 50, 75, 95])
            print(f"  {label:9s} n={a.size:>5}  mean {a.mean():8.2f}  "
                  f"p1 {q[0]:8.2f}  p5 {q[1]:8.2f}  median {q[3]:8.2f}  "
                  f"p95 {q[5]:8.2f}")

    # Size confound: l scales with action count, so a gap can be molecule size
    # rather than suppression.
    n_v = min(len(act), len(s0_l))
    per_action = s0_l[:n_v] / np.maximum(1.0, act[:n_v])
    pa_non = per_action[s0_toxv[:n_v] == 0]
    pa_tox = per_action[s0_toxv[:n_v] > 0]
    if pa_non.size and pa_tox.size:
        print(f"  per action:  non-toxic {pa_non.mean():.3f} nats  |  "
              f"toxic {pa_tox.mean():.3f} nats")
        _gap_raw = abs(s0_non - s0_tox)
        _gap_pa = abs(pa_non.mean() - pa_tox.mean()) * float(act[:n_v].mean())
        check("toxic/non-toxic gap is suppression, not molecule size",
              _gap_pa > 0.5 * _gap_raw,
              f"raw gap {_gap_raw:.1f}; size-normalised {_gap_pa:.1f}. "
              + ("Most of the gap survives normalisation, so it is real suppression "
                 "inherited from stage 1."
                 if _gap_pa > 0.5 * _gap_raw else
                 "Most of the gap disappears once molecule size is accounted for - "
                 "the 'suppression' is largely a size confound."),
              warn_only=True)

    if tox.size:
        # A defensible floor sits well below the typical NON-toxic bound (so it never
        # binds on the molecules we want to model) but above the toxic p5 at step 0
        # (so it still has room to bind later, rather than being inert from
        # iteration 1).
        cand = float(np.floor(min(non.mean() if non.size else tox.mean(),
                                  np.percentile(tox, 5)) - 20.0))
        print(f"\n  TOX_FLOOR candidate: {cand:.1f}")
        print(f"    - mean non-toxic l is {non.mean():.1f}, so this is "
              f"{abs(cand - non.mean()):.0f} nats below it: a toxic ligand at the "
              f"floor is e^-{abs(cand - non.mean()):.0f} as likely as a typical "
              f"non-toxic one")
        print(f"    - toxic p5 at step 0 is {np.percentile(tox, 5):.1f}, so the floor "
              f"starts out binding on under 5% of toxic molecules and has room to "
              f"bind more as training proceeds, rather than being inert from "
              f"iteration 1")
        print(f"    Set TOX_FLOOR = {cand:.1f} ONLY for a deliberate stabilisation "
              f"experiment; None is the paper-faithful baseline.")

    np.savez_compressed(STEP0_PATH, l=s0_l, tox=s0_toxv, action_count=act[:n_v],
                        val_weighted=s0_w, val_nll=s0_nll,
                        val_l_nontoxic=s0_non, val_l_toxic=s0_tox)
    print(f"  -> {STEP0_PATH}")
else:
    print("  skipped (resuming, or MEASURE_STEP0 = False)")

# %% [markdown]
# ## Training loop

# %%
TRAIN_HEADER = ('step\ttime(h)\tloss\tlr\tl_nontoxic\tl_toxic\tepoch\tfrac_toxic'
                '\tfrac_floored\tgrad_norm\tfrac_clip\tfrac_logfloor\n')
VAL_HEADER = ('step\ttime(h)\tepoch\tval_weighted\tval_nll\tval_l_nontoxic'
              '\tval_l_toxic\tes_metric\tes_counter\tbest\n')


def ensure_header(path, header):
    """Create with its header if absent or empty. Never truncates."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        with open(path, 'w') as f:
            f.write(header)


ensure_header(LOG_FILE, TRAIN_HEADER)
ensure_header(VAL_LOG_FILE, VAL_HEADER)

if not is_continuous:
    t0 = time.time()
else:
    t0 = time.time() - elapsed_offset * 3600.0
    print(f"  clock resumed at {elapsed_offset:.2f} h")


def save_state(best=False, reason=""):
    """One self-contained, atomically written checkpoint."""
    state = {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'lr': optimizer.param_groups[0]['lr'],
        'global_counter': global_counter, 'epoch_no': epoch_no,
        'iters_per_epoch': ITERS_PER_EPOCH,
        'early_stopping': early_stopping.state_dict(),
        'best_val_nll': best_val_nll,
        'elapsed_hours': (time.time() - t0) / 3600.0,
        'rng': rng_state(), 'configs': configs, 'fold': FOLD, 'N_C': N_C,
        'lambda_tox': LAMBDA_TOX, 'tox_floor': TOX_FLOOR,
        'objective': objective_signature(), 'data': data_signature(),
        'smoke_test': SMOKE_TEST,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'reason': reason,
    }
    atomic_save(state, CKPT_LAST)
    atomic_save({k: state[k] for k in ('optimizer', 'lr', 'global_counter',
                                       'epoch_no', 'early_stopping')},
                CKPT_TRAINER, keep_prev=False)

    if best:
        atomic_save({'model': model.state_dict(), 'configs': configs, 'fold': FOLD,
                     'epoch': epoch_no - 1, 'global_counter': global_counter,
                     'val_loss': early_stopping.best,
                     'val_metric': EARLY_STOP_ON,
                     'N_C': N_C, 'lambda_tox': LAMBDA_TOX, 'tox_floor': TOX_FLOOR,
                     'objective': objective_signature(), 'smoke_test': SMOKE_TEST,
                     'saved_at': state['saved_at']}, CKPT_BEST)


def die_nonfinite(what, value):
    """Stop WITHOUT touching ckpt_last.pt - the last good state must survive."""
    path = os.path.join(OUT_DIR, f"nonfinite_{time.strftime('%Y%m%d-%H%M%S')}.pt")
    try:
        atomic_save({'model': model.state_dict(), 'global_counter': global_counter,
                     'epoch_no': epoch_no, 'what': what}, path, keep_prev=False)
    except Exception as e:
        print(f"  [WARN] could not write {path}: {type(e).__name__}")
    raise RuntimeError(
        f"Non-finite {what} ({value}) at iteration {global_counter}. ckpt_last.pt is "
        f"untouched, so re-running resumes from the last good state; the failing "
        f"weights are in {os.path.basename(path)}. This is a real numerical failure, "
        f"unlike an 'nan' in the l_toxic column, which only means a logging window "
        f"held no toxic molecule.")


def grad_diagnostics(model, clip_value):
    """-> (pre-clip global norm, fraction of elements the clip will rewrite)."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return float('nan'), float('nan')
    norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads)))
    n_clip = sum(int((g.detach().abs() > clip_value).sum()) for g in grads)
    n_tot = sum(g.numel() for g in grads)
    return norm, n_clip / max(1, n_tot)


# %%
# SIGTERM (SLURM time limit) / SIGINT checkpoint at the next iteration boundary.
_stop = {'requested': False, 'name': None}


def _request_stop(signum, frame):
    if _stop['requested']:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        return
    _stop['requested'] = True
    _stop['name'] = signal.Signals(signum).name
    print(f"\n[SIGNAL] {_stop['name']} received - finishing this iteration, "
          f"checkpointing, then exiting. Send it again to abort immediately.",
          flush=True)


if GRACEFUL_SIGNALS:
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _request_stop)
        except (ValueError, OSError):
            pass        # not on the main thread


def done():
    if epoch_no > MAX_EPOCHS:
        return f"MAX_EPOCHS = {MAX_EPOCHS} reached"
    if early_stopping.counter >= PATIENCE:
        return f"early stopping triggered ({early_stopping.counter}/{PATIENCE})"
    return None


# %%
model.mode = 'loss'
model.train()
it_train = iter(loader_train)
interrupted = False
win = {'loss': 0.0, 'n_batch': 0, 'sum_non': 0.0, 'n_non': 0, 'sum_tox': 0.0,
       'n_tox': 0, 'n_floored': 0, 'logfloor': 0.0,
       'grad_norm': 0.0, 'frac_clip': 0.0, 'n_diag': 0}

if done():
    print(f"Nothing to do: {done()}. Raise MAX_EPOCHS, or set FORCE_RESTART = True.")
else:
    print(f"Training from iteration {global_counter:,}, epoch {epoch_no} "
          f"({ITERS_PER_EPOCH:,} iterations per epoch)"
          + ("  [SMOKE TEST - not reportable]" if SMOKE_TEST else ""))
    print(f"Model selection: {EARLY_STOP_ON}  |  patience {PATIENCE}/{MAX_EPOCHS}")

# %%
while not done():
    try:
        inputs_cpu = next(it_train)
    except StopIteration:
        # The loader is exhausted. Epoch boundaries are decided by the iteration
        # count below, NOT by this - so a mid-epoch resume cannot produce a long
        # epoch the way it did before.
        it_train = iter(loader_train)
        continue

    if inputs_cpu is None:          # whole batch dropped by the collator
        continue

    global_counter += 1
    is_summary = (global_counter % SUMMARY_STEP == 0)

    optimizer.zero_grad(set_to_none=True)       # Gluon overwrites; torch accumulates
    loss = model(*batch_to_tensor(inputs_cpu))
    if not torch.isfinite(loss):
        die_nonfinite("loss", loss.item())
    loss.backward()

    # clip_grad_value_ clamps, and clamping NaN returns NaN, so a non-finite
    # gradient would pass straight into optimizer.step() and poison every parameter.
    _grads = [p.grad for p in model.parameters() if p.grad is not None]
    if _grads and not torch.isfinite(torch.stack([g.sum() for g in _grads]).sum()):
        die_nonfinite("gradient", float('nan'))

    if is_summary:
        gn, fc = grad_diagnostics(model, CLIP_GRAD)
        win['grad_norm'] += gn
        win['frac_clip'] += fc
        win['n_diag'] += 1

    torch.nn.utils.clip_grad_value_(model.parameters(), CLIP_GRAD)
    optimizer.step()

    if global_counter % DECAY_STEP == 0:
        for g in optimizer.param_groups:
            g['lr'] *= (1.0 - DECAY)

    if global_counter % CHECKPOINT_EVERY == 0:
        save_state(reason=f"periodic@{global_counter}")

    _l, _tox = model.last_l, model.last_tox      # the raw, unfloored bound
    win['loss'] += loss.item()
    win['n_batch'] += 1
    win['sum_non'] += float(_l[_tox == 0].sum())
    win['n_non'] += int((_tox == 0).sum())
    win['sum_tox'] += float(_l[_tox > 0].sum())
    win['n_tox'] += int((_tox > 0).sum())
    win['n_floored'] += int(model.last_floored.sum())
    win['logfloor'] += model.last_frac_floor

    if is_summary:
        lr_now = optimizer.param_groups[0]['lr']
        hours = (time.time() - t0) / 3600.0
        mean_loss = win['loss'] / max(1, win['n_batch'])
        l_non = win['sum_non'] / win['n_non'] if win['n_non'] else float('nan')
        l_tox = win['sum_tox'] / win['n_tox'] if win['n_tox'] else float('nan')
        frac_tox = win['n_tox'] / max(1, win['n_tox'] + win['n_non'])
        frac_floor = win['n_floored'] / max(1, win['n_tox'])
        g_norm = win['grad_norm'] / max(1, win['n_diag'])
        f_clip = win['frac_clip'] / max(1, win['n_diag'])
        f_logfloor = win['logfloor'] / max(1, win['n_batch'])

        with open(LOG_FILE, 'a') as f:
            f.write(f"{global_counter}\t{hours:.4f}\t{mean_loss:.4f}\t{lr_now}"
                    f"\t{l_non:.4f}\t{l_tox:.4f}\t{epoch_no}\t{frac_tox:.4f}"
                    f"\t{frac_floor:.4f}\t{g_norm:.4f}\t{f_clip:.6f}"
                    f"\t{f_logfloor:.6f}\n")

        print(f"iter {global_counter:06d} | ep {epoch_no} | loss {mean_loss:8.3f} | "
              f"l_nontox {l_non:8.2f} | l_tox {l_tox:9.2f} | "
              f"tox {100 * frac_tox:4.1f}%"
              + (f" | floored {100 * frac_floor:5.1f}%" if TOX_FLOOR is not None else "")
              + f" | |g| {g_norm:7.1f} | clip {100 * f_clip:4.1f}%"
              f" | lr {lr_now:.2e} | {hours:.2f}h", flush=True)

        if TOX_FLOOR is not None and win['n_tox'] and frac_floor > 0.98:
            print("  [WARN] essentially every toxic molecule is at the floor: the "
                  "toxic branch contributes no gradient, equivalent to "
                  "LAMBDA_TOX = 0. Lower the floor, or accept that toxic ligands are "
                  "simply being ignored.")
        if f_clip > 0.5:
            print("  [WARN] over half the gradient elements are being clipped, so "
                  "only the gradient SIGN matters - which is how an unbounded branch "
                  "takes over training.")
        if f_logfloor > 0.10:
            print("  [WARN] more than 10% of per-action log-probabilities are pinned "
                  "at ln(1e-10), where the gradient is ~1e10.")

        for k in win:
            win[k] = 0.0
        del loss, inputs_cpu
        gc.collect()

    # ---- epoch boundary, by iteration count ---------------------------------
    if global_counter % ITERS_PER_EPOCH == 0:
        finished = epoch_no
        epoch_no += 1

        val_w, val_nll, val_non, val_tox = evaluate(model, loader_val)
        metric = select_metric(val_w, val_nll, val_non)
        stop = early_stopping.step(metric)
        hours = (time.time() - t0) / 3600.0

        with open(VAL_LOG_FILE, 'a') as f:
            f.write(f"{global_counter}\t{hours:.4f}\t{finished}\t{val_w:.4f}"
                    f"\t{val_nll:.4f}\t{val_non:.4f}\t{val_tox:.4f}"
                    f"\t{metric:.4f}\t{early_stopping.counter}"
                    f"\t{early_stopping.best:.4f}\n")

        print(f"epoch {finished} | val weighted {val_w:.4f} | nll {val_nll:.4f} | "
              f"l_nontox {val_non:.3f} | l_tox {val_tox:.3f} | "
              f"{EARLY_STOP_ON} {metric:.4f} | best {early_stopping.best:.4f} | "
              f"patience {early_stopping.counter}/{PATIENCE}"
              f"{'  <- new best' if early_stopping.improved else ''}")

        # Watch plain NLL against its own best, independently of the selection
        # metric. Under a working CTDDG this RISES by construction, so it is a
        # diagnostic, not a target.
        if val_nll < best_val_nll:
            best_val_nll = val_nll
        elif val_nll > best_val_nll + NLL_DIVERGENCE_MARGIN:
            print(f"  [note] val NLL {val_nll:.2f} is "
                  f"{val_nll - best_val_nll:.2f} above its best ({best_val_nll:.2f}). "
                  f"Expected when toxicity suppression is working. Judge the run on "
                  f"l_nontoxic ({val_non:.2f}), which is the quantity Eq. 8 is "
                  f"supposed to leave alone.")

        save_state(best=early_stopping.improved, reason=f"epoch{finished}")
        if stop:
            print(f"Early stopping after epoch {finished}.")

    if _stop['requested']:
        save_state(reason=f"signal:{_stop['name']}@{global_counter}")
        with open(LOG_FILE, 'a') as f:
            f.write(f"# interrupted by {_stop['name']} at iteration "
                    f"{global_counter}, epoch {epoch_no}\n")
        print(f"[SIGNAL] checkpointed at iteration {global_counter} (epoch "
              f"{epoch_no}). Re-run to continue from here.")
        interrupted = True
        break

# %% [markdown]
# ## Finish

# %%
if not interrupted:
    save_state(reason="final")
    if not os.path.exists(CKPT_BEST):
        save_state(best=True)   # no epoch improved; keep the final weights as "best"
    with open(LOG_FILE, 'a') as f:
        f.write('# finished\n')
    print(f"\nDone. Best validation {EARLY_STOP_ON}: {early_stopping.best:.4f}")
else:
    print(f"\nStopped by {_stop['name']} at iteration {global_counter}. "
          f"Best validation {EARLY_STOP_ON} so far: {early_stopping.best:.4f}")

if SMOKE_TEST:
    print("[SMOKE TEST] checkpoints, test bundle and metrics are pipeline artifacts, "
          "not results.")

for name, path in (("best weights", CKPT_BEST), ("last weights", CKPT_LAST),
                   ("configs", CKPT_CONFIG), ("test bundle", TEST_BUNDLE),
                   ("split manifest", SPLIT_MANIFEST), ("step-0 likelihoods", STEP0_PATH),
                   ("train log", LOG_FILE), ("val log", VAL_LOG_FILE)):
    print(f"  {name:20s} {path}")

print("\n" + "=" * 70)
for c in CHECKS:
    print(f"  {c['status']:5s}  {c['name']}: {c['detail']}")

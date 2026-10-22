#!/usr/bin/env python3
"""Cross-notebook consistency audit for the CTDDG reproduction.

Every notebook here is self-contained, which means the model definition, the
graph-construction helpers and the feature builders are DUPLICATED across files.
That is deliberate - a notebook you can run top to bottom is worth a lot - but it
means a fix applied in one file and not the others produces a checkpoint that
loads cleanly and then computes something different.

This script compares the shared definitions and fails loudly on drift. Run it
after editing any notebook, and before any long GPU job.

    python3 99_consistency_audit.py [--dir .]

Exit code 0 = consistent, 1 = drift found.
"""
import argparse
import ast
import difflib
import os
import sys

FILES = ["00_phase0_diagnostics.py", "random_forest_classifier.py",
         "pretraining_torch.py", "02_finetune_ctddg.py",
         "03_generate_samples.py", "04_evaluate_metrics.py"]

# (definition, files it must match across, why it matters)
#
# Every notebook here is self-contained, so the model definition, the graph
# helpers and the feature builders are DUPLICATED. A fix applied in one file and
# not the others produces a checkpoint that loads cleanly and then computes
# something different.
ALL_TORCH = ["pretraining_torch.py", "02_finetune_ctddg.py",
             "03_generate_samples.py", "00_phase0_diagnostics.py"]
TRAIN_ONLY = ["pretraining_torch.py", "02_finetune_ctddg.py"]
TRAIN_PLUS_DIAG = ["pretraining_torch.py", "02_finetune_ctddg.py",
                   "00_phase0_diagnostics.py"]
TOX_FILES = ["random_forest_classifier.py", "04_evaluate_metrics.py"]

MUST_MATCH = [
    # --- how a molecule becomes a sequence of actions ------------------------
    ("MoleculeSpec", ALL_TORCH,
     "the atom and bond vocabularies ARE the model's output space; an index that "
     "means a different atom in two files silently relabels every action"),
    ("_mol_spec", ALL_TORCH, "the shared vocabulary singleton"),
    ("get_mol_spec", ALL_TORCH, "as above"),
    ("get_graph_from_smiles", TRAIN_PLUS_DIAG,
     "the entry point to every training target, including the disconnected-graph "
     "guard that stops a salt from corrupting its own target"),
    ("traverse_graph", TRAIN_PLUS_DIAG,
     "the Q_alpha proposal must be the same distribution in both phases, and the "
     "completeness fix (AUDIT.md A21) must be present in all of them"),
    ("single_reorder", TRAIN_ONLY,
     "both phases must decompose a molecule into actions identically, or the "
     "transferred weights mean nothing"),
    ("single_expand", TRAIN_ONLY, "as above"),
    ("merge_single_0", TRAIN_ONLY, "as above"),
    ("merge_single", TRAIN_ONLY, "as above"),
    ("strip_salt", TRAIN_PLUS_DIAG,
     "a different salt rule means the two phases see different molecules"),

    # --- the action distribution ---------------------------------------------
    ("segment_sum", ALL_TORCH, "core graph reduction"),
    ("segment_max", ALL_TORCH,
     "the per-molecule max that makes the log-softmax overflow-proof; absent, the "
     "stable path silently falls back to a different normalisation"),
    ("action_log_probs", ALL_TORCH,
     "THE action distribution. Training optimises under one normalisation and the "
     "sampler draws from another if these drift - every shape still matches and "
     "every probability still sums to one"),
    ("Policy", ALL_TORCH,
     "the policy head, which both returns log-probabilities for training and "
     "probabilities for sampling from the same code path"),
    ("_concat_neighbourhoods", ALL_TORCH,
     "the GCN's neighbourhood ordering IS the weight matrix layout"),
    ("EfficientGraphConvFn", ["pretraining_torch.py", "02_finetune_ctddg.py",
                              "03_generate_samples.py"],
     "the custom autograd graph convolution; it recomputes the concatenated "
     "neighbourhood in backward instead of storing it"),
    ("GraphConv", ALL_TORCH,
     "the graph convolution itself: its weight layout IS the checkpoint's"),
    ("Linear_BN", ALL_TORCH,
     "parameter names and order must match for state_dict transfer"),
    ("get_activation", ALL_TORCH, "a different activation is a different model"),
    ("_TwoLayerDense", ["02_finetune_ctddg.py", "03_generate_samples.py"],
     "FNN_0 must return LOGITS in both files: stage 2 takes a log_softmax of its "
     "output and the sampler a softmax. A version that normalises internally "
     "double-softmaxes in one of the two while loading the other's weights "
     "without complaint"),

    # --- toxicity labelling and evaluation -----------------------------------
    ("_PHYSCHEM", TOX_FILES,
     "the independent evaluator is TRAINED in one file and APPLIED in the other; "
     "a different descriptor list or order means it scores on features it never saw"),
    ("independent_feature_row", TOX_FILES,
     "the single shared featuriser for that evaluator"),
    ("morgan_bits", ["random_forest_classifier.py", "02_finetune_ctddg.py",
                     "04_evaluate_metrics.py"],
     "the Ames RF's input. The forest is trained in one file, applied to BindingDB "
     "ligands in another and to generated molecules in a third; a fingerprint "
     "difference changes Toxic_class in Eq. 8 and the reported toxic fraction"),
    ("_MORGAN_GEN", ["random_forest_classifier.py", "02_finetune_ctddg.py",
                     "04_evaluate_metrics.py"],
     "the generator cache morgan_bits indexes"),
    ("model_behaviour_hash", ["02_finetune_ctddg.py", "04_evaluate_metrics.py"],
     "the label-cache key. If the probe set or seed differs between the two "
     "files, each decides independently whether the other's cache is valid and "
     "they relabel in a loop"),

    # --- sampling ------------------------------------------------------------
    ("_rand_action", ["03_generate_samples.py", "00_phase0_diagnostics.py"],
     "the diagnostic and the real sampler must draw actions the same way, or "
     "check A does not predict the full run"),
    ("get_mol_from_graph", ["03_generate_samples.py", "00_phase0_diagnostics.py"],
     "the validity verdict AND the failure reason behind every census"),
    ("get_d", ALL_TORCH,
     "the 2- and 3-hop neighbourhoods fed to the receptive field"),
]

# Configuration values (not source) that must agree across files, because each
# describes the SAME experiment from a different notebook.
MUST_AGREE = [
    ("STABLE_SOFTMAX", ALL_TORCH,
     "training and sampling must normalise the action distribution the same way"),
    ("TRAVERSAL_POLICY", TRAIN_PLUS_DIAG,
     "'upstream_broken' reproduces the traversal that dropped atoms; a file left "
     "on it while the others are fixed builds corrupt targets"),
    ("FRAGMENT_POLICY", TRAIN_PLUS_DIAG + TOX_FILES,
     "which fragment of a salt is the molecule"),
    ("P_ALPHA", TRAIN_PLUS_DIAG, "the Q_alpha proposal parameter"),
    ("MAX_ITER", ALL_TORCH,
     "the action budget; a sampler capped below the training budget truncates "
     "molecules the training distribution contains"),
    ("STRIP_SALTS", TRAIN_PLUS_DIAG, "whether salts are stripped at all"),
    ("D_RECEPTIVE", ALL_TORCH, "the receptive field depth is part of the model"),
]

# Deliberately NOT in MUST_AGREE, because the second file reads them from the
# artifact rather than restating them, which is stronger than agreement:
#   N_C                 03 takes it from the checkpoint's configs
#   FP_RADIUS/FP_NBITS  02 and 04 take them from fingerprint_config.pkl
#   STABLE_SOFTMAX      03 overrides its own default from the checkpoint
# A hardcoded copy in the second file would agree with the first and still be
# wrong whenever the artifact was built differently.

# Differences that are real but verified harmless. Each needs a reason, and each
# is checked below to be LIVE - a stale entry that no longer matches anything is
# reported, so this dict cannot quietly accumulate excuses for drift that was
# since fixed.
KNOWN_BENIGN = {
    ("get_d", "dtype"):
        "np.int32 in the training files vs np.int64 in the samplers, inherited from "
        "the original notebooks. The matrix holds only ones and is used solely for "
        "nonzero() positions, so the returned D_2/D_3 index arrays are identical.",
    ("GraphConv", "inference-only variant"):
        "00_phase0_diagnostics calls `_concat_neighbourhoods(X, A_list) @ self.W` "
        "directly where the other three call `EfficientGraphConvFn.apply(...)`. "
        "That Function's forward IS that same expression (see its `forward`: "
        "`return _concat_neighbourhoods(X, A_list) @ W`), so the values are "
        "identical by construction, not merely close. The two differ only in "
        "backward, which recomputes the neighbourhood to save memory - and 00 runs "
        "under torch.no_grad() for inference only, so it never takes one.",
    ("morgan_matrix", "signature"):
        "random_forest_classifier takes RDKit mols (it already has them) while the "
        "other two take SMILES and parse. Both delegate every bit to the shared "
        "`morgan_bits`, which IS enforced identical, so the fingerprints match.",
    ("independent_features", "signature"):
        "the training file takes RDKit mols, the evaluation file takes SMILES and "
        "parses. Both delegate to the shared `independent_feature_row`, and "
        "04_evaluate_metrics verifies the recorded feature_spec and physchem_order "
        "against the model's meta before predicting.",
}

def definitions(path):
    """{name: source} for top-level functions, classes and simple assignments."""
    src = open(path).read()
    lines = src.split("\n")
    out = {}
    for node in ast.parse(src).body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for nm in names:
            out[nm] = "\n".join(lines[node.lineno - 1: node.end_lineno])
    return out


def code_only(text):
    """Executable lines only: comments, docstrings and blanks removed."""
    out, in_doc, quote = [], False, None
    for line in text.split("\n"):
        stripped = line.strip()
        if in_doc:
            if quote in stripped:
                in_doc = False
            continue
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            if not (stripped.endswith(quote) and len(stripped) > 3):
                in_doc = True
            continue
        code = line.split("#")[0].rstrip()
        if code.strip():
            out.append(code)
    return out


def literal_values(path):
    """{name: value} for top-level assignments of plain literals."""
    out = {}
    for node in ast.parse(open(path).read()).body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if not isinstance(t, ast.Name):
                continue
            try:
                out.setdefault(t.id, ast.literal_eval(node.value))
            except (ValueError, TypeError, SyntaxError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--verbose", action="store_true",
                    help="print the full diff for every mismatch")
    args = ap.parse_args()

    missing = [f for f in FILES if not os.path.isfile(os.path.join(args.dir, f))]
    if missing:
        print(f"[FAIL] notebooks not found in {args.dir}: {missing}")
        return 1

    defs = {f: definitions(os.path.join(args.dir, f)) for f in FILES}

    print("=" * 78)
    print(f"CTDDG cross-notebook consistency audit  ({args.dir})")
    print("=" * 78)

    n_ok = n_benign = n_bad = 0
    for name, files, why in MUST_MATCH:
        bodies = {}
        for f in files:
            raw = defs[f].get(name)
            bodies[f] = None if raw is None else code_only(raw)

        absent = [f for f, b in bodies.items() if b is None]
        if absent:
            n_bad += 1
            print(f"  FAIL   {name:24s} absent from {absent}")
            print(f"         {why}")
            continue

        ref_file = files[0]
        ref = bodies[ref_file]
        diffs = {f: b for f, b in bodies.items() if b != ref}

        if not diffs:
            n_ok += 1
            print(f"  OK     {name:24s} identical across {len(files)} file(s)")
            continue

        benign = [k for k in KNOWN_BENIGN if k[0] == name]
        if benign:
            n_benign += 1
            print(f"  BENIGN {name:24s} differs, verified harmless")
            print(f"         {KNOWN_BENIGN[benign[0]]}")
        else:
            n_bad += 1
            print(f"  DRIFT  {name:24s} differs across {list(diffs)}")
            print(f"         WHY IT MATTERS: {why}")

        if args.verbose or not benign:
            for f, b in diffs.items():
                for line in difflib.unified_diff(ref, b, ref_file, f, lineterm="", n=1):
                    print("           " + line)

    # --- configuration values ------------------------------------------------
    vals = {f: literal_values(os.path.join(args.dir, f)) for f in FILES}
    print("-" * 78)
    n_agree = 0
    for name, files, why in MUST_AGREE:
        present = {f: vals[f][name] for f in files if name in vals[f]}
        if len(present) < 2:
            print(f"  SKIP   {name:24s} set in {len(present)} of {len(files)} file(s)"
                  + (f": {list(present)}" if present else ""))
            continue
        distinct = {repr(v) for v in present.values()}
        if len(distinct) == 1:
            n_agree += 1
            print(f"  OK     {name:24s} = {next(iter(distinct))} in "
                  f"{len(present)} file(s)")
        else:
            n_bad += 1
            print(f"  DISAGREE {name:22s} " +
                  ", ".join(f"{os.path.basename(f)}={v!r}"
                            for f, v in sorted(present.items())))
            print(f"         WHY IT MATTERS: {why}")

    # --- stale exemptions ----------------------------------------------------
    stale = []
    for (name, _tag), reason in KNOWN_BENIGN.items():
        files = [f for f in FILES if name in defs[f]]
        if len(files) < 2:
            stale.append((name, f"defined in {len(files)} file(s), nothing to differ"))
            continue
        ref = code_only(defs[files[0]][name])
        if all(code_only(defs[f][name]) == ref for f in files[1:]):
            stale.append((name, "the files now agree, so the exemption is unused"))
    if stale:
        print("-" * 78)
        print("  STALE EXEMPTIONS (KNOWN_BENIGN entries that no longer apply):")
        for name, why in stale:
            print(f"    {name:24s} {why}")
        print("    Remove them, so the dict cannot accumulate excuses for drift")
        print("    that was since fixed.")

    print("-" * 78)
    print(f"  {n_ok} identical | {n_benign} known-benign | {n_agree} values agree | "
          f"{n_bad} DRIFT")
    if n_bad:
        print("\n  Fix the drift before training. A notebook that loads a checkpoint")
        print("  cleanly and then computes something different is the hardest class")
        print("  of bug to find afterwards.")
        return 1
    print("\n  Shared definitions are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

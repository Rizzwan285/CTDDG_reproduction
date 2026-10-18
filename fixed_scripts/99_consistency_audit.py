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
MUST_MATCH = [
    ("single_reorder", ["pretraining_torch.py", "02_finetune_ctddg.py"],
     "both phases must decompose a molecule into actions identically, or the "
     "transferred weights mean nothing"),
    ("single_expand", ["pretraining_torch.py", "02_finetune_ctddg.py"],
     "as above"),
    ("merge_single_0", ["pretraining_torch.py", "02_finetune_ctddg.py"],
     "as above"),
    ("merge_single", ["pretraining_torch.py", "02_finetune_ctddg.py"],
     "as above"),
    ("traverse_graph", ["pretraining_torch.py", "02_finetune_ctddg.py"],
     "the Q_alpha proposal must be the same distribution in both phases"),
    ("strip_salt", ["pretraining_torch.py", "02_finetune_ctddg.py"],
     "a different salt rule means the two phases see different molecules"),
    ("segment_sum", ["pretraining_torch.py", "02_finetune_ctddg.py",
                     "03_generate_samples.py", "00_phase0_diagnostics.py"],
     "core graph reduction"),
    ("_concat_neighbourhoods", ["pretraining_torch.py", "02_finetune_ctddg.py",
                                "03_generate_samples.py", "00_phase0_diagnostics.py"],
     "the GCN's neighbourhood ordering IS the weight matrix layout"),
    ("Linear_BN", ["pretraining_torch.py", "02_finetune_ctddg.py",
                   "03_generate_samples.py", "00_phase0_diagnostics.py"],
     "parameter names and order must match for state_dict transfer"),
    ("_PHYSCHEM", ["random_forest_classifier.py", "04_evaluate_metrics.py"],
     "the independent evaluator is TRAINED in one file and APPLIED in the other; "
     "a different descriptor list or order means it scores on features it never saw"),
    ("_rand_action", ["03_generate_samples.py", "00_phase0_diagnostics.py"],
     "the diagnostic and the real sampler must draw actions the same way, or "
     "check A does not predict the full run"),
]

# Differences that are real but verified harmless. Each needs a reason.
KNOWN_BENIGN = {
    ("get_d", "dtype"):
        "np.int32 in the training files vs np.int64 in the samplers, inherited from "
        "the original notebooks. The matrix holds only ones and is used solely for "
        "nonzero() positions, so the returned D_2/D_3 index arrays are identical.",
    ("independent_features", "signature"):
        "the training file takes RDKit mols (it already has them), the evaluation "
        "file takes SMILES and parses. The FEATURE COMPUTATION and column order are "
        "identical, and 04_evaluate_metrics.py verifies the recorded feature_spec "
        "and physchem_order against the model's meta before predicting.",
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

    print("-" * 78)
    print(f"  {n_ok} identical | {n_benign} known-benign | {n_bad} DRIFT")
    if n_bad:
        print("\n  Fix the drift before training. A notebook that loads a checkpoint")
        print("  cleanly and then computes something different is the hardest class")
        print("  of bug to find afterwards.")
        return 1
    print("\n  Shared definitions are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Why do pair rows reference proteins that are not in the fold's unique-protein list?

Reports what the unmatched sequences are, so you can tell the three cases apart:
  (a) they are train proteins the unique list omits  -> regenerate the list and re-embed
  (b) they belong to the other split                 -> the pair files are not split-pure
  (c) they are near-matches of listed sequences      -> a normalisation bug; all rows recoverable
"""
import argparse
import glob
import os
from collections import Counter

FOLD_SUFFIX = {"d1": "4_org_1042_104", "d2": "4_org_1000_112", "d3": "4_org_1004_122",
               "d4": "4_org_1002_103", "d5": "4_org_1036_124"}


def norm(line):
    """Whitespace-stripped only - exactly what the notebook's seq_to_row does."""
    return "".join(line.split())


def read_norm(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            s = norm(line)
            if s:
                out.append(s)
            if limit and len(out) >= limit:
                break
    return out


def split_dir(repo, fold, split):
    return os.path.join(repo, "datasets/data/bindingdb",
                        f"{split}_dataset", f"{split}_{FOLD_SUFFIX[fold]}")


def one(d, pattern):
    hits = sorted(glob.glob(os.path.join(d, pattern)))
    if not hits:
        raise FileNotFoundError(f"no match for {pattern!r} in {d}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/muhamed/repo/CTDDG")
    ap.add_argument("--fold", default="d1")
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--samples", type=int, default=5)
    a = ap.parse_args()

    d = split_dir(a.repo, a.fold, a.split)
    other = "test" if a.split == "train" else "train"
    d_other = split_dir(a.repo, a.fold, other)
    tag = "tr" if a.split == "train" else "te"
    tag_other = "te" if a.split == "train" else "tr"

    prot_pairs = one(d, "proteins_*_corrected")
    lig_pairs = one(d, "ligands_*_corrected")
    uniq = os.path.join(d, f"{a.fold}_{tag}_unique_proteins.txt")
    uniq_other = os.path.join(d_other, f"{a.fold}_{tag_other}_unique_proteins.txt")

    print(f"fold {a.fold} / {a.split}")
    print(f"  pairs   {os.path.basename(prot_pairs)}")
    print(f"  unique  {os.path.basename(uniq)}\n")

    rows = read_norm(prot_pairs)
    ligs = read_norm(lig_pairs)
    listed = read_norm(uniq)
    listed_set = set(listed)
    listed_other = set(read_norm(uniq_other)) if os.path.isfile(uniq_other) else set()

    print(f"  pair rows            {len(rows)}")
    print(f"  ligand rows          {len(ligs)}"
          + ("   [!] MISALIGNED" if len(ligs) != len(rows) else ""))
    print(f"  listed proteins      {len(listed)}  ({len(listed_set)} distinct)")
    print(f"  listed in {other:5s}      {len(listed_other)}\n")

    unmatched_rows = [s for s in rows if s not in listed_set]
    distinct_unmatched = Counter(unmatched_rows)
    pct = 100.0 * len(unmatched_rows) / max(1, len(rows))
    print(f"  UNMATCHED rows       {len(unmatched_rows)} / {len(rows)}  ({pct:.1f}%)")
    print(f"  distinct unmatched   {len(distinct_unmatched)}\n")

    if not unmatched_rows:
        print("  Nothing unmatched. The pair files and the unique list agree.")
        return

    # (b) do they belong to the other split?
    in_other = {s for s in distinct_unmatched if s in listed_other}
    rows_in_other = sum(distinct_unmatched[s] for s in in_other)
    print(f"  [b] unmatched sequences found in the {other} split's list: "
          f"{len(in_other)} distinct / {rows_in_other} rows")
    if in_other:
        print(f"      -> the {a.split} pair files contain {other}-split proteins. "
              f"If this is the train split, that is leakage.")

    # (c) near-matches. Matching above is case-sensitive, like the notebook, so this
    # measures how many rows case folding alone would bring back.
    upper_listed = {s.upper() for s in listed_set}
    case_only = {s for s in distinct_unmatched if s.upper() in upper_listed}
    rows_case = sum(distinct_unmatched[s] for s in case_only)
    print(f"  [c] recovered by case folding alone: {len(case_only)} distinct / {rows_case} rows")

    prefix_hits = 0
    sample = list(distinct_unmatched)[:200]
    for s in sample:
        for t in listed:
            if s and t and (s.startswith(t) or t.startswith(s)) and s != t:
                prefix_hits += 1
                break
    print(f"  [c] prefix/truncation relation to a listed sequence: "
          f"{prefix_hits}/{len(sample)} of a {len(sample)}-sequence sample")

    alpha_listed = set("".join(listed[:50]))
    alpha_unmatched = set("".join(list(distinct_unmatched)[:50]))
    extra = alpha_unmatched - alpha_listed
    print(f"  [c] residue letters in unmatched but not listed: "
          f"{sorted(extra) if extra else 'none'}")

    # (a) coverage the other way
    used = {s for s in rows if s in listed_set}
    print(f"\n  [a] listed proteins never used by any pair row: {len(listed_set - used)}")

    ln_u = [len(s) for s in distinct_unmatched]
    ln_l = [len(s) for s in listed_set]
    f = lambda v: f"min {min(v)} / median {sorted(v)[len(v)//2]} / max {max(v)}"
    print(f"  lengths  unmatched: {f(ln_u)}")
    print(f"           listed:    {f(ln_l)}")

    print(f"\n  most frequent unmatched sequences:")
    for s, n in distinct_unmatched.most_common(a.samples):
        where = "in " + other if s in listed_other else "unknown"
        print(f"    {n:7d} rows | len {len(s):5d} | {where} | {s[:60]}...")

    print("\n  VERDICT")
    if rows_in_other > 0.5 * len(unmatched_rows):
        print("    Mostly the other split's proteins -> the fold's pair files are not "
              "split-pure. Fix the fold before training.")
    elif rows_case > 0.5 * len(unmatched_rows):
        print("    Mostly a case mismatch -> normalise case on both sides and the rows return.")
    elif len(distinct_unmatched) < 50:
        print("    A small number of proteins account for the loss -> add them to the unique "
              "list and re-embed those proteins.")
    else:
        print("    Many distinct proteins absent from the list -> the unique list was built "
              "from a different population than the pair files. Regenerate it from these pair "
              "files, then re-run 01_generate_prose_embeddings.ipynb.")


if __name__ == "__main__":
    main()

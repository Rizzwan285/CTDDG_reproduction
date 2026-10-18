import random, numpy as np, networkx as nx
from rdkit import Chem, RDLogger

from _nbsource import load_defs, source_path

RDLogger.DisableLog('rdApp.*')
print(f"source: {source_path('pretraining_torch.py')}\n")

def traverse_OLD(graph, atom_ranks, current_node=None, step_ids=None, p=0.8, log_p=0.0):
    if current_node is None:
        next_nodes = range(len(atom_ranks)); step_ids=[-1]*len(next_nodes); nr=atom_ranks
    else:
        next_nodes=[n for n in graph.neighbors(current_node) if step_ids[n]<0]
        nr=[atom_ranks[n] for n in next_nodes]
    next_nodes=[n for n,r in sorted(zip(next_nodes,nr),key=lambda x:x[1])]
    while next_nodes:
        if len(next_nodes)==1: nn=next_nodes[0]
        elif random.random()>=(1.0-p): nn=next_nodes[0]; log_p+=np.log(p)
        else:
            nn=next_nodes[random.randint(1,len(next_nodes)-1)]
            log_p+=np.log((1.0-p)/(len(next_nodes)-1))
        step_ids[nn]=max(step_ids)+1
        _,log_p = traverse_OLD(graph,atom_ranks,nn,step_ids,p,log_p)
        next_nodes=[n for n in graph.neighbors(nn) if step_ids[n]<0]
        nr=[atom_ranks[n] for n in next_nodes]
        next_nodes=[n for n,r in sorted(zip(next_nodes,nr),key=lambda x:x[1])]
    return step_ids, log_p

# The fixed traversal is NOT copied here: it is loaded from the shipped notebook
# source, so this measurement cannot pass while the shipped code differs. That
# divergence is exactly what 99_consistency_audit.py exists to prevent, and an
# evidence script with its own copy would reintroduce it.
_ns = {"np": np, "random": random, "P_ALPHA": 0.8,
       "TRAVERSAL_POLICY": "complete_dfs"}
load_defs("pretraining_torch.py", ["traverse_graph"], _ns)
traverse_NEW = _ns["traverse_graph"]


def with_policy(policy):
    """Run the shipped traversal under one TRAVERSAL_POLICY value."""
    _ns["TRAVERSAL_POLICY"] = policy
    return traverse_NEW


def g_of(smi):
    m=Chem.MolFromSmiles(smi); ranks=list(Chem.CanonicalRankAtoms(m))
    g=nx.Graph(); g.add_nodes_from(range(m.GetNumAtoms()))
    g.add_edges_from([[b.GetBeginAtomIdx(),b.GetEndAtomIdx()] for b in m.GetBonds()])
    return m,g,ranks

# realistic drug-like sample
smis = ["CC(=O)Oc1ccccc1C(=O)O","CC(C)(C)c1ccccc1","CC(=O)Nc1ccc(O)cc1",
 "CN1C=NC2=C1C(=O)N(C)C(=O)N2C","CC(C)Cc1ccc(cc1)C(C)C(=O)O","OC(=O)c1ccccc1O",
 "CN(C)CCOC(c1ccccc1)c1ccccc1","CC(C)NCC(O)COc1cccc2ccccc12",
 "Cc1ccc(cc1)S(=O)(=O)N","CC(=O)OC1=CC=CC=C1C(=O)O","CN1CCC[C@H]1c1cccnc1",
 "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1","COc1ccc2cc(ccc2c1)C(C)C(=O)O",
 "CCN(CC)CCNC(=O)c1ccc(N)cc1","CC(C)(N)Cc1ccccc1","CN(C)C(=O)Nc1ccccc1",
 "c1ccc2c(c1)cccn2","C1CC2CCC1CC2","CCO","c1ccccc1"]
random.seed(0)
old_fail = new_fail = 0; mols_old_fail = []
N = 100
for smi in smis:
    m,g,ranks = g_of(smi)
    of = nf = 0
    for _ in range(N):
        ids,_ = traverse_OLD(g,ranks)
        if min(ids)<0: of += 1
        try:
            ids2,_ = traverse_NEW(g,ranks)
            if min(ids2)<0 or sorted(ids2)!=list(range(len(ranks))): nf += 1
        except ValueError: nf += 1
    old_fail += of; new_fail += nf
    if of: mols_old_fail.append((smi, of))
print(f"OLD traversal: {old_fail}/{len(smis)*N} traversals leave atoms unvisited")
print(f"NEW traversal: {new_fail}/{len(smis)*N}")
print(f"\nmolecules affected by OLD ({len(mols_old_fail)}/{len(smis)}):")
for smi, of in sorted(mols_old_fail, key=lambda x:-x[1]):
    print(f"  {of:>3}/{N}  {smi}")

# does NEW give a valid permutation and a sane log_p?
print("\nNEW traversal validity over 2000 random draws on the full set:")
bad = 0; lps = []
for smi in smis:
    m,g,ranks = g_of(smi)
    for _ in range(100):
        ids,lp = traverse_NEW(g,ranks)
        if sorted(ids)!=list(range(len(ranks))): bad += 1
        lps.append(lp)
print(f"  {bad} invalid permutations; log_p range [{min(lps):.3f}, {max(lps):.3f}]"
      f" (must be <= 0)")

# Does TRAVERSAL_POLICY = 'upstream_broken' actually reproduce traverse_OLD?
# Without this, the A/B above compares the fix against a story about the old
# code rather than against the old code itself.
print("\n'upstream_broken' vs the historical traverse_OLD, same seed per draw:")
tg_old_policy = with_policy('upstream_broken')
agree = disagree = 0
for smi in smis:
    m, g, ranks = g_of(smi)
    for trial in range(50):
        random.seed(1000 + trial)
        ids_hist, lp_hist = traverse_OLD(g, ranks)
        random.seed(1000 + trial)
        try:
            ids_pol, lp_pol = tg_old_policy(g, ranks)
        except ValueError:
            # The shipped function refuses to RETURN a corrupt permutation; the
            # historical one returned it. Agreement is then judged on whether
            # both left atoms unvisited.
            if min(ids_hist) < 0:
                agree += 1
            else:
                disagree += 1
            continue
        if ids_pol == ids_hist and abs(lp_pol - lp_hist) < 1e-12:
            agree += 1
        else:
            disagree += 1
with_policy('complete_dfs')
print(f"  {agree}/{agree + disagree} draws match the historical traversal exactly "
      f"(or agree that it left atoms unvisited)")
if disagree:
    print(f"  {disagree} disagree — the 'upstream_broken' policy does NOT faithfully "
          f"reproduce the old code, so the A/B above is not measuring what it claims")

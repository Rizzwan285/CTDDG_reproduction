"""What does the legacy clamp+epsilon path actually do to gradients?

Runs the SHIPPED `action_log_probs` out of pretraining_torch.py; paths resolve
relative to this file, so it works from any directory.
"""
import torch

from _nbsource import load_defs, source_path

SRC = "pretraining_torch.py"
ns = load_defs(SRC, ["segment_sum", "segment_max", "action_log_probs"],
               {"torch": torch})
alp, ssum = ns["action_log_probs"], ns["segment_sum"]
print(f"source: {source_path(SRC)}\n")
rep = torch.tensor([0,0,0,0,1,1,1,2,2,2,2])

print("Gradient of the CHOSEN action's log-prob wrt its own logit.")
print("This is the corrective signal for 'make this action more likely'.\n")
print(f"{'logit scale':>12} {'p(chosen)':>12} {'stable |g|':>12} {'legacy |g|':>12}  note")
for scale in (1.0, 5.0, 20.0, 60.0, 100.0, 1000.0):
    torch.manual_seed(3)
    base_x = torch.randn(11,1,264)*scale
    base_t = torch.randn(3,1,1)*scale
    # pick the LEAST likely action in molecule 0 - what a suppressed action looks like
    flat = base_x[0,0]
    j = int(flat.argmin())
    out = {}
    for stable in (True, False):
        rx = base_x.clone().requires_grad_(True); rt = base_t.clone().requires_grad_(True)
        a,_ = alp(rx, rt, rep, 3, stable=stable)
        a[0,0,j].backward()
        out[stable] = (float(a[0,0,j].exp()), float(rx.grad[0,0,j].abs()))
    note = ""
    if out[False][1] < 1e-8 <= out[True][1]: note = "legacy signal is dead"
    print(f"{scale:>12.0f} {out[True][0]:>12.3e} {out[True][1]:>12.4g} {out[False][1]:>12.4g}  {note}")

print("\nPer-action log-prob value (what enters `l`):")
print(f"{'logit scale':>12} {'stable logp':>14} {'legacy logp':>14}  note")
for scale in (1.0, 20.0, 60.0, 100.0, 1000.0):
    torch.manual_seed(3)
    rx = torch.randn(11,1,264)*scale; rt = torch.randn(3,1,1)*scale
    j = int(rx[0,0].argmin())
    a1,_ = alp(rx, rt, rep, 3, stable=True)
    a0,_ = alp(rx, rt, rep, 3, stable=False)
    v1, v0 = float(a1[0,0,j]), float(a0[0,0,j])
    note = "legacy FLOORED at ln(1e-10) = -23.026" if abs(v0 + 23.0259) < 1e-3 else ""
    print(f"{scale:>12.0f} {v1:>14.3f} {v0:>14.3f}  {note}")

print("\nClamp gradient: fraction of logits outside +/-80 that receive EXACTLY zero grad")
torch.manual_seed(3)
rx = (torch.randn(11,1,264)*100).requires_grad_(True); rt=(torch.randn(3,1,1)*100).requires_grad_(True)
a0,_ = alp(rx, rt, rep, 3, stable=False)
a0.sum().backward()
outside = (rx.detach().abs() >= 80.0)
print(f"  {int(outside.sum())}/{outside.numel()} logits outside +/-80 "
      f"({100*float(outside.float().mean()):.1f}%)")
print(f"  of those, {int((rx.grad[outside]==0).sum())} have exactly zero gradient")
rx2 = (torch.randn(11,1,264)*100).requires_grad_(True); rt2=(torch.randn(3,1,1)*100).requires_grad_(True)
a1,_ = alp(rx2, rt2, rep, 3, stable=True)
a1.sum().backward()
out2 = (rx2.detach().abs() >= 80.0)
print(f"  stable path: {int((rx2.grad[out2]==0).sum())} of {int(out2.sum())} zero")

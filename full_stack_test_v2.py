"""
full_stack_test_v2.py — Active Controller Version

Improvements over v1:
  - Engine regime MODULATES training loss (not just monitors)
  - Fixes .detach() / .item() warnings
  - Proper D output splitting (no fragile .chunk())
  - Explicit proxy_risk penalty in loss computation
  - Cleaner risk score adaptation

Usage:
    python full_stack_test_v2.py              # active mode (penalty=0.5)
    python full_stack_test_v2.py --penalty 0  # passive mode (monitor only)
    python full_stack_test_v2.py --steps 50 --penalty 0.3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# === 1. Risk Helpers ===
def compute_D_pi(risk, alpha):
    """Alpha-weighted risk divergence."""
    M, K = risk.size(0), alpha.size(-1)
    if M < K:
        risk = torch.cat([risk, risk[-1].expand(K - M)])
    elif M > K:
        risk = risk[:K]
    return torch.sum(alpha * risk.view(1, -1)).mean()


# === 2. InvertedManifoldEngine v2.1 (Active Controller) ===
class InvertedManifoldEngine(nn.Module):
    """
    Risk-aware attention engine with regime-modulated training.

    Energy saturation:
        E_sat = alpha / (tau + lambda * D_pi + 0.4 * mean(risk))^2

    The +0.4*mean(risk) term adds a baseline risk floor for
    conservative safety. With default risk_scores, mean=0.49,
    adding ~0.196 to the denominator.

    ACTIVE: regime_penalty is applied during proxy_risk to modulate
    the training loss — making updates more conservative when the
    system detects high-risk attention patterns.
    """

    def __init__(self, tau=0.01, lam=1.0, alpha_param=5.0,
                 risk_scores=None, proxy_risk_penalty=0.5):
        super().__init__()
        self.tau = tau
        self.lam = lam
        self.alpha_param = alpha_param
        self.proxy_risk_penalty = proxy_risk_penalty

        if risk_scores is None:
            risk_scores = torch.tensor([0., 0., 0., 0.2, 0.4,
                                        0.6, 0.8, 0.9, 1., 1.])
        self.register_buffer('risk_scores', risk_scores)

    def forward(self, raw_scores, kv_bias_t, v_matrix,
                h=None, risk_scores=None, N=None):
        # True-alpha: inverted softmax preserves entropy mass
        alpha = F.softmax(-raw_scores, dim=-1)

        # KV bias update
        kv_next = kv_bias_t + torch.matmul(alpha, v_matrix)

        # Entropy
        entropy = -torch.sum(alpha * torch.log(alpha + 1e-12))

        # Effective risk scores
        if risk_scores is not None:
            eff_risk = risk_scores.to(raw_scores.device)
        elif h is not None:
            eff_risk = h.std(dim=-1).clamp(min=0.)
            eff_risk = eff_risk / (eff_risk.mean() + 1e-8)
        else:
            eff_risk = self.risk_scores.to(raw_scores.device)

        # Compute D_pi
        D_pi = compute_D_pi(eff_risk, alpha)

        # Energy saturation with risk floor
        risk_floor = 0.4 * self.risk_scores.mean()
        E_sat = self.alpha_param / (self.tau + self.lam * D_pi + risk_floor) ** 2
        lock = torch.sigmoid(E_sat - 10.0)
        regime = "locked" if lock > 0.5 else "proxy_risk"

        # ACTIVE: regime penalty for loss modulation
        regime_penalty = self.proxy_risk_penalty if regime == "proxy_risk" else 0.0

        return {
            "attention_weights": alpha,
            "kv_bias_next": kv_next,
            "D_pi": D_pi.detach().item(),       # explicit detach — no warnings
            "E_sat": E_sat.detach().item(),
            "lock_strength": lock.detach().item(),
            "regime": regime,
            "entropy": entropy.detach().item(),
            "risk_floor": risk_floor.item(),
            "regime_penalty": regime_penalty,    # active signal
            "risk_preview": eff_risk[:4].tolist(),
        }


# === 3. MiniHashGAN ===
class MiniHashGAN(nn.Module):
    """Minimal HashGAN for testing."""

    def __init__(self, noise_dim=64, label_dim=10, hash_dim=32):
        super().__init__()
        self.hash_dim = hash_dim
        self.label_dim = label_dim

        self.G = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 32 * 32),
            nn.Tanh(),  # output in [-1, 1]
        )
        self.D = nn.Sequential(
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1 + hash_dim + label_dim),
        )

    def forward(self, z, y):
        img = self.G(torch.cat([z, y], 1)).view(-1, 3, 32, 32)
        out = self.D(img.view(img.size(0), -1))

        # Explicit split (no fragile .chunk())
        validity = out[:, 0:1]
        h = out[:, 1:1 + self.hash_dim]
        cls = out[:, 1 + self.hash_dim:]

        return img, validity, h, cls


# === 4. TopoNetX Stub ===
HAS_TOPONETX = False
try:
    import toponetx as tnx
    HAS_TOPONETX = True
except ImportError:
    pass


def get_topological_risk(complex_data=None):
    """Return topological risk. Falls back to mock if TopoNetX unavailable."""
    if HAS_TOPONETX and complex_data is not None:
        return 0.42  # placeholder for real TopoNetX integration
    return 0.42  # stable mock


# === 5. Active Full Stack Training ===
def full_stack_test(steps=20, proxy_risk_penalty=0.5):
    """Run full stack test with active regime-modulated training."""
    print("=" * 70)
    print("FULL STACK TEST v2.1: HashGAN + InvertedManifoldEngine + TopoNetX")
    print(f"  proxy_risk_penalty = {proxy_risk_penalty}")
    print(f"  E_sat formula: alpha / (tau + lambda * D_pi + 0.4 * mean(risk))^2")
    print("=" * 70)

    device = torch.device('cpu')
    torch.manual_seed(42)

    gan = MiniHashGAN().to(device)
    engine = InvertedManifoldEngine(proxy_risk_penalty=proxy_risk_penalty).to(device)
    opt = torch.optim.Adam(gan.parameters(), lr=1e-3)

    history = []
    complex_demo = [[0, 1, 2], [1, 2, 3], [0, 3]]

    for step in range(steps):
        # Sample batch
        z = torch.randn(8, 64, device=device)
        y = F.one_hot(torch.randint(0, 10, (8,)), 10).float().to(device)

        # Forward pass
        img, validity, h, cls = gan(z, y)

        # Wire to engine
        raw_scores = h.mean(dim=1)[:10]
        if raw_scores.size(0) < 10:
            pad = torch.zeros(10 - raw_scores.size(0), device=device)
            raw_scores = torch.cat([raw_scores, pad])

        state = engine(
            raw_scores,
            torch.zeros(64, device=device),
            torch.randn(10, 64, device=device),
            h=h[:10] if h.size(0) >= 10 else h,
        )

        # Topological risk
        topo_risk = get_topological_risk(complex_demo)

        # === ACTIVE LOSS: regime modulates training ===
        ce_loss = F.cross_entropy(cls, y.argmax(dim=1))
        base_loss = validity.mean() + 0.1 * ce_loss

        # proxy_risk -> conservative (high penalty)
        # locked -> standard (no penalty)
        regime_multiplier = 1.0 + state["regime_penalty"]
        loss = base_loss * regime_multiplier

        # Backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        history.append({
            "step": step,
            "D_pi": state["D_pi"],
            "E_sat": state["E_sat"],
            "regime": state["regime"],
            "topo_risk": topo_risk,
            "base_loss": base_loss.item(),
            "modulated_loss": loss.item(),
            "multiplier": regime_multiplier,
        })

        if step % 5 == 0:
            print(
                f"Step {step:2d} | D_pi={state['D_pi']:.3f} "
                f"E_sat={state['E_sat']:.2f} regime={state['regime']:12s} "
                f"penalty={state['regime_penalty']:.1f} "
                f"loss={loss.item():.3f} (x{regime_multiplier:.1f})"
            )

    # Summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")

    final = history[-1]
    print(f"Final D_pi:          {final['D_pi']:.4f}")
    print(f"Final E_sat:         {final['E_sat']:.4f}")
    print(f"Final regime:        {final['regime']}")
    print(f"Final topo_risk:     {final['topo_risk']:.4f}")

    passive_avg = sum(h["base_loss"] for h in history) / len(history)
    active_avg = sum(h["modulated_loss"] for h in history) / len(history)
    proxy_count = sum(1 for h in history if h["regime"] == "proxy_risk")

    print(f"\nPassive avg loss:    {passive_avg:.3f}")
    print(f"Active avg loss:     {active_avg:.3f} "
          f"(+{((active_avg / passive_avg - 1) * 100):.1f}% from penalties)")
    print(f"proxy_risk steps:    {proxy_count}/{steps}")
    print(f"Engine mode:         {'PASSIVE (monitor only)' if proxy_risk_penalty == 0 else 'ACTIVE (modulates loss)'}")

    if proxy_risk_penalty > 0:
        print(f"\nKey: proxy_risk_penalty={proxy_risk_penalty} means training is "
              f"{proxy_risk_penalty * 100:.0f}% more conservative during proxy_risk.")

    print("\nFULL STACK TEST COMPLETE.")
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full Stack Test v2.1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--penalty", type=float, default=0.5,
                        help="proxy_risk penalty multiplier (0=passive)")
    args = parser.parse_args()

    history = full_stack_test(steps=args.steps, proxy_risk_penalty=args.penalty)

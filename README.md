# InvertedManifoldEngine

Risk-aware attention mechanism with entropy-based safety gating, Lyapunov stability, and composite reward signaling for AI alignment.

## Overview

This project integrates three key components:

1. **InvertedManifoldEngine** - Risk-aware attention with true-alpha D_pi
2. **HashGAN (PyTorch port)** - Deep learning to hash with Wasserstein GAN
3. **Entropy-GAT** - Graph Attention with entropy-based lock prevention

## Core Results

```
D_pi (true alpha):  0.9202  (HIGH risk signal, entropy drives caution)
entropy_bits:       1.5260  (moderate entropy, meaningful spread)
E_sat:              5.78    (below lock threshold)
lock_strength:      0.0145  (very low - staying cautious)
regime:             proxy_risk (Entropy-GAT PROTECTED)
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py --mode sanity    # Run all checks
python main.py --mode report    # System report
python main.py --mode train --cfg cifar_step1 --iters 1000
```

## Architecture

### InvertedManifoldEngine
- True-alpha: softmax(-scores) preserves entropy mass
- Risk-weighted divergence D_pi = sum(alpha * risk_scores)
- Energy saturation E_sat = alpha_param / (tau + lam * D_pi)^2
- Regime gating: locked | proxy_risk
- Lyapunov stability T_lyap for alignment monitoring

### HashGAN
- Generator: Residual blocks with label conditioning
- Discriminator: ResNet-style or AlexNet
- Loss: WGAN-GP + ACGAN cross-entropy

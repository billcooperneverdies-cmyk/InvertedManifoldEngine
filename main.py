#!/usr/bin/env python
"""
InvertedManifoldEngine - Main Entry Point

Usage:
    python main.py --mode train --cfg cifar_step1
    python main.py --mode eval --cfg cifar_eval
    python main.py --mode sanity
    python main.py --mode report
"""
import argparse
import sys
import torch

from config import Config, cifar_step1_config, cifar_step2_config, cifar_eval_config
from inverted_manifold_engine import InvertedManifoldEngine, EntropyGATLayer, create_default_engine
from hashgan_pytorch import create_hashgan_model
from train import CIFAR10Trainer


def run_sanity_check():
    print("=" * 70)
    print("InvertedManifoldEngine - Full Pipeline Sanity Check")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"PyTorch: {torch.__version__}")

    # 1. Engine Check
    print("\n" + "-" * 40)
    print("[1/5] InvertedManifoldEngine")
    print("-" * 40)
    engine = create_default_engine().to(device)
    raw_scores = torch.tensor([5.00, 4.50, 4.00, 3.00, 2.00,
                               1.00, 0.50, 0.10, -0.50, -1.00], device=device)
    v_matrix = torch.randn(10, 64, device=device)
    kv_bias = torch.zeros(64, device=device)
    state = engine(raw_scores, kv_bias, v_matrix)
    print(f"  D_pi:          {state['D_pi']:.4f} (expected ~0.9202)")
    print(f"  entropy_bits:  {state['entropy_bits']:.4f} (expected ~1.5260)")
    print(f"  regime:        {state['regime']} (expected proxy_risk)")
    assert abs(state['D_pi'] - 0.9202) < 0.01
    assert state['regime'] == 'proxy_risk'
    print("  PASS")

    # 2. EntropyGAT Check
    print("\n" + "-" * 40)
    print("[2/5] EntropyGATLayer")
    print("-" * 40)
    gat = EntropyGATLayer(in_features=128, out_features=64, num_heads=4).to(device)
    x = torch.randn(10, 128, device=device)
    out, gat_state = gat(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Regime: {gat_state['regime']}")
    assert out.shape == (10, 64)
    print("  PASS")

    # 3. HashGAN Check
    print("\n" + "-" * 40)
    print("[3/5] HashGAN (Generator + Discriminator)")
    print("-" * 40)
    G, D = create_hashgan_model(label_dim=10, hash_dim=64,
                                 g_arch="NORM", d_arch="NORM", output_size=32)
    G, D = G.to(device), D.to(device)
    z = torch.randn(4, 256, device=device)
    labels = torch.randint(0, 10, (4,), device=device)
    labels_oh = torch.nn.functional.one_hot(labels, 10).float()
    fake = G(z, labels_oh)
    wgan, hash_c = D(fake)
    print(f"  Generated: {fake.shape} range [{fake.min():.2f}, {fake.max():.2f}]")
    print(f"  WGAN:      {wgan.shape}")
    print(f"  Hash:      {hash_c.shape}")
    assert fake.shape == (4, 3, 32, 32)
    print("  PASS")

    # 4. Config Check
    print("\n" + "-" * 40)
    print("[4/5] Configuration System")
    print("-" * 40)
    cfg = cifar_step1_config()
    d = cfg.to_dict()
    cfg2 = Config.from_dict(d)
    assert cfg.MODEL.HASH_DIM == cfg2.MODEL.HASH_DIM
    print("  Round-trip: OK")
    print("  PASS")

    # 5. Integration Check
    print("\n" + "-" * 40)
    print("[5/5] Integration: Engine + GAT + HashGAN")
    print("-" * 40)
    disc_features = torch.randn(16, 512, device=device)
    gat_integration = EntropyGATLayer(512, 256, num_heads=2).to(device)
    processed, int_state = gat_integration(disc_features)
    print(f"  Disc features: {disc_features.shape}")
    print(f"  GAT processed: {processed.shape}")
    print(f"  Regime: {int_state['regime']}, D_pi: {int_state['D_pi']:.4f}")
    assert processed.shape == (16, 256)
    print("  PASS")

    engine.entropy_gat_report()
    print("=" * 70)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 70)
    return 0


def run_report():
    print("=" * 70)
    print("InvertedManifoldEngine - System Report")
    print("=" * 70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nPyTorch: {torch.__version__}")
    print(f"Device:  {device}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")

    print("\n[HashGAN Model Sizes]")
    for g_arch in ["NORM", "GOOD"]:
        for d_arch in ["NORM", "ALEXNET"]:
            G, D = create_hashgan_model(g_arch=g_arch, d_arch=d_arch, output_size=32)
            g_p = sum(p.numel() for p in G.parameters())
            d_p = sum(p.numel() for p in D.parameters())
            print(f"  G={g_arch:5s} D={d_arch:8s} | G: {g_p:>10,}  D: {d_p:>10,}  Total: {g_p+d_p:>10,}")

    print("\n[Configuration Presets]")
    for name, fn in [("Step 1", cifar_step1_config), ("Step 2", cifar_step2_config), ("Eval", cifar_eval_config)]:
        c = fn()
        print(f"  {name:20s} | Iters: {c.TRAIN.ITERS:>6}  Batch: {c.TRAIN.BATCH_SIZE:>4}")
    print("=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(description='InvertedManifoldEngine')
    parser.add_argument('--mode', type=str, default='sanity',
                        choices=['train', 'eval', 'sanity', 'report'])
    parser.add_argument('--cfg', type=str, default='cifar_step1',
                        choices=['cifar_step1', 'cifar_step2', 'cifar_eval'])
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()

    cfg_map = {
        'cifar_step1': cifar_step1_config,
        'cifar_step2': cifar_step2_config,
        'cifar_eval': cifar_eval_config
    }

    if args.mode == 'sanity':
        return run_sanity_check()
    elif args.mode == 'report':
        return run_report()
    elif args.mode == 'train':
        cfg = cfg_map[args.cfg]()
        if args.iters:
            cfg.TRAIN.ITERS = args.iters
        trainer = CIFAR10Trainer(cfg)
        trainer.train()
        return 0
    elif args.mode == 'eval':
        cfg = cfg_map[args.cfg]()
        print(f"Evaluation mode: {args.cfg}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

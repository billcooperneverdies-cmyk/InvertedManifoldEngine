"""
Training Pipeline for InvertedManifoldEngine + HashGAN on CIFAR-10
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config
from inverted_manifold_engine import InvertedManifoldEngine, EntropyGATLayer
from hashgan_pytorch import create_hashgan_model, WGANGPLoss, ACGANLoss


class CIFAR10Trainer:
    """Unified trainer for HashGAN + InvertedManifoldEngine on CIFAR-10."""

    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg.ensure_dirs()

        self.trainloader, self.testloader = self._build_dataloaders()

        self.G, self.D = create_hashgan_model(
            label_dim=cfg.DATA.LABEL_DIM, hash_dim=cfg.MODEL.HASH_DIM,
            g_arch=cfg.MODEL.G_ARCHITECTURE, d_arch=cfg.MODEL.D_ARCHITECTURE,
            output_size=cfg.DATA.WIDTH_HEIGHT,
            dim_g=cfg.MODEL.DIM_G, dim_d=cfg.MODEL.DIM_D)
        self.G.to(self.device)
        self.D.to(self.device)

        if cfg.ENGINE.ENABLED:
            self.engine = InvertedManifoldEngine(
                tau=cfg.ENGINE.TAU, lam=cfg.ENGINE.LAM,
                alpha_param=cfg.ENGINE.ALPHA_PARAM,
                entropy_threshold=cfg.ENGINE.ENTROPY_THRESHOLD).to(self.device)
        else:
            self.engine = None

        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.TRAIN.LR, betas=(0.0, 0.9))
        self.opt_g = None
        if cfg.TRAIN.G_LR > 0:
            self.opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.TRAIN.G_LR, betas=(0.0, 0.9))

        self.wgan_loss = WGANGPLoss()
        self.acgan_loss = ACGANLoss(alpha=cfg.TRAIN.CROSS_ENTROPY_ALPHA)

        self.metrics = {
            'd_loss': [], 'g_loss': [], 'w_dist': [],
            'D_pi': [], 'entropy': [], 'regime': [],
            'lock_strength': [], 'T_lyap': []
        }

        self._load_pretrained()

    def _build_dataloaders(self):
        transform = transforms.Compose([
            transforms.Resize(self.cfg.DATA.WIDTH_HEIGHT),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(
            root=self.cfg.DATA.DATA_ROOT, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=self.cfg.DATA.DATA_ROOT, train=False, download=True, transform=transform)

        if len(trainset) > self.cfg.DATA.DB_SIZE:
            indices = torch.randperm(len(trainset))[:self.cfg.DATA.DB_SIZE]
            trainset = Subset(trainset, indices)

        trainloader = DataLoader(trainset, batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                 shuffle=True, num_workers=2, drop_last=True)
        testloader = DataLoader(testset, batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                shuffle=False, num_workers=2)
        return trainloader, testloader

    def _load_pretrained(self):
        if self.cfg.MODEL.G_PRETRAINED_MODEL_PATH and os.path.exists(self.cfg.MODEL.G_PRETRAINED_MODEL_PATH):
            self.G.load_state_dict(torch.load(self.cfg.MODEL.G_PRETRAINED_MODEL_PATH, map_location=self.device))
            print(f"Loaded generator: {self.cfg.MODEL.G_PRETRAINED_MODEL_PATH}")
        if self.cfg.MODEL.D_PRETRAINED_MODEL_PATH and os.path.exists(self.cfg.MODEL.D_PRETRAINED_MODEL_PATH):
            self.D.load_state_dict(torch.load(self.cfg.MODEL.D_PRETRAINED_MODEL_PATH, map_location=self.device))
            print(f"Loaded discriminator: {self.cfg.MODEL.D_PRETRAINED_MODEL_PATH}")

    def _lr_decay(self, iteration):
        if self.cfg.TRAIN.DECAY:
            return max(0., 1. - iteration / self.cfg.TRAIN.ITERS)
        return 1.0

    def _compute_engine_state(self, real_images, fake_images):
        if self.engine is None:
            return None
        with torch.no_grad():
            _, real_hash = self.D(real_images)
            _, fake_hash = self.D(fake_images)
            real_flat = real_hash.view(-1)[:10]
            fake_flat = fake_hash.view(-1)[:10]
            raw_scores = torch.cat([real_flat, fake_flat])
            v_matrix = torch.randn(min(10, raw_scores.numel()), 64, device=self.device)
            kv_bias = torch.zeros(64, device=self.device)
            state = self.engine(raw_scores[:10], kv_bias, v_matrix)
        return state

    def _train_step(self, real_images, real_labels, iteration):
        batch_size = real_images.size(0)
        decay = self._lr_decay(iteration)

        for param_group in self.opt_d.param_groups:
            param_group['lr'] = self.cfg.TRAIN.LR * decay
        if self.opt_g:
            for param_group in self.opt_g.param_groups:
                param_group['lr'] = self.cfg.TRAIN.G_LR * decay

        for _ in range(self.cfg.TRAIN.N_CRITIC):
            self.opt_d.zero_grad()
            real_score, real_hash = self.D(real_images)

            noise = torch.randn(batch_size, 256, device=self.device)
            fake_labels = F.one_hot(torch.randint(0, 10, (batch_size,), device=self.device), 10).float()
            fake_images = self.G(noise, fake_labels).detach()
            fake_score, fake_hash = self.D(fake_images)

            d_loss_wgan, w_dist, gp = self.wgan_loss.discriminator_loss(
                real_score, fake_score, real_images, fake_images, self.D,
                gp_weight=self.cfg.TRAIN.WGAN_SCALE_GP)

            d_loss_acgan = self.acgan_loss(real_hash, real_labels)
            if self.cfg.TRAIN.ACGAN_SCALE_FAKE != 0:
                d_loss_acgan += self.cfg.TRAIN.ACGAN_SCALE_FAKE * \
                    self.acgan_loss(real_hash, real_labels, fake_hash, fake_labels, partial=True)

            d_loss = self.cfg.TRAIN.WGAN_SCALE * d_loss_wgan + self.cfg.TRAIN.ACGAN_SCALE * d_loss_acgan
            d_loss.backward()
            self.opt_d.step()

        g_loss = torch.tensor(0.0)
        if self.opt_g and iteration > 0:
            self.opt_g.zero_grad()
            noise = torch.randn(batch_size, 256, device=self.device)
            gen_labels = F.one_hot(torch.randint(0, 10, (batch_size,), device=self.device), 10).float()
            gen_images = self.G(noise, gen_labels)
            gen_score, gen_hash = self.D(gen_images)
            g_loss_wgan = self.wgan_loss.generator_loss(gen_score)
            g_loss_acgan = self.acgan_loss(gen_hash, gen_labels)
            g_loss = self.cfg.TRAIN.WGAN_SCALE_G * g_loss_wgan + self.cfg.TRAIN.ACGAN_SCALE_G * g_loss_acgan
            g_loss.backward()
            self.opt_g.step()

        engine_state = self._compute_engine_state(real_images, fake_images)

        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item() if isinstance(g_loss, torch.Tensor) else 0.0,
            'w_dist': w_dist,
            **({k: engine_state[k] for k in ['D_pi', 'entropy_bits', 'regime',
                                              'lock_strength', 'T_lyap']}
               if engine_state else {})
        }

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training: {self.cfg.DATA.OUTPUT_DIR}")
        print(f"Device: {self.device}")
        print(f"Iters: {self.cfg.TRAIN.ITERS}")
        print(f"Engine: {'ENABLED' if self.engine else 'DISABLED'}")
        print(f"{'='*60}\n")

        data_iter = iter(self.trainloader)
        pbar = tqdm(range(self.cfg.TRAIN.ITERS), desc='Training')

        for iteration in pbar:
            try:
                real_images, real_labels_idx = next(data_iter)
            except StopIteration:
                data_iter = iter(self.trainloader)
                real_images, real_labels_idx = next(data_iter)

            real_images = real_images.to(self.device)
            real_labels = F.one_hot(real_labels_idx, self.cfg.DATA.LABEL_DIM).float().to(self.device)

            metrics = self._train_step(real_images, real_labels, iteration)
            desc = f"D:{metrics['d_loss']:.3f} G:{metrics['g_loss']:.3f} W:{metrics['w_dist']:.3f}"
            if 'D_pi' in metrics:
                desc += f" D_pi:{metrics['D_pi']:.3f} R:{metrics['regime'][:4]}"
            pbar.set_description(desc)

            for k, v in metrics.items():
                if k in self.metrics:
                    self.metrics[k].append(v)

            if (iteration + 1) % self.cfg.TRAIN.SAMPLE_FREQUENCY == 0:
                self._save_samples(iteration)
            if (iteration + 1) % self.cfg.TRAIN.EVAL_FREQUENCY == 0:
                self._evaluate(iteration)
            if (iteration + 1) % self.cfg.TRAIN.CHECKPOINT_FREQUENCY == 0:
                self._save_checkpoint(iteration)

        self._save_checkpoint(self.cfg.TRAIN.ITERS - 1)

    @torch.no_grad()
    def _save_samples(self, iteration, n_samples=64):
        self.G.eval()
        noise = torch.randn(n_samples, 256, device=self.device)
        labels = F.one_hot(torch.arange(10, device=self.device).repeat(n_samples // 10 + 1),
                           10)[:n_samples].float()
        samples = self.G(noise, labels)
        samples = (samples + 1) / 2
        torchvision.utils.save_image(
            samples, f"{self.cfg.DATA.IMAGE_DIR}/samples_{iteration}.png", nrow=8)
        self.G.train()

    @torch.no_grad()
    def _evaluate(self, iteration):
        self.D.eval()
        hashes = []
        for images, _ in self.testloader:
            images = images.to(self.device)
            _, hash_codes = self.D(images)
            hashes.append(hash_codes.cpu())
        hashes = torch.cat(hashes)
        hash_entropy = -torch.sum(hashes * torch.log(torch.abs(hashes) + 1e-12), dim=1).mean()
        print(f"\n[Eval@{iteration}] Hash entropy: {hash_entropy:.4f}")
        self.D.train()

    def _save_checkpoint(self, iteration):
        g_path = os.path.join(self.cfg.DATA.MODEL_DIR, f"G_{iteration}.pth")
        d_path = os.path.join(self.cfg.DATA.MODEL_DIR, f"D_{iteration}.pth")
        torch.save(self.G.state_dict(), g_path)
        torch.save(self.D.state_dict(), d_path)
        print(f"\nSaved: G_{iteration}.pth, D_{iteration}.pth")

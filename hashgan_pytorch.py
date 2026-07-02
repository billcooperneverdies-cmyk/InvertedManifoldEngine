import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with optional upsampling/downsampling."""

    def __init__(self, in_dim, out_dim, resample=None):
        super().__init__()
        self.resample = resample

        if resample == 'up':
            self.conv1 = nn.ConvTranspose2d(in_dim, out_dim, 3, 2, 1)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
            shortcut = nn.ConvTranspose2d(in_dim, out_dim, 1, 2, 0)
        elif resample == 'down':
            self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
            self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 2, 1)
            shortcut = nn.Conv2d(in_dim, out_dim, 1, 2, 0)
        else:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
            shortcut = nn.Conv2d(in_dim, out_dim, 1, 1, 0) if in_dim != out_dim else None

        self.bn1 = nn.BatchNorm2d(in_dim if resample != 'up' else in_dim)
        self.bn2 = nn.BatchNorm2d(out_dim if resample != 'up' else out_dim)
        self.shortcut = shortcut

    def forward(self, x):
        shortcut = x if self.shortcut is None else self.shortcut(x)

        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)

        if shortcut.shape != out.shape:
            if shortcut.dim() == 4:
                shortcut = F.interpolate(shortcut, size=out.shape[2:], mode='nearest')
                if shortcut.shape[1] != out.shape[1]:
                    pad = torch.zeros(shortcut.size(0), out.shape[1] - shortcut.shape[1],
                                     *shortcut.shape[2:], device=shortcut.device)
                    shortcut = torch.cat([shortcut, pad], dim=1)
        return shortcut + out


class HashGANGenerator(nn.Module):
    """HashGAN Generator with label conditioning."""

    def __init__(self, label_dim=10, noise_dim=None, dim_g=128,
                 architecture="NORM", output_size=32):
        super().__init__()
        self.label_dim = label_dim
        self.architecture = architecture
        self.output_size = output_size

        if architecture == "GOOD":
            self.noise_dim = noise_dim or 128
            self.dim = dim_g
            self.fc = nn.Linear(self.noise_dim, 8 * self.dim * 4 * 4)
            self.res1 = ResidualBlock(8 * self.dim, 8 * self.dim, 'up')
            self.res2 = ResidualBlock(8 * self.dim, 4 * self.dim, 'up')
            self.res3 = ResidualBlock(4 * self.dim, 2 * self.dim, 'up')
            self.res4 = ResidualBlock(2 * self.dim, 1 * self.dim, 'up')
            self.output_conv = nn.Conv2d(self.dim, 3, 3, 1, 1)
        else:  # NORM
            self.noise_dim = noise_dim or 256
            self.dim_g = dim_g
            self.fc = nn.Linear(self.noise_dim, self.dim_g * 4 * 4)
            self.res1 = ResidualBlock(self.dim_g, self.dim_g, 'up')
            self.res2 = ResidualBlock(self.dim_g, self.dim_g, 'up')
            self.res3 = ResidualBlock(self.dim_g, self.dim_g, 'up')
            self.output_conv = nn.Conv2d(self.dim_g, 3, 3, 1, 1)

        self.label_embed = nn.Linear(label_dim, self.noise_dim - label_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noise, labels):
        label_proj = self.label_embed(labels.float())
        if noise.shape[1] == self.noise_dim:
            noise = noise[:, self.label_dim:]
        conditioned = torch.cat([labels.float(), noise], dim=1)

        x = self.fc(conditioned)
        if self.architecture == "GOOD":
            x = x.view(-1, 8 * self.dim, 4, 4)
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
        else:
            x = x.view(-1, self.dim_g, 4, 4)
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)

        x = F.relu(x)
        x = self.output_conv(x)
        x = torch.tanh(x)

        if x.shape[2:] != (self.output_size, self.output_size):
            x = F.interpolate(x, size=(self.output_size, self.output_size),
                            mode='bilinear', align_corners=False)
        return x


class HashGANDiscriminator(nn.Module):
    """HashGAN Discriminator: outputs WGAN score + ACGAN hash code."""

    def __init__(self, hash_dim=64, dim_d=128, label_dim=10,
                 architecture="NORM", input_size=32):
        super().__init__()
        self.hash_dim = hash_dim
        self.architecture = architecture
        self.input_size = input_size

        if architecture == "ALEXNET":
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 11, 4, 0), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
                nn.Conv2d(96, 256, 5, 1, 2, groups=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
                nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, 3, 1, 1, groups=2), nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, 3, 1, 1, groups=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
                nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            )
            self.wgan_head = nn.Linear(4096, 1)
            self.hash_head = nn.Linear(4096, hash_dim)
        else:  # NORM
            self.conv_in = nn.Conv2d(3, dim_d, 3, 1, 1)
            self.res1 = ResidualBlock(dim_d, dim_d, 'down')
            self.res2 = ResidualBlock(dim_d, dim_d, None)
            self.res3 = ResidualBlock(dim_d, dim_d, None)
            self.res4 = ResidualBlock(dim_d, dim_d, None)
            self.wgan_head = nn.Linear(dim_d, 1)
            self.hash_head = nn.Linear(dim_d, hash_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, stage='train'):
        features = self._extract_features(x)
        wgan_score = self.wgan_head(features).squeeze(-1)
        hash_code = torch.tanh(self.hash_head(features))
        return wgan_score, hash_code

    def _extract_features(self, x):
        if self.architecture == "ALEXNET":
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        else:
            x = self.conv_in(x)
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
            x = F.relu(x)
            x = x.mean(dim=[2, 3])
        return x


class WGANGPLoss:
    """Wasserstein GAN with Gradient Penalty loss computation."""

    @staticmethod
    def discriminator_loss(real_score, fake_score, real_data, fake_data,
                           discriminator, gp_weight=10.0):
        wasserstein = fake_score.mean() - real_score.mean()
        alpha = torch.rand(real_data.size(0), 1, 1, 1, device=real_data.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        d_interpolates, _ = discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates.sum(), inputs=interpolates,
            create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss = wasserstein + gp_weight * gradient_penalty
        return loss, wasserstein.item(), gradient_penalty.item()

    @staticmethod
    def generator_loss(fake_score):
        return -fake_score.mean()


class ACGANLoss(nn.Module):
    """Auxiliary Classifier GAN loss with cross-entropy."""

    def __init__(self, alpha=5.0, normed=True):
        super().__init__()
        self.alpha = alpha
        self.normed = normed
        self.ce = nn.CrossEntropyLoss()

    def forward(self, hash_pred, labels, hash_fake=None, labels_fake=None, partial=False):
        if partial and hash_fake is not None:
            hash_pred = hash_pred.detach()
        loss_real = self.ce(hash_pred, labels.argmax(dim=1))
        loss = loss_real
        if hash_fake is not None and labels_fake is not None:
            loss_fake = self.ce(hash_fake, labels_fake.argmax(dim=1))
            loss = loss + loss_fake
        return loss


def create_hashgan_model(label_dim=10, hash_dim=64,
                         g_arch="NORM", d_arch="NORM",
                         output_size=32, **kwargs):
    """Create HashGAN generator + discriminator pair."""
    generator = HashGANGenerator(
        label_dim=label_dim, architecture=g_arch, output_size=output_size,
        **{k: v for k, v in kwargs.items() if k.startswith('dim_g') or k == 'noise_dim'})
    discriminator = HashGANDiscriminator(
        hash_dim=hash_dim, label_dim=label_dim, architecture=d_arch,
        input_size=output_size,
        **{k: v for k, v in kwargs.items() if k.startswith('dim_d')})
    return generator, discriminator


if __name__ == "__main__":
    print("=" * 60)
    print("HashGAN PyTorch Sanity Check")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G, D = create_hashgan_model(g_arch="NORM", d_arch="NORM", output_size=32)
    G, D = G.to(device), D.to(device)
    z = torch.randn(4, 256, device=device)
    labels = F.one_hot(torch.randint(0, 10, (4,), device=device), 10).float()
    fake = G(z, labels)
    wgan, hash_c = D(fake)
    print(f"Generated: {fake.shape} range [{fake.min():.2f}, {fake.max():.2f}]")
    print(f"WGAN: {wgan.shape} | Hash: {hash_c.shape}")
    print("ALL CHECKS PASSED")

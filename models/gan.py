import torch.nn as nn


class GANGenerator(nn.Module):
    """GAN Generator: noise -> composition (40-dim)"""
    def __init__(self, noise_dim=100, comp_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, comp_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


class GANDiscriminator(nn.Module):
    """GAN Discriminator: composition (40-dim) -> real/fake"""
    def __init__(self, comp_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(comp_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

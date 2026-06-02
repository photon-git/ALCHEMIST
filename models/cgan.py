import torch.nn as nn


class CGANGenerator(nn.Module):
    """CGAN Generator: (noise + properties) -> composition (40-dim)

    Conditions generation on 26-dim thermodynamic properties.
    Input: concat(z[noise_dim], prop[prop_dim])
    """
    def __init__(self, noise_dim=5, prop_dim=26, comp_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + prop_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, comp_dim),
            nn.Sigmoid()
        )

    def forward(self, z, properties):
        import torch
        x = torch.cat([z, properties], dim=1)
        return self.net(x)


class CGANDiscriminator(nn.Module):
    """CGAN Discriminator: (composition + properties) -> real/fake

    Input: concat(comp[comp_dim], prop[prop_dim])
    """
    def __init__(self, comp_dim=40, prop_dim=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(comp_dim + prop_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, properties):
        import torch
        xp = torch.cat([x, properties], dim=1)
        return self.net(xp)

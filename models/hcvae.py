"""
HCVAE — Hierarchical Conditional Variational Autoencoder
Architecture (matches ALCHEMIST paper):
  - ElementImportanceScorer (AEA module)
  - MacroEncoder  ->  z_macro
  - MicroEncoder  ->  z_micro  (conditioned on z_macro)
  - LatentGFA     ->  z_gfa   (from target properties)
  - PhysicsGuidedDecoder  (z_gfa + z_macro + z_micro -> composition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ElementImportanceScorer(nn.Module):
    """AEA module: learn per-element importance scores in [0, 1]."""
    def __init__(self, comp_dim, hidden_dim=64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(comp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, comp_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """x: (batch, comp_dim) -> scores: (batch, comp_dim)"""
        return self.scorer(x)


class MacroEncoder(nn.Module):
    """Encode high-importance (macro) elements into z_macro."""
    def __init__(self, comp_dim, hidden_dim, latent_dim, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(comp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class MicroEncoder(nn.Module):
    """Encode low-importance (micro) elements into z_micro, conditioned on z_macro."""
    def __init__(self, comp_dim, macro_latent_dim, hidden_dim, latent_dim, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(comp_dim + macro_latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x_trace, z_macro):
        h = self.net(torch.cat([x_trace, z_macro], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)


class LatentGFA(nn.Module):
    """Transform target properties into GFA conditioning vector z_gfa."""
    def __init__(self, prop_dim, latent_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prop_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim)
        )

    def forward(self, properties):
        return self.net(properties)


class PhysicsGuidedDecoder(nn.Module):
    """Decode [z_gfa ; z_macro ; z_micro] -> composition logits."""
    def __init__(self, latent_gfa_dim, latent_structure_dim, hidden_dim, comp_dim, dropout=0.3):
        super().__init__()
        total = latent_gfa_dim + latent_structure_dim
        self.net = nn.Sequential(
            nn.Linear(total, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(hidden_dim, comp_dim)
        )

    def forward(self, z_gfa, z_structure):
        return self.net(torch.cat([z_gfa, z_structure], dim=1))


class HierarchicalCVAE(nn.Module):
    """
    Full HCVAE model.

    Args:
        comp_dim:     number of elements (default 40)
        prop_dim:     number of thermodynamic properties (default 26)
        latent_macro: dimension of z_macro (default 48)
        latent_micro: dimension of z_micro (default 24)
        latent_gfa:   dimension of z_gfa   (default 24)
        hidden_dim:   hidden layer width    (default 128)
        beta:         KL weight in ELBO     (default 2.0)
        dropout:      dropout rate          (default 0.4)
    """
    def __init__(self, comp_dim=40, prop_dim=26,
                 latent_macro=48, latent_micro=24, latent_gfa=24,
                 hidden_dim=128, beta=2.0, dropout=0.4):
        super().__init__()
        self.comp_dim = comp_dim
        self.beta = beta
        self.latent_structure_dim = latent_macro + latent_micro

        self.element_scorer = ElementImportanceScorer(comp_dim, hidden_dim=64)
        self.encoder_macro = MacroEncoder(comp_dim, hidden_dim, latent_macro, dropout)
        self.encoder_micro = MicroEncoder(comp_dim, latent_macro, hidden_dim, latent_micro, dropout)
        self.latent_gfa = LatentGFA(prop_dim, latent_gfa, dropout=dropout * 0.75)
        self.decoder = PhysicsGuidedDecoder(
            latent_gfa, self.latent_structure_dim, hidden_dim, comp_dim, dropout=dropout * 0.75
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x_comp, x_prop):
        """
        Returns:
            x_recon:       (batch, comp_dim)
            mu_macro:      (batch, latent_macro)
            logvar_macro:  (batch, latent_macro)
            mu_micro:      (batch, latent_micro)
            logvar_micro:  (batch, latent_micro)
            element_scores:(batch, comp_dim)  — AEA importance scores
        """
        scores = self.element_scorer(x_comp)
        x_macro = x_comp * scores
        x_micro = x_comp * (1.0 - scores)

        mu_macro, logvar_macro = self.encoder_macro(x_macro)
        z_macro = self.reparameterize(mu_macro, logvar_macro)

        mu_micro, logvar_micro = self.encoder_micro(x_micro, z_macro)
        z_micro = self.reparameterize(mu_micro, logvar_micro)

        z_gfa = self.latent_gfa(x_prop)
        z_structure = torch.cat([z_macro, z_micro], dim=1)
        x_recon = self.decoder(z_gfa, z_structure)

        return x_recon, mu_macro, logvar_macro, mu_micro, logvar_micro, scores

    def generate(self, target_properties, num_samples=1, temperature=1.0):
        """Inverse design: sample compositions conditioned on target properties.

        Args:
            target_properties: (1, prop_dim) normalized property tensor
            num_samples:        number of candidates
            temperature:        sampling temperature (higher = more diverse)

        Returns:
            compositions: (num_samples, comp_dim)
        """
        self.eval()
        with torch.no_grad():
            z_gfa = self.latent_gfa(target_properties)
            z_structure = torch.randn(num_samples, self.latent_structure_dim,
                                      device=z_gfa.device) * temperature
            x_recon = self.decoder(z_gfa.expand(num_samples, -1), z_structure)
        return x_recon

    def get_element_importance(self, dataloader, device):
        """Return mean and per-sample AEA importance scores."""
        self.eval()
        all_scores = []
        with torch.no_grad():
            for batch in dataloader:
                comp = batch['composition'].to(device)
                all_scores.append(self.element_scorer(comp).cpu().numpy())
        import numpy as np
        all_scores = np.concatenate(all_scores, axis=0)
        return all_scores.mean(axis=0), all_scores

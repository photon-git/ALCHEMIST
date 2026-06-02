"""
Unified training entry for AlloyGAN / HCVAE.

Usage examples:
    # Train HCVAE (default)
    python train.py --model HCVAE --dataroot /path/to/Alloy_train.csv

    # Train GAN
    python train.py --model GAN --dataroot /path/to/Alloy_train.csv \
        --epochs 200 --batch_size 64 --lr_d 5e-4 --lr_g 5e-4

    # Train CGAN
    python train.py --model CGAN --dataroot /path/to/Alloy_train.csv \
        --noise_dim 5 --epochs 200 --batch_size 64
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import parse_args
from utils.data_loader import get_data_loader
from utils.metrics import evaluate_all
from models import GANGenerator, GANDiscriminator, \
    CGANGenerator, CGANDiscriminator, HierarchicalCVAE


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_device(use_cuda):
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[device] CPU")
    return device


def save_checkpoint(state, path, filename):
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, filename)
    torch.save(state, fpath)
    print(f"  ✓ Checkpoint saved → {fpath}")


# ── GAN trainer ────────────────────────────────────────────────────────────────

def train_gan(args, device):
    train_loader, _, _ = get_data_loader(args)

    G = GANGenerator(noise_dim=args.noise_dim, comp_dim=40).to(device)
    D = GANDiscriminator(comp_dim=40).to(device)
    criterion = nn.BCELoss()
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, weight_decay=1e-4)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, weight_decay=1e-4)

    print(f"\n{'='*60}\nTraining GAN  (noise_dim={args.noise_dim})\n{'='*60}")

    for epoch in range(1, args.epochs + 1):
        d_losses, g_losses = [], []
        for batch in train_loader:
            real = batch[:, :40].to(device)         # composition only
            bs = real.size(0)
            real_lbl = torch.ones(bs, device=device)
            fake_lbl = torch.zeros(bs, device=device)

            # ── Discriminator ──
            z = torch.randn(bs, args.noise_dim, device=device)
            fake = G(z).detach()
            d_loss = criterion(D(real).flatten(), real_lbl) + \
                     criterion(D(fake).flatten(), fake_lbl)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # ── Generator ──
            z = torch.randn(bs, args.noise_dim, device=device)
            g_loss = criterion(D(G(z)).flatten(), real_lbl)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        if epoch % args.log_interval == 0:
            print(f"[GAN] Epoch {epoch:4d}/{args.epochs}  "
                  f"D_loss={np.mean(d_losses):.4f}  G_loss={np.mean(g_losses):.4f}")

    save_checkpoint({
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'args': args,
    }, args.save_path, 'gan_final.pth')
    return G, D


# ── CGAN trainer ───────────────────────────────────────────────────────────────

def train_cgan(args, device):
    train_loader, _, _ = get_data_loader(args)

    G = CGANGenerator(noise_dim=args.noise_dim, prop_dim=26, comp_dim=40).to(device)
    D = CGANDiscriminator(comp_dim=40, prop_dim=26).to(device)
    criterion = nn.BCELoss()
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, weight_decay=1e-5)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, weight_decay=1e-5)

    print(f"\n{'='*60}\nTraining CGAN  (noise_dim={args.noise_dim})\n{'='*60}")

    for epoch in range(1, args.epochs + 1):
        d_losses, g_losses = [], []
        for batch in train_loader:
            batch = batch.to(device)
            real_comp = batch[:, :40]
            real_prop = batch[:, 40:]
            bs = real_comp.size(0)
            real_lbl = torch.ones(bs, device=device)
            fake_lbl = torch.zeros(bs, device=device)

            # ── Discriminator ──
            z = torch.randn(bs, args.noise_dim, device=device)
            fake_comp = G(z, real_prop).detach()
            d_loss = criterion(D(real_comp, real_prop).flatten(), real_lbl) + \
                     criterion(D(fake_comp, real_prop).flatten(), fake_lbl)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # ── Generator ──
            z = torch.randn(bs, args.noise_dim, device=device)
            fake_comp = G(z, real_prop)
            g_loss = criterion(D(fake_comp, real_prop).flatten(), real_lbl)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        if epoch % args.log_interval == 0:
            print(f"[CGAN] Epoch {epoch:4d}/{args.epochs}  "
                  f"D_loss={np.mean(d_losses):.4f}  G_loss={np.mean(g_losses):.4f}")

    save_checkpoint({
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'args': args,
    }, args.save_path, 'cgan_final.pth')
    return G, D


# ── HCVAE trainer ──────────────────────────────────────────────────────────────

def hcvae_loss(x_recon, x_target, mu_macro, logvar_macro, mu_micro, logvar_micro, beta):
    recon = F.mse_loss(x_recon, x_target, reduction='sum') / x_target.size(0)
    kl_macro = -0.5 * torch.sum(1 + logvar_macro - mu_macro.pow(2) - logvar_macro.exp()) \
               / x_target.size(0)
    kl_micro = -0.5 * torch.sum(1 + logvar_micro - mu_micro.pow(2) - logvar_micro.exp()) \
               / x_target.size(0)
    kl = beta * (kl_macro + kl_micro)
    return recon + kl, recon, kl_macro, kl_micro


def run_epoch(model, loader, optimizer, device, beta, train=True):
    model.train() if train else model.eval()
    totals = dict(loss=0, recon=0, kl_macro=0, kl_micro=0)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            comp = batch['composition'].to(device)
            prop = batch['properties'].to(device)
            x_recon, mu_mac, lv_mac, mu_mic, lv_mic, _ = model(comp, prop)
            loss, recon, kl_mac, kl_mic = hcvae_loss(
                x_recon, comp, mu_mac, lv_mac, mu_mic, lv_mic, beta
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            totals['loss']     += loss.item()
            totals['recon']    += recon.item()
            totals['kl_macro'] += kl_mac.item()
            totals['kl_micro'] += kl_mic.item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def train_hcvae(args, device):
    train_loader, val_loader, train_ds = get_data_loader(args)

    model = HierarchicalCVAE(
        comp_dim=40, prop_dim=26,
        latent_macro=args.latent_macro,
        latent_micro=args.latent_micro,
        latent_gfa=args.latent_gfa,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=False
    )

    best_val, patience_cnt = float('inf'), 0
    history = {k: [] for k in ('train_loss', 'train_recon', 'train_kl_macro',
                                'train_kl_micro', 'val_loss', 'val_recon')}

    print(f"\n{'='*60}\nTraining HCVAE  (β={args.beta})\n{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t = run_epoch(model, train_loader, optimizer, device, args.beta, train=True)
        v = run_epoch(model, val_loader,   optimizer, device, args.beta, train=False)
        scheduler.step(v['loss'])

        history['train_loss'].append(t['loss'])
        history['train_recon'].append(t['recon'])
        history['train_kl_macro'].append(t['kl_macro'])
        history['train_kl_micro'].append(t['kl_micro'])
        history['val_loss'].append(v['loss'])
        history['val_recon'].append(v['recon'])

        if epoch % args.log_interval == 0:
            print(f"[HCVAE] Epoch {epoch:4d}/{args.epochs}  "
                  f"train_loss={t['loss']:.4f}  val_loss={v['loss']:.4f}  "
                  f"recon={t['recon']:.4f}  kl_mac={t['kl_macro']:.4f}  kl_mic={t['kl_micro']:.4f}")

        if v['loss'] < best_val:
            best_val = v['loss']
            patience_cnt = 0
            save_checkpoint({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val,
                'comp_scaler': train_ds.comp_scaler,
                'prop_scaler': train_ds.prop_scaler,
                'args': args,
            }, args.save_path, 'hcvae_best.pth')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    return model, history, train_ds


# ── Inverse design helper ──────────────────────────────────────────────────────

def inverse_design(model, target_properties_raw, train_ds, num_candidates=100,
                   temperature=1.0, device='cpu'):
    """Generate candidate alloy compositions from raw target properties.

    Args:
        model:                 trained HierarchicalCVAE
        target_properties_raw: (26,) array of target thermo properties (raw scale)
        train_ds:              MetallicGlassDataset — carries scalers
        num_candidates:        number of candidates to generate
        temperature:           sampling temperature (higher = more diverse)
        device:                torch device

    Returns:
        compositions: (num_candidates, 40) in at-% (non-negative, sums to 100)
    """
    import numpy as np
    target_norm = train_ds.prop_scaler.transform(
        target_properties_raw.reshape(1, -1)
    ).astype(np.float32)
    target_t = torch.from_numpy(target_norm).to(device)

    model.eval()
    with torch.no_grad():
        raw = model.generate(target_t, num_samples=num_candidates,
                             temperature=temperature)

    comps = train_ds.comp_scaler.inverse_transform(raw.cpu().numpy())
    comps = np.maximum(comps, 0)
    comps = comps / comps.sum(axis=1, keepdims=True) * 100
    return comps


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = make_device(args.cuda)

    if args.model == 'GAN':
        # Override noise_dim default to match original GAN (100)
        if args.noise_dim == 5:      # user left CGAN default, fix silently
            args.noise_dim = 100
        train_gan(args, device)

    elif args.model == 'CGAN':
        if args.noise_dim == 100:    # user left GAN default, fix silently
            args.noise_dim = 5
        train_cgan(args, device)

    elif args.model == 'HCVAE':
        model, history, train_ds = train_hcvae(args, device)

        # Quick evaluation on training distribution
        print("\n[Evaluation] Generating 200 samples for metrics...")
        import numpy as np
        # Collect real denormalised compositions
        all_real = []
        for batch in torch.utils.data.DataLoader(
                train_ds, batch_size=64, shuffle=False):
            all_real.append(batch['composition'].numpy())
        all_real = np.concatenate(all_real, axis=0)
        real_denorm = train_ds.comp_scaler.inverse_transform(all_real)

        # Generate
        all_props = np.concatenate(
            [batch['properties'].numpy() for batch in
             torch.utils.data.DataLoader(train_ds, batch_size=len(train_ds))],
            axis=0
        )
        model.eval()
        generated = []
        with torch.no_grad():
            for _ in range(4):                  # 4 × 50 = 200
                idx = np.random.randint(0, len(all_props))
                prop_t = torch.from_numpy(
                    all_props[idx:idx+1].astype(np.float32)
                ).to(device)
                s = model.generate(prop_t, num_samples=50)
                generated.append(s.cpu().numpy())
        fake = np.concatenate(generated, axis=0)
        fake_denorm = train_ds.comp_scaler.inverse_transform(fake)
        fake_denorm = np.maximum(fake_denorm, 0)
        fake_denorm = fake_denorm / fake_denorm.sum(axis=1, keepdims=True) * 100

        m = evaluate_all(
            np.maximum(real_denorm, 0),
            fake_denorm
        )
        print("\n[Metrics]")
        print(f"  Wasserstein   : {m['wasserstein']:.4f} ± {m['wasserstein_std']:.4f}")
        print(f"  KL divergence : {m['kl']:.4f}")
        print(f"  JS divergence : {m['js']:.4f}")
        print(f"  Mode coverage : {m['mode_coverage']:.2%}")
        print(f"  Diversity     : {m['diversity']:.4f}")

    else:
        print(f"Unknown model: {args.model}")


if __name__ == '__main__':
    main()

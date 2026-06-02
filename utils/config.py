import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='AlloyGAN / HCVAE unified training entry.'
    )

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='HCVAE',
                        choices=['GAN', 'CGAN', 'HCVAE'],
                        help='Model to train')

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to Alloy_train.csv')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data used for validation (HCVAE only)')
    parser.add_argument('--seed', type=int, default=42)

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (HCVAE / GAN-G)')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU if available')

    # ── HCVAE hyperparameters ─────────────────────────────────────────────────
    parser.add_argument('--latent_macro', type=int, default=48)
    parser.add_argument('--latent_micro', type=int, default=24)
    parser.add_argument('--latent_gfa',   type=int, default=24)
    parser.add_argument('--hidden_dim',   type=int, default=128)
    parser.add_argument('--beta',         type=float, default=2.0,
                        help='KL weight in HCVAE ELBO')
    parser.add_argument('--dropout',      type=float, default=0.4)
    parser.add_argument('--patience',     type=int, default=50,
                        help='Early-stopping patience (HCVAE)')

    # ── GAN / CGAN hyperparameters ────────────────────────────────────────────
    parser.add_argument('--noise_dim', type=int, default=100,
                        help='GAN noise dimension (5 for CGAN, 100 for GAN)')
    parser.add_argument('--lr_d', type=float, default=2e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--lr_g', type=float, default=2e-4,
                        help='Generator learning rate')

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Print loss every N epochs')

    return parser.parse_args()

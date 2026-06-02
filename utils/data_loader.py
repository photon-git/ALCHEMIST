import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AlloyDataset(Dataset):
    """Alloy dataset for GAN / CGAN.

    Loads all rows from CSV (skips the first 'source' column).
    Each sample is the raw row vector: comp(40) + prop(26).
    """
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        data.drop(columns=['source'], errors='ignore', inplace=True)
        self.data = data.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


class MetallicGlassDataset(Dataset):
    """Alloy dataset for HCVAE.

    Splits composition (cols 0-39) and properties (cols 40-65).
    Applies StandardScaler to both. Train/val split is done here.

    Usage:
        train_ds = MetallicGlassDataset(csv_path, split='train')
        val_ds   = MetallicGlassDataset(csv_path, split='val',
                                        comp_scaler=train_ds.comp_scaler,
                                        prop_scaler=train_ds.prop_scaler)
    """
    def __init__(self, csv_path, split='train', test_size=0.2, random_state=42,
                 comp_scaler=None, prop_scaler=None):
        data = pd.read_csv(csv_path)
        data.drop(columns=['source'], errors='ignore', inplace=True)

        compositions = data.iloc[:, :40].values.astype(np.float32)
        properties   = data.iloc[:, 40:].values.astype(np.float32)

        comp_train, comp_test, prop_train, prop_test = train_test_split(
            compositions, properties, test_size=test_size, random_state=random_state
        )

        if split == 'train':
            self.comp_scaler = StandardScaler().fit(comp_train)
            self.prop_scaler = StandardScaler().fit(prop_train)
            self.compositions = self.comp_scaler.transform(comp_train)
            self.properties   = self.prop_scaler.transform(prop_train)
        else:
            if comp_scaler is None or prop_scaler is None:
                raise ValueError("Provide comp_scaler and prop_scaler for val/test split.")
            self.comp_scaler  = comp_scaler
            self.prop_scaler  = prop_scaler
            self.compositions = comp_scaler.transform(comp_test)
            self.properties   = prop_scaler.transform(prop_test)

        print(f"[Dataset/{split}] {len(self.compositions)} samples")

    def __len__(self):
        return len(self.compositions)

    def __getitem__(self, idx):
        return {
            'composition': torch.from_numpy(self.compositions[idx]),
            'properties':  torch.from_numpy(self.properties[idx]),
        }


def get_data_loader(args):
    """Build train/val DataLoaders from args.

    For GAN / CGAN: uses AlloyDataset (raw rows).
    For HCVAE:      uses MetallicGlassDataset (split comp/prop, normalised).

    Returns:
        (train_loader, val_loader, extra)
        extra = None for GAN/CGAN; train_dataset for HCVAE (carries scalers).
    """
    if args.model in ('GAN', 'CGAN'):
        train_ds = AlloyDataset(args.dataroot)
        train_loader = data_utils.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        return train_loader, None, None

    elif args.model == 'HCVAE':
        train_ds = MetallicGlassDataset(args.dataroot, split='train',
                                        test_size=args.test_size,
                                        random_state=args.seed)
        val_ds = MetallicGlassDataset(args.dataroot, split='val',
                                      test_size=args.test_size,
                                      random_state=args.seed,
                                      comp_scaler=train_ds.comp_scaler,
                                      prop_scaler=train_ds.prop_scaler)
        train_loader = data_utils.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        val_loader = data_utils.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False
        )
        return train_loader, val_loader, train_ds

    else:
        raise ValueError(f"Unknown model: {args.model}")

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import os
import numpy as np

class TradingDataset(Dataset):
    def __init__(self, csv_file, schema_file, seq_len=32, transform=None):
        self.data = pd.read_csv(csv_file)
        with open(schema_file, 'r') as f:
            self.schema = yaml.safe_load(f)
        self.transform = transform
        self.seq_len = seq_len
        self._validate_schema()

    def _validate_schema(self):
        required_cols = [k for k in self.schema.keys() if k != 'features']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get a window of seq_len ending at idx
        start = max(0, idx - self.seq_len + 1)
        window = self.data.iloc[start:idx+1].drop(['timestamp'], axis=1).values.astype(float)
        # Pad if needed
        if window.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - window.shape[0], window.shape[1]))
            window = np.vstack([pad, window])
        if self.transform:
            window = self.transform(window)
        return torch.tensor(window, dtype=torch.float32)


def get_dataloader(csv_file, schema_file, batch_size=64, shuffle=True, seq_len=32, transform=None):
    dataset = TradingDataset(csv_file, schema_file, seq_len=seq_len, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 
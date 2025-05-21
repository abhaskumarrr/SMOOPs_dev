import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import os

class TradingDataset(Dataset):
    def __init__(self, csv_file, schema_file, transform=None):
        self.data = pd.read_csv(csv_file)
        with open(schema_file, 'r') as f:
            self.schema = yaml.safe_load(f)
        self.transform = transform
        self._validate_schema()

    def _validate_schema(self):
        required_cols = [k for k in self.schema.keys() if k != 'features']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row.drop(['timestamp']).values.astype(float)
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features, dtype=torch.float32)


def get_dataloader(csv_file, schema_file, batch_size=64, shuffle=True, transform=None):
    dataset = TradingDataset(csv_file, schema_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 
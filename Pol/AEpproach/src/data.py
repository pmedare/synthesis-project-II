import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = self.data.columns[2:]
        self.vectors = self.data[self.features]
        self.vectors = torch.tensor(self.data[self.features].values, dtype=torch.float32)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]
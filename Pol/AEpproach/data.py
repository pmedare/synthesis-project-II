from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, data, label=0):
        self.data = data
        self.vectors = self.data[:, 2:]

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
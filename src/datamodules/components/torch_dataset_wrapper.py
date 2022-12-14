from torch.utils.data import Dataset
from datasets import Dataset as HFDataset


class DatasetWrapper(Dataset):
    def __init__(self, dataset: HFDataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

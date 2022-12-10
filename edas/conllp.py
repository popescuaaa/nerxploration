from datasets import load_dataset
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset


class DatasetWrapper(Dataset):
    def __init__(self, dataset: HFDataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = load_dataset("conllpp")
    print(dataset)
    print(type(dataset))
    for e in dataset:
        print(e)
        break

    tds = DatasetWrapper(dataset=dataset)
    print(len(tds))
    print(tds[10])

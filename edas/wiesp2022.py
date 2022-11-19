from datasets import load_dataset

if __name__ == '__main__':
    dataset = load_dataset("adsabs/WIESP2022-NER")
    print(dataset)
    print(dataset["train"][0])
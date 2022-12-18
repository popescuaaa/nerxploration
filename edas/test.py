from transformers import BertTokenizerFast
from datasets import load_dataset
from pprint import pprint
from typing import Dict
from torch.utils.data import Dataset

def create_tag_lookup_table() -> Dict:
    iob_labels = ["B", "I"]
    ner_labels = ['Formula', 'Mission', 'Citation', 'Software', 'EntityOfFutureInterest', 'Software',
                  'ObservationalTechniques', 'ComputingFacility', 'Proposal', 'Organization', 'CelestialObjectRegion',
                  'Collaboration', 'ComputingFacility', 'Survey', 'O', 'Fellowship', 'Wavelength',
                  'CelestialObjectRegion', 'Telescope', 'Survey', 'Dataset', 'CelestialRegion', 'Archive', 'Instrument',
                  'Wavelength', 'Collaboration', 'URL', 'CelestialRegion', 'Tag', 'Instrument', 'TextGarbage',
                  'Telescope', 'ObservationalTechniques', 'Database', 'TextGarbage', 'Grant', 'Formula', 'Database',
                  'Identifier', 'Mission', 'CelestialObject', 'Model', 'Person', 'CelestialObject', 'Observatory',
                  'Tag', 'Model', 'Fellowship', 'Organization', 'Grant', 'Event', 'Dataset', 'URL', 'Citation',
                  'Location', 'EntityOfFutureInterest', 'Event', 'Archive', 'Proposal', 'Location', 'Observatory',
                  'Person', 'Identifier']
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels
    return dict(zip(range(0, len(all_labels) + 1), all_labels))

label_all_tokens = False

def align_label(texts, labels, labels_to_ids):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=128, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(Dataset):

    def __init__(self, dataset):

        # Get all ner tags
        self.labels = [e['ner_tags'] for e in dataset]
        self.tokens = [e['tokens'] for e in dataset]
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels

if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    dataset = load_dataset("adsabs/WIESP2022-NER")

    # ids_to_labels = create_tag_lookup_table()
    # labels_to_ids = {v: k for k, v in ids_to_labels.items()}
    #
    # print("Aligning labels...")
    # dataset = dataset.map(lambda examples: {'labels': align_label(examples['tokens'], examples['ner_tags'])})
    # print("Done.")
    # print(dataset['train'][0])
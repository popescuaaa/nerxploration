"""
Pseudoimplementation of the Transformer model from "Attention is All You Need"
from: https://keras.io/examples/nlp/ner_transformers/
"""

## NN
import torch
import torch.nn as nn
import numpy as np

## Data
import datasets
from datasets import load_dataset
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
import transformers

## Transformers
from torch.nn import MultiheadAttention

# Types
from typing import List, Dict, Tuple, Optional, Union


## Compute block
class TransformerBock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float, dropout_attn: bool = True) -> None:
        super(TransformerBock, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dropout_attn = dropout_attn

        # Define a simple architecture
        if self.dropout_attn:
            self.attention = MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        else:
            self.attention = MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

        # A linear with dropout from input dim  + attention dim
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

        self.dropout_layer_0 = nn.Dropout(p=dropout)
        self.dropout_layer_1 = nn.Dropout(p=dropout)

        self.linear_layer_norm_0 = nn.LayerNorm(embedding_dim)
        self.linear_layer_norm_1 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("input shape: ", x.shape)
        attn_output, _ = self.attention(query=x, key=x, value=x)
        print("attn_output shape: ", attn_output.shape)
        attn_output = self.dropout_layer_0(attn_output)
        print("attn_output shape: ", attn_output.shape)
        out = self.linear_layer_norm_0(x + attn_output)
        print("out shape: ", out.shape)
        out = self.linear(out)
        print("out shape: ", out.shape)
        out = self.dropout_layer_1(out)
        print("out shape: ", out.shape)
        out = self.linear_layer_norm_1(out + x)
        print("out shape: ", out.shape)

        return out

    @property
    def device(self):
        return next(self.parameters()).device


## Encoder block ~ this should be replaced with CNN at the end
class TokenPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, embedding_dim: int, vocab_size: int) -> None:
        super(TokenPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(0, self.max_seq_len, dtype=torch.long, device=self.device)
        positions = positions.unsqueeze(0).expand(x.shape[0], self.max_seq_len)
        out = self.token_embedding(x) + self.position_embedding(positions)
        return out

    @property
    def device(self):
        return next(self.parameters()).device


## Model
class TransformerNER(nn.Module):
    def __init__(self,
                 num_tags: int,
                 vocab_size: int,
                 max_seq_len: int = 128,
                 embedding_dim: int = 128,
                 num_heads: int = 4
                 ):
        super(TransformerNER, self).__init__()
        self.num_tags = num_tags
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.embedding = TokenPositionalEmbedding(self.max_seq_len, self.embedding_dim, self.vocab_size)
        self.transformer_block = TransformerBock(self.embedding_dim, self.num_heads, dropout=0.1)
        self.pre_dropout = nn.Dropout(p=0.1)

        # Linear layer for classification
        self.linear = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.embedding_dim, self.num_tags),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("input", x.shape)
        out = self.embedding(x)
        print("embedding", out.shape)
        out = self.transformer_block(out)
        print("transformer", out.shape)
        out = self.pre_dropout(out)
        print("pre_dropout", out.shape)
        out = self.linear(out)
        print("linear", out.shape)
        return out

    @property
    def device(self):
        return next(self.parameters()).device


## Data
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


def align_label(
        tokens: List[str],
        ner_tags: List[str],
        tag_lookup_table: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        max_seq_len: int = 128
) -> List[str]:
    tokenized_inputs = tokenizer("".join(tokens), padding='max_length', max_length=max_seq_len, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(tag_lookup_table[ner_tags[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(tag_lookup_table[ner_tags[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


class DatasetWrapper(Dataset):
    def __init__(self,
                 dataset: datasets.Dataset,
                 max_seq_len: int
                 ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len

        self.tag_lookup_table = create_tag_lookup_table()
        self.tag_lookup_table = {v: k for k, v in self.tag_lookup_table.items()}

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    def __getitem__(self, item):
        # Tokens
        tokens = self.dataset[item]["tokens"]
        # lowercase
        tokens = [token.lower() for token in tokens]
        encoded_tokens = self.tokenizer("".join(tokens), padding='max_length', max_length=self.max_seq_len, truncation=True,
                                        return_tensors="pt")
        # tags
        ner_tags = self.dataset[item]["ner_tags"]
        # align tags
        ner_tags = align_label(tokens=tokens,
                               ner_tags=ner_tags,
                               tag_lookup_table=self.tag_lookup_table,
                               tokenizer=self.tokenizer,
                               max_seq_len=self.max_seq_len)
        # return list of tokens and tags

        return encoded_tokens["input_ids"].numpy().flatten(), np.array(ner_tags)

    def __len__(self):
        return len(self.dataset)


class TokenLoss(nn.Module):
    def __init__(self):
        super(TokenLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(y_pred, y_true)
        mask = y_true != 0
        loss = loss[mask]
        return loss.mean()


if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("adsabs/WIESP2022-NER")
    print(dataset["train"][0])

    # Create a tag lookup table
    tag_lookup_table = create_tag_lookup_table()

    all_tokens = sum(dataset["train"]["tokens"], [])
    all_tokens_array = np.array(list(map(str.lower, all_tokens)))

    counter = Counter(all_tokens_array)
    print(len(counter))

    num_tags = len(tag_lookup_table)
    print("num_tags", num_tags)
    vocab_size = 20000

    vocabulary = [token for token, count in counter.most_common(vocab_size)]

    # Create a dataset wrapper over the original dataset
    train_dataset = DatasetWrapper(dataset=dataset["train"], max_seq_len=64)
    print(train_dataset[0])
    exit(0)
    val_dataset = DatasetWrapper(dataset=dataset["validation"], max_seq_len=64)

    model = TransformerNER(num_tags, vocab_size, max_seq_len=64, embedding_dim=64, num_heads=4)
    # Print the number of trainable parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = TokenLoss()

    num_epochs = 10
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Empty cuda cache
    torch.cuda.empty_cache()
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            tokens, ner_tags = batch
            optimizer.zero_grad()
            x = tokens.to(device)
            y = ner_tags.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                tokens, ner_tags = batch
                optimizer.zero_grad()
                x = tokens.to(device)
                y = ner_tags.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                val_losses.append(loss.item())
            print(f"Epoch {epoch + 1} - val loss: {np.mean(val_losses)}")

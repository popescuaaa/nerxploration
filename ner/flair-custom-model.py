from flair.data import Sentence
from flair.nn import Classifier

import pandas as pd
import numpy as np
import os
import spacy

nlp = spacy.load("en_core_web_sm")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found')

    return pd.read_csv(path, encoding='utf-8')


"""
part of speech tagging
"""


def insert_pos_tag(df: pd.DataFrame, src_filed: str) -> pd.DataFrame:
    df['Pos'] = df[src_filed].apply(lambda x: nlp(x))
    return df


if __name__ == "__main__":
    # load data
    df = load_data('../data/trivia10k13.csv')
    df = insert_pos_tag(df=df, src_filed='Word')

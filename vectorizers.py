import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime as dt
from string import whitespace
from typing import Dict, Iterable, List

import numpy as np
from numpy import log

from _stopwords import STOPWORDS

np.seterr(divide="ignore")


class CountVectorizer:
    def __init__(
        self, stop_words=None, max_words=500, preprocesser=None, tokenizer=None
    ):
        self.max_words = max_words

        if stop_words:
            STOPWORDS = set(stop_words)
        if preprocesser:
            self.preprocess = preprocesser
        if tokenizer:
            self.tokenize = tokenizer

    @property
    def vocab(self):
        return list(self.vocab_encoder.keys())

    def fit(self, corpus: Iterable[str]) -> None:
        corpus = self.preprocess(corpus)

        all_counts = Counter()
        [all_counts.update(doc) for doc in corpus]

        top_words = all_counts.most_common(self.max_words)
        self.vocab_encoder = {}
        for i, (word, _) in enumerate(top_words):
            self.vocab_encoder[word] = i

    def transform(self, corpus: Iterable[str]) -> np.ndarray:
        corpus = self.preprocess(corpus)
        counts = [Counter(doc) for doc in corpus]

        count_matrix = []
        for count in counts:
            row = np.zeros(len(self.vocab_encoder))
            for word in self.vocab_encoder:
                row[self.vocab_encoder[word]] = count[word]
            count_matrix.append(row)

        return np.array(count_matrix)

    def fit_transform(self, corpus: Iterable[str]) -> np.ndarray:
        corpus = self.preprocess(corpus)
        counts = [Counter(doc) for doc in corpus]
        all_counts = Counter()
        [all_counts.update(doc) for doc in counts]

        top_words = all_counts.most_common(self.max_words)
        self.vocab_encoder = {}
        for i, (word, _) in enumerate(top_words):
            self.vocab_encoder[word] = i

        count_matrix = []
        for count in counts:
            row = np.zeros(len(self.vocab_encoder))
            for word in self.vocab_encoder:
                row[self.vocab_encoder[word]] = count[word]
            count_matrix.append(row)

        return np.array(count_matrix)

    def tokenize(self, doc: str) -> List[str]:
        return doc.split()

    def preprocess(self, corpus: Iterable[str]) -> Iterable:
        for doc in corpus:
            doc = doc.lower()
            # Subbing whitespace for regular space
            doc = re.sub(f"[{whitespace}]", " ", doc)
            # Subbing out all non-alpha chars
            doc = re.sub("[^a-z ]", "", doc)
            # Subbing out multiple whitespace chars
            tokens = self.tokenize(doc)
            yield (token for token in tokens if token not in STOPWORDS)


class TfidfVectorizer(CountVectorizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, corpus: Iterable[str]):
        self.counts_matrix = super().fit_transform(corpus)
        self.dfs = np.sum(self.count_matrix > 0, axis=0)
        self.idfs = log((self.count_matrix.shape[0] + 1) / (self.dfs + 1)) + 1

    def transform(self, corpus: Iterable[str]):
        """ Transforms text into a tfidf Vector matrix """
        count_matrix = super().transform(corpus)

        tfidfs = np.zeros_like(count_matrix, dtype="float64")
        for i in range(count_matrix.shape[0]):
            tfidfs[i] = euclidian_normalization(count_matrix[i] * self.idfs)

        return tfidfs

    def fit_transform(self, corpus: Iterable[str]):
        # Fit
        self.count_matrix = super().fit_transform(corpus)
        self.dfs = np.sum(self.count_matrix > 0, axis=0)
        self.idfs = log((self.count_matrix.shape[0] + 1) / (self.dfs + 1)) + 1
        # Transform
        tfidfs = np.zeros_like(self.count_matrix, dtype="float64")
        for i in range(self.count_matrix.shape[0]):
            tfidfs[i] = euclidian_normalization(self.count_matrix[i] * self.idfs)

        return tfidfs


def euclidian_normalization(arr):
    """ Perform Euclidian Normalization on an array """
    return arr / np.sqrt((arr ** 2).sum())


if __name__ == "__main__":
    from pathlib import Path
    from tqdm import tqdm
    from datetime import datetime as dt

    corpus = [
        path.read_text(encoding="utf-8")
        for path in tqdm(list(Path(r"E:\tfidf_search_data\opinions").glob("*"))[:10000])
    ]

    vectorizer = TfidfVectorizer()
    s = dt.now()
    print(vectorizer.fit_transform(corpus))
    print(dt.now() - s)

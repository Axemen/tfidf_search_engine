import re
from collections import Counter
from datetime import datetime as dt
from string import whitespace
from typing import Iterable, List

import numpy as np
import scipy.sparse as sp
from numpy import log

from ._stopwords import STOPWORDS

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

    def fit(self, corpus: Iterable[str]) -> None:
        self.fit_transform(corpus)

    def transform(self, corpus: Iterable[str]) -> np.ndarray:
        corpus = self.preprocess(corpus)

        indptr = [0]
        indices = []
        data = []
        for doc in corpus:
            for term in doc:
                try:
                    indices.append(self.vocab[term])
                    data.append(1)
                except KeyError:
                    pass
            indptr.append(len(indices))

        return sp.csr_matrix((data, indices, indptr), dtype=float)
        

    def fit_transform(self, corpus: Iterable[str]) -> np.ndarray:
        corpus = self.preprocess(corpus)

        indptr = [0]
        indices = []
        data = []
        vocab = {}

        for doc in corpus:
            for term in doc:
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))

        # Creating unfiltered count_matrix 
        count_matrix =  sp.csr_matrix((data, indices, indptr), dtype=float)

        # Filtering Count matrix for the top used words
        counts = count_matrix.sum(axis=0).A.flatten()
        top_n = np.argsort(counts)[-self.max_words:]

        # Filtering vocab
        self.vocab = self.filter_dict(vocab, top_n)
        

        # ? refactor?
        trans_table = {}
        for k, v in self.vocab.items():
            for nk, nv in vocab.items():
                if nk == k:
                    trans_table[nv] = v

        # Sorting the order in which they will be 
        # returned in the filtered_matrix
        top_n = sorted(top_n, key=lambda x: trans_table[x])

        filtered_counts = count_matrix[:, top_n]
        self.shape = filtered_counts.shape # Saving shape for later use
        return filtered_counts


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

    def filter_dict(self, d, l):
        result = {}
        i = 0
        for k, v in d.items():
            if v in l:
                result[k] = i
                i += 1

        return result


class TfidfVectorizer(CountVectorizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, corpus: Iterable[str]):
        count_matrix = super().fit_transform(corpus)
        self.dfs = np.sum(count_matrix > 0, axis=0).A.flatten()
        self.idfs = sp.csr_matrix(log((count_matrix.shape[0] + 1) / (self.dfs + 1)) + 1)

    def transform(self, corpus: Iterable[str]):
        """ Transforms text into a tfidf Vector matrix """
        count_matrix = super().transform(corpus)

        tfidfs = []
        for i in range(count_matrix.shape[0]):
            n = euclidian_normalization(count_matrix[i, :].multiply(self.idfs))
            tfidfs.append(n)

        tfidfs = sp.vstack(tfidfs)
        return tfidfs

    def fit_transform(self, corpus: Iterable[str]):
        # Fit
        count_matrix = super().fit_transform(corpus)
        self.dfs = np.sum(count_matrix > 0, axis=0).A.flatten()
        self.idfs = sp.csr_matrix(log((count_matrix.shape[0] + 1) / (self.dfs + 1)) + 1)
        # Transform
        tfidfs = []
        for i in range(count_matrix.shape[0]):
            n = euclidian_normalization(count_matrix[i, :].multiply(self.idfs))
            tfidfs.append(n)

        tfidfs = sp.vstack(tfidfs)
        return tfidfs


def euclidian_normalization(row):
    """ Perform Euclidian Normalization on an array """
    return row / np.sqrt(row.power(2).sum())


import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime as dt
from math import log
from string import whitespace
from typing import Dict, Iterable, List

import numpy as np

from _stopwords import STOPWORDS

np.seterr(divide="ignore")


class CountVectorizer:
    def __init__(self, stop_words=None, max_words=500, preprocesser=None, tokenizer=None):
        self.max_words = max_words
        self.vocab_encoder = {}

        if stop_words:
            STOPWORDS = set(stop_words)
        if preprocesser:
            self.preprocess = preprocesser
        if tokenizer:
            self.tokenize = tokenizer

    @property
    def vocab(self):
        return list(self.vocab_encoder.keys())

    def fit(self, corpus: Iterable[str]):
        corpus = self.preprocess(corpus)

        all_counts = Counter()
        [all_counts.update(doc) for doc in corpus]

        top_words = all_counts.most_common(self.max_words)
        vocab = set(i[0] for i in top_words)
        for i, (word, _) in enumerate(top_words):
            self.vocab_encoder[word] = i

    def fit_transform(self, corpus: Iterable[str]) -> np.ndarray:
        corpus = self.preprocess(corpus)
        counts = [Counter(doc) for doc in corpus]

        all_counts = Counter()
        [all_counts.update(c) for c in (Counter(count) for count in counts)]
        vocab = set(i[0] for i in all_counts.most_common(self.max_words))

        for i, (word, _) in enumerate(all_counts.most_common(self.max_words)):
            self.vocab_encoder[word] = i

        count_matrix = []
        for count in counts:
            row = np.zeros(self.max_words)
            for word in self.vocab_encoder:
                row[self.vocab_encoder[word]] = count[word]
            count_matrix.append(row)

        return np.array(count_matrix)

    def transform(self, corpus: Iterable[str]) -> np.ndarray:
        corpus = self.preprocess(corpus)
        counts = [Counter(doc) for doc in corpus]

        count_matrix = []
        for count in counts:
            row = np.zeros(self.max_words)
            for word in self.vocab_encoder:
                row[self.vocab_encoder[word]] = count[word]
            count_matrix.append(row)

        return np.array(count_matrix)

    def tokenize(self, doc: str) -> List[str]:
        return doc.split()

    def preprocess(self, corpus: List[str]) -> Iterable:
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
        corpus = self.preprocess(corpus)

        term_frequencies = Counter()
        [term_frequencies.update(doc) for doc in corpus]

        top_words = term_frequencies.most_common(self.max_words)
        vocab = set(i[0] for i in top_words)
        for i, (word, _) in enumerate(top_words):
            self.vocab_encoder[word] = i

        self.term_frequencies = {k:term_frequencies[k] for k in self.vocab_encoder}
        self.vocab_decoder = {v:k for k, v in self.vocab_encoder.items()}

    def fit_transform(self, corpus: Iterable[str]):
        count_matrix = super().fit_transform(corpus)
        tfidf_matrix = np.zeros(count_matrix.shape)

        # Calculate tf-idf score for top words
        n = count_matrix.shape[0]
        for i, doc in enumerate(count_matrix):
            tf = sum(x[i] for x in count_matrix)
            for j, df in enumerate(doc):
                idf = np.log(n / df) + 1
                if idf == np.inf:
                    idf = 0
                tfidf_matrix[i][j] = tf * idf

        return tfidf_matrix

    def transform(self, corpus: Iterable[str]):
        """ Transforms text into a tfidf Vector matrix """
        count_matrix = super().transform(corpus)
        tfidf_matrix = np.zeros(count_matrix.shape)

        # Calculate tf-idf score for top words
        n = count_matrix.shape[0]

        for i, doc in enumerate(count_matrix):
            tf = self.term_frequencies[self.vocab_decoder[i]]
            for j, df in enumerate(doc):
                tfidf_matrix[i][j] = tf * self._idf(n, df)
        return tfidf_matrix

    def _idf(self, n_docs, document_freq):
        idf = np.log(n_docs / document_freq) + 1
        if idf == np.inf:
            return 0
        return idf


if __name__ == "__main__":
    from pathlib import Path
    from tqdm import tqdm

    corpus = [p.read_text(encoding="utf-8") for p in tqdm(list(Path("E:/tfidf_search_data/opinions").glob("*")))]
    vectorizer = TfidfVectorizer()
    print("Fitting")
    vectorizer.fit(corpus)
    print("Transforming")
    print(vectorizer.transform(corpus))

import re
from collections import Counter, defaultdict
from string import whitespace
from typing import Dict, Iterable, List
from math import log

import numpy as np
from nltk.corpus import stopwords
from datetime import datetime as dt


class TfidfVectorizer:
    def __init__(self, stop_words=None, max_words=500):
        # self.stop_words = set(stop_words)
        self.max_words = max_words
        self.vocab_encoder = {}

        if stop_words:
            self.stop_words = set(stop_words)
        else:
            self.stop_words = set(stopwords.words("english"))

    @property
    def vocab(self):
        return list(self.vocab_encoder.keys())

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
            yield (token for token in tokens if token not in self.stop_words)

    def fit_transform(self, corpus: Iterable[str]):
        corpus = self.preprocess(corpus)
        counts = [Counter(doc) for doc in corpus]
        # Get top 500 used words
        # Combining all counts into one Counter() obj
        all_counts = Counter()
        [all_counts.update(c) for c in (Counter(count) for count in counts)]
        # Grabbing top self.max_words word_ids
        vocab = set(i[0] for i in all_counts.most_common(self.max_words))

        for i, (word, _) in enumerate(all_counts.most_common(self.max_words)):
            self.vocab_encoder[word] = i

        count_matrix = []
        for count in counts:
            row = np.zeros(self.max_words)
            for word in self.vocab_encoder:
                row[self.vocab_encoder[word]] = count[word]
            count_matrix.append(row)

        count_matrix = np.array(count_matrix)
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

    def transform(self, corpus: Iterable[str]):
        pass


if __name__ == "__main__":
    from pathlib import Path

    corpus = (p.read_text(encoding="utf-8") for p in Path("test_docs").glob("*"))

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)

from sklearn.feature_extraction.text import TfidfVectorizer
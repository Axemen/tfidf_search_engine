import re
from string import whitespace
from typing import Iterable, List

from nltk.stem.snowball import SnowballStemmer

from search._stopwords import STOPWORDS


class Preprocesser:
    def __init__(self, stemmer=None, tokenizer=None, preprocess=None) -> None:

        if stemmer is None:
            stemmer = SnowballStemmer("english")
        self.stemmer = stemmer

        if tokenizer:
            self.tokenize = tokenizer

        if preprocess:
            self.preprocess = preprocess

    def preprocess_corpus(self, corpus: Iterable[str]) -> Iterable[Iterable[str]]:
        for doc in corpus:
            yield self.preprocess(doc)

    def tokenize(self, doc: str):
        return doc.split()

    def preprocess(self, doc: str) -> Iterable[str]:
        doc = doc.lower()
        doc = re.sub(f"[{whitespace}]", " ", doc)
        doc = re.sub("[^a-z ]", "", doc)
        tokens = self.tokenize(doc)
        tokens = (self.stemmer.stem(token) for token in tokens)
        return (token for token in tokens if token not in STOPWORDS)

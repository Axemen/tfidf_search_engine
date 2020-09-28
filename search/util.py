import re
from pathlib import Path
from ._stopwords import STOPWORDS
from typing import Iterable, List
from string import whitespace
from nltk.stem.snowball import SnowballStemmer

class Preprocesser:
    def __init__(self, stemmer=None, tokenizer=None, preprocess=None) -> None:

        if stemmer is None:
            self.stemmer = SnowballStemmer('english')
        self.stemmer = stemmer
        
        if tokenizer is None:
            self.tokenize = lambda doc: doc.split()
        self.tokenize = tokenizer

        if preprocess is None:
            def preprocess(doc:str) -> Iterable[str]:
                doc = doc.lower()
                doc = re.sub(f"[{whitespace}]", " ", doc)
                doc = re.sub("[^a-z ]", "", doc)
                tokens = self.tokenize(doc)
                return (token for token in tokens if token not in STOPWORDS)
        self.preprocess = preprocess()

    def preprocess_corpus(self, corpus:Iterable[str]) -> Iterable[Iterable[str]]:
        for doc in corpus:
            yield self.preprocess(doc)

    
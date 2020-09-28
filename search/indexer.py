from __future__ import annotations

import json
import re
from collections import defaultdict, Counter, namedtuple
from io import FileIO
from pathlib import Path
from string import whitespace
from typing import Dict, Iterable, List, Union
from unittest.loader import defaultTestLoader

import numpy as np
import scipy.sparse as sp
import joblib
from tqdm import tqdm

from search.util import Preprocesser
from search.vectorizers import TfidfVectorizer


class Indexer:
    def __init__(self, index_dir: str) -> None:
        self.index_dir = Path(index_dir)
        self.vectorizer = TfidfVectorizer()
        self.doc_id_counter = 0
        self.documents = []

        if not self.index_dir.exists():
            self.index_dir.mkdir()

        self.load_reverse_lookup()

    def load_reverse_lookup(self):
        self.reverse_lookup = defaultdict(list)
        lookup_path = self.index_dir.joinpath("reverse_lookup.json")
        if lookup_path.exists():
            data = json.load(lookup_path)
            for k in data:
                self.reverse_lookup[k] += data[k]

    def save_reverse_lookup(self):
        lookup_path = self.index_dir.joinpath("reverse_lookup.json")
        json.dump(self.reverse_lookup, lookup_path)

    def index_document(self, path: str, encoding="utf-8") -> None:
        path = Path(path)

        text = path.read_text(encoding=encoding)
        text = self.preprocess(text)
        tokens = self.tokenize(text)

        doc_id = self.doc_id_counter
        self.doc_id_counter += 1

        vocab = set(tokens)
        for term in vocab:
            self.reverse_lookup[term].append(doc_id)

        self.documents.append(Document(id=doc_id, path=path))
        print(self.documents)

    def index_corpus(self, path: str) -> None:
        raise NotImplementedError()

    def tokenize(self, doc: str) -> List[str]:
        return doc.split()

    def preprocess(self, doc: str) -> str:
        doc = doc.lower()
        # Subbing whitespace for regular space
        doc = re.sub(f"[{whitespace}]", " ", doc)
        # Subbing out all non-alpha chars
        doc = re.sub("[^a-z ]", "", doc)
        # Subbing out multiple whitespace chars
        return doc

    def load_index(self):
        raise NotImplementedError()

class Doc:
    def __init__(self, id, tf) -> None:
        self.id = id
        self.tf = tf

class InvertedIndex:
    def __init__(self, data: Dict = None) -> None:
        self._table = defaultdict(set)
        self._doc_paths = {}
        self.preprocesser = Preprocesser()

    @staticmethod
    def load(path: str) -> InvertedIndex:
        return joblib.load(path)

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(self, path)

    def lookup(self, terms: Iterable[str]) -> set:
        scores = defaultdict(int)
        for term in terms:
            term = self.preprocesser.stemmer.stem(term)
            for d in self._table[term]:
                scores[d.id] += d.tf * self.idf_scores[term]

        results = sorted(scores, key=scores.get, reverse=True)
        return [self._doc_paths[k] for k in results]

    def index_file(self, path: Union[str, Path], encoding="utf-8", update_idfs=True) -> None:
        path = Path(path)
        doc = path.read_text(encoding)
        doc = list(self.preprocesser.preprocess(doc))
        doc_id = len(self._doc_paths)
        self._doc_paths[doc_id] = str(path)
        tfs = Counter(doc)
        for term in doc:
            self._table[term].add(Doc(id=doc_id, tf=tfs[term]))

        if update_idfs:
            self.calc_idfs()

    def index_dir(self, path:Union[str, Path], verbose=True) -> None:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError("This is a file not a directory")
        
        print("Indexing Files")
        for p in tqdm(list(path.glob("*"))[:10000], disable=not verbose):
            self.index_file(p, update_idfs=False)

        self.calc_idfs()

    def remove(self, doc_id: int) -> None:
        for k in self._table:
            for d in self._table[k]:
                if d.id == doc_id:
                    self._table[k].remove(d)

    def calc_idfs(self):
        # idf = log( num_docs + 1 / doc_freq + 1 ) + 1
        num_docs = len(self._doc_paths)

        self.idf_scores = {
            term: (np.log((num_docs + 1) / len(term) + 1) + 1) for term in self._table
        }

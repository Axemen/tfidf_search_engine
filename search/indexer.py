from __future__ import annotations

from io import FileIO
import json
import re
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
from string import whitespace
from typing import Dict, Iterable, List, Union, Type
from _pytest.nodes import File

import numpy as np
import scipy.sparse as sp

# from ._stopwords import STOPWORDS
from .vectorizers import TfidfVectorizer


Document = namedtuple("Document", ["id", "path"])


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


class InvertedIndex:
    """ 
    {
        "term": set[ids]
    }
    """
    def __init__(self, data: Dict = None) -> None:
        self._table = defaultdict(set)
        self._doc_paths = {}
        

    @staticmethod
    def load(path: Union[str, Path, FileIO]) -> InvertedIndex:
        ii = InvertedIndex()

        if type(path) == str or type(path) == Path:
            data = json.load(open(path, 'r'))
        else:
            data = json.load(path)
        
        for k in data:
            data[k] = set(data[k])

        ii._table = data
        return ii

    def save(self, path:Union[str, Path]) -> None:

        data = {k:list(v) for k, v in self._table.items()}

        json.dump(data, open(path, 'w'))

    def lookup(self, terms:Iterable[str]) -> set:
        results = set()

        for term in terms:
            results.update(self._table[term])

        return results



    def add(self, path:Union[str, Path], encoding='utf-8') -> None:

        if type(path) == str:
            doc = Path(path).read_text(encoding=encoding)

        doc_id = len(self._doc_paths)
        self._doc_paths[doc_id] = path

        for term in doc:
            self._table[term].add(doc_id)

    def remove(self, doc_id:int) -> None:
        pass


if __name__ == "__main__":

    indexer = Indexer("./indices")

    doc = Path("E:/tfidf_search_data/opinions/A  A ACOUSTICS INC P")

    indexer.index_document(doc)

from __future__ import annotations

from collections import defaultdict, Counter, namedtuple
from pathlib import Path
from typing import Iterable, Union
from heapq import nlargest
import json

import numpy as np
from tqdm import tqdm

from search.util import Preprocesser


# * Replacing with dictionaries.
""" 
index = {
    term: {
        doc_id: term_freq 
    }
}
"""


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(dict)
        self._doc_paths = {}
        self.preprocesser = Preprocesser()

    def lookup(self, terms: Iterable[str]) -> set:
        # TODO redefine
        scores = Counter()

        for term in terms:
            score = self.fast_cosine_scores(term)
            scores.update(score)

        return [i[0] for i in nlargest(10, scores.items(), key=lambda x: x[1])]

    def fast_cosine_scores(self, term):
        # * This can be speed up significantly by iterating through all of the
        # * queries at the same time and then only performing the tfidf calculation
        # * on the matching documents between the queries.
        score = Counter()
        N = len(self.index)
        df = len(self.index[term])
        for doc_id in self.index[term]:
            tfidf = self.index[term][doc_id] * (np.log(N / df) + 1)
            score[doc_id] += tfidf
        return score

    def index_file(
        self, path: Union[str, Path], encoding="utf-8", update_idfs=True
    ) -> None:
        path = Path(path)
        doc = path.read_text(encoding)
        doc = list(self.preprocesser.preprocess(doc))
        doc_id = len(self._doc_paths)
        self._doc_paths[doc_id] = str(path)
        tfs = Counter(doc)
        for term in doc:
            self.index[term][doc_id] = tfs[term]

        if update_idfs:
            self.calc_idfs()

    def index_dir(self, path: Union[str, Path], verbose=True) -> None:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError("This is a file not a directory")

        print("Indexing Files")
        for p in tqdm(list(path.glob("*"))[:10], disable=not verbose):
            self.index_file(p, update_idfs=False)

        self.calc_idfs()

    def remove(self, doc_id: int) -> None:
        for term in self.index:
            del self.index[term][doc_id]

    def calc_idfs(self):
        num_docs = len(self._doc_paths)
        self.idf_scores = {term: self._idf(num_docs, len(term))
                           for term in self.index}

    def _idf(self, n, df):
        return (np.log((n + 1) / df + 1) + 1)

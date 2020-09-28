from search.vectorizers import CountVectorizer, TfidfVectorizer
from pathlib import Path
from tqdm import tqdm

corpus = [
        path.read_text(encoding="utf-8")
        for path in tqdm(list(Path(r"E:\tfidf_search_data\opinions").glob("*"))[:200])
    ]


def test_count_results():

    v = CountVectorizer()
    
    v.fit(corpus)
    a = v.transform(corpus)
    b = v.fit_transform(corpus)

    assert not (a-b).toarray().any(), "Matricies are not equivalent"

def test_tfidf_vectorizer_idfs():
    v = TfidfVectorizer()
    v.fit(corpus)
    a = v.idfs
    v.fit_transform(corpus)
    b = v.idfs

    assert not (a-b).toarray().any(), "Matricies are not equivalent"

def test_tfidf_dfs():
    v = TfidfVectorizer()
    v.fit(corpus)
    a = v.dfs
    v.fit_transform(corpus)
    b = v.dfs
    assert not (a-b).any(), "Matricies are not equivalent"

def test_tfidf_results():
    v = TfidfVectorizer()
    v.fit(corpus)
    a = v.transform(corpus)
    b = v.fit_transform(corpus)

    assert not (a-b).toarray().any(), "Matricies are not equivalent"
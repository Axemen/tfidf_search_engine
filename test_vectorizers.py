import pytest
from pathlib import Path

from vectorizers import CountVectorizer, TfidfVectorizer

def load_test_files():
    return (p.read_text(encoding='utf-8') for p in Path('test_docs').glob('*'))

class TestCountVectorizer:
    def test_fit(self):
        cv = CountVectorizer()
        files = load_test_files()
        cv.fit(files)
        
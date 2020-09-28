import pytest
from ..indexer import InvertedIndex
from io import StringIO


data = """ 
    {
  "lookup_table": {
    "one": [1, 2, 3, 4, 5],
    "two": [3, 2, 1],
    "three": [8, 3, 9, 10],
    "four": [3, 8, 9],
    "five": [2, 7, 3, 5]
  },
  "doc_paths": {
    "0": "0.json",
    "1": "1.json",
    "2": "2.json",
    "3": "3.json",
    "4": "4.json",
    "5": "5.json",
    "6": "6.json",
    "7": "7.json",
    "8": "8.json",
    "9": "9.json",
    "10": "10.json"
  }
}
"""


def test_ii_load():
    ii = InvertedIndex.load(StringIO(data))
    assert ii, "InvertedIndex failed to load"
    assert ii._table, "InvertedIndex failed to create _table"


def test_ii_save():
    ii = InvertedIndex.load(StringIO(data))
    ii.save("test.json")


def test_ii_lookup():
    ii = InvertedIndex.load(StringIO(data))
    assert ii._table
    results = ii.lookup(['three', 'four', 'five', 'zero'])
    assert results, "results were not returned"

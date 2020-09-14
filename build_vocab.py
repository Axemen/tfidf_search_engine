import json
import re
from collections import Counter
from functools import partial
from pathlib import Path
from string import whitespace

from joblib import Parallel, delayed


def preprocess_text(text):
    # Replaces all whitespace chars with char ' '
    text = re.sub(f'[{whitespace}]', ' ', text)
    # Removes all characters except ascii and whitespace chars
    text = re.sub('[^a-zA-z ]', '', text)
    return text.lower()

def count_tokens_in_file(path, preprocess_fn=None):
    text = Path(path).read_text(encoding='utf-8')

    if preprocess_fn:
        text = preprocess_fn(text)

    return Counter(text.split())

if __name__ == "__main__":
    paths = Path('data/opinions').glob('*')
    
    do = delayed(partial(count_tokens_in_file, preprocess_fn=preprocess_text))
    tasks = (do(path) for path in paths)
    token_counts = Parallel(n_jobs=-1)(tasks)
    
    c = Counter()
    for count in token_counts:
        c.update(count)

    json.dump(dict(c), open('word_counts.json', 'w'))

from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path   
from joblib import dump
from tqdm import tqdm

def iter_files(paths):
    for path in tqdm(paths):
        yield Path(path).read_text(encoding='utf-8')

paths = list(Path('data/opinions').glob('*'))

vectorizer = TfidfVectorizer()

vectorizer.fit(iter_files(paths))

dump(vectorizer, open("tfidf_vectorizer.pkl", 'wb'))
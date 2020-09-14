import json
from pathlib import Path
from string import ascii_letters

from joblib import Parallel, delayed


def case_generator(path):
    with open(path, 'r', encoding='utf') as f:
        for line in f:
            yield json.loads(line)

def save_opinions(case, i):
    print(i, end='\r')
    opinions = case['casebody']['data']['opinions']
    all_opinions = "\n\n".join(opinion['text'] for opinion in opinions)
    name = "".join(c for c in case['name'] if c in ascii_letters + " ")[:20]
    Path(f"data/opinions/{name}").write_text(all_opinions, encoding='utf-8')

data_file_paths = Path("data").glob('*.json')

if __name__ == "__main__":

    do = delayed(save_opinions)

    for path in data_file_paths:
        print(path)
        tasks = (do(case, i) for i, case in enumerate(case_generator(path)))
        x = Parallel(n_jobs=-1)(tasks)

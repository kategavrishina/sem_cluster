from sklearn.cluster import Birch
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from ..utils import load_embedding, return_vec


def run_birch_baseline(path_to_dataset: str, path_to_model: str):
    model = load_embedding(path_to_model)
    dataset = pd.read_csv(path_to_dataset, sep='\t')
    result = pd.DataFrame()

    for word in dataset['word'].unique():
        part = dataset[dataset['word'] == word].copy()
        part['vector'] = part['context'].apply(return_vec, model=model)
        birch = Birch(n_clusters=2, threshold=0.1).fit(list(part['vector']))
        part['cluster'] = birch.labels_
        result = result.append(part)
    print(result)
    print(f"ARI Birch clustering: {adjusted_rand_score(result['cluster'], result['gold_sense_id'])}")
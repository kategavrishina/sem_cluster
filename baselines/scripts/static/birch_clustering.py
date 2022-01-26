from sklearn.cluster import Birch
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from ..utils import load_embedding, return_vec
import numpy as np


def run_birch_baseline(path_to_dataset: str, path_to_model: str):
    model = load_embedding(path_to_model)
    dataset = pd.read_csv(path_to_dataset, sep='\t')
    result = pd.DataFrame()
    ari = []
    print("Birch clustering")
    for word in dataset['word'].unique():
        part = dataset[dataset['word'] == word].copy()
        part['vector'] = part['context'].apply(return_vec, model=model)
        birch = Birch(n_clusters=2, threshold=0.1).fit(list(part['vector']))
        part['predict_sense_id'] = birch.labels_
        result = result.append(part)
        ari.append(adjusted_rand_score(part['predict_sense_id'], part['gold_sense_id']))
        print(f"Word: {word}, ARI: {round(adjusted_rand_score(part['predict_sense_id'], part['gold_sense_id']), 2)}")
    print(f"Average ARI Birch clustering: {round(np.mean(ari), 2)}\n")
    return result

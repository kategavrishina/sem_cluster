from sklearn.cluster import Birch
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from ..utils import load_embedding, return_vec
import numpy as np


def clusterize_search(vecs, gold_sense_ids=None,
                      ncs=list(range(1, 5, 1)) + list(range(5, 12, 2)),
                      affinity='cosine', linkage='average', ncs_search=True):
    """
    Gets word, vectors, gold_sense_ids and provides clustering.
    """
    print(vecs)
    zero_vecs = ((vecs ** 2).sum(axis=-1) == 0)
    if zero_vecs.sum() > 0:
        vecs = np.concatenate((vecs,
                               zero_vecs[:, np.newaxis].astype(vecs.dtype)),
                              axis=-1)

    best_clids = None
    best_silhouette = 0
    distances = []

    # matrix with computed distances between each pair of the two collections of inputs
    distance_matrix = cdist(vecs, vecs, metric=affinity)
    distances.append(distance_matrix)
    for nc in ncs:
        if nc >= len(vecs):
            print(f"We have only {len(vecs)} samples")
            break
        # clusterization
        clr = Birch(n_clusters=nc, threshold=0.1).fit(vecs)

        clids = clr.fit_predict(distance_matrix).labels_ if nc > 1 \
            else np.zeros(len(vecs))

        # computing metrics
        ari = adjusted_rand_score(gold_sense_ids, clids) if gold_sense_ids is not None else np.nan
        sil_cosine = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids, metric='cosine')

        if sil_cosine > best_silhouette:
            best_silhouette = sil_cosine
            best_clids = clids
            best_nc = nc
            best_ari = ari

    return best_clids, best_nc, best_ari


def run_birch_baseline(path_to_dataset: str, path_to_model: str):
    model = load_embedding(path_to_model)
    dataset = pd.read_csv(path_to_dataset, sep='\t')
    result = pd.DataFrame()
    ari = []
    print("Birch clustering")
    for word in dataset['word'].unique():
        part = dataset[dataset['word'] == word].copy()
        part['vector'] = part['context'].apply(return_vec, model=model)
        print(list(part['vector']))
        best_clids, best_nc, best_ari = clusterize_search(part['vector'],
                                                          gold_sense_ids=part['gold_sense_id'].tolist())
        part['predict_sense_id'] = best_clids
        result = result.append(part)
        ari.append(best_ari)
        print(f"Word: {word}, ARI: {round(best_ari, 2)}", f"Number of clusters {best_nc}")
    print(f"Average ARI Birch clustering: {round(np.mean(ari), 2)}\n")
    return result

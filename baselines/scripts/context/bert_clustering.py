from sklearn.cluster import KMeans
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from transformers import AutoTokenizer, AutoModel, BertConfig
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
from ..utils import return_bert_single_vec, return_bert_avg_vec


def clusterize_search(vecs, gold_sense_ids=None,
                      ncs=list(range(1, 5, 1)) + list(range(5, 12, 2)),
                      affinity='cosine'):
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
        best_clids = np.zeros(len(vecs))
        if nc >= len(vecs):
            print(f"We have only {len(vecs)} samples")
            break
        # clusterization
        clr = KMeans(n_clusters=nc, random_state=40).fit(vecs)

        clids = clr.fit_predict(distance_matrix) if nc > 1 \
            else np.zeros(len(vecs))

        # computing metrics
        ari = adjusted_rand_score(gold_sense_ids, clids) if gold_sense_ids is not None else np.nan
        sil_cosine = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids, metric='cosine')

        if sil_cosine > best_silhouette:
            best_silhouette = sil_cosine
            best_clids = clids
            best_nc = nc
            best_ari = ari
    print(best_ari)
    return best_clids, best_nc, best_ari


def run_bert_baseline(path_to_dataset: str, model_name: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config).to(device)

    dataset = pd.read_csv(path_to_dataset, sep='\t')
    result = pd.DataFrame()
    ari = []

    for word in dataset['word'].unique():
        try:
            part = dataset[dataset['word'] == word].copy()
            part['vector'] = part.apply(return_bert_single_vec, model=model, tokenizer=tokenizer, device=device, axis=1)
            best_clids, best_nc, best_ari = clusterize_search(preprocessing.normalize(list(part['vector'])),
                                                              gold_sense_ids=list(part['gold_sense_id']))
            part['predict_sense_id'] = best_clids
            result = pd.concat([result, part])
            ari.append(best_ari)
            print(f"Word: {word}, ARI: {round(best_ari, 2)}", f"Number of clusters {best_nc}")
        except Exception as E:
            print(word, E)
    print(f"Average ARI Bert clustering: {round(np.mean(ari), 2)}\n")
    return result
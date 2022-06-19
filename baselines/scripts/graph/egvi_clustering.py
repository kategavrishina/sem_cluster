import pandas as pd
from sklearn.metrics import adjusted_rand_score
from ..utils import load_embedding, return_vec
import numpy as np
from sklearn.cluster import KMeans

def remove_target_word(df):
    return df['context'].replace(df['word'], '')

def run_egvi_baseline(path_to_dataset: str, path_to_model: str):
    sense_inventories = pd.read_csv('http://ltdata1.informatik.uni-hamburg.de/158/ru/cc.ru.300.vec.gz.top100.inventory.tsv.gz', sep='\t')

    model = load_embedding(path_to_model)
    dataset = pd.read_csv(path_to_dataset, sep='\t')
    dataset['processed_context'] = dataset.apply(remove_target_word, axis=1)

    result = pd.DataFrame()
    ari = []
    print("Egvi clustering")
    for word in dataset['word'].unique():
        part = dataset[dataset['word'] == word].copy()
        sense_inventories_part = sense_inventories[sense_inventories['word'] == word].copy()
        # print(sense_inventories_part, word)

        part['vector'] = part['processed_context'].apply(return_vec, model=model)
        sense_inventories_part['vector'] = sense_inventories_part['cluster'].apply(return_vec, model=model)

        init = np.array([np.array(el) for el in sense_inventories_part['vector']])
        kmeans = KMeans(random_state=42, init=init, n_clusters=init.shape[0], n_init=1)
        cluster_names = kmeans.fit(np.array([np.array(el) for el in part['vector']]))
        part['predict_sense_id'] = cluster_names.labels_
        result = result.append(part)
        ari.append(adjusted_rand_score(part['predict_sense_id'], part['gold_sense_id']))
        print(f"Word: {word}, ARI: {round(adjusted_rand_score(part['predict_sense_id'], part['gold_sense_id']), 2)}")
    print(f"Average ARI Egvi clustering: {round(np.mean(ari), 2)}\n")
    return result

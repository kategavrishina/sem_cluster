import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score


def first_sense(dataset: pd.DataFrame):
    ari = []
    accuracy = []
    print("First sense for each")
    for word in dataset['word'].unique():
        part = dataset[dataset['word'] == word].copy()
        part['predict_sense_id'] = [1] * len(part)
        ari.append(adjusted_rand_score(part['predict_sense_id'], part['gold_sense_id']))
        accuracy.append(accuracy_score(part['predict_sense_id'], part['gold_sense_id']))
        print(f"Word: {word}, Accuracy: {round(accuracy_score(part['predict_sense_id'], part['gold_sense_id']), 2)} ARI: {round(adjusted_rand_score(part['predict_sense_id'], part['gold_sense_id']), 2)}")
    print(f"Average Accuracy: {np.mean(accuracy)}\nAverage ARI: {np.mean(ari)}\n")


def random_sense(dataset: pd.DataFrame):
    ari = []
    accuracy = []
    print("Random sense")
    for word in dataset['word'].unique():
        part = dataset[dataset['word'] == word].copy()
        part['predict_sense_id'] = [np.random.randint(1, 3) for _ in range(len(part))]
        ari.append(adjusted_rand_score(part['predict_sense_id'], part['gold_sense_id']))
        accuracy.append(accuracy_score(part['predict_sense_id'], part['gold_sense_id']))
        print(f"Word: {word}, Accuracy: {round(accuracy_score(part['predict_sense_id'], part['gold_sense_id']), 2)} ARI: {round(adjusted_rand_score(part['predict_sense_id'], part['gold_sense_id']), 2)}")
    print(f"Average Accuracy: {np.mean(accuracy)}\nAverage ARI: {np.mean(ari)}\n")


def run_all_naive_baselines(path_to_dataset: str):
    ds = pd.read_csv(path_to_dataset, sep='\t')
    first_sense(ds)
    random_sense(ds)

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def first_sense(dataset: pd.DataFrame):
    dummy_pred = [1] * len(dataset)
    print(f"Accuracy first sense for each: {accuracy_score(dummy_pred, dataset['gold_sense_id'])}")


def random_sense(dataset: pd.DataFrame):
    random_pred = [np.random.randint(1, 3) for _ in range(len(dataset))]
    print(f"Accuracy random sense: {accuracy_score(random_pred, dataset['gold_sense_id'])}")


def run_all_naive_baselines(path_to_dataset: str):
    ds = pd.read_csv(path_to_dataset, sep='\t')
    first_sense(ds)
    random_sense(ds)

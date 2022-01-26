from sklearn.cluster import KMeans
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, BertConfig
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
from ..utils import return_bert_vec


def run_bert_baseline(path_to_dataset: str, model_name: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config).to(device)

    dataset = pd.read_csv(path_to_dataset, sep='\t')
    result = pd.DataFrame()
    ari = []

    for word in dataset['word'].unique():
        part = dataset[dataset['word'] == word].copy()
        part['vector'] = part.apply(return_bert_vec, model=model, tokenizer=tokenizer, device=device, axis=1)
        clst = KMeans(n_clusters=2, random_state=40).fit(preprocessing.normalize(list(part['vector'])))
        part['predict_sense_id'] = clst.labels_
        result = result.append(part)
        ari.append(adjusted_rand_score(part['predict_sense_id'], part['gold_sense_id']))
        print(f"Word: {word}, ARI: {round(adjusted_rand_score(part['cluster'], part['gold_sense_id']), 2)}")
    # print(result['predict_sense_id'].value_counts())
    print(f"Average ARI Bert clustering: {round(np.mean(ari), 2)}\n")

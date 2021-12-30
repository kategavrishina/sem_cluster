from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from ..utlis import return_vec
from pymystem3 import Mystem

ms = Mystem()


def get_senses_vecs(word: str, model: KeyedVectors):
    lemma = ms.lemmatize(word)[0]
    lst_1 = [w[0] for w in model.similar_by_word(lemma + '_NOUN')][:1]
    total_vec_1 = [0] * 300
    for w in lst_1:
        total_vec_1 += model[w]
    first_sense_vec = total_vec_1 / len(lst_1)

    lst_2 = [w[0] for w in model.similar_by_vector((model[lemma + '_NOUN'] - first_sense_vec))]
    total_vec_2 = [0] * 300
    for w in lst_2:
        total_vec_2 += model[w]
    second_sense_vec = total_vec_2 / len(lst_2)
    return first_sense_vec, second_sense_vec


def compare_vecs(vector, first_sense_vec, second_sense_vec):
    first_sense_cosine = 1 - cosine(vector, first_sense_vec)
    second_sense_cosine = 1 - cosine(vector, second_sense_vec)
    if first_sense_cosine > second_sense_cosine:
        return 1
    else:
        return 0


def run_jamsic_baseline(path_to_dataset: str, path_to_model: str):
    model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
    dataset = pd.read_csv(path_to_dataset, sep='\t')
    result = pd.DataFrame()
    for word in dataset['word'].unique():
        part = dataset[dataset['word'] == word].copy()
        part['vector'] = part['context'].apply(return_vec, model=model)
        fsv, ssv = get_senses_vecs(word, model)
        part['cluster'] = part['vector'].apply(compare_vecs, first_sense_vec=fsv, second_sense_vec=ssv)
        result = result.append(part)
    print(f"ARI jamsic method: {adjusted_rand_score(result['cluster'], result['gold_sense_id'])}")

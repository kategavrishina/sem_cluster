from razdel import tokenize
from pymorphy2 import MorphAnalyzer
from string import punctuation
from nltk.corpus import stopwords
import numpy as np
from gensim.models import KeyedVectors
# import nltk

# nltk.download('stopwords')
russian_stops = stopwords.words('russian')
morph = MorphAnalyzer()
cached_dict = dict()


def return_vec(string: str, model: KeyedVectors):
    vec = [0.0] * 300
    length = 0
    for token in tokenize(string.lower()):
        if token.text not in punctuation and token.text not in russian_stops:
            if token.text in cached_dict:
                lemma = cached_dict[token.text]
            else:
                parsed = morph.parse(token.text)
                try:
                    pos = parsed[0].tag.POS
                    if pos in ('ADJF', 'ADJS'):
                        pos = 'ADJ'
                    if pos == 'ADVB':
                        pos = 'ADV'
                    if pos in ('INFN', 'PRTS', 'PRTF', 'GRND'):
                        pos = 'VERB'
                    lemma = parsed[0].normal_form.replace('ё', 'е') + '_' + pos
                except TypeError:
                    continue
                cached_dict[token] = lemma
            try:
                vec += model[lemma]
            except KeyError:
                continue
            length += 1
    if np.isnan(np.array(vec) / length).any():
        return [0.0] * 300
    return np.array(vec) / length

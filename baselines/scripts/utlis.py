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


def return_bert_vec(dframe, model, tokenizer, device):

    start_id = int(dframe.positions.split(',')[0].split('-')[0].strip())
    end_id = int(dframe.positions.split(',')[0].split('-')[1].strip())

    word = dframe.context[start_id: end_id]  # Extract word
    input = tokenizer.encode(dframe.context, return_tensors="pt").to(device)  # Encode sentence
    tokenized_sent = tokenizer.tokenize(dframe.context)  # Tokenize sentence
    sent_logits = model(input, return_dict=True)["last_hidden_state"]
    if word in tokenized_sent:
        word_index = list(np.where(np.array(tokenized_sent) == word)[0])[0]  # Get first instance of word
        word_embedding = sent_logits[:, word_index, :].cpu().detach().numpy()
    else:
        # in case the word is divided in pieces:
        prev_token = ""
        word_embedding = []
        for i, token_i in enumerate(tokenized_sent):
            token_i = token_i.lower()
            if word.startswith(token_i):
                word_embedding.append(sent_logits[:, i, :].cpu().detach().numpy())
                prev_token = token_i
                continue
            if prev_token and token_i.startswith("##"):
                word_embedding.append(sent_logits[:, i, :].cpu().detach().numpy())
                word_embedding = np.mean(word_embedding, axis=0)
                break
            else:
                prev_token = ""
                word_embedding = []
        if len(word_embedding) == 0:
            return np.zeros((1, 312))

    return word_embedding[0]

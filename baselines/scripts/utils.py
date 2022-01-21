from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import numpy as np
from .preprocessing_udpipe import udpipe_preprocessor
import gensim
import zipfile

russian_stops = stopwords.words('russian')
morph = MorphAnalyzer()
cached_dict = dict()


def load_embedding(modelfile):
    # Detect the model format by its extension:
    # Binary word2vec format:
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
        modelfile.endswith(".txt.gz")
        or modelfile.endswith(".txt")
        or modelfile.endswith(".vec.gz")
        or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith(".zip"):
        with zipfile.ZipFile(modelfile, "r") as archive:
            # Loading the model itself:
            stream = archive.open(
                "model.bin"  # or model.txt, if you want to look at the model
            )
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors="replace"
            )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    # Unit-normalizing the vectors (if they aren't already):
    #emb_model.init_sims(
    #    replace=True
    #)
    return emb_model


def return_vec(string: str, model: gensim.models.KeyedVectors):
    vec = [0.0] * 300
    length = 0
    TEST = []
    for token in udpipe_preprocessor(string):
        if token.split('_')[0] not in russian_stops:
            try:
                vec += model[token]
            except KeyError:
                TEST.append(token)
                continue
        length += 1
    if np.isnan(np.array(vec) / length).any():
        print(string)
        return [0.0] * 300
    return np.array(vec) / length


def return_bert_vec(dframe, model, tokenizer, device):
    start_id = int(dframe.positions.split(',')[0].split('-')[0].strip())
    end_id = int(dframe.positions.split(',')[0].split('-')[1].strip())

    word = dframe.context[start_id: end_id]  # Extract word
    bert_input = tokenizer.encode(dframe.context, return_tensors="pt").to(device)  # Encode sentence
    tokenized_sent = tokenizer.tokenize(dframe.context)  # Tokenize sentence
    sent_logits = model(bert_input, return_dict=True)["last_hidden_state"]
    if word in tokenized_sent:
        # Get first instance of word:
        word_index = list(np.where(np.array(tokenized_sent) == word)[0])[0]
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
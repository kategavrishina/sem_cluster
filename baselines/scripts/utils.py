from nltk.corpus import stopwords
import numpy as np
from .preprocessing_udpipe import udpipe_preprocessor
import gensim
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from sklearn.manifold import TSNE
import plotly.express as px

sns.set()

russian_stops = stopwords.words('russian')
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
    for token in udpipe_preprocessor(string):
        if token.split('_')[0] not in russian_stops:
            try:
                vec += model[token]
            except KeyError:
                continue
        length += 1
    if np.isnan(np.array(vec) / length).any():
        # print(string)
        return [0.0] * 300
    return list(np.array(vec) / length)


def return_bert_single_vec(dframe, model, tokenizer, device):
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
            return np.zeros((1, 1024))[0]
    return word_embedding[0]


def return_bert_avg_vec(dframe, model, tokenizer, device):
    word_average = []
    if ',' in dframe.positions:
        start_ids = [int(position.split('-')[0].strip()) for position in dframe.positions.split(',')]
        end_ids = [int(position.split('-')[1].strip()) for position in dframe.positions.split(',')]
    else:
        start_ids = [int(dframe.positions.split('-')[0].strip())]
        end_ids = [int(dframe.positions.split('-')[1].strip())]

    for start_id, end_id in zip(start_ids, end_ids):

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
                    word_embedding = np.mean(word_embedding, axis=0)[0]
                    break
                else:
                    prev_token = ""
                    word_embedding = []
            if len(word_embedding) == 0:
                word_embedding = np.zeros((1, 1024))

        word_average.append(word_embedding[0])

    return word_average[0]


def make_picture(data, method):
    shapes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    figs, axs = plt.subplots(2, 2, figsize=(20, 20))
    for word, shape in zip(data['word'].unique(), shapes):
        sub = data[data['word'] == word].copy()
        reducer = umap.UMAP()
        Y = reducer.fit_transform(StandardScaler().fit_transform(list(sub['vector'])))
        sns.scatterplot(x=Y[:, 0], y=Y[:, 1], ax=axs[shape[0], shape[1]], s=100, hue=sub['gold_sense_id'],
                        style=sub['predict_sense_id'])
        axs[shape[0], shape[1]].set_title(word)
    plt.savefig(f'{method}_visualization.png')


def format_str(df):
    string = df['context']
    target_word = df['word']
    ret = []
    i = 0
    processed_string = []
    for word in string.split():
        if word == target_word:
            processed_string.append('<b>')
            processed_string.append(word)
            processed_string.append('</b>')
        else:
            processed_string.append(word)
    for word in processed_string:
        if i == 12:
            ret.append('<br>')
            i = 0
        i += 1
        ret.append(word)
    return " ".join(ret)


def make_html_picture(df, method):
    df['format_context'] = df.apply(format_str, axis=1)
    df['gold_sense_id'] = df['gold_sense_id'].astype(str)
    df['predict_sense_id'] = df['predict_sense_id'].astype(str)

    tsne = TSNE(n_components=2, init='pca').fit_transform(list(df['vector']))
    df['tsne_x'] = [el[0] for el in tsne]
    df['tsne_y'] = [el[1] for el in tsne]

    fig = px.scatter(df,
                     x="tsne_x",
                     y="tsne_y",
                     color="gold_sense_id",
                     facet_col='word',
                     symbol='predict_sense_id',
                     text='format_context')

    fig.update_traces(mode="markers", hovertemplate=None)
    fig.write_html(f'visualizations/{method}_visualization.html')
import os
import re
import gzip
import csv
from bs4 import BeautifulSoup as bs
from tqdm.auto import tqdm
import pandas as pd



def make_content(file):
    with gzip.open(file, "rb") as f:
        data = f.readlines()
        data = [d.decode() for d in data]
        data = "".join(data)
    bs_data = bs(data, "lxml")
    sentences = bs_data.find_all("se")
    return sentences



def get_data(file, content_arr, idx):
    sentences = make_content(file)
    for _, sent in enumerate(sentences):
        sent_arr = []
        sent_arr.append(str(idx))
        lex_sent = []
        full_sent = []
        for word in sent.find_all('w'):
            ana = word.find('ana')
            lemma = ana['lex']
            try:
                gram = ana['gr'].split(',')[0]
                result = lemma + '_' + gram
            except KeyError:
                gram = ana.next_sibling["gr"]
                result = lemma + '_' + gram
            lex_sent.append(result)
        lex_str = ' '.join(lex_sent)
        sent_arr.append(lex_str)        
        full_sent = sent.text.strip().replace("\n", " ")
        sent_arr.append(full_sent)
        content_arr.append(sent_arr)
        idx += 1
    return idx





count = 0
idx = 0
content_arr = []
pbar = tqdm(total=124661)
for root, dirs, files in os.walk("./nkrya_full_source_merged_gzip"):
    for file in files:
        pbar.update(1)
        if not file.startswith(".") and file.endswith("xml.gz"):
            in_path = root + '/' + file
            idx = get_data(in_path, content_arr, idx)
            count += 1
        if count % 1000 == 0:
            print(count)
pbar.close()

df = pd.DataFrame(content_arr, columns = ['ID','LEMMAS', 'RAW'])
df.to_csv("corpus.csv.gz", encoding = 'utf-8', index = False, compression="gzip", line_terminator="\n")

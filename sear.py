import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim import matutils
import string
from pymystem3 import Mystem
mystem = Mystem()
from gensim.models.fasttext import FastText
from tqdm import tqdm_notebook
from judicial_splitter import splitter as sp
import pickle
import os
import json

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

names = []
for root, dirs, files in os.walk('avito_corpus'):
    for name in tqdm(files):
        names.append(os.path.join(root, name))

def preprocessing(input_text, del_stopwords=True, del_digit=False):
    """
    :input: raw text
        1. lowercase, del punctuation, tokenize
        2. normal form
        3. del stopwords
        4. del digits
    :return: lemmas
    """
    russian_stopwords = set(stopwords.words('russian'))
    words = [x.lower().strip(string.punctuation + '»«–…—') for x in word_tokenize(input_text)]
    lemmas = [mystem.lemmatize(x)[0] for x in words if x]

    lemmas_arr = []
    for lemma in lemmas:
        if del_stopwords:
            if lemma in russian_stopwords:
                continue
        if del_digit:
            if lemma.isdigit():
                continue
        lemmas_arr.append(lemma)
    return lemmas_arr

model_w2v = Word2Vec.load('araneum_none_fasttextcbow_300_5_2018.model')


def get_w2v_vectors(model_w2v, lemmas):
    vec_list = []

    for word in lemmas:
        try:
            vec = model_w2v.wv[word]
            vec_list.append(vec)
        except:
            continue

    vec = sum(vec_list) / len(vec_list)
    return vec


wv_base = []
    
for i, file in tqdm_notebook(enumerate(names)):
    if file.endswith('.txt'):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = file
        file = i
            
    pr_par = preprocessing(text)
    vector = get_w2v_vectors(model_w2v, pr_par)
    wv_base.append({'id' : [i], 'text': text, 'vector': vector})

from gensim import matutils
import numpy as np

def similarity(v1, v2):
    v1_norm = matutils.unitvec(np.array(v1))
    v2_norm = matutils.unitvec(np.array(v2))
    return np.dot(v1_norm, v2_norm)


def search_w2v(request, model_w2v, wv_base):
    request = preprocessing(request, del_stopwords=False)

    vec = get_w2v_vectors(model_w2v, request)
    similarity_dict = {}

    for elem in wv_base:
        sim = similarity(vec, elem['vector'])
        similarity_dict[sim] = elem['text']

    result = [similarity_dict[sim] for sim in sorted(similarity_dict, reverse=True)[:5]]
    return result


tagged_data = []
file = {}
i = 0


def train_doc2vec(names):
    for i, file in tqdm_notebook(enumerate(names)):
        if file.endswith('.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = file
            file = i

        tagged_data.append(TaggedDocument(words=preprocessing(text), tags=[i]))

    d2v_model = Doc2Vec(vector_size=100, min_count=5, alpha=0.025, min_alpha=0.025, epochs=100, workers=4, dm=1,
                        seed=42)
    d2v_model.build_vocab(tagged_data)
    d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs, report_delay=60)

    return d2v_model

model_d2v = train_doc2vec(names)

fname = get_tmpfile("doc2vec_model_answers")
model_d2v.save(fname)

def get_d2v_vectors(lemmas):
    vec = d2v_model.infer_vector(lemmas)
    return vec


base_dv = []
    
for i, file in tqdm_notebook(enumerate(names)):
    if file.endswith('.txt'):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = file
        file = i
            
    pr_par = preprocessing(text)
    vector = get_w2v_vectors(model_d2v, pr_par)
    base_dv.append({'id' : [i], 'text': text, 'vector': vector})


def search_d2v(request, model_d2v, base_dv):
    
    vec = get_w2v_vectors(model_d2v, request)
    similarity_dict = {}

    for elem in base_dv:
        sim = similarity(vec, elem['vector'])
        similarity_dict[sim] = elem['text']

    result = [similarity_dict[sim] for sim in sorted(similarity_dict, reverse=True)[:5]]
    return result


def search(request, search_method):

    if search_method == 'word2vec':
        search_result = search_w2v(request, model_w2v, wv_base)

    elif search_method == 'doc2vec':
        search_result = search_d2v(request, model_d2v, base_dv)

    else:
        raise TypeError('unsupported search method')

    return search_result


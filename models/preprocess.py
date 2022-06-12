import contractions
import copy
import nltk
import numpy as np
import pandas as pd
import re
import spacy
import string
import unicodedata

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from collections import Counter

# résumé -> resume
def remove_accent(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

# i'm -> i am
def remove_contraction(text):
    return contractions.fix(text)

# emojis, dll
def remove_special_char(text):
    return re.sub(r'[^a-zA-z0-9.,!?/:;\"\'\s]' , ' ', text)

# hello    world -> hello world
def remove_redundant_space(text):
    # trailing spaces
    text = re.sub(' +$', '', text)
    # leading spaces
    text = re.sub('^ +', '', text)
    # redundant spaces
    text = re.sub(' +', ' ', text)
    return text

# hello 123 -> hello
def remove_num(text):
    return re.sub(r'\d+', ' ', text)

# hello! -> hello
def remove_punctuation(text):
    text = re.sub('_', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text;

def preprocess_basic(text):
    text = remove_accent(text)
    text = remove_contraction(text)
    text = remove_special_char(text)
    text = remove_num(text)
    text = remove_punctuation(text)
    text = text.lower()
    text = remove_redundant_space(text)
    return text

dict_slang = pd.read_csv("https://raw.githubusercontent.com/jfcjaya/dataset/main/slang.csv")
columns = ['slang', 'formal']
dict_slang.sort_values('slang', inplace = True)
dict_slang.drop_duplicates(subset = columns, keep = False, inplace = True)
slang_word = pd.Series(dict_slang['formal'].values, index = dict_slang['slang']).to_dict()

def slang_to_formal(text):
    return " ".join([slang_word[word] if word in slang_word
                     else word
                     for word in text.split(' ')])

stopword_list = stopwords.words('english')
stopword_list.remove("not")

def remove_stopwords(text):
    words = word_tokenize(text)
    filter = [word
              for word in words
              if not word in stopword_list]
    return ' '.join(filter)

lemma = spacy.load('en_core_web_sm', parse = True, tag = True, entity = True)

def lemmatize_text(text):
    text = lemma(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def preprocess_advanced(text: str):
    text = slang_to_formal(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def preprocess_all(text: str):
    clean = copy.deepcopy(text)
    clean = preprocess_basic(clean)
    clean = preprocess_advanced(clean)
    return clean

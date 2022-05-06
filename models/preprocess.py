import re
import string
import unicodedata
import contractions

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

def preprocess(text):
    text = remove_accent(text)
    text = remove_contraction(text)
    text = remove_special_char(text)
    text = remove_num(text)
    text = remove_punctuation(text)
    text = text.lower()
    text = remove_redundant_space(text)
    return text

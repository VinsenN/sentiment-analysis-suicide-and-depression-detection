from sklearn.preprocessing import RobustScaler
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec

import numpy as np

w2v_param = {
    'size': 300,
    'min_count': 2,
    'window': 1,
    'scaling': 1
}

d2v_param = {
    'size': 200,
    'max_epochs': 50,
    'alpha': 0.025,
    'scaling': 1
}

def get_w2v_arr(tokenized_x, w2v_model):
    res = np.zeros((len(tokenized_x), w2v_param['size']))
    for i in range(len(tokenized_x)):
        for word in tokenized_x[i]:
            try:
                res[i] += w2v_model[word]
            except KeyError:
                continue
    if w2v_param['scaling'] == 1:
        res = RobustScaler().fit_transform(res)
    return res

def get_d2v_arr(tokenized_x, d2v_model):
    res = np.zeros((1, d2v_param['size']))
    res[0] = d2v_model.infer_vector(tokenized_x[0].words)
    if d2v_param['scaling'] == 1:
        res = RobustScaler().fit_transform(res)
    return res

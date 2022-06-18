import contractions
import math
import nltk
import numpy as np
import pandas as pd
import pickle
import re
import spacy
import string
import unicodedata

from collections import Counter

from copy import deepcopy

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import effective_n_jobs

from itertools import product

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix, accuracy_score, classification_report, precision_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

class SKLearn_Model:
    def __init__(self, 
                 threshold: int,
                 split: float, 
                 feature: str, 
                 model_type: str, 
                 **kwargs):
        
        # initialize parameter
        self.feature = feature
        self.model_type = model_type
        self.kwargs = kwargs

        # get dataset and feature extraction
        self.x_train, self.x_test, self.y_train, self.y_test = data[(threshold, split)]
        self._create_feature()

        # create model
        self._train_model()
    
    def _create_feature(self):
        if self.feature == "tf-idf":
            tfidf = TfidfVectorizer(max_features = 500, ngram_range = (1, 5))
            self.x_train = tfidf.fit_transform(self.x_train.values)
            self.x_test = tfidf.transform(self.x_test.values)
            self.transformer = tfidf

        elif self.feature == "w2v":
            self.w2v_param = {
                'size': 500,
                'window': 5
            }
            
            x_train_tok = self.x_train.apply(lambda x_train: x_train.split())
            x_train_tok = self.x_train.reset_index(drop = True)
            x_test_tok = self.x_test.apply(lambda x_test: x_test.split())
            x_test_tok = self.x_test.reset_index(drop = True)

            self.w2v_model = self._get_w2v_model(x_train_tok, **self.w2v_param)
            self.x_train = self._get_w2v_arr(x_train_tok, **self.w2v_param)
            self.x_test = self._get_w2v_arr(x_test_tok, **self.w2v_param)

    def _get_w2v_model(self, tokenized_x, **kwargs):
        w2v_model = Word2Vec(
                        tokenized_x, 
                        size = kwargs['size'], 
                        window = kwargs['window'], 
                        workers = effective_n_jobs(-1), 
                        sg = 1, 
                        alpha = 0.01, 
                        min_alpha = 0.0001, 
                        seed = 777
                    )
        return w2v_model

    def _get_w2v_arr(self, tokenized_x, **kwargs):
        res = np.zeros((len(tokenized_x), kwargs['size']))
        for i in range(len(tokenized_x)):
            for word in tokenized_x[i]:
                try:
                    res[i] += self.w2v_model[word]
                except KeyError:
                    continue

        res = MinMaxScaler(feature_range = (-1, 1)).fit_transform(res)
        return res

    def _train_model(self):
        if self.model_type == "mnb":
            # set default grid search parameter
            self.mnb_param = {
                'alpha': [0.0001, 0.001, 0.01, 1.0],
                'fit_prior': [True, False],
            }
            # if custom parameter exists, use it instead
            if 'mnb_param' in self.kwargs:
                self.mnb_param = self.kwargs['mnb_param']

            if(self.feature == 'w2v'): # rescale as naive bayes only accepts non-negative values
                self.x_train = MinMaxScaler(feature_range = (0, 1)).fit_transform(self.x_train)
                self.x_test = MinMaxScaler(feature_range = (0, 1)).fit_transform(self.x_test)

            mnb = MultinomialNB()
            mnb_grid = GridSearchCV(mnb, param_grid = self.mnb_param, scoring = "accuracy", verbose = 2, n_jobs = -1)
            mnb_grid.fit(self.x_train, self.y_train)
            self.best_params = mnb_grid.best_params_
            self.model = mnb_grid.best_estimator_

        elif self.model_type == "svm":
            # set default grid search parameter
            self.svm_param = {
                'C': [0.01, 0.1, 1, 10, 100],
                'kernel': ["linear"],
                'gamma': [0.01, 0.1, 1, 10, 'auto', 'scale'],
                'max_iter': np.logspace(0, 5, num = 6),
                'verbose': [True]
            }
            # if custom parameter exists, use it instead
            if 'svm_param' in self.kwargs:
                self.svm_param = self.kwargs['svm_param']

            svm = SVC()
            svm_grid = GridSearchCV(svm, param_grid = self.svm_param, scoring = "accuracy", verbose = 2, n_jobs = -1)
            svm_grid.fit(self.x_train, self.y_train)
            self.best_params = svm_grid.best_params_
            self.model = svm_grid.best_estimator_

        elif self.model_type == "rf":
            # set default grid search parameter
            self.rf_param = {
                'bootstrap': [True],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [None, 80, 90, 100, 110],
                'min_samples_split': [2, 8, 10, 12]
            }
            # if custom parameter exists, use it instead
            if 'rf_param' in self.kwargs:
                self.rf_param = self.kwargs['rf_param']

            rf = RandomForestClassifier()
            rf_grid = GridSearchCV(rf, param_grid = self.rf_param, scoring = "accuracy", verbose = 2, n_jobs = -1)
            rf_grid.fit(self.x_train, self.y_train)
            self.best_params = rf_grid.best_params_
            self.model = rf_grid.best_estimator_

    def get_result(self):
        result = dict()
        y_pred = self.model.predict(self.x_train)
        result['train_acc'] = accuracy_score(self.y_train, y_pred) * 100
        result['train_pre'] = precision_score(self.y_train, y_pred) * 100
        result['train_rec'] = recall_score(self.y_train, y_pred) * 100
        result['train_f1']  = f1_score(self.y_train, y_pred) * 100

        y_pred = self.model.predict(self.x_test)
        result['test_acc'] = accuracy_score(self.y_test, y_pred) * 100
        result['test_pre'] = precision_score(self.y_test, y_pred) * 100
        result['test_rec'] = recall_score(self.y_test, y_pred) * 100
        result['test_f1']  = f1_score(self.y_test, y_pred) * 100

        return result

    def evaluate_model(self):
        print("\nBest Parameter")
        print("=====================================================")
        print(self.best_params)
        
        y_pred = self.model.predict(self.x_train)
        print("\nTraining Dataset")
        print("=====================================================")
        print(f"accuracy score  : {accuracy_score(self.y_train, y_pred) * 100}")
        print(f"precision score : {precision_score(self.y_train, y_pred, average = 'macro') * 100}")
        print(f"recall score    : {recall_score(self.y_train, y_pred, average = 'macro') * 100}")
        print(f"f1-score        : {f1_score(self.y_train, y_pred, average = 'macro') * 100}")

        y_pred = self.model.predict(self.x_test)
        print("\nTesting Dataset")
        print("=====================================================")
        plot_confusion_matrix(self.model, self.x_test, self.y_test)
        print(f"accuracy score  : {accuracy_score(self.y_test, y_pred) * 100}")
        print(f"precision score : {precision_score(self.y_test, y_pred, average = 'macro') * 100}")
        print(f"recall score    : {recall_score(self.y_test, y_pred, average = 'macro') * 100}")
        print(f"f1-score        : {f1_score(self.y_test, y_pred, average = 'macro') * 100}")
        print("\n")

    def predict(self, text):
        if self.feature == "tf-idf":
            text = self.transformer.transform([text])
        else:
            text = text.split()
            text = self._get_w2v_arr([text], **self.w2v_param)

        prediction = self.model.predict(text)[0]
        return prediction

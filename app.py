from flask import Flask, request, render_template

from models.preprocess import preprocess_all
from models.vectorizer import get_w2v_arr, get_d2v_arr
from models.sklearn_model import SKLearn_Model
import pickle
import numpy as np

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk.tokenize import word_tokenize

app = Flask(__name__)

# count_vect = pickle.load(open("models/vectorizer.pickle", 'rb')) # CountVectorizer
# w2v_model = Word2Vec.load("models/word2vec.model") # Word2Vec
# d2v_model = Doc2Vec.load("models/doc2vec.model") # Doc2Vec

model = pickle.load(open('models/testing.sav','rb')) # SVM

@app.route('/', methods=['GET'])
def show_html():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    message = request.form.get('message')
    filtered_message = preprocess_all(message)
    filtered_message = [filtered_message]

    # transformed_message = [TaggedDocument(words=word_tokenize(w), tags=[str(i)]) for i, w in enumerate(filtered_message)]
    # print(transformed_message)
    # # vectorized_message = get_w2v_arr(filtered_message, w2v_model)
    # vectorized_message = get_d2v_arr(transformed_message, d2v_model)

    result = model.predict(filtered_message)
    return render_template('index.html', predicted_message = message, prediction = result)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

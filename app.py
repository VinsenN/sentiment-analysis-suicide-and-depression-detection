# notes:
# before running in local, make sure flask, pickle, and contractions installed

from flask import Flask, request, render_template

from models.preprocess import *
import pickle

app = Flask(__name__)

count_vect = pickle.load(open("models/vectorizer.pickle", 'rb')) # CountVectorizer
model = pickle.load(open('models/model.sav','rb')) # MultinomialNB

@app.route('/', methods=['GET'])
def show_html():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    message = request.form.get('message')
    filtered_message = preprocess(message)
    result = model.predict(count_vect.transform([filtered_message]))[0]
    return render_template('index.html', predicted_message = message, prediction = result)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

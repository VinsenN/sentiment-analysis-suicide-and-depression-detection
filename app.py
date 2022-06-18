from flask import Flask, request, render_template

from models.preprocess import preprocess_all
import pickle

app = Flask(__name__)

feature_extraction = pickle.load(open(r'models/tfidf_transfomer.pkl', 'rb'))
model = pickle.load(open(r'models/svm_tfidf.sav', 'rb'))

@app.route('/', methods=['GET'])
def show_html():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    message = request.form.get('message')

    if len(message) > 100:
        return render_template('index.html', warning_message = 'exceeded_char_length')
    elif message.isspace():
        return render_template('index.html')

    filtered_message = preprocess_all(message)
    vectorized_message = feature_extraction.transform([message])

    result = model.predict(vectorized_message)[0]
    return render_template('index.html', predicted_message = message.strip(), prediction = result)


if __name__ == '__main__':
    app.run(port=3000, debug=True)

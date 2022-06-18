from flask import Flask, request, render_template

from models.preprocess import preprocess_all
from models.sklearn_model import SKLearn_Model
import pickle

app = Flask(__name__)
class_model = pickle.load(open('models/svm_tfidf.obj.','rb'))

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
    result = class_model.predict(filtered_message)
    return render_template('index.html', predicted_message = message.strip(), prediction = result)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

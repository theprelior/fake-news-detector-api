# Import necessary libraries and modules
from flask import Flask, request, jsonify
from flask_cors import CORS
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

# Create a Flask app
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

# Load the trained models and other required data
LR = pickle.load(open('lrModel.sav', 'rb'))
DT = pickle.load(open('dtModel.sav', 'rb'))
GB = pickle.load(open('gbModel.sav', 'rb'))
RF = pickle.load(open('rfModel.sav', 'rb'))

x_train = pickle.load(open('x_train.sav', 'rb'))
x_test = pickle.load(open('x_test.sav', 'rb'))
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Define a function for text preprocessing
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Define a function to output the label based on the predicted class
def output_label(n):
    if n == 0:
        return 'Fake News'
    elif n == 1:
        return 'Not a fake news'

# Define a route for the '/detect' endpoint and specify the allowed methods
@app.route('/detect', methods=['POST'])
def detect_fake_news():
    # Get the text from the request's JSON payload
    text = request.json['text']
    
    # Prepare the input text for prediction
    testing_news = {'text': [text]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    # Make predictions using the loaded models
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GB.predict(new_xv_test)
    pred_RFC = RF.predict(new_xv_test)

    # Create a dictionary containing the predictions
    prediction = {
        'news': text,
        'LR_prediction': output_label(pred_LR[0]),
        'DT_prediction': output_label(pred_DT[0]),
        'GB_prediction': output_label(pred_GBC[0]),
        'RF_prediction': output_label(pred_RFC[0])
    }

    # Return the prediction as a JSON response
    return jsonify(prediction)

# Run the Flask app on the specified host and port
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

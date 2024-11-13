from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load the dataset
data = pd.read_csv('dataset.csv', encoding='latin1')

X = data['Input Message']
y = data['Response']

# Create a pipeline with TF-IDF Vectorizer and a Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['message']
    response = model.predict([user_input])
    return jsonify({'response': response[0]})

if __name__ == "__main__":
    app.run(debug=True)

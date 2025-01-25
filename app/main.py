from flask import Flask, request, jsonify
import tensorflow as tf
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle
import os
import subprocess
import logging
import nltk


app = Flask(__name__)

# Initialisation des outils NLTK
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Charger le modèle
model_path = os.path.join(BASE_DIR, 'model_lstm_compatible.h5')
model = tf.keras.models.load_model(model_path)


# Charger le tokenizer
tokenizer_path = os.path.join(BASE_DIR, 'tokenizer.pickle')
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)


# Dictionnaire pour stocker la prédiction associée à un tweet
prediction_cache = {}


def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', 'mention', text)
    text = re.sub(r'\#\w+', 'hashtag', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


@app.route('/')
def home():
    return "API d'analyse de sentiments en ligne"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        clean_tokens = clean_text(data['text'])
        text_cleaned = ' '.join(clean_tokens)
        sequences = tokenizer.texts_to_sequences([text_cleaned])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=512,
            padding='post',
            truncating='post'
        )
        
        prediction = model.predict(padded_seq)
        sentiment_score = float(prediction[0][0])
        sentiment = "positif" if sentiment_score >= 0.5 else "négatif"
        
        return jsonify({
            'text': data['text'],
            'sentiment': sentiment,
            'score': sentiment_score,
            'processed_text': text_cleaned
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedbackpositif', methods=['POST'])
def feedbackpositif():
    return "true"


@app.route('/feedbacknegatif', methods=['POST'])
def feedbacknegatif():
    data = request.get_json()
    tweet_text = data.get('text', 'Texte inconnu')
    with open(os.path.join(BASE_DIR, 'feedback_negatif.txt'), 'a') as file:
        file.write(f'{tweet_text}: {prediction_cache.get(tweet_text, "Non prédit")}\n')
    return jsonify({'status': 'Feedback enregistré'}), 200


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

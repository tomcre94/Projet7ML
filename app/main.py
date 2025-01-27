from flask import Flask, request, jsonify
import tensorflow as tf
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pickle
import os
import logging
import nltk
import emoji
import contractions

app = Flask(__name__)

# Initialisation des outils NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = './app/trained_model/model_lstm_compatible.h5'

# Initialisation des outils de prétraitement
stop_words = set(stopwords.words('english'))
stop_words -= {'no', 'not', 'nor', 'none', 'never', 'nothing', 'nowhere', 'hardly', 'barely', 'scarcely'}
lemmatizer = WordNetLemmatizer()
tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# Chargement du modèle
model = tf.keras.models.load_model(MODEL_PATH)

# Chargement du tokenizer
with open(os.path.join(BASE_DIR, 'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)

# Cache pour les prédictions
prediction_cache = {}

def get_wordnet_pos(word):
    """Obtenir la partie du discours pour la lemmatisation"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text):
    """Fonction de prétraitement du texte"""
    # Conversion des emojis en texte
    text = emoji.demojize(text)
    
    # Expansion des contractions
    text = contractions.fix(text)
    
    # Nettoyage des URLs, mentions et hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', 'mention', text)
    text = re.sub(r'\#(\w+)', r'\1', text)
    
    # Gestion des répétitions de lettres
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Tokenisation
    tokens = tweet_tokenizer.tokenize(text)
    
    # Nettoyage et normalisation
    tokens = [token for token in tokens if (
        token not in stop_words and
        len(token) > 1 and
        not token.isnumeric()
    )]
    
    # Lemmatisation avec POS tagging
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) 
             for token in tokens]
    
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
        
        # Prétraitement
        clean_tokens = clean_text(data['text'])
        text_cleaned = ' '.join(clean_tokens)
        
        # Tokenization et padding
        sequences = tokenizer.texts_to_sequences([text_cleaned])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=512,
            padding='post',
            truncating='post'
        )
        
        # Prédiction
        prediction = model.predict(padded_seq)
        sentiment_score = float(prediction[0][0])
        sentiment = "positif" if sentiment_score >= 0.5 else "négatif"
        
        # Mise en cache de la prédiction
        prediction_cache[data['text']] = sentiment_score
        
        return jsonify({
            'text': data['text'],
            'sentiment': sentiment,
            'score': sentiment_score,
            'processed_text': text_cleaned
        })
        
    except Exception as e:
        app.logger.error(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedbackpositif', methods=['POST'])
def feedbackpositif():
    return jsonify({'status': 'Feedback positif reçu'}), 200

@app.route('/feedbacknegatif', methods=['POST'])
def feedbacknegatif():
    try:
        data = request.get_json()
        tweet_text = data.get('text', 'Texte inconnu')
        with open(os.path.join(BASE_DIR, 'feedback_negatif.txt'), 'a') as file:
            file.write(f'{tweet_text}: {prediction_cache.get(tweet_text, "Non prédit")}\n')
        return jsonify({'status': 'Feedback enregistré'}), 200
    except Exception as e:
        app.logger.error(f"Erreur lors de l'enregistrement du feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Configuration du logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    
    # Lancement de l'application
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
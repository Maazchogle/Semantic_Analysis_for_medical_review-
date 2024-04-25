from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the Keras model and TF-IDF vectorizer
model = load_model('model.keras')

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        
        # Vectorize the input review using the TF-IDF vectorizer
        review_tfidf = tfidf_vectorizer.transform([review])
        
        # Make a prediction using the Keras model
        prediction = model.predict(review_tfidf)
        
        # Map the prediction to a human-readable label and emoji
        if prediction > 0.5:
            sentiment_label = 'Positive 😃'
            emoji = '😃'
        else:
            sentiment_label = 'Negative 😞'
            emoji = '😞'

        return render_template('result.html', review=review, sentiment=sentiment_label, emoji=emoji)

if __name__ == '__main__':
    app.run(debug=True)

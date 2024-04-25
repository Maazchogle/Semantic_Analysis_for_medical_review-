from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

app = Flask(__name__)

# Load the SVM model and TF-IDF vectorizer
with open('svm_model_subset.pkl', 'rb') as model_file:
    svm_model_subset = pickle.load(model_file)

with open('tfidf_vectorizer_subset.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer_subset = pickle.load(vectorizer_file)

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
        review_tfidf = tfidf_vectorizer_subset.transform([review])
        
        # Make a prediction using the SVM model
        prediction = svm_model_subset.predict(review_tfidf)[0]
        
        # Map the prediction to a human-readable label and emoji
        if prediction == 'positive':
            sentiment_label = 'Positive 😃'
            emoji = '😃'
        else:
            sentiment_label = 'Negative 😞'
            emoji = '😞'

        return render_template('result.html', review=review, sentiment=sentiment_label, emoji=emoji)

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

app = Flask(__name__)

# Load the SVM model and TF-IDF vectorizer
with open('svm_model_subset.pkl', 'rb') as model_file:
    svm_model_subset = pickle.load(model_file)

with open('tfidf_vectorizer_subset.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer_subset = pickle.load(vectorizer_file)

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
        review_tfidf = tfidf_vectorizer_subset.transform([review])
        
        # Make a prediction using the SVM model
        prediction = svm_model_subset.predict(review_tfidf)[0]
        
        # Map the prediction to a human-readable label and emoji
        if prediction == 'positive':
            sentiment_label = 'Positive 😃'
            emoji = '😃'
        else:
            sentiment_label = 'Negative 😞'
            emoji = '😞'

        return render_template('result.html', review=review, sentiment=sentiment_label, emoji=emoji)

if __name__ == '__main__':
    app.run(debug=True)



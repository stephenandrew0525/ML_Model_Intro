from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Model information dictionary
MODEL_INFO = {
    'naive_bayes': {
        'name': 'Naive Bayes',
        'overview': 'Naive Bayes is a probabilistic classifier based on Bayes\' theorem with the "naive" assumption of conditional independence between every pair of features.',
        'how_it_works': 'The model calculates the probability of a message being spam based on the presence of certain words, assuming that the occurrence of each word is independent of other words.',
        'code_example': 'from sklearn.naive_bayes import MultinomialNB\n\nmodel = MultinomialNB()\nmodel.fit(X_train, y_train)\npredictions = model.predict(X_test)',
        'strengths': [
            'Fast training and prediction',
            'Works well with high-dimensional data',
            'Requires less training data',
            'Good for real-time applications'
        ],
        'limitations': [
            'Assumes features are independent',
            'May miss complex patterns',
            'Sensitive to feature correlations',
            'Requires careful feature engineering'
        ],
        'best_practices': [
            'Use TF-IDF vectorization for text features',
            'Handle class imbalance with class weights',
            'Regularize the model if needed',
            'Monitor feature independence assumption'
        ]
    },
    'random_forest': {
        'name': 'Random Forest',
        'overview': 'Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes of the individual trees.',
        'how_it_works': 'The model creates multiple decision trees, each trained on a random subset of the data and features. The final prediction is made by majority voting of all trees.',
        'code_example': 'from sklearn.ensemble import RandomForestClassifier\n\nmodel = RandomForestClassifier(n_estimators=100)\nmodel.fit(X_train, y_train)\npredictions = model.predict(X_test)',
        'strengths': [
            'High accuracy',
            'Handles non-linear relationships',
            'Feature importance analysis',
            'Resistant to overfitting'
        ],
        'limitations': [
            'More complex',
            'Slower training time',
            'Requires more memory',
            'Less interpretable'
        ],
        'best_practices': [
            'Tune number of trees and depth',
            'Use feature importance for feature selection',
            'Handle class imbalance',
            'Monitor memory usage'
        ]
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'overview': 'Logistic Regression is a statistical model that uses a logistic function to model a binary dependent variable.',
        'how_it_works': 'The model learns a linear decision boundary in the feature space and uses the sigmoid function to convert the output into probabilities.',
        'code_example': 'from sklearn.linear_model import LogisticRegression\n\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)\npredictions = model.predict(X_test)',
        'strengths': [
            'Interpretable',
            'Works well with linearly separable data',
            'Fast training and prediction',
            'Good for baseline models'
        ],
        'limitations': [
            'May struggle with complex patterns',
            'Requires feature scaling',
            'Sensitive to outliers',
            'Limited to linear decision boundaries'
        ],
        'best_practices': [
            'Scale features before training',
            'Handle class imbalance',
            'Regularize to prevent overfitting',
            'Monitor feature importance'
        ]
    },
    'bert': {
        'name': 'BERT',
        'overview': 'BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that has revolutionized natural language processing.',
        'how_it_works': 'The model uses a transformer architecture to process text bidirectionally, understanding context from both directions simultaneously.',
        'code_example': 'from transformers import BertTokenizer, BertForSequenceClassification\n\nmodel = BertForSequenceClassification.from_pretrained("bert-base-uncased")\ntokenizer = BertTokenizer.from_pretrained("bert-base-uncased")',
        'strengths': [
            'State-of-the-art performance',
            'Understands context',
            'Works well with complex language',
            'Transfer learning capability'
        ],
        'limitations': [
            'Requires large computational resources',
            'Needs fine-tuning for specific tasks',
            'Large model size',
            'Slower inference time'
        ],
        'best_practices': [
            'Fine-tune on specific task',
            'Use appropriate batch size',
            'Monitor GPU memory usage',
            'Regularize during fine-tuning'
        ]
    }
}

# Initialize models and vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
models = {
    'naive_bayes': MultinomialNB(),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Function to load and preprocess data
def load_data(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Rename columns to match our expected format
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        
        # Convert labels to binary (spam=1, ham=0)
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Split the data
        X = df['text'].values
        y = df['label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorize the text data for traditional models
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        
        # Train traditional models
        for model_name, model in models.items():
            model.fit(X_train_vectorized, y_train)
            train_score = model.score(X_train_vectorized, y_train)
            test_score = model.score(X_test_vectorized, y_test)
            print(f"{model_name} - Training accuracy: {train_score:.2f}, Testing accuracy: {test_score:.2f}")
        
        # Save the trained models and vectorizer
        for model_name, model in models.items():
            joblib.dump(model, f'{model_name}_model.joblib')
        joblib.dump(vectorizer, 'vectorizer.joblib')
        print("Models and vectorizer saved successfully!")
        
        return True
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

# Load the models and vectorizer if they exist, otherwise train new ones
model_files = {name: f'{name}_model.joblib' for name in models.keys()}
vectorizer_path = 'vectorizer.joblib'

if all(os.path.exists(path) for path in model_files.values()) and os.path.exists(vectorizer_path):
    print("Loading existing models and vectorizer...")
    for model_name in models.keys():
        models[model_name] = joblib.load(f'{model_name}_model.joblib')
    vectorizer = joblib.load(vectorizer_path)
else:
    print("Training new models with spam_sms.csv...")
    if load_data('spam_sms.csv'):
        print("Models trained successfully!")
    else:
        print("Failed to train models. Please check the dataset format.")
        # Initialize empty models if training fails
        models = {
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

def predict_bert(text):
    try:
        # Tokenize and prepare input
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get prediction
        with torch.no_grad():
            outputs = bert_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][prediction].item()
        
        return prediction, confidence
    except Exception as e:
        print(f"Error in BERT prediction: {str(e)}")
        return 0, 0.5  # Default to "Not Spam" with 50% confidence if error occurs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model/<model_type>')
def model_details(model_type):
    if model_type not in MODEL_INFO:
        return "Model not found", 404
    
    # Get model performance metrics
    if model_type in models:
        model = models[model_type]
        X_test = vectorizer.transform(pd.read_csv('spam_sms.csv')['v2'].values)
        y_test = pd.read_csv('spam_sms.csv')['v1'].map({'spam': 1, 'ham': 0}).values
        training_accuracy = model.score(X_test, y_test) * 100
        testing_accuracy = model.score(X_test, y_test) * 100
    else:
        training_accuracy = 0
        testing_accuracy = 0

    return render_template('model_details.html',
                         model_name=MODEL_INFO[model_type]['name'],
                         overview=MODEL_INFO[model_type]['overview'],
                         how_it_works=MODEL_INFO[model_type]['how_it_works'],
                         code_example=MODEL_INFO[model_type]['code_example'],
                         strengths=MODEL_INFO[model_type]['strengths'],
                         limitations=MODEL_INFO[model_type]['limitations'],
                         best_practices=MODEL_INFO[model_type]['best_practices'],
                         training_accuracy=round(training_accuracy, 2),
                         testing_accuracy=round(testing_accuracy, 2))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_type = data.get('model_type', 'naive_bayes')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        if model_type == 'bert':
            prediction, confidence = predict_bert(text)
        else:
            if model_type not in models:
                return jsonify({'error': 'Invalid model type'}), 400
                
            # Vectorize the input text
            text_vectorized = vectorizer.transform([text])
            
            # Make prediction using selected model
            model = models[model_type]
            prediction = model.predict(text_vectorized)[0]
            probability = model.predict_proba(text_vectorized)[0]
            confidence = float(max(probability))
        
        return jsonify({
            'prediction': 'Spam' if prediction == 1 else 'Not Spam',
            'confidence': confidence,
            'model_used': model_type
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render dynamically sets the PORT
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"  # Enable debug only if explicitly set

    app.run(host="0.0.0.0", port=port, debug=debug_mode)

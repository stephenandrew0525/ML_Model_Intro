# Spam Message Detector

This is a simple web application that uses machine learning to detect whether a given message is spam or not. The application uses a Naive Bayes classifier with TF-IDF vectorization for text classification.

## Features

- Modern and responsive web interface
- Real-time spam detection
- Confidence score for predictions
- Easy to use text input

## Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- joblib

## Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Open a terminal in the project directory
2. Run the Flask application:
   ```
   python app.py
   ```
3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## How to Use

1. Enter the message you want to check in the text area
2. Click the "Check for Spam" button
3. The result will be displayed below the button, showing whether the message is spam or not, along with a confidence score

## Note

This application uses a simple training dataset for demonstration purposes. For better accuracy in a production environment, you should:

1. Use a larger, more diverse training dataset
2. Implement proper model persistence
3. Add more sophisticated text preprocessing
4. Consider using more advanced models like BERT or other transformer-based models 
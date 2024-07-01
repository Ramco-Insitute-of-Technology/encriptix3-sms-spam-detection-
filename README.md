# encriptix3-sms-spam-detection-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path, encoding='latin1'):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
        return data
    except UnicodeDecodeError:
        raise ValueError("File encoding issue. Try a different encoding.")
    except FileNotFoundError:
        raise ValueError("File not found. Check the file path.")

def preprocess_data(data):
    # Keep only the relevant columns and rename them
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    return data

def split_data(data):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
    )
    return X_train, X_test, y_train, y_test

def extract_features(X_train, X_test):
    # Preprocess the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

def train_model(X_train_tfidf, y_train):
    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model

def evaluate_model(model, X_test_tfidf, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, conf_matrix

def main():
    file_path = 'path/to/your/spam[1].csv'
    
    # Load and preprocess data
    data = load_data(file_path)
    data = preprocess_data(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Extract features
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    
    # Train the model
    model = train_model(X_train_tfidf, y_train)
    
    # Evaluate the model
    accuracy, report, conf_matrix = evaluate_model(model, X_test_tfidf, y_test)
    
    # Print results
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')
    print(f'Confusion Matrix:\n{conf_matrix}')

if __name__ == "__main__":
    main()

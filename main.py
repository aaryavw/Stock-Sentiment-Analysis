import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download essential NLP data components from NLTK
nltk.download('stopwords')

# ==============================================================================
# 1. LOAD AND INSPECT THE DATA
# ==============================================================================
print("Loading financial news dataset...")

# Note: The original Kaggle file 'all-data.csv' often lacks headers and uses ISO-8859-1 encoding
try:
    df = pd.read_csv('all-data.csv', names=['sentiment', 'headline'], encoding='ISO-8859-1')
except FileNotFoundError:
    # Fallback name if you renamed your file
    df = pd.read_csv('FinancialData.csv', names=['sentiment', 'headline'], encoding='ISO-8859-1')

print(f"Dataset Shape: {df.shape}")
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# ==============================================================================
# 2. TEXT PREPROCESSING (The NLP Pipeline)
# ==============================================================================
print("\nCleaning text data...")

stop_words = set(stopwords.words('english'))

def clean_headline(text):
    # A. Convert text to lowercase
    text = text.lower()
    
    # B. Remove punctuation, special characters, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # C. Tokenize (split into words) and remove common stopwords (e.g., 'the', 'is', 'at')
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    # D. Re-join words back into a single string
    return " ".join(cleaned_words)

# Apply the cleaning pipeline to every headline
df['clean_headline'] = df['headline'].apply(clean_headline)

print("\nOriginal Headline sample:")
print(df['headline'].iloc[0])
print("\nCleaned Headline sample:")
print(df['clean_headline'].iloc[0])

# ==============================================================================
# 3. SPLIT DATA & TEXT VECTORIZATION
# ==============================================================================
X = df['clean_headline']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nVectorizing text using TF-IDF...")
# TF-IDF converts words into a matrix of numbers based on their importance and frequency
vectorizer = TfidfVectorizer(max_features=5000) 
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==============================================================================
# 4. TRAIN THE SENTIMENT MODEL
# ==============================================================================
print("Training Logistic Regression Model...")
# Logistic Regression handles sparse text matrix dimensions beautifully and rapidly
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# ==============================================================================
# 5. EVALUATE THE MODEL
# ==============================================================================
y_pred = model.predict(X_test_tfidf)

print("\n================ EVALUATION REPORT ================")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Matrix Metrics:")
print(classification_report(y_test, y_pred))

# ==============================================================================
# 6. TEST WITH YOUR OWN CUSTOM HEADLINES
# ==============================================================================
print("\n================ LIVE MODEL INFERENCE ================")
custom_headlines = [
    "Apple shares skyrocket after beating quarterly earnings expectations.",
    "Tesla faces supply chain bottlenecks, production halts expected next week.",
    "Microsoft announces completely flat revenue growth for the third quarter."
]

# Clean, transform, and predict on the new live phrases
cleaned_custom = [clean_headline(h) for h in custom_headlines]
custom_tfidf = vectorizer.transform(cleaned_custom)
predictions = model.predict(custom_tfidf)

for headline, sentiment in zip(custom_headlines, predictions):
    print(f"\nHeadline: '{headline}'")
    print(f"AI Predicted Sentiment: {sentiment.upper()}")

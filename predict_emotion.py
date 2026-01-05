"""
Emotion Detection from Text - Interactive Prediction Script
This script loads the data, trains models, and allows interactive predictions
"""

import pandas as pd
import numpy as np
import nltk
import string
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load emoji dictionary
emojis = pd.read_csv('Dataset/emojis.txt', sep=',', header=None)
emojis_dict = {i: j for i, j in zip(emojis[0], emojis[1])}
pattern = '|'.join(sorted(re.escape(k) for k in emojis_dict))

def replace_emojis(text):
    text = re.sub(pattern, lambda m: emojis_dict.get(m.group(0)), text, flags=re.IGNORECASE)
    return text

def remove_punct(text):
    text = replace_emojis(text)
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def tokenization(text):
    text = text.lower()
    text = re.split('\W+', text)
    return text

stopword = nltk.corpus.stopwords.words('english')
stopword.extend(['yr', 'year', 'woman', 'man', 'girl', 'boy', 'one', 'two', 'sixteen', 
                'yearold', 'fu', 'weeks', 'week', 'treatment', 'associated', 'patients', 
                'may', 'day', 'case', 'old', 'u', 'n', 'didnt', 'ive', 'ate', 'feel', 
                'keep', 'brother', 'dad', 'basic', 'im'])

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

def clean_text(text):
    text = remove_punct(text)
    text = tokenization(text)
    text = remove_stopwords(text)
    text = lemmatizer(text)
    return text

print("Loading dataset...")
df = pd.read_csv('Dataset/text_emotions.csv')
print(f"Dataset loaded: {df.shape[0]} samples")

print("\nPreprocessing data...")
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['sentiment'], 
                                                      test_size=0.3, random_state=0)

print("Extracting features...")
countVectorizer1 = CountVectorizer(analyzer=clean_text)
countVector1 = countVectorizer1.fit_transform(X_train)
countVector2 = countVectorizer1.transform(X_test)

tfidf_transformer_xtrain = TfidfTransformer()
x_train = tfidf_transformer_xtrain.fit_transform(countVector1)

tfidf_transformer_xtest = TfidfTransformer()
x_test = tfidf_transformer_xtest.fit_transform(countVector2)

print("\nTraining models...")
# Encode labels for XGBoost
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("1. Training SVM...")
svm = SGDClassifier()
svm.fit(x_train, y_train)

print("2. Training Logistic Regression...")
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

print("3. Training Random Forest...")
rfc = RandomForestClassifier(n_estimators=1, random_state=0)
rfc.fit(x_train, y_train)

print("4. Training XGBoost (this may take a few minutes)...")
xgbc = XGBClassifier(max_depth=16, n_estimators=100, nthread=6, verbose=0)
xgbc.fit(x_train, y_train_encoded)

print("5. Training Naive Bayes...")
mnb = MultinomialNB()
mnb.fit(x_train, y_train)

print("6. Training Decision Tree...")
dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)

print("\n" + "="*60)
print("All models trained successfully!")
print("="*60)
print("\nYou can now enter text to detect emotions.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    input_str = input("What's in your mind: ").strip()
    
    if input_str.lower() in ['quit', 'exit', 'nothing', '']:
        print("Thank you for using Emotion Detection!")
        break
    
    if not input_str:
        continue
    
    try:
        # Process the input text
        processed_text = tfidf_transformer_xtest.fit_transform(
            countVectorizer1.transform([input_str])
        )
        
        print("\n" + "-"*60)
        print("Emotion Predictions:")
        print("-"*60)
        svm_pred = svm.predict(processed_text)[0]
        lr_pred = logisticRegr.predict(processed_text)[0]
        rfc_pred = rfc.predict(processed_text)[0]
        xgbc_pred_encoded = xgbc.predict(processed_text)[0]
        xgbc_pred = label_encoder.inverse_transform([xgbc_pred_encoded])[0]
        mnb_pred = mnb.predict(processed_text)[0]
        dt_pred = dt.predict(processed_text)[0]
        
        print(f'SVM:                {svm_pred}')
        print(f'Logistic Regression: {lr_pred}')
        print(f'Random Forest:       {rfc_pred}')
        print(f'XGBoost:             {xgbc_pred}')
        print(f'Naive Bayes:         {mnb_pred}')
        print(f'Decision Tree:       {dt_pred}')
        print("-"*60 + "\n")
        
    except Exception as e:
        print(f"Error processing input: {e}\n")


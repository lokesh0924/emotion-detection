"""
Emotion Detection from Text - Simple Prediction Script
Usage: python predict_emotion_simple.py "your text here"
"""

import pandas as pd
import numpy as np
import nltk
import string
import re
import sys
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
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

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

def predict_emotion(input_text, models_dict, countVectorizer1, tfidf_transformer_xtest, label_encoder):
    """Predict emotion for given text"""
    try:
        processed_text = tfidf_transformer_xtest.fit_transform(
            countVectorizer1.transform([input_text])
        )
        
        predictions = {}
        predictions['SVM'] = models_dict['svm'].predict(processed_text)[0]
        predictions['Logistic Regression'] = models_dict['lr'].predict(processed_text)[0]
        predictions['Random Forest'] = models_dict['rfc'].predict(processed_text)[0]
        
        xgbc_pred_encoded = models_dict['xgbc'].predict(processed_text)[0]
        predictions['XGBoost'] = label_encoder.inverse_transform([xgbc_pred_encoded])[0]
        
        predictions['Naive Bayes'] = models_dict['mnb'].predict(processed_text)[0]
        predictions['Decision Tree'] = models_dict['dt'].predict(processed_text)[0]
        
        return predictions
    except Exception as e:
        return {'error': str(e)}

# Load and prepare models
print("Loading dataset and training models...")
print("This may take a few minutes...\n")

df = pd.read_csv('Dataset/text_emotions.csv')
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['sentiment'], 
                                                      test_size=0.3, random_state=0)

countVectorizer1 = CountVectorizer(analyzer=clean_text)
countVector1 = countVectorizer1.fit_transform(X_train)
countVector2 = countVectorizer1.transform(X_test)

tfidf_transformer_xtrain = TfidfTransformer()
x_train = tfidf_transformer_xtrain.fit_transform(countVector1)

tfidf_transformer_xtest = TfidfTransformer()
x_test = tfidf_transformer_xtest.fit_transform(countVector2)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Train models
print("Training models...")
svm = SGDClassifier()
svm.fit(x_train, y_train)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

rfc = RandomForestClassifier(n_estimators=1, random_state=0)
rfc.fit(x_train, y_train)

xgbc = XGBClassifier(max_depth=16, n_estimators=100, nthread=6, verbose=0)
xgbc.fit(x_train, y_train_encoded)

mnb = MultinomialNB()
mnb.fit(x_train, y_train)

dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)

models_dict = {
    'svm': svm,
    'lr': logisticRegr,
    'rfc': rfc,
    'xgbc': xgbc,
    'mnb': mnb,
    'dt': dt
}

print("Models trained successfully!\n")

# Get input from command line or use example
if len(sys.argv) > 1:
    input_text = ' '.join(sys.argv[1:])
else:
    # Example texts if no input provided
    examples = [
        "I am so happy today!",
        "This is terrible, I'm so angry!",
        "I'm scared of what might happen.",
        "I love you so much!",
        "I feel so sad and lonely.",
        "Wow, that's surprising!"
    ]
    print("No input provided. Using example texts:\n")
    input_text = examples[0]  # Use first example

print(f"Input text: '{input_text}'")
print("\n" + "="*60)
print("Emotion Predictions:")
print("="*60)

predictions = predict_emotion(input_text, models_dict, countVectorizer1, 
                              tfidf_transformer_xtest, label_encoder)

if 'error' in predictions:
    print(f"Error: {predictions['error']}")
else:
    for model_name, emotion in predictions.items():
        print(f"{model_name:20s}: {emotion}")
    print("="*60)


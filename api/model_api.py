"""
Simple Flask API to expose the emotion detection models.

Endpoint:
    POST /predict
    Body: { "text": "your sentence here" }
    Response:
    {
        "final_emotion": "joy",
        "per_model": {
            "SVM": "joy",
            "Logistic Regression": "joy",
            "Random Forest": "joy",
            "XGBoost": "joy",
            "Naive Bayes": "joy",
            "Decision Tree": "joy"
        }
    }
"""

import os
import re
import string
from collections import Counter

from flask import Flask, jsonify, request
import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from xgboost import XGBClassifier

app = Flask(__name__)


def ensure_nltk():
    """Download required NLTK data if missing."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


def build_pipeline():
    """Load data, build preprocessing pipeline and train models (one‑time)."""
    ensure_nltk()

    # Load emoji dictionary
    emojis_path = os.path.join(os.path.dirname(__file__), "..", "Dataset", "emojis.txt")
    emojis = pd.read_csv(emojis_path, sep=",", header=None)
    emojis_dict = {i: j for i, j in zip(emojis[0], emojis[1])}
    pattern = "|".join(sorted(re.escape(k) for k in emojis_dict))

    def replace_emojis(text: str) -> str:
        return re.sub(pattern, lambda m: emojis_dict.get(m.group(0)), text, flags=re.IGNORECASE)

    def remove_punct(text: str):
        text = replace_emojis(text)
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub("[0-9]+", "", text)
        return text

    def tokenization(text: str):
        text = text.lower()
        return re.split(r"\W+", text)

    stopword = nltk.corpus.stopwords.words("english")
    stopword.extend(
        [
            "yr",
            "year",
            "woman",
            "man",
            "girl",
            "boy",
            "one",
            "two",
            "sixteen",
            "yearold",
            "fu",
            "weeks",
            "week",
            "treatment",
            "associated",
            "patients",
            "may",
            "day",
            "case",
            "old",
            "u",
            "n",
            "didnt",
            "ive",
            "ate",
            "feel",
            "keep",
            "brother",
            "dad",
            "basic",
            "im",
        ]
    )

    def remove_stopwords(tokens):
        return [word for word in tokens if word and word not in stopword]

    wn = nltk.WordNetLemmatizer()

    def lemmatizer(tokens):
        return [wn.lemmatize(word) for word in tokens]

    def clean_text(text: str):
        text = remove_punct(text)
        tokens = tokenization(text)
        tokens = remove_stopwords(tokens)
        tokens = lemmatizer(tokens)
        return tokens

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "Dataset", "text_emotions.csv")
    df = pd.read_csv(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["sentiment"], test_size=0.3, random_state=0
    )

    count_vectorizer = CountVectorizer(analyzer=clean_text)
    count_train = count_vectorizer.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()
    x_train = tfidf_transformer.fit_transform(count_train)

    # Label encoder for XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Train models (same as notebook)
    svm = SGDClassifier()
    svm.fit(x_train, y_train)

    logistic_regr = LogisticRegression()
    logistic_regr.fit(x_train, y_train)

    rfc = RandomForestClassifier(n_estimators=1, random_state=0)
    rfc.fit(x_train, y_train)

    xgbc = XGBClassifier(max_depth=16, n_estimators=100, nthread=6, verbose=0)
    xgbc.fit(x_train, y_train_encoded)

    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)

    dt = tree.DecisionTreeClassifier()
    dt.fit(x_train, y_train)

    models = {
        "SVM": svm,
        "Logistic Regression": logistic_regr,
        "Random Forest": rfc,
        "XGBoost": xgbc,
        "Naive Bayes": mnb,
        "Decision Tree": dt,
    }

    return {
        "vectorizer": count_vectorizer,
        "tfidf": tfidf_transformer,
        "label_encoder": label_encoder,
        "models": models,
    }


# Build pipeline once at startup
PIPELINE = build_pipeline()


def majority_vote(per_model: dict) -> str:
    """
    Combine per‑model predictions into a single label.
    Priority order in case of tie: SVM > XGBoost > others.
    """
    counts = Counter(per_model.values())
    if not counts:
        return ""

    # Basic majority
    most_common = counts.most_common()
    top_count = most_common[0][1]
    candidate_emotions = [e for e, c in most_common if c == top_count]

    if len(candidate_emotions) == 1:
        return candidate_emotions[0]

    # Tie‑breaking using preferred models
    priority_models = ["SVM", "XGBoost"]
    for model_name in priority_models:
        emotion = per_model.get(model_name)
        if emotion in candidate_emotions:
            return emotion

    # Fallback to first candidate
    return candidate_emotions[0]


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be a non‑empty string."}), 400

    vectorizer = PIPELINE["vectorizer"]
    tfidf = PIPELINE["tfidf"]
    label_encoder = PIPELINE["label_encoder"]
    models = PIPELINE["models"]

    # Transform input text
    count_vec = vectorizer.transform([text])
    x_vec = tfidf.transform(count_vec)

    per_model = {}
    for name, model in models.items():
        if name == "XGBoost":
            encoded_pred = model.predict(x_vec)[0]
            pred = label_encoder.inverse_transform([encoded_pred])[0]
        else:
            pred = model.predict(x_vec)[0]
        per_model[name] = str(pred)

    final_emotion = majority_vote(per_model)

    return jsonify({"final_emotion": final_emotion, "per_model": per_model})


if __name__ == "__main__":
    # For local testing only. In production, use gunicorn or a WSGI server.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)



# Emotion Detection from Text using Machine Learning
## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Dataset](#dataset)
* [Models](#models)
* [Result](#result)
* [Conclusion](#conclusion)

## General info 
Emotion is one of the basic instincts of a human being. Emotion detection plays a vital role in the field of textual analysis. At present, people’s expressions and emotional states have turned into the leading topic for research works. Emotion Detection and Recognition from texts are recent fields of research that are closely related to Emotion Analysis. Emotion Analysis aims at detecting and recognizing feelings through the expressions from sentences, such as anger, surprise, joy, fear, sadness, love etc.
	
## Technologies
Project is created with:
* **Google Colab**

Language used in this project is:
* **Python**
	
## Dataset
Here tweet emotions from SemEval-2018 Affect in Tweets Distant Supervision Corpus (AIT-2018 Dataset) is used.This dataset has two columns content and sentiment. It has 20000 unique text classified with 6 emotions such as anger, love, surprise, fear, joy, sadness. This dataset contains lots of emotions but they mainly clustered those emotions into 4 basic emotions such as anger, fear, joy, sadness.

## Models
* Support Vector Machine (SVM)
* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier
* Multinomial Naive Bayes
* Decision Tree Classifier

## Result
Accuracy of the six models - Support Vector Machine (SVM), Logistic Regression, Random
Forest Classifier, XGBoost Classifier, Multinomial Naive Bayes and Decision Tree Classifier
are `89.52%`, `86.2%`, `75.97%`, `89.1%`, `67.63%` and `86.82%` respectively.

## Conclusion 
In this project, we have used "tweet emotions from SemEval-2018 Affect in Tweets Distant Supervision Corpus (AIT-2018 Dataset)". As it’s a noisy dataset, first we had to do the preprocessing such as punctuation remove and converted emojis into texts, tokenization, remove stopwords, and lemmatization.

Then we have implemented six models and compared them. After comparing them, we observed that **SVM** model gave the highest accuracy which is **89.52%**.

After completion of our project, we observed that our project can detect emotions from texts.




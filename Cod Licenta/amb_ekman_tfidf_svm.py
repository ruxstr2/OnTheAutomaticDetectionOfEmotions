# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:12:43 2023

@author: ruxan
"""

import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import svm
import datetime
from nltk.stem import PorterStemmer
from sklearn.metrics import zero_one_loss
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier

dataset = pd.read_csv("all_emotions_ekman.csv")
X = dataset['text']
ylabels = dataset['classes']

stopwords_set = stopwords.words('english')
corpus = []
nr_texte = 54263
ps = PorterStemmer()
for i in range(nr_texte):
    review = X[i]
    review = re.sub('[^a-zA-Z]',' ',review) #eliminate non-letter characters
    review = review.lower() #lower case
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords_set]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating the model
vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(corpus).toarray()
X_train, X_test, y_train, y_test = train_test_split(X[:nr_texte], ylabels[:nr_texte], test_size = 0.20, random_state = 0)

#Classifier
y_train_tuple_list = [tuple(map(int, x.split(','))) for x in y_train]
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train_tuple_list)

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time:", datetime_string)

classifier = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
multilabel_classifier = MultiOutputClassifier(classifier).fit(X_train, y_train)
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time after fit:", datetime_string)

y_pred = multilabel_classifier.predict(X_test)
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time after predict:", datetime_string)

#Evaluation => TODO add messages, timer; positive-negative scale

y_test_tuple_list = [tuple(map(int, x.split(','))) for x in y_test]
y_test = mlb.transform(y_test_tuple_list)

print(classification_report(y_test, y_pred, target_names=['anger','disgust','fear','joy','sadness','suprise','neutral']))

f1 = f1_score(y_test, y_pred,average = 'macro')
print("F1 macro: %.2f%%" % (f1 * 100.0))

f1 = f1_score(y_test, y_pred,average = 'weighted')
print("F1 weighted: %.2f%%" % (f1 * 100.0))

f1 = f1_score(y_test, y_pred,average = 'micro')
print("F1 micro: %.2f%%" % (f1 * 100.0))

zo = zero_one_loss(y_test, y_pred, normalize = False)
print("Zero-one-loss: %.2f%%" % (zo * 100.0))

hl = hamming_loss(y_test, y_pred)
print("Hamming loss: %.2f%%" % (hl * 100.0))
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:00:03 2023

@author: ruxan
"""

import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import svm
import numpy as np
import gensim
from gensim.models import Word2Vec,KeyedVectors

df = pd.read_csv('all_emotions_ekman_unambiguous.csv')
#df = pd.read_csv('all_sentiments_unambiguous.csv')
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['classes'], test_size=0.2, random_state=0)

X_train = X_train.apply(gensim.utils.simple_preprocess)
X_test = X_test.apply(gensim.utils.simple_preprocess)

model_cbow = Word2Vec(sentences=X_train, vector_size=400,sg=1, min_count=10, workers=4, window =3, epochs = 20)

model_cbow.wv.most_similar("ok")

vec = model_cbow.wv['king'] - model_cbow.wv['man'] + model_cbow.wv['woman']
model_cbow.wv.most_similar([vec])

def vectorize(sentence, w2v_model):
    words_vecs = [w2v_model.wv[word] for word in sentence if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(400)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)

X_train = np.array([vectorize(sentence,model_cbow) for sentence in X_train])
X_test = np.array([vectorize(sentence,model_cbow) for sentence in X_test])

classifier = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
y_pred = classifier.predict(X_test)


print(classification_report(y_test, y_pred, target_names=['anger','disgust','fear','joy','sadness','suprise','neutral']))
#print(classification_report(y_test, y_pred, target_names=['neutral','positive','negative','ambiguous'])) 

f1 = f1_score(y_test, y_pred,average = 'macro')
print("F1 macro: %.2f%%" % (f1 * 100.0))

f1 = f1_score(y_test, y_pred,average = 'weighted')
print("F1 weighted: %.2f%%" % (f1 * 100.0))

f1 = f1_score(y_test, y_pred,average = 'micro')
print("F1 micro: %.2f%%" % (f1 * 100.0))


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=['anger','disgust','fear','joy','sadness','suprise','neutral'])
cm_display.plot()
plt.show()

              precision    recall  f1-score   support

       anger       0.50      0.06      0.10      1097
     disgust       0.00      0.00      0.00       124
        fear       0.00      0.00      0.00       161
         joy       0.70      0.71      0.71      3796
     sadness       0.73      0.11      0.19       622
     suprise       0.33      0.00      0.00       952
     neutral       0.44      0.82      0.57      3142

    accuracy                           0.55      9894
   macro avg       0.39      0.24      0.22      9894
weighted avg       0.54      0.55      0.48      9894

Accuracy: 54.54%
F1: 47.59%

              precision    recall  f1-score   support

       anger       0.51      0.05      0.10      1097
     disgust       0.00      0.00      0.00       124
        fear       0.00      0.00      0.00       161
         joy       0.70      0.71      0.71      3796
     sadness       0.76      0.11      0.19       622
     suprise       0.00      0.00      0.00       952
     neutral       0.44      0.82      0.57      3142

    accuracy                           0.55      9894
   macro avg       0.34      0.24      0.22      9894
weighted avg       0.51      0.55      0.48      9894

Accuracy: 54.60%
F1: 47.62%

              precision    recall  f1-score   support

     neutral       0.45      0.78      0.58      3210
    positive       0.73      0.68      0.71      3817
    negative       0.57      0.26      0.35      2023
   ambiguous       0.00      0.00      0.00       963

    accuracy                           0.56     10013
   macro avg       0.44      0.43      0.41     10013
weighted avg       0.54      0.56      0.52     10013

F1 macro: 40.88%
F1 weighted: 52.50%
F1 micro: 56.34%

              precision    recall  f1-score   support

       anger       0.45      0.05      0.09      1097
     disgust       0.00      0.00      0.00       124
        fear       0.00      0.00      0.00       161
         joy       0.70      0.71      0.70      3796
     sadness       0.75      0.11      0.19       622
     suprise       0.20      0.00      0.00       952
     neutral       0.44      0.81      0.57      3142

    accuracy                           0.54      9894
   macro avg       0.36      0.24      0.22      9894
weighted avg       0.52      0.54      0.47      9894

F1 macro: 22.29%
F1 weighted: 47.37%
F1 micro: 54.34%


vs

skipgram

[('women', 0.5436233878135681),
 ('king', 0.4199151396751404),
 ('areas', 0.379663348197937),
 ('influence', 0.3765932321548462),
 ('attraction', 0.36790233850479126),
 ('safety', 0.36440062522888184),
 ('sorts', 0.3629583418369293),
 ('programs', 0.3597966432571411),
 ('poverty', 0.3574797809123993),
 ('species', 0.35629522800445557)]

model_cbow.wv.most_similar("ok")
Out[13]: 
[('gotcha', 0.5610153675079346),
 ('alright', 0.5331026315689087),
 ('okay', 0.527996838092804),
 ('flex', 0.4807935655117035),
 ('yah', 0.4780280292034149),
 ('nsfw', 0.46878886222839355),
 ('ahh', 0.4575332701206207),
 ('sheep', 0.4571068286895752),
 ('fam', 0.45686861872673035),
 ('woah', 0.4474829137325287)]

              precision    recall  f1-score   support

     neutral       0.46      0.76      0.58      3210
    positive       0.74      0.70      0.72      3817
    negative       0.61      0.34      0.43      2023
   ambiguous       0.67      0.00      0.00       963

    accuracy                           0.58     10013
   macro avg       0.62      0.45      0.43     10013
weighted avg       0.62      0.58      0.55     10013

F1 macro: 43.33%
F1 weighted: 54.69%
F1 micro: 57.94%

              precision    recall  f1-score   support

       anger       0.54      0.09      0.16      1097
     disgust       0.62      0.04      0.08       124
        fear       0.00      0.00      0.00       161
         joy       0.70      0.72      0.71      3796
     sadness       0.79      0.15      0.25       622
     suprise       0.50      0.02      0.03       952
     neutral       0.45      0.81      0.58      3142

    accuracy                           0.55      9894
   macro avg       0.52      0.26      0.26      9894
weighted avg       0.58      0.55      0.49      9894

Accuracy: 55.49%
F1: 49.36%

              precision    recall  f1-score   support

       anger       0.56      0.10      0.16      1097
     disgust       0.67      0.05      0.09       124
        fear       0.00      0.00      0.00       161
         joy       0.70      0.72      0.71      3796
     sadness       0.76      0.14      0.24       622
     suprise       0.36      0.01      0.03       952
     neutral       0.45      0.80      0.58      3142

    accuracy                           0.55      9894
   macro avg       0.50      0.26      0.26      9894
weighted avg       0.56      0.55      0.49      9894

F1 macro: 25.85%
F1 weighted: 49.36%
F1 micro: 55.48%

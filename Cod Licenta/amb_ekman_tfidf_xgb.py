# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 16:43:33 2023

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
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer
import datetime
from nltk.stem import PorterStemmer
from sklearn.metrics import zero_one_loss
from sklearn.metrics import hamming_loss

#dataset = pd.read_csv("all_emotions_ekman.csv")
dataset = pd.read_csv("all_emotions.csv")
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

#xgb_cl = OneVsRestClassifier(xgb.XGBClassifier())
xgb_cl = xgb.XGBClassifier()
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time:", datetime_string)

xgb_cl.fit(X_train,y_train)
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time after fit:", datetime_string)

y_pred = xgb_cl.predict(X_test)
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time after predict:", datetime_string)


y_test_tuple_list = [tuple(map(int, x.split(','))) for x in y_test]
y_test = mlb.transform(y_test_tuple_list)

#print(classification_report(y_test, y_pred, target_names=['anger','disgust','fear','joy','sadness','suprise','neutral']))
#print(classification_report(y_test, y_pred, target_names=['neutral','positive','negative','ambiguous']))
print(classification_report(y_test, y_pred, target_names=['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral']))

f1 = f1_score(y_test, y_pred,average = 'macro')
print("F1 macro: %.2f%%" % (f1 * 100.0))

f1 = f1_score(y_test, y_pred,average = 'weighted')
print("F1 weighted: %.2f%%" % (f1 * 100.0))

f1 = f1_score(y_test, y_pred,average = 'micro')
print("F1 micro: %.2f%%" % (f1 * 100.0))

zo = zero_one_loss(y_test, y_pred)
print("Zero-one-loss: %.2f%%" % (zo * 100.0))

hl = hamming_loss(y_test, y_pred)
print("Hamming loss: %.2f%%" % (hl * 100.0))

              precision    recall  f1-score   support

       anger       0.66      0.19      0.29      1385
     disgust       0.78      0.25      0.38       202
        fear       0.69      0.40      0.50       177
         joy       0.84      0.67      0.75      4329
     sadness       0.76      0.35      0.48       811
     suprise       0.68      0.14      0.24      1308
     neutral       0.60      0.19      0.28      3569

   micro avg       0.77      0.37      0.50     11781
   macro avg       0.72      0.31      0.42     11781
weighted avg       0.72      0.37      0.47     11781
 samples avg       0.40      0.39      0.39     11781

F1 macro: 41.59%
F1 weighted: 46.71%
F1 micro: 50.26%
Zero-one-loss: 64.22%
Hamming loss: 11.47%

----sentiment
              precision    recall  f1-score   support

     neutral       0.60      0.19      0.28      3569
    positive       0.84      0.67      0.75      4329
    negative       0.75      0.35      0.48      2457
   ambiguous       0.68      0.14      0.24      1308

   micro avg       0.77      0.39      0.52     11663
   macro avg       0.72      0.34      0.44     11663
weighted avg       0.73      0.39      0.49     11663
 samples avg       0.42      0.40      0.41     11663

F1 macro: 43.54%
F1 weighted: 49.05%
F1 micro: 52.24%
Zero-one-loss: 62.19%
Hamming loss: 19.37%

Current date and time: 11-08-2023 12:12:58
Current date and time after fit: 11-08-2023 13:33:58
Current date and time after predict: 11-08-2023 13:34:00

--- allemotions

                precision    recall  f1-score   support

    admiration       0.75      0.47      0.58      1038
     amusement       0.76      0.72      0.74       585
         anger       0.61      0.20      0.30       384
     annoyance       0.53      0.04      0.07       634
      approval       0.62      0.08      0.14       738
        caring       0.50      0.07      0.12       272
     confusion       0.79      0.07      0.12       327
     curiosity       0.82      0.06      0.12       570
        desire       0.59      0.34      0.43       147
disappointment       0.66      0.07      0.12       312
   disapproval       0.41      0.01      0.03       493
       disgust       0.78      0.25      0.38       202
 embarrassment       0.64      0.33      0.43        64
    excitement       0.74      0.16      0.26       204
          fear       0.69      0.41      0.51       145
     gratitude       0.94      0.87      0.90       696
         grief       0.00      0.00      0.00        22
           joy       0.55      0.34      0.42       372
          love       0.73      0.76      0.74       469
   nervousness       0.47      0.18      0.26        39
      optimism       0.68      0.43      0.53       373
         pride       0.61      0.33      0.43        33
   realization       0.59      0.09      0.15       248
        relief       0.50      0.07      0.12        45
       remorse       0.59      0.49      0.53       135
       sadness       0.67      0.25      0.37       340
      surprise       0.57      0.25      0.35       252
       neutral       0.60      0.19      0.28      3569

     micro avg       0.70      0.28      0.40     12708
     macro avg       0.62      0.27      0.34     12708
  weighted avg       0.65      0.28      0.35     12708
   samples avg       0.31      0.29      0.29     12708

F1 macro: 33.72%
F1 weighted: 35.22%
F1 micro: 39.98%
Zero-one-loss: 75.44%
Hamming loss: 3.51%
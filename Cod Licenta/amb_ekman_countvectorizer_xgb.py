# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 14:41:34 2023

@author: ruxan
"""

import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import xgboost as xgb
import datetime
from nltk.stem import PorterStemmer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
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
    #review = [word for word in review if not word in stopwords_set]
    review = [ps.stem(word) for word in review if not word in stopwords_set]
    review = ' '.join(review)
    corpus.append(review)

# Creating the bag of Word Model
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
       
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
     disgust       0.79      0.22      0.34       202
        fear       0.68      0.39      0.50       177
         joy       0.84      0.67      0.75      4329
     sadness       0.75      0.35      0.48       811
     suprise       0.70      0.14      0.24      1308
     neutral       0.61      0.20      0.31      3569

   micro avg       0.77      0.38      0.51     11781
   macro avg       0.72      0.31      0.41     11781
weighted avg       0.72      0.38      0.47     11781
 samples avg       0.40      0.39      0.39     11781

F1 macro: 41.40%
F1 weighted: 47.40%
F1 micro: 50.84%
Zero-one-loss: 63.32%
Hamming loss: 11.41%

Current date and time: 08-08-2023 17:42:32
Current date and time after fit: 08-08-2023 19:50:33
Current date and time after predict: 08-08-2023 19:50:37



#stemmed: 19565
              precision    recall  f1-score   support

     neutral       0.61      0.20      0.31      3569
    positive       0.84      0.67      0.75      4329
    negative       0.75      0.35      0.47      2457
   ambiguous       0.70      0.14      0.24      1308

   micro avg       0.77      0.40      0.53     11663
   macro avg       0.72      0.34      0.44     11663
weighted avg       0.73      0.40      0.50     11663
 samples avg       0.42      0.41      0.41     11663

F1 macro: 44.09%
F1 weighted: 49.72%
F1 micro: 52.77%
Zero-one-loss: 61.42%
Hamming loss: 19.28%

Current date and time: 08-08-2023 20:30:08
Current date and time after fit: 08-08-2023 21:48:10
Current date and time after predict: 08-08-2023 21:48:13

  
---allemotions
Current date and time: 12-08-2023 00:19:11
Current date and time after fit: 12-08-2023 06:39:59
Current date and time after predict: 12-08-2023 06:40:04

                precision    recall  f1-score   support

    admiration       0.76      0.45      0.56      1038
     amusement       0.76      0.74      0.75       585
         anger       0.50      0.12      0.20       384
     annoyance       0.58      0.04      0.07       634
      approval       0.67      0.08      0.14       738
        caring       0.47      0.07      0.12       272
     confusion       0.75      0.07      0.13       327
     curiosity       0.86      0.06      0.12       570
        desire       0.55      0.36      0.44       147
disappointment       0.71      0.08      0.14       312
   disapproval       0.33      0.01      0.03       493
       disgust       0.79      0.22      0.34       202
 embarrassment       0.69      0.31      0.43        64
    excitement       0.80      0.17      0.28       204
          fear       0.70      0.41      0.52       145
     gratitude       0.94      0.87      0.90       696
         grief       0.25      0.05      0.08        22
           joy       0.58      0.35      0.43       372
          love       0.73      0.76      0.74       469
   nervousness       0.53      0.21      0.30        39
      optimism       0.70      0.44      0.54       373
         pride       0.69      0.33      0.45        33
   realization       0.60      0.10      0.17       248
        relief       0.67      0.04      0.08        45
       remorse       0.65      0.60      0.62       135
       sadness       0.70      0.24      0.36       340
      surprise       0.57      0.22      0.32       252
       neutral       0.61      0.20      0.31      3569

     micro avg       0.71      0.28      0.40     12708
     macro avg       0.65      0.27      0.34     12708
  weighted avg       0.66      0.28      0.36     12708
   samples avg       0.31      0.29      0.29     12708

F1 macro: 34.23%
F1 weighted: 35.72%
F1 micro: 40.44%
Zero-one-loss: 75.41%
Hamming loss: 3.49%
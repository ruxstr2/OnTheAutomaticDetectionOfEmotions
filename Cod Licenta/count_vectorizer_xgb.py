# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:58:22 2023

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
import numpy as np
import math
import datetime
from nltk.stem import PorterStemmer
#dataset = pd.read_csv("all_sentiments_unambiguous.csv")
dataset = pd.read_csv("all_emotions_unambiguous.csv")
X = dataset['text']
ylabels = dataset['classes']

stopwords_set = stopwords.words('english')
corpus = []
nr_texte = 45446
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


#print(classification_report(y_test, y_pred, target_names=['anger','disgust','fear','joy','sadness','suprise','neutral']))
#print(classification_report(y_test, y_pred, target_names=['neutral','positive','negative','ambiguous']))
print(classification_report(y_test, y_pred, target_names=['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral']))

f1 = f1_score(y_test, y_pred,average = 'macro')
print("F1 macro: %.2f%%" % (f1 * 100.0))

f1 = f1_score(y_test, y_pred,average = 'weighted')
print("F1 weighted: %.2f%%" % (f1 * 100.0))

f1 = f1_score(y_test, y_pred,average = 'micro')
print("F1 micro: %.2f%%" % (f1 * 100.0))


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


#Results
              precision    recall  f1-score   support

       anger       0.61      0.26      0.36      1097
     disgust       0.65      0.43      0.52       124
        fear       0.60      0.47      0.53       161
         joy       0.82      0.70      0.76      3796
     sadness       0.75      0.43      0.54       622
     suprise       0.66      0.23      0.34       952
     neutral       0.50      0.85      0.63      3142

    accuracy                           0.63      9894
   macro avg       0.66      0.48      0.53      9894
weighted avg       0.67      0.63      0.61      9894

Accuracy: 63.00%
F1: 61.41%

Current date and time: 05-08-2023 11:38:30
Current date and time after fit: 05-08-2023 16:14:40
Current date and time after predict: 05-08-2023 16:14:42

vs stemming:
    27224 vs 18764
    
              precision    recall  f1-score   support

       anger       0.60      0.29      0.39      1097
     disgust       0.65      0.41      0.50       124
        fear       0.67      0.50      0.57       161
         joy       0.81      0.71      0.76      3796
     sadness       0.71      0.44      0.55       622
     suprise       0.63      0.24      0.35       952
     neutral       0.52      0.83      0.64      3142

    accuracy                           0.63      9894
   macro avg       0.66      0.49      0.54      9894
weighted avg       0.67      0.63      0.62      9894

F1 macro: 53.70%
F1 weighted: 62.06%
F1 micro: 63.46%

Current date and time: 05-08-2023 22:38:16
Current date and time after fit: 06-08-2023 00:32:52
Current date and time after predict: 06-08-2023 00:32:54
--Current date and time: 07-08-2023 08:39:07
Current date and time after fit: 07-08-2023 10:27:02
Current date and time after predict: 07-08-2023 10:27:04

----sentiment
              precision    recall  f1-score   support

     neutral       0.53      0.81      0.64      3210
    positive       0.82      0.72      0.77      3817
    negative       0.67      0.47      0.55      2023
   ambiguous       0.66      0.24      0.35       963

    accuracy                           0.65     10013
   macro avg       0.67      0.56      0.58     10013
weighted avg       0.68      0.65      0.64     10013

F1 macro: 57.65%
F1 weighted: 64.25%
F1 micro: 65.10%

Current date and time: 07-08-2023 10:35:28
Current date and time after fit: 07-08-2023 11:39:25
Current date and time after predict: 07-08-2023 11:39:26

--- allemotions
Current date and time: 12-08-2023 14:40:46
Current date and time after fit: 12-08-2023 20:35:25
Current date and time after predict: 12-08-2023 20:35:27

                precision    recall  f1-score   support

    admiration       0.62      0.58      0.60       671
     amusement       0.68      0.82      0.75       415
         anger       0.45      0.36      0.40       271
     annoyance       0.45      0.10      0.16       355
      approval       0.52      0.12      0.19       467
        caring       0.41      0.09      0.15       160
     confusion       0.62      0.11      0.18       216
     curiosity       0.93      0.08      0.14       343
        desire       0.54      0.42      0.48        73
disappointment       0.46      0.11      0.18       175
   disapproval       0.29      0.03      0.05       355
       disgust       0.61      0.41      0.49       124
 embarrassment       0.45      0.37      0.41        41
    excitement       0.50      0.16      0.25       129
          fear       0.63      0.59      0.61       113
     gratitude       0.93      0.89      0.91       501
         grief       0.00      0.00      0.00         8
           joy       0.51      0.44      0.47       206
          love       0.71      0.78      0.75       348
   nervousness       0.11      0.09      0.10        11
      optimism       0.67      0.59      0.63       209
         pride       0.60      0.43      0.50        14
   realization       0.66      0.17      0.27       134
        relief       0.00      0.00      0.00        17
       remorse       0.52      0.71      0.60        87
       sadness       0.62      0.36      0.45       199
      surprise       0.53      0.47      0.49       193
       neutral       0.53      0.88      0.66      3255

      accuracy                           0.58      9090
     macro avg       0.52      0.36      0.39      9090
  weighted avg       0.58      0.58      0.52      9090

F1 macro: 38.77%
F1 weighted: 51.89%
F1 micro: 57.61%
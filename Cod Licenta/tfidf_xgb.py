# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:45:21 2023

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
import xgboost as xgb
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
    review = [ps.stem(word) for word in review if not word in stopwords_set]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating the model
vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(corpus).toarray()
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

f1 = f1_score(y_test, y_pred,average = 'weighted')
print("F1 micro: %.2f%%" % (f1 * 100.0))


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral'])
cm_display.plot()
plt.show()


              precision    recall  f1-score   support

       anger       0.61      0.25      0.36      1097
     disgust       0.63      0.43      0.51       124
        fear       0.62      0.44      0.52       161
         joy       0.82      0.71      0.76      3796
     sadness       0.75      0.43      0.54       622
     suprise       0.64      0.23      0.34       952
     neutral       0.51      0.85      0.64      3142

    accuracy                           0.63      9894
   macro avg       0.65      0.48      0.52      9894
weighted avg       0.67      0.63      0.62      9894

Accuracy: 63.26%
F1: 61.55%

vs stemmed
              precision    recall  f1-score   support

       anger       0.58      0.28      0.37      1097
     disgust       0.58      0.36      0.45       124
        fear       0.68      0.50      0.58       161
         joy       0.81      0.72      0.77      3796
     sadness       0.72      0.45      0.56       622
     suprise       0.65      0.23      0.34       952
     neutral       0.52      0.83      0.64      3142

    accuracy                           0.64      9894
   macro avg       0.65      0.48      0.53      9894
weighted avg       0.67      0.64      0.62      9894

F1 macro: 52.90%
F1 weighted: 62.12%
F1 micro: 62.12%

Current date and time: 06-08-2023 23:12:41
Current date and time after fit: 07-08-2023 01:02:16
Current date and time after predict: 07-08-2023 01:02:18

---sentiment
              precision    recall  f1-score   support

     neutral       0.53      0.81      0.64      3210
    positive       0.82      0.72      0.77      3817
    negative       0.68      0.46      0.55      2023
   ambiguous       0.65      0.23      0.34       963

    accuracy                           0.65     10013
   macro avg       0.67      0.56      0.57     10013
weighted avg       0.68      0.65      0.64     10013

F1 macro: 57.34%
F1 weighted: 64.12%
F1 micro: 64.12%

Current date and time: 07-08-2023 07:25:25
Current date and time after fit: 07-08-2023 08:20:50
Current date and time after predict: 07-08-2023 08:20:52

----allemotions
                precision    recall  f1-score   support

    admiration       0.64      0.56      0.60       671
     amusement       0.69      0.79      0.74       415
         anger       0.46      0.35      0.40       271
     annoyance       0.45      0.12      0.19       355
      approval       0.52      0.12      0.19       467
        caring       0.49      0.14      0.22       160
     confusion       0.43      0.06      0.11       216
     curiosity       0.69      0.08      0.15       343
        desire       0.52      0.38      0.44        73
disappointment       0.44      0.11      0.18       175
   disapproval       0.31      0.03      0.05       355
       disgust       0.63      0.40      0.49       124
 embarrassment       0.48      0.32      0.38        41
    excitement       0.44      0.16      0.23       129
          fear       0.67      0.55      0.60       113
     gratitude       0.94      0.88      0.91       501
         grief       0.00      0.00      0.00         8
           joy       0.53      0.44      0.48       206
          love       0.72      0.82      0.76       348
   nervousness       0.20      0.09      0.13        11
      optimism       0.65      0.58      0.61       209
         pride       0.57      0.29      0.38        14
   realization       0.63      0.18      0.28       134
        relief       0.00      0.00      0.00        17
       remorse       0.50      0.66      0.57        87
       sadness       0.63      0.35      0.45       199
      surprise       0.53      0.41      0.47       193
       neutral       0.53      0.88      0.66      3255

      accuracy                           0.57      9090
     macro avg       0.51      0.35      0.38      9090
  weighted avg       0.57      0.57      0.52      9090

F1 macro: 38.14%
F1 weighted: 51.81%
F1 micro: 51.81%
Current date and time: 13-08-2023 00:40:36
Current date and time after fit: 13-08-2023 05:44:34
Current date and time after predict: 13-08-2023 05:44:36
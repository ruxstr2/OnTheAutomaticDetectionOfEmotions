# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 19:37:17 2023

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
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import PorterStemmer
import datetime

dataset = pd.read_csv("all_emotions_ekman_unambiguous.csv")
#dataset = pd.read_csv("all_sentiments_unambiguous.csv")
X = dataset['text']
ylabels = dataset['classes']

stopwords_set = stopwords.words('english')
corpus = []
nr_texte = 49469
#nr_texte = 50063
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
classifier=RandomForestClassifier()
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time:", datetime_string)

classifier.fit(X_train,y_train)
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time after fit:", datetime_string)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time after predict:", datetime_string)


print(classification_report(y_test, y_pred, target_names=['anger','disgust','fear','joy','sadness','suprise','neutral']))
#print(classification_report(y_test, y_pred, target_names=['neutral','positive','negative','ambiguous']))

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

              precision    recall  f1-score   support

       anger       0.49      0.32      0.39      1097
     disgust       0.58      0.31      0.41       124
        fear       0.59      0.48      0.53       161
         joy       0.73      0.78      0.75      3796
     sadness       0.66      0.39      0.49       622
     suprise       0.45      0.24      0.31       952
     neutral       0.53      0.68      0.60      3142

    accuracy                           0.61      9894
   macro avg       0.57      0.46      0.50      9894
weighted avg       0.60      0.61      0.60      9894

Accuracy: 61.05%
F1: 59.60%

vs stemming

              precision    recall  f1-score   support

       anger       0.51      0.33      0.40      1097
     disgust       0.59      0.33      0.42       124
        fear       0.60      0.51      0.55       161
         joy       0.73      0.78      0.75      3796
     sadness       0.64      0.40      0.49       622
     suprise       0.45      0.23      0.31       952
     neutral       0.54      0.69      0.61      3142

    accuracy                           0.62      9894
   macro avg       0.58      0.47      0.50      9894
weighted avg       0.61      0.62      0.60      9894

Accuracy: 61.62%
F1: 60.10%

Current date and time: 06-08-2023 17:27:07
Current date and time after fit: 06-08-2023 18:14:35
Current date and time after predict: 06-08-2023 18:14:40

--stemmed again
              precision    recall  f1-score   support

       anger       0.51      0.34      0.41      1097
     disgust       0.56      0.32      0.41       124
        fear       0.58      0.51      0.54       161
         joy       0.72      0.78      0.75      3796
     sadness       0.67      0.40      0.50       622
     suprise       0.45      0.24      0.31       952
     neutral       0.54      0.69      0.61      3142

    accuracy                           0.62      9894
   macro avg       0.58      0.47      0.50      9894
weighted avg       0.61      0.62      0.60      9894

F1 macro: 50.41%
F1 weighted: 60.11%
F1 micro: 61.57%

Current date and time: 07-08-2023 12:54:09
Current date and time after fit: 07-08-2023 13:43:10
Current date and time after predict: 07-08-2023 13:43:16

--- sentiment
              precision    recall  f1-score   support

     neutral       0.54      0.65      0.59      3210
    positive       0.74      0.78      0.76      3817
    negative       0.62      0.50      0.55      2023
   ambiguous       0.47      0.24      0.32       963

    accuracy                           0.63     10013
   macro avg       0.59      0.54      0.55     10013
weighted avg       0.63      0.63      0.62     10013

Accuracy: 63.01%
F1: 62.10%

Current date and time: 06-08-2023 19:08:39
Current date and time after fit: 06-08-2023 20:04:08
Current date and time after predict: 06-08-2023 20:04:15

better sentiment
              precision    recall  f1-score   support

     neutral       0.55      0.65      0.59      3210
    positive       0.74      0.78      0.76      3817
    negative       0.62      0.51      0.56      2023
   ambiguous       0.48      0.25      0.33       963

    accuracy                           0.63     10013
   macro avg       0.60      0.55      0.56     10013
weighted avg       0.63      0.63      0.63     10013

F1 macro: 56.12%
F1 weighted: 62.52%
F1 micro: 63.36%

Current date and time: 07-08-2023 11:52:44
Current date and time after fit: 07-08-2023 12:49:37
Current date and time after predict: 07-08-2023 12:49:44
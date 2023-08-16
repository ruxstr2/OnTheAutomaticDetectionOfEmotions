# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:04:31 2023

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

#dataset = pd.read_csv("all_emotions_ekman_unambiguous.csv")
dataset = pd.read_csv("all_sentiments_unambiguous.csv")
X = dataset['text']
ylabels = dataset['classes']

stopwords_set = stopwords.words('english')
corpus = []
#nr_texte = 49469
nr_texte = 50063
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
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time:", datetime_string)

classifier = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time after fit:", datetime_string)

y_pred = classifier.predict(X_test)
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S") 
print("Current date and time after predict:", datetime_string)


#print(classification_report(y_test, y_pred, target_names=['anger','disgust','fear','joy','sadness','suprise','neutral']))
print(classification_report(y_test, y_pred, target_names=['neutral','positive','neutral','ambiguous']))
 
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

       anger       0.56      0.31      0.40      1097
     disgust       0.63      0.36      0.46       124
        fear       0.67      0.49      0.57       161
         joy       0.79      0.76      0.77      3796
     sadness       0.72      0.43      0.54       622
     suprise       0.68      0.20      0.31       952
     neutral       0.53      0.80      0.64      3142

    accuracy                           0.64      9894
   macro avg       0.65      0.48      0.53      9894
weighted avg       0.66      0.64      0.62      9894

Accuracy: 63.98%
F1: 62.26%

vs stemming
              precision    recall  f1-score   support

       anger       0.55      0.32      0.41      1097
     disgust       0.68      0.40      0.51       124
        fear       0.69      0.53      0.60       161
         joy       0.79      0.75      0.77      3796
     sadness       0.71      0.47      0.56       622
     suprise       0.67      0.20      0.31       952
     neutral       0.53      0.81      0.64      3142

    accuracy                           0.64      9894
   macro avg       0.66      0.50      0.54      9894
weighted avg       0.66      0.64      0.63      9894

F1 macro: 54.23%
F1 weighted: 62.66%
F1 micro: 64.31%

Current date and time: 06-08-2023 07:21:50
Current date and time after fit: 06-08-2023 13:50:26
Current date and time after predict: 06-08-2023 14:38:48

--sentiment
              precision    recall  f1-score   support

     neutral       0.54      0.77      0.63      3210
    positive       0.81      0.75      0.77      3817
     neutral       0.65      0.51      0.57      2023
   ambiguous       0.68      0.21      0.32       963

    accuracy                           0.65     10013
   macro avg       0.67      0.56      0.57     10013
weighted avg       0.67      0.65      0.64     10013

F1 macro: 57.29%
F1 weighted: 64.33%
F1 micro: 65.31%
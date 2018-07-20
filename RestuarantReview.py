# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:09:59 2018

@author: ABHIJIT
"""
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# importing dataset
dataset = pd.read_csv(r'''D:\MS\UDEMY COURSES\Machine Learning\Machine_Learning_AZ_Template_Folder\Machine Learning A-Z Template Folder\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing\Restaurant_Reviews.tsv''',delimiter='\t',quoting=3)

#c cleaning texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# creating bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values

#splitting the data set into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# fitting classifier to training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

# predicting the test set results
yPred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, yPred)

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
review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0])
review = review.lower()
review = review.split()
review = [word for word in review if not word in set(stopwords.words('english'))]
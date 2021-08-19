#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pa
import numpy as np
import nltk

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import string
import enchant
import warnings
warnings.filterwarnings('ignore')


# # Step-1 Load the data

# In[2]:


df = pa.read_csv('data.csv')
df.head()


# In[3]:


LABELS = list(df.columns)  # getting all columns
CLASS_LABELS = LABELS[2:]  # selecting only classes
CLASS_LABELS


# In[4]:


#the data is multilabeled i.e each comment contains more than one class.
#Below is the count of each class for the number of comments.
count = df[CLASS_LABELS].sum()
print(count)
count.plot.bar(figsize=(8, 6))


# In[5]:


new_df = df.drop('id',axis=1)
new_df.head(11)


# In[6]:


new_df.tail()


# In[7]:


label_list = []
for i in range(0,1384):
    toxic = new_df['toxic'][i]
    severe_toxic = new_df['severe_toxic'][i]
    obscene = new_df['obscene'][i]
    threat = new_df['threat'][i]
    insult = new_df['insult'][i]
    identity_hate = new_df['identity_hate'][i]
    ans = 'none'
    if severe_toxic == 1:
        ans = 'severe_toxic'
    elif obscene == 1:
        ans = 'obscene'
    elif threat == 1:
        ans = 'threat'
    elif insult == 1:
        ans = 'insult'
    elif identity_hate == 1:
        ans = 'identity_hate'
    elif toxic == 1:
        ans = 'toxic'
    else:
        ans = 'none'
    label_list.append(ans)     
    
new_df['final_label'] = label_list


# In[8]:


new_df['final_label'].unique()


# In[9]:


# getting all comment_text in single list to perform preprocessing easily
comment_list = []
new_list = []
for w in new_df['comment_text']:
    comment_list.append(w)

import re
for w in comment_list:
    n = re.sub("[^A-Za-z]+"," ",w)
    new_list.append(n)


# # Step-2 Pre-Processing

# In[10]:


# PreProcessing steps
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
filtered_list=[]
for w in new_list:
    word=word_tokenize(w)
    for a in word:
        if a not in stop_words:
            if a not in string.punctuation:
                if a.startswith("n't"):
                    a=a.replace("n't",'not')
                if a not in filtered_list:
                    filtered_list.append(a) 


# # Step-3 Feature Extraction

# In[11]:


from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
count = count_vec.fit(filtered_list)


# # Step-4 Build and Train the Model

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# models
knn = KNeighborsClassifier(n_neighbors=1)
log_reg = LogisticRegression()
rfc = RandomForestClassifier(n_estimators=100)
svc = SVC()
naive = MultinomialNB()

X_train, X_test, y_train, y_test = train_test_split(new_df['comment_text'],new_df['final_label'], test_size=0.20, random_state = 0)


# In[13]:


q = count.transform(X_train)


# In[14]:


X_test = count.transform(X_test)


# # K-nearest Neighbour

# In[15]:


knn.fit(q,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# # Logistic Regression

# In[16]:


log_reg.fit(q,y_train)
pred = log_reg.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# # Random Forest Classifier

# In[17]:


rfc.fit(q,y_train)
pred = rfc.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# # Support Vector Machine

# In[18]:


svc.fit(q,y_train)
pred = svc.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# # Multinomial Naive Bayes 

# In[19]:


naive.fit(q,y_train)
pred = naive.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


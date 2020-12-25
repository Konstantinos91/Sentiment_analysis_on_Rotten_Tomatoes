
# coding: utf-8

# In[1]:


# import all of the needed libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


#read the given train and the test datasets (.tsv files). 
train_dataset = pd.read_csv('/home/konstantinos/Desktop/train.tsv', sep = '\t') 
test_dataset = pd.read_csv('/home/konstantinos/Desktop/test.tsv', sep = '\t')


# In[3]:


#use read() to view train data(top five instances)
train_dataset.head()


# In[4]:


#use read() to view test data(top five instances)
test_dataset.head()


# In[5]:


#print the unique sentiment labels 
train_dataset['Sentiment'].unique()


# In[6]:


#print the type of the train dataset 
type(train_dataset)


# In[7]:


#use describe() to take statistical informations 
train_dataset.describe()


# In[9]:


#tokenize train dataset, by using the scikit learn CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[10]:


c_vector = CountVectorizer()


# In[11]:


count_train_x = c_vector.fit_transform(train_dataset['Phrase'])


# In[12]:


#print the dimensions of the indext of the training data count vector
count_train_x.shape


# In[13]:


#retrieve the index of common words/continuous characters/ngrams as for example the word ‘movie
c_vector.vocabulary_.get(u'movie')


# In[14]:


#print the name of the features
c_vector.get_feature_names()


# In[15]:


#convert the event(occurences) to frequencies
from sklearn.feature_extraction.text import TfidfTransformer


# In[16]:


#use fit()method to fit estimator to the data
tfreq_transformer = TfidfTransformer(use_idf = False).fit(count_train_x)


# In[17]:


#to transform the count-matrix to tf representation, we use the transform() method. 
tfreq_train_x = tfreq_transformer.transform(count_train_x)


# In[18]:


#transform the count matrix to Tfidf representation
tfidfreq_transformer = TfidfTransformer()


# In[19]:


x_train_tfidf = tfidfreq_transformer.fit_transform(count_train_x)


# In[20]:


#doing the classifier’s training, in order to forecast a phrase’s sentiment label
from sklearn.naive_bayes import MultinomialNB


# In[21]:


classifier = MultinomialNB().fit(x_train_tfidf, train_dataset['Sentiment'])


# In[22]:


count_test_x = c_vector.transform(test_dataset['Phrase'])


# In[23]:


# use transform() method for the transformation of the test count-matrix to Tfidf representation
tfidfreq_test_x = tfidfreq_transformer.transform(count_test_x)      


# In[24]:


#calculate and print of predictions 
predicted = classifier.predict(tfidfreq_test_x)


# In[25]:


for i, j in zip(test_dataset['PhraseId'], predicted):
    print(i, predicted[j])


# In[26]:


import csv
#create a csv file with our output 


# In[27]:


with open('Rotten_Sentiment.csv', 'w') as csvfile:
    csvfile.write('PhraseId,Sentiment\n')
    for i, j in zip(test_dataset['PhraseId'], predicted):
         csvfile.write('{}, {}\n'.format(i, j))


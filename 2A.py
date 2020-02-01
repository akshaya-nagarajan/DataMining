
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import string
import math

training_set = pd.read_csv('~/Downloads/titanic/train.csv')
test_set = pd.read_csv('~/Downloads/titanic/test.csv')
#training_set.head()

titanic_dataset = pd.concat([training_set,test_set])
#print(titanic_dataset)

# Getting all column infos
#titanic_dataset.info()

# dropping the unwanted col
#cols = ['PassengerId']
#titanic_dataset = titanic_dataset.drop(cols, axis=1)

# Optional for small dataset. Can be done for large dataset
#titanic_dataset = titanic_dataset.dropna()

# Using forward or backward interpolation to fill the age column
#training_set['Age'] = training_set['Age'].interpolate(limit_direction ='forward')
#training_set['Age'] = training_set['Age'].interpolate(limit_direction ='backward')
print(training_set['Age'])
#training_set.info()

# Or finding the median for the age column - not a good option
#training_set['Age'] = training_set.loc[:,"Age"].median()
#print(training_set['Age'])

# Group by sex  to get the age using the grouped median
groupedDataset = training_set.groupby(['Sex'])
groupedDataset.Age.median()
training_set['Age'] = groupedDataset.Age.apply(lambda x: x.fillna(x.median()))
print(training_set['Age'])

for value in training_set['Cabin']:
    if value != value:
        str = ''.join(random.choice(string.digits) for _ in range(2))
        val = random.choice("ABCDE") + str
        training_set['Cabin'].fillna(val, inplace = True) 
print(training_set['Cabin'])        
        
    
    


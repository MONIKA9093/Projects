#!/usr/bin/env python
# coding: utf-8

# # Cancer Disease Prediction

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.datasets import load_breast_cancer


# In[2]:


dscancer=load_breast_cancer()


# In[3]:


dscancer


# In[4]:


dscancer.keys()


# In[5]:


dscancer.data


# In[6]:


dscancer.target


# In[7]:


dscancer.feature_names


# In[8]:


dscancer.DESCR


# In[9]:


dfcancer=pd.DataFrame(data=dscancer.data,columns=dscancer.feature_names)


# In[10]:


dfcancer


# In[11]:


dfcancer['target']=dscancer.target


# In[12]:


dfcancer


# In[13]:


dfcancer.head()


# In[14]:


dfcancer.tail()


# In[15]:


dfcancer.info()


# # Splitting the data

# In[16]:


x=dfcancer.iloc[:,0:-1]


# In[17]:


y=dfcancer.iloc[:,-1]


# # Breaking the data for the testing

# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)


# In[19]:


x_train.shape


# In[20]:


x_test.shape


# In[21]:


y_train.shape


# In[22]:


y_test.shape


# # Training

# In[23]:


lg=LogisticRegression()


# In[24]:


lg.fit(x_train,y_train)


# # Prediction

# In[25]:


predlg=lg.predict(x_test)


# In[26]:


predlg


# In[27]:


accuracy_score(y_test,predlg)


# In[28]:


confusion_matrix(y_test,predlg)


# In[29]:


print(classification_report(y_test,predlg));


# In[ ]:





# In[ ]:





# In[ ]:





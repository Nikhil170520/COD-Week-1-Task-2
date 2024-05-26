#!/usr/bin/env python
# coding: utf-8

# In[27]:


# Importing Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split as TTS
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE,r2_score as RS,accuracy_score as AS
import numpy as np


# In[2]:


# Load Dataset
df=pd.read_csv(r'D:\Student_Marks.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


# Split The Data To Train And Test
x=df.drop(['Marks'],axis=1)
y=df['Marks']


# In[12]:


x_train,x_test,y_train,y_test=TTS(x,y,test_size=0.2,random_state=10)


# In[13]:


# Fit The Model With LinearRegression
lm=LR()
lm.fit(x_train,y_train)


# In[14]:


y_pred=lm.predict(x_test)


# In[18]:


# evaluate its performance using metrics like mean squared And R2 Score
mse=MSE(y_test,y_pred)
print(f"Mean Squared error = {mse}")
rs=RS(y_test,y_pred)
print(f"R2 Score = {rs}")      


# In[22]:


# Visualize the regression line and actual vs. predicted values
plt.scatter(y_test,y_pred)
plt.title('Actual And Predict Values')
plt.show()


# In[21]:


plt.plot(y_test,y_pred)
plt.title('Actual And Predict Values')
plt.show()


# In[30]:


# Inserting New Data To Predict
new = np.array([[7,9]])
pred = lm.predict(new)
print(f"Predict New Values = {pred[0]}")


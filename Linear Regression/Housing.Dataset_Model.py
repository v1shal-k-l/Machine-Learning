#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[90]:


df=pd.read_csv("Housing.csv")


# In[91]:


dataset=pd.DataFrame(df)
dataset


# In[ ]:





# In[92]:


d5={"yes":0,"no":1}
df['mainroad']=df['mainroad'].map(d5)
d6={"yes":0,"no":1}
df['guestroom']=df['guestroom'].map(d6)
d7={"yes":0,"no":1}
df['basement']=df['basement'].map(d7)
d8={"yes":0,"no":1}
df['hotwaterheating']=df['hotwaterheating'].map(d8)
d9={"yes":0,"no":1}
df['airconditioning']=df['airconditioning'].map(d9)
d11={"yes":0,"no":1}
df["prefarea"]=df['prefarea'].map(d11)
d12={"semi-furnished":0,"furnished":1,"unfurnished":2}
df["furnishingstatus"]=df['furnishingstatus'].map(d12)


df.head()


# In[127]:


X=dataset.iloc[:,1:12] ##independent featres
y=dataset.iloc[:,:-12]    ## dependent features


# In[128]:


y


# In[129]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.90, random_state=42)


# In[130]:


y_test


# In[131]:


## Standardizing the Data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)


# In[132]:


X_test=scaler.transform(X_test)


# In[133]:


X_train


# In[134]:


from sklearn.linear_model  import LinearRegression


# In[135]:


from sklearn.model_selection import cross_val_score


# In[136]:


regression=LinearRegression()
regression.fit(X,y)


# In[137]:


mse=cross_val_score(regression,X_train,y_train,scoring="neg_mean_squared_error",cv=5)


# In[138]:


np.mean(mse)


# In[139]:


reg_predict=regression.predict(X_test)


# In[140]:


reg_predict


# In[141]:


import seaborn as sns
sns.displot(reg_predict-y_test,kind="kde")


# In[142]:


from sklearn.metrics import r2_score


# In[143]:


score=r2_score(reg_predict,y_test)


# In[144]:


score


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[54]:


data=pd.read_csv("dataset (1).csv")


# In[55]:


data


# In[56]:


data=data.drop(["filename"],axis=1)


# In[57]:


data.info()


# In[58]:


data["label"] = [int(x) if x.isnumeric() else x for x in data["label"] ]


# In[59]:


d1={"normal":0,"murmur":1,"extrastole":2,"extrahls":3,"artifact":4}
data['label']=data['label'].map(d1)


# In[60]:


data.dtypes


# In[61]:


data.isnull().sum()


# In[62]:


## diving the dataset into depending and independent features
X = data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[63]:


#apply SelectKBest class to extract top 10 best features
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers


# In[64]:


import matplotlib.pyplot as plt
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[65]:


##data=data.drop(["mfcc1"],axis=1)


# In[66]:


from sklearn.model_selection import train_test_split
#split the data qet into 75% training and 25% testing
X_train, X_test, y_train, y_test = train_test_split (X,y,test_size=0.2, random_state=0)


# In[67]:


## standardizing the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[68]:


from sklearn.svm import SVC


# In[69]:


from sklearn.svm import SVC


# In[70]:


classifier=SVC(kernel="linear")


# In[71]:


classifier.fit(X_train,y_train)


# In[72]:


from sklearn.metrics import accuracy_score
y_pred=classifier.predict(X_test)


# In[73]:


accuracy_score(y_test,y_pred)


# In[74]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





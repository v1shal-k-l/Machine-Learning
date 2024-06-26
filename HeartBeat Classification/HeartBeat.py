#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


RSH=pd.read_csv("RSH murmur normal category.csv")


# In[3]:


df=RSH.drop(["filename"],axis=1)


# In[4]:


df.head(100)


# In[5]:


X=df.loc[:, "chroma_stft":"mfcc13"]


# In[6]:


X


# In[99]:


y=df["label"]


# In[100]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[101]:


corr_features = correlation(X, 0.7)
len(set(corr_features))


# In[102]:


corr_features


# In[103]:


X=X.drop(corr_features,axis=1)


# In[104]:


# from autoviz.AutoViz_Class import AutoViz_Class

# AV = AutoViz_Class()


# In[105]:


# filename = "RSH.csv"
# sep = ","
# dfte = AV.AutoViz(
#     filename,
#     sep=",",
#     depVar="",
#     dfte=None,
#     header=0,
#     verbose=0,
#     lowess=False,
#     chart_format="svg",
#     max_rows_analyzed=150000,
#     max_cols_analyzed=30,
# )
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline


# In[106]:


y


# In[107]:


d1={"normal":0,"Murmurs":1,"Murmurs1":1}


# In[108]:


y=y.map(d1)


# In[109]:


y = y.fillna(0)


# In[110]:


y=y.astype(int)


# In[111]:


(len(y))


# In[112]:


X


# In[113]:


import scipy.stats as stats
def diagnostic_plots(df,variable):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.show()


# In[114]:


diagnostic_plots(X,"spectral_centroid")


# # Logarithmic Transformation

# In[115]:


X['Log_Spectral_centroid']=np.log(X['spectral_centroid']+1)
diagnostic_plots(X,'Log_Spectral_centroid')


# In[116]:


diagnostic_plots(X,"mfcc6")


# # Sqaure Root Tansformation

# In[117]:


X['sqr_spectral_centroid']=X['Log_Spectral_centroid']**(1/2)
diagnostic_plots(X,'sqr_spectral_centroid')


# In[118]:


diagnostic_plots(X,"mfcc2")


# In[119]:


X['sqr_mfcc6']=X['mfcc6']**(1/2)
diagnostic_plots(X,'sqr_mfcc6')


# In[120]:


diagnostic_plots(X,"mfcc10")


# In[121]:


diagnostic_plots(X,"rmse")


# In[122]:


X['sqr_rmse']=X['rmse']**(1/2)
diagnostic_plots(X,'sqr_rmse')


# In[123]:


diagnostic_plots(X,"chroma_stft")


# In[124]:


X['sqr_chroma_stft']=X['chroma_stft']**(2.5)
diagnostic_plots(X,'sqr_chroma_stft')


# In[125]:


X


# In[126]:


X=X.drop(["rmse","mfcc6","mfcc2","Log_Spectral_centroid","spectral_centroid","chroma_stft"],axis=1)


# In[127]:


X


# In[128]:


# X=X.drop(["mfcc2"],axis=1)


# In[129]:


from sklearn.model_selection import train_test_split
#split the data qet into 75% training and 25% testing
X_train, X_test, y_train, y_test = train_test_split (X,y,test_size=0.3, random_state=0)


# In[130]:


X_test


# In[ ]:





# In[131]:


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train=scaler.fit_transform(X_train)
# X_test=scaler.transform(X_test)


# In[ ]:





# In[132]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the actual values
X_train = scaler.fit_transform(X_train)

# Transform the predicted values
X_test = scaler.transform(X_test)


# In[133]:


y_train.isnull().sum()


# In[134]:


import numpy as np
# Assuming your X_train contains infinity values
X_train[np.isinf(X_train)] = np.finfo(np.float64).max


# In[135]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # Replace NaN with the mean of the column
X_train = imputer.fit_transform(X_train)


# In[136]:


from sklearn.svm import SVC


# In[137]:


classifier=SVC(kernel="linear")
classifier.fit(X_train,y_train)


# In[138]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # Replace NaN with the mean of the column
X_test= imputer.fit_transform(X_test)


# In[139]:


from sklearn.metrics import accuracy_score
y_pred=classifier.predict(X_test)


# In[140]:


accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
# Assuming you have ground truth labels (y_true) and predicted labels (y_pred)
accuracy 


# In[141]:


accuracy


# In[142]:


from sklearn.metrics import mean_squared_error
# Calculate the Mean Squared Error (MSE) between the scaled predicted and actual values
mse = mean_squared_error(y_pred,y_test)
# Print the MSE
print("Mean Squared Error (MSE):", mse)


# In[143]:


from math import sqrt
rmse = sqrt(mse)
rmse


# In[144]:


from sklearn.metrics import mean_absolute_percentage_error
mae=mean_absolute_percentage_error(y_test,y_pred)


# In[145]:


maep=mae*100


# In[146]:


maep


# In[147]:


from sklearn.tree import DecisionTreeClassifier
## Postpruning
treemodel=DecisionTreeClassifier(max_depth=4)
treemodel.fit(X_train,y_train)
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)


# In[148]:


y_pred=treemodel.predict(X_test)


# In[149]:


score=accuracy_score(y_pred,y_test)
print(score)


# In[150]:


from sklearn.model_selection import GridSearchCV
treemodel=DecisionTreeClassifier()
## Preprunning
parameter={
 'criterion':['entropy'],
  'splitter':['best'],
  'max_depth':[4],
  'max_features':['log2']
    
}
cv=GridSearchCV(treemodel,param_grid=parameter,cv=5,scoring='accuracy')
cv.fit(X_train,y_train)
cv.best_params_


# In[151]:


y_pred=cv.predict(X_test)
score=accuracy_score(y_pred,y_test)
score


# In[152]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')  # Replace NaN with the mean of the column
X= imputer.fit_transform(X)


# In[153]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(random_state=0)

# Define a parameter grid for n_estimators
param_grid = {'n_estimators': [10, 50, 100, 200, 500,600]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X, y)

# Get the best value for n_estimators
best_n_estimators = grid_search.best_params_['n_estimators']


# In[154]:


best_n_estimators


# In[155]:


randomforest_classifier= RandomForestClassifier(n_estimators=600)
from sklearn.model_selection import cross_val_score
score=cross_val_score(randomforest_classifier,X,y,cv=5)


# In[156]:


score.mean()


# In[157]:


# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators and other hyperparameters

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)


# In[158]:


# Calculate the Mean Squared Error (MSE) between the scaled predicted and actual values
mse = mean_squared_error(y_pred,y_test)
# Print the MSE
print("Mean Squared Error (MSE) in Random Forest Classifier:", mse)
rmse = sqrt(mse)
print("Root Mean Squared Error (RMSE) in Random Forest Classifier:", rmse)
from sklearn.metrics import mean_absolute_percentage_error
mae=mean_absolute_percentage_error(y_test,y_pred)
print("Mean absolute percentage Error (MAE) in Random Forest Classifier:", mae)


# In[159]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Generate sample data (you can replace this with your own data)
X, labels = make_blobs(n_samples=47, centers=2, random_state=0)

# Create a DBSCAN clustering model
dbscan = DBSCAN(eps=0.5, min_samples=4)

# Fit the model to the data
dbscan.fit(X)

# Retrieve the cluster labels
cluster_labels = dbscan.labels_

# Number of clusters, ignoring noise if present (-1 represents noise)
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

# Plot the data points with different colors for each cluster
unique_labels = set(cluster_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for label, color in zip(unique_labels, colors):
    if label == -1:
        # Noise points are black
        color = [0, 0, 0, 1]
    class_member_mask = (cluster_labels == label)
    xY = X[class_member_mask]
    plt.scatter(xY[:, 0], xY[:, 1], s=50, c=[color], label=f'Cluster {label}')

plt.title(f'DBSCAN Clustering - Estimated {num_clusters} Clusters')
plt.legend()
plt.show()


# In[160]:


X_train


# In[161]:


import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate a synthetic binary classification dataset
X, y = make_classification(
    n_samples=47,
    n_features=8,
    n_classes=2,  # Set to 2 for binary classification
    n_clusters_per_class=2,
    n_informative=4,
    random_state=0
)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost classifier
model = xgb.XGBClassifier(learning_rate=0.2, n_estimators=600, max_depth=4)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)


# In[162]:


import seaborn as sns
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Create a count plot to compare actual and predicted values
plt.figure(figsize=(15,5))
sns.countplot(x='Actual', hue='Predicted', data=df)
plt.title(f'Actual vs. Predicted (Accuracy: {accuracy:.2f})')
plt.show()


# In[163]:


from sklearn.metrics import confusion_matrix
# Create a confusion matrix
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap of the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[164]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


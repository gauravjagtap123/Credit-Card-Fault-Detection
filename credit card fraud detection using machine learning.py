#!/usr/bin/env python
# coding: utf-8

# # 1.Importing Library

# In[74]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.rc("font", size=14)
plt.rcParams['axes.grid'] = True
plt.figure(figsize=(6,3))
plt.gray()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import  PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, export
from sklearn.ensemble import BaggingClassifier, BaggingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor 
#from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# # 2.Loading the dataset

# In[4]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[5]:


df.tail()


# 1.Lets understand the data now so the data has features such as time (secs),amount (USD), class and V1-V28 are transactions.
# 
# 2.The time columns give us the elapsed time of every transaction from the first transaction.
# 
# 3.V1-V28 features of every transaction as the data is of credit card the feature has been converted to numerical values through principal component analysis.
# 
# 4.Class column gives us the details about the transaction is fraud or legit as '0' represent the transaction is legit and '1' represent the transaction is fraud

# # 3.Understand the Data

# In[7]:


df.info()


# In[6]:


df.isnull().sum()


# In[8]:


df.drop(['Time'],axis=1,inplace=True)


# In[9]:


df.head()


# In[11]:


# we will check for any duplicate data and drop it
df.shape


# In[13]:


df.drop_duplicates(inplace=True)


# In[14]:


df.shape   # So the duplicate value which was 9144 values was dropped. Now we have clean data


# In[15]:


df.describe()


# In[17]:


round(100*(df.isnull().sum()/len(df)),2).sort_values(ascending=False)


# In[18]:


# Lets check the class column

df['Class'].value_counts()


# 0 - Legit transaction 1 - Fraud transaction
# We can also see that the data is highly unbalanced. As most of the data is in legit transaction and if we feed this data to machine learning model it will always predict normal transaction only

# In[21]:


# separating the class data
legit=df[df.Class == 0]
fraud=df[df.Class == 1]


# In[22]:


print(legit.shape)
print(fraud.shape)


# In[23]:


legit.Amount.describe()


# In[24]:


fraud.Amount.describe()


# In[25]:


#Now lets compare the values of both the transactions
df.groupby('Class').mean()


# # 4. Handling Unbalanced Data

# In[26]:


# We use under-sampling technique to handle unbalanced data. This will give us even distribution of the data
legit_sample=legit.sample(n=473) # 473 because as the number of fraud transaction is 473


# In[27]:


# Concatenating two dataframes
df1 = pd.concat([legit_sample,fraud],axis=0)


# In[28]:


df1.head()


# In[30]:


df1['Class'].value_counts() # So here we have unifromly distributed the data


# In[31]:


df1.groupby('Class').mean()  # Here we get to know if we have got good or bad samples,incase of bas samples the mean value will be very different


# # 5.Splitting the data into features and target

# In[33]:


X = df1.drop(columns='Class',axis=1)
Y = df1['Class']


# In[34]:


print(X)


# In[35]:


print(Y)


# # 6.Split the data into Training data & Test data

# In[37]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=1)


# In[38]:


print(X.shape,X_train.shape,X_test.shape)


# # 7. Model Development

# # 1.Logistic Regression

# In[41]:


#training the model
model = LogisticRegression()
model.fit(X_train,Y_train)


# # 6. Model Evaluation

# In[44]:


X_train_prediction=model.predict(X_train)
train_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data:',train_accuracy)


# In[45]:


# 94.1 prediction score means our model can predict 94 correct prediction out of 100 sample


# In[46]:


X_test_prediction=model.predict(X_test)
test_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy on testing data:',test_accuracy)


# In[49]:


print('F1 score of the Decision Tree model is {}'.format(f1_score(Y_test,X_test_prediction)))


# In[51]:


confusion_matrix(Y_test,X_test_prediction,labels = [0,1])


# # Decision Tree

# In[57]:


dt_train_prediction=DecisionTreeClassifier(max_depth=4, criterion = 'entropy')
dt_train_prediction.fit(X_train, Y_train)
dt_yhat=dt_train_prediction.predict(X_test)


# In[58]:


print('Accuracy score of the Decision.Tree model is {}'.format(accuracy_score(Y_test,dt_yhat)))


# In[59]:


print('F1 score of the Decision Tree model is {}'.format(f1_score(Y_test, dt_yhat)))


# In[60]:


confusion_matrix(Y_test, dt_yhat, labels = [0, 1])


# In[61]:


# So the first row represent positive value where as the second row represent negative value.
# So in total we have 112 positive value and 6 negative value so only out 118 total transaction only 6 where classified as false prediction


# # K- nearest Neighbors

# In[63]:


n = 7
knn=KNeighborsClassifier(n_neighbors=n)
knn.fit(X_train,Y_train)
knn_yhat=knn.predict(X_test)


# In[64]:


print('Accuracy score of the K-Nearest model is{}'.format(accuracy_score(Y_test,knn_yhat)))


# In[65]:


print('F1 score of the K-Nearest Neighbors model is {}'.format(f1_score(Y_test, knn_yhat)))


# In[66]:


confusion_matrix(Y_test, knn_yhat, labels = [0, 1])


# # Random Forest

# In[70]:


rf=RandomForestClassifier(max_depth=4)
rf.fit(X_train,Y_train)
rf_yhat=rf.predict(X_test)


# In[71]:


print('Accuracy score of the Random Forest model is {}'.format(accuracy_score(Y_test, rf_yhat)))


# In[72]:


print('F1 score of the Random Forest model is {}'.format(f1_score(Y_test, rf_yhat)))


# In[73]:


confusion_matrix(Y_test, rf_yhat, labels = [0, 1])


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("winequality-red.csv")


# In[ ]:


data.head()


# In[ ]:


data.corr


# In[ ]:


data.columns


# In[ ]:


data.info() 


# In[ ]:


data['quality'].unique() 


# In[ ]:


sns.pairplot(data)


# In[ ]:


from collections import Counter
Counter(data['quality'])


# In[ ]:


sns.countplot(x='quality', data=data)


# In[ ]:


sns.boxplot(x='quality', y='fixed acidity', data = data)


# In[ ]:


sns.boxplot(x='quality', y='volatile acidity', data = data)


# In[ ]:


sns.boxplot(x='quality', y='citric acid', data = data)


# In[ ]:


sns.boxplot(x='quality', y='residual sugar', data = data)


# In[ ]:


sns.boxplot(x='quality', y='chlorides', data = data)


# In[ ]:


sns.boxplot(x='quality', y='free sulfur dioxide', data = data)


# In[ ]:


sns.boxplot(x='quality', y='density', data = data)


# In[ ]:


sns.boxplot(x='quality', y='pH', data = data)


# In[ ]:


sns.boxplot(x='quality', y='sulphates', data = data)


# In[ ]:


sns.boxplot(x='quality', y='alcohol', data = data)


# In[ ]:


data.describe()


# In[ ]:



reviews = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
data['Reviews'] = reviews


# In[ ]:


data.columns


# In[ ]:


data['Reviews'].unique()
Counter(data['Reviews'])


# In[ ]:


x = data.iloc[:,:11]
y = data['Reviews']
x.head(10)
y.head(10)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()


# In[ ]:


df=pd.read_csv("winequality-red.csv")
df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]

X = df.drop(['quality','goodquality'], axis = 1)
y = df['goodquality']
df['goodquality'].value_counts()


# In[ ]:


from sklearn.preprocessing import StandardScaler
X_features = X
X = StandardScaler().fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(random_state=1)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print(classification_report(y_test, y_pred1))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Social_Network_Ads.csv")


# In[3]:


#check the head
df.head()


# In[4]:


#Check tail
df.tail()


# In[5]:


df.dtypes


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


#checking the duplicacy
df.duplicated().sum()


# In[9]:


df.isnull().sum()


# In[10]:


df.corr()


# In[11]:


df["Gender"].value_counts(normalize = True)*100


# In[12]:


sns.countplot(x = "Gender",data = df)


# In[13]:


df["Gender"].value_counts().plot(kind ="pie",autopct = "%.2f")


# In[14]:


def age_group(pi):
    if pi > 20 and pi <= 30:
        Age_group = "21-30"
    elif pi > 30 and pi <= 40:
        Age_group = "31-40"
    elif pi > 40 and pi <= 50:
        Age_group = "41-50"
    else:
        Age_group = "Above 50"
    return (Age_group)


# In[15]:


df["Age_group"] = df["Age"].apply(age_group)


# In[16]:


df


# In[17]:


df.columns


# In[18]:


sns.countplot(data = df, x = "Age_group")


# In[19]:


pd.crosstab(df["Age_group"],df["Purchased"],margins=True)


# In[20]:


sns.distplot(df["Age"])


# In[21]:


sns.catplot(data=df, y="Purchased",x="Age_group", kind="bar")


# In[22]:


sns.set_theme(style="darkgrid")
sns.countplot(y="Purchased",data=df, palette = "flare")
plt.ylabel("Purchased")
plt.xlabel("Total no of customers")
plt.show()


# In[23]:


sns.set_theme(style = "darkgrid")
sns.countplot(data=df,x="Gender", palette = "rocket")
plt.xlabel("Gender")
plt.ylabel("Total customers")
plt.show()


# In[24]:


pd.crosstab(df["Age_group"],df["Purchased"]).plot(kind="bar",figsize=(12,5))
plt.title("Purchase by Age_group")
plt.xlabel("Age_group")
plt.ylabel("Total No of People Purchased")
plt.show()


# In[25]:


sns.jointplot(data = df,x="EstimatedSalary",y="Purchased",hue = "Gender",kind = "scatter")


# In[26]:


sns.pairplot(data = df,hue="Gender")


# In[27]:


df


# In[28]:


x = df.iloc[:,2:4].values


# In[29]:


y = df.iloc[:,-2].values


# In[30]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3 , random_state = (0))

from sklearn.neighbors import KNeighborsClassifier
KNClassifier = KNeighborsClassifier(n_neighbors = 17)
KNClassifier.fit(x_train,y_train)
from sklearn.metrics import classification_report

y_pred = KNClassifier.predict(x_test)

print(classification_report(y_test,y_pred))


# In[31]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
KNAcc = accuracy_score(y_pred,y_test)
print(confusion_matrix(y_test,y_pred))
print("K neighbors Accuracy score is: {:.2f}%".format(KNAcc*100))


# In[ ]:





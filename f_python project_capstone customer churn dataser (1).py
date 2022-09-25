#!/usr/bin/env python
# coding: utf-8

# ## Capstone  project : “Customer Churn Dataset”

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error,confusion_matrix


# ## A.Data manupulation

# In[9]:


df=pd.read_csv(r"/Users/rathoreaditya/Downloads/customer_churn (2).csv")


# In[10]:


df


# In[11]:


#a. Extract the 5th column & store it in ‘customer_5’
customer_5=df.iloc[:,4]
customer_5


# ### 

# In[12]:


### b. Extract the 15th column & store it in ‘customer_15’
customer_15=df.iloc[:,14]
customer_15

# c. Extract all the male senior citizens whose Payment Method is Electronic check &
store the result in ‘senior_male_electronic’
# In[13]:


senior_male_electronic=df[(df['gender']=='Male')&(df['SeniorCitizen']==1)&(df['PaymentMethod']=='Electronic check')]


# In[14]:


senior_male_electronic


# In[15]:


df.columns

## d. Extract all those customers whose tenure is greater than 70 months or their
Monthly charges is more than 100$ & store the result in ‘customer_total_tenure’
# In[16]:


customer_total_tenure=df[(df['tenure']>70) | (df['MonthlyCharges']>100)]


# In[17]:


customer_total_tenure

###  e. Extract all the customers whose Contract is of two years, payment method is Mailed
check & the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’
# In[18]:


two_mail_yes=df[(df['Contract']=='Two year')& (df['PaymentMethod']=='Mailed check')& (df['Churn']=='Yes')]


# In[19]:


two_mail_yes

### f. Extract 333 random records from the customer_churndataframe& store the result in
‘customer_333’
# In[20]:


customer_333= df.sample(n=333)
customer_333

## g. Get the count of different levels from the ‘Churn’ column
# In[21]:


df['Churn'].value_counts()


# ## B. data visualization
a. build a bar-plot for the ’InternetService’ column:
1. Set x-axis label to ‘Categories of Internet Service’
2. Set y-axis label to ‘Count of Categories’
3. Set the title of plot to be ‘Distribution of Internet Service’
4. Set the color of the bars to be ‘orange’
# In[22]:


x=df['InternetService'].value_counts().keys().tolist()
y=df['InternetService'].value_counts().tolist()


# In[23]:


plt.bar(x,y,color='orange')
plt.xlabel("Categories of Internet Service’")
plt.ylabel("‘Count of Categories’")
plt.title("Distribution of Internet Service")

## b. build histogram for the ‘tenure’ column:
1. Set the number of bins to be 30

2. Set the color of the bins to be ‘green’

3. Assign the title ‘Distribution of tenure’
# In[24]:


plt.hist(df['tenure'],bins=30,color='g')
plt.title("Distribution of tenure")

c. Build a scatter-plot between ‘MonthlyCharges’ & ‘tenure’. Map ‘MonthlyCharges’ to the y-axis & ‘tenure’ to the ‘x-axis’:

i. Assign the points a color of ‘brown’
ii. Set the x-axis label to ‘Tenure of customer’
iii. Set the y-axis label to ‘Monthly Charges of customer’
iv. Set the title to ‘Tenure vs Monthly Charges’
# In[25]:


x=df['tenure'].head(10)
y=df['MonthlyCharges'].head(10)
plt.scatter(x,y,color='brown')
plt.xlabel("Tenure of customer")
plt.ylabel("Monthly Charges of customer")
plt.title("Tenure vs Monthly Charges")
plt.grid(True)


# d. ### Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the y-axis & ‘Contract’ on the x-axis.

# In[26]:


x=df['Contract']
y=df['tenure']
sns.boxplot(x,y,data=df,width=0.3)


# ### C) Linear Regression:


a. Build a simple linear model where dependent variable is ‘MonthlyCharges’ and independent variable is ‘tenure’

i. Divide the dataset into train and test sets in 70:30 ratio.
ii. Build the model on train set and predict the values on test set
iii. After predicting the values, find the root mean square error
iv. Find out the error in prediction & store the result in ‘error’
v. Find the root mean square error
# In[27]:


y=pd.DataFrame(df['MonthlyCharges'])
x=pd.DataFrame(df['tenure'])


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)


# In[29]:


obj1=LinearRegression()
obj1


# In[30]:


obj1.fit(x_train,y_train)


# In[31]:


y_pred=obj1.predict(x_test)
y_pred


# In[32]:


error=mean_squared_error(y_test,y_pred)
error


# In[33]:


rmse=np.sqrt(error)
rmse


# ### D) Logistic Regression:

# 
# a. Build a simple logistic regression modelwhere dependent variable is ‘Churn’ & independent variable is ‘MonthlyCharges’
# 
# i. Divide the dataset in 65:35 ratio
# 
# ii. Build the model on train set and predict the values on test set
# 
# iii. Build the confusion matrix and get the accuracy score

# In[34]:


y=pd.DataFrame(df['Churn'])
x=pd.DataFrame(df['MonthlyCharges'])


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)


# In[36]:


obj2=LogisticRegression()


# In[37]:


obj2


# In[38]:


obj2.fit(x_train,y_train)


# In[39]:


y_pred=obj2.predict(x_test)


# In[40]:


confusion_matrix(y_test,y_pred)


# In[41]:


confusion_matrix(y_test,y_pred)


# In[42]:


score=accuracy_score(y_test,y_pred)
score

## b. Build a multiple logistic regression model where dependent variable is ‘Churn’ & independent variables are ‘tenure’ & ‘MonthlyCharges’

i. Divide the dataset in 80:20 ratio

ii. Build the model on train set and predict the values on test set

iii. Build the confusion matrix and get the accuracy score
# In[43]:


x=pd.DataFrame(df.loc[:,['MonthlyCharges','tenure']])
y=df['Churn']


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)


# In[45]:


obj3=LogisticRegression()
obj3.fit(x_train,y_train)


# In[46]:


y_pred=obj3.predict(x_test)
y_pred


# In[47]:


confusion_matrix(y_pred,y_test)


# In[48]:


score=accuracy_score(y_pred,y_test)
score


# ## E). Decision Tree
a. Build a decision tree model where dependent variable is ‘Churn’ & independent variable is ‘tenure’

i. Divide the dataset in 80:20 ratio
ii. Build the model on train set and predict the values on test set
iii. Build the confusion matrix and calculate the accuracy
# In[49]:


x=pd.DataFrame(df['tenure'])
y=df['Churn']


# In[50]:


obj5=DecisionTreeClassifier()
obj5


# In[51]:


obj5.fit(x_train,y_train)


# In[52]:


confusion_matrix(y_pred,y_test)


# In[53]:


score=accuracy_score(y_pred,y_test)
score


# ### F) Random Forest:
# a. Build a Random Forest model where dependent variable is ‘Churn’ & independent
# variables are ‘tenure’ and ‘MonthlyCharges’
# 
# i. Divide the dataset in 70:30 ratio
# 
# ii. Build the model on train set and predict the values on test set
# 
# iii. Build the confusion matrix and calculate the accuracy 

# In[54]:


y4=pd.DataFrame(df['Churn'])
x4=pd.DataFrame(df.loc[:,['MonthlyCharges','tenure']])


# In[55]:


x_train,x_test,y_train,y_test=train_test_split(x4,y4,test_size=0.30,random_state=0)


# In[56]:


obj7=RandomForestClassifier(n_estimators=100)
obj7.fit(x_train,y_train)


# In[57]:


y_pred = obj7.predict(x_test)
y_pred


# In[58]:


confusion_matrix(y_pred,y_test)


# In[59]:


score=accuracy_score(y_pred,y_test)
score


# In[ ]:





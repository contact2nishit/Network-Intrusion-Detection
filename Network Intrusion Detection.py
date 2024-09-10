#!/usr/bin/env python
# coding: utf-8

# # MACHINE LEARNING PROJECT FOR NETWORK INTRUSION DETECTION
# ### Dataset: Kaggle dataset
# ### Technique: Using multiple models with majority voting
# ### ML Problem type: Classification problem

# In[ ]:


''' This project analyses a dataset on network intrusion obtained by simulating intrusions on a military network environment
It was published '''


# In[56]:


#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import VotingClassifier


# ### Reading Source Data File

# In[2]:


train=pd.read_csv('C:\Project_NID\Train_data.csv')


# ### Exploratory Data Analysis on the Source data

# In[3]:


train.head(3)


# In[4]:


train.info()
#Noticed that there are total 25192 records with 42 columns. 
#out of 42 4 columns are non-number (or Object type)
#All coulmns have non-null entries count same as total row counts, hence no null values present in the data set


# In[5]:


#Checking meausures of central tendency
train.describe()


# In[6]:


train.describe(include='object')


# In[7]:


col_list = []
for col in train.columns:
    if train[col].dtype != 'object':
        col_list.append(col)

fig = plt.figure(figsize =(10, 2))

# show plot
plt.boxplot(train[col_list])
plt.show()


# In[8]:


# Noticed outlier in the src_bytes column
train = train[train.src_bytes<3790900]
plt.boxplot(train[col_list])
plt.show()


# In[9]:


#Noticed more outliers in the dst_bytes column
train = train[train.dst_bytes<579090]
plt.boxplot(train[col_list])
plt.show()


# In[10]:


# Successfully completed outliers treatment


# In[11]:


#Checking heatmap for correlation in x-variables
sns.heatmap(train[col_list].corr())


# In[12]:


#dropping correlated variables and checking heatmap again
train1 = train.drop(columns = ["num_outbound_cmds","srv_serror_rate","srv_rerror_rate","protocol_type","service","flag","class","is_host_login","rerror_rate","num_root","serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate"])
sns.heatmap(train1.corr())


# In[13]:


# Creating a new dataframe to exclude correlated variables but include categorical variables
train2 = train.drop(columns = ["num_outbound_cmds","srv_serror_rate","srv_rerror_rate","is_host_login","rerror_rate","num_root","serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate"])


# In[14]:


# Checking class column's distribution
train2["class"].value_counts()


# In[15]:


#Encoding all object type variables
def fn_OneHotEncoder(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            dummy = pd.get_dummies(df[col], prefix = col, drop_first = True)
            df = pd.concat([df, dummy], axis =1)
            df = df.drop(columns = [col], axis =1)
    return df
train2 = fn_OneHotEncoder(train2)
train2


# In[16]:


#Making sure that dummy encoder has not changed the y-value distribution
train2["class_normal"].value_counts()


# In[17]:


#Splitting x and y variables
Y_train = train2["class_normal"]
X_train = train2.drop(["class_normal"], axis=1)


# In[18]:


#Selecting top 10 features that relate to the y-variable
clf = RandomForestClassifier(n_estimators = 100, random_state=42)
clf.fit(X_train, Y_train)
importances = clf.feature_importances_
feature_imp_df = pd.DataFrame({"Feature":X_train.columns, "feature_importance": importances}).sort_values("feature_importance",ascending = False)
feature_imp_df.reset_index(inplace=True, drop =True)
print(feature_imp_df.head(10))


# In[19]:


#Appending features to a list
feature_top10 = [i for i in (feature_imp_df.loc[0:9,"Feature"])]
print(feature_top10)


# In[20]:


#Selecting the required columns
X_train = X_train[feature_top10]
X_train


# In[21]:


#Scaling the data
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns = [feature_top10])
X_train.head(2)


# In[22]:


#Dataset splitting into train and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)
X_train


# ### Random Forest Classifier Model

# In[49]:


clf = RandomForestClassifier(n_estimators =100)


# ### SVM Model

# In[48]:


#Build the model
svm = SVC(kernel="rbf", gamma=0.5, C=1.0)


# ### K-Nearest Neighbors (KNN)

# In[47]:


knn = KNeighborsClassifier(n_neighbors =1)


# ### Naive Bayes Model

# In[46]:


gnb = GaussianNB()


# ### Decision Tree Model

# In[45]:


DTC = tree.DecisionTreeClassifier(criterion='entropy', random_state = 0)


# In[66]:


models = []
models.append(("Random Forest",clf))
models.append(("Support Vector Machine",svm))
models.append(("K-Nearest Neighbors",knn))
models.append(("Naive Bayes",gnb))
models.append(("Decision Tree",DTC))


# In[61]:


for i,v in models:
    v.fit(x_train,y_train)
    accuracy = metrics.accuracy_score(y_test,v.predict(x_test))
    confusion_matrix = metrics.confusion_matrix(y_test,v.predict(x_test))
    classification= metrics.classification_report(y_test,v.predict(x_test))
    print()
    print(f"Evaluating {i} model:")
    print(f"The model is {accuracy*100}% accurate")
    print("Confusion matrix")
    print(confusion_matrix)
    print("Classification report:")
    print(classification)
    print()


# ### Voting Model

# In[67]:


voting = VotingClassifier(estimators = models)
voting.fit(x_train,y_train)
accuracy = metrics.accuracy_score(y_test,voting.predict(x_test))
confusion_matrix = metrics.confusion_matrix(y_test,voting.predict(x_test))
classification= metrics.classification_report(y_test,voting.predict(x_test))
print("Evaluating Voting Classifier model:")
print(f"The model is {accuracy*100}% accurate")
print("Confusion matrix")
print(confusion_matrix)
print("Classification report:")
print(classification)


# In[ ]:





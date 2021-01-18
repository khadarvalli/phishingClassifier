
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("phising.csv")
data


# Data Description:
# 
# The dataset consists of different columns with information regarding whether a website is a phising website or not.
# 
# The columns are :
# 
# having_IP_Address [-1  1] 
# 
# URL_Length [ 1  0 -1]
# 
# Shortining_Service [ 1 -1]
# 
# having_At_Symbol [ 1 -1]
# 
# double_slash_redirecting [-1  1]
# 
# Prefix_Suffix [-1  1]
# 
# having_Sub_Domain [-1  0  1]
# 
# SSLfinal_State [-1  1  0]
# 
# Domain_registeration_length [-1  1]
# 
# Favicon [ 1 -1]
# 
# port [ 1 -1]
# 
# HTTPS_token [-1  1]
# 
# Request_URL [ 1 -1]
# 
# URL_of_Anchor [-1  0  1]
# 
# Links_in_tags [ 1 -1  0]
# 
# SFH [-1  1  0]
# 
# Submitting_to_email [-1  1]
# 
# Abnormal_URL [-1  1]
# 
# Redirect [0 1]
# 
# on_mouseover [ 1 -1]
# 
# RightClick [ 1 -1]
# 
# popUpWidnow [ 1 -1]
# 
# Iframe [ 1 -1]
# 
# age_of_domain [-1  1]
# 
# DNSRecord [-1  1]
# 
# web_traffic [-1  0  1]
# 
# Page_Rank [-1  1]
# 
# Google_Index [ 1 -1]
# 
# Links_pointing_to_page [ 1  0 -1]
# 
# Statistical_report [-1  1]
# 
# Result [-1  1]
# 

# In[3]:


data.info()


# In[4]:


data.describe()


# Let's check if there are any missing values:

# In[5]:


data.isna().sum()


# In[ ]:





# Great! There are no missing values.
# 
# Also, the data already contains categorical values encoded. So we don't need to do that.

# let's see the data distribution of the respective columns:

# In[6]:


plot.figure(figsize=(15,50), facecolor='white')
plotnumber =1

for column in data.drop(['Result'],axis=1):
    ax = plot.subplot(12,3,plotnumber)
    sns.countplot(data[column])
    plot.xlabel(column,fontsize=10)
    plotnumber+=1
plot.show()


# In[7]:


plot.figure(figsize=(15,50), facecolor='white')
plotnumber =1

for column in data.drop(['Result'],axis=1):
    ax = plot.subplot(12,3,plotnumber)
    sns.violinplot(data= data,x=data[column],y =data["Result"] )
    plot.xlabel(column,fontsize=10)
    plotnumber+=1
plot.show()


# Well the data looks decently distributed from the violin plot above.

# Let's check how balanced our dataset is:

# In[8]:


sns.countplot(data['Result'])


# Great! our dataset is balanced. 
# 
# We can go ahead with training our model.

# In[9]:


#divide our data
x=data.drop('Result',axis=1)
y=data['Result']


# In[10]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plot.plot(range(1,11),wcss)
plot.title('The Elbow Method')
plot.xlabel('Number of clusters')
plot.ylabel('WCSS')
plot.show()


# In[11]:


from kneed import KneeLocator
s=KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
s.knee


# In[12]:


import pickle
kmeans = KMeans(n_clusters =s.knee, init = 'k-means++', random_state = 42)
with open( 'kmeans.sav', 'wb') as f:
    pickle.dump(kmeans,f)
y_kmeans = kmeans.fit_predict(x)


# In[13]:


x['Cluster']=y_kmeans
x['Labels']=y
list_of_clusters=x['Cluster'].unique()
x.head()


# In[14]:


from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split


# In[15]:


list_of_clusters


# In[39]:


model_name=[]
for i in list_of_clusters:
    cluster_data=x[x['Cluster']==i] 
    cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
    cluster_label= cluster_data['Labels']
    x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=.25, random_state=36)
    
# creating a new model with SVC
    sv_classifier = SVC()
    sv_classifier.fit(x_train, y_train)
    # Predictions using the SVC Model
    prediction_svm=sv_classifier.predict(x_test) 
    if len(y_test.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
        svm_score = accuracy_score(y_test,prediction_svm)
        print('svm_score',svm_score)
    else:
        svm_score = roc_auc_score(y_test, prediction_svm) # AUC for Random Forest
        print('svm_score',svm_score)
               

# creating a new model with XGBClassifier
    xgb = XGBClassifier(criterion='gini',max_depth=5, n_estimators=50)
    xgb.fit(x_train, y_train)
    # Predictions using the XGBClassifier Model
    prediction_xgboost = xgb.predict(x_test)  
    if len(y_test.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
        xgboost_score = accuracy_score(y_test, prediction_xgboost)
        print('xgboost_score',xgboost_score)
    else:
        xgboost_score = roc_auc_score(y_test, prediction_xgboost) # AUC for XGBoost
        print('xgboost_score',xgboost_score)
                
    
   #comparing the two models
    if(svm_score <  xgboost_score):
        print('XGBoost'+str(i))
        model_name.append('XGBoost'+str(i) +'.sav')
        with open( 'XGBoost'+str(i)+'.sav', 'wb') as f:
            pickle.dump(xgb,f)
       
        
    else:
        print('SVM'+str(i))
        model_name.append('XGBoost'+str(i) +'.sav')
        with open( 'SVM'+str(i)+'.sav', 'wb') as f:
            pickle.dump(sv_classifier,f)
l=model_name


# In[45]:


class prediction:
    def model_prediction(data):
        clus=open('kmeans.sav','rb')
        kmeans=pickle.load(clus)


        clusters=kmeans.fit_predict(data)
        data['clusters']=clusters
        clusters=data['clusters'].unique()
        result=[]


        def find_correct_model_file(p):
            for u in l:
                for i in u:
                    for k in i:
                        if str(p)==str(k):
                            return u

        for i in clusters:
            cluster_data= data[data['clusters']==i]
            cluster_data = cluster_data.drop(['clusters'],axis=1)
            model_name=find_correct_model_file(i)
            model=open(model_name,'rb')
            model=pickle.load(model)
            for val in (model.predict(cluster_data)):
                result.append(val)
        result = pd.DataFrame(result,columns=['Predictions'])
        path="Predictions.csv"
        result.to_csv("Predictions.csv",header=True)
        print(result)


# In[48]:


a=model_prediction(x_test)


# In[75]:


a=pd.read_csv('Predictions.csv')
del a['Unnamed: 0']
a.replace(1,'yes',inplace=True)
a.replace(-1,'no',inplace=True)
a


# In[ ]:





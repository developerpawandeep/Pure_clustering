#!/usr/bin/env python
# coding: utf-8

# ### Reading

# In[98]:


import pandas as pd


# In[99]:


df=pd.read_csv("C:/Users/Dragneel/Desktop/Biotechniques-Articles - Papers.csv")


# In[100]:


df.head()


# ### Preparing Data

# In[101]:


df.columns


# In[102]:


df.drop(['Unnamed: 2','Volume'],axis=1)


# ### Applying Algo 

# In[103]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# In[104]:


doc=df['Title'].tolist()


# In[105]:


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(doc)


# In[106]:


terms = vectorizer.get_feature_names()


# In[107]:


print(X)


# #### Checking the best number of clusters

# In[108]:


import matplotlib.pyplot as plt
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')
def plot_elbow_graph(km_init="k-means++"):
    wcss = []

    for i in range(1, 15):
        kmeans = KMeans(n_clusters = i, init = km_init, random_state = 42,max_iter = 100, n_init = 1) #random,k-means++
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 15), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
        
plot_elbow_graph()


# ##### So possible number of clusters can be : 5 / 11

# In[109]:


true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


# In[110]:


labels = model.predict(X)


# In[111]:


clusters={}
n=0


# In[112]:


labels


# In[113]:


model.labels_


# In[114]:


import collections 
temp=collections.defaultdict(list)
for idx,word in enumerate(labels):
    temp[word].append(terms[idx])


# In[115]:


temp[0]


# In[125]:


clus1=pd.DataFrame(data=temp[0])
clus2=pd.DataFrame(data=temp[1])
clus3=pd.DataFrame(data=temp[2])
clus4=pd.DataFrame(data=temp[3])
clus5=pd.DataFrame(data=temp[4])


# In[140]:


export_csv = clus1.to_csv (r'C:/Users/Dragneel/Desktop/cluster1.csv', index = None, header=True)
export_csv = clus2.to_csv (r'C:/Users/Dragneel/Desktop/cluster2.csv', index = None, header=True)
export_csv = clus3.to_csv (r'C:/Users/Dragneel/Desktop/cluster3.csv', index = None, header=True)
export_csv = clus4.to_csv (r'C:/Users/Dragneel/Desktop/cluster4.csv', index = None, header=True)
export_csv = clus5.to_csv (r'C:/Users/Dragneel/Desktop/cluster5.csv', index = None, header=True)


# ###### We can do 11 clusters too 
# true_k = 11
# model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
# model.fit(X)

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


movie_data = pd.read_csv('Movie.csv')


# In[4]:


movie_data


# In[5]:


movie_data.shape


# In[6]:


movie_data.isna().sum()


# In[7]:


movie_data['userId'].nunique()


# In[8]:


movie_data['movie'].nunique()


# In[9]:


print(movie_data['movie'].unique())


# ### provide ubcf

# ## use correlation metric

# In[10]:


corr_piv_table=pd.pivot_table(data=movie_data,values='rating',index='movie',columns='userId').fillna(0)
corr_piv_table


# In[11]:


corr_piv_table.columns = movie_data['userId'].unique()
corr_piv_table


# In[12]:


corr_piv_table.corr().round(2)


# In[13]:


movie_data.columns


# from sklearn.metrics import pairwise_distances

# ##  euclidean distence  as a metric  for UBCF

# In[14]:


from sklearn.metrics import pairwise_distances


# In[15]:


euclidean_pivot_table = pd.pivot_table(data=movie_data,values='rating',index='userId',columns='movie').fillna(0)
euclidean_pivot_table


# In[16]:


euclidean_pivot_table.index  = movie_data['userId'].unique()
euclidean_pivot_table


# In[17]:


euclidean_pivot_table.values


# In[18]:


euclidean_uBcf =  pairwise_distances(X = euclidean_pivot_table.values,metric='euclidean')


# In[19]:


euclidean_uBcf


# In[20]:


ubcf_euclidean_metric = pd.DataFrame(data =euclidean_uBcf)
ubcf_euclidean_metric.index=movie_data['userId'].unique()
ubcf_euclidean_metric.columns=movie_data['userId'].unique()
ubcf_euclidean_metric.round(2)


# ##  use cosine as a metric for UBDF

# In[21]:


cosine_pivot_table = pd.pivot_table( data = movie_data, values='rating',index='userId',columns='movie',).fillna(0)


# In[22]:


cosine_pivot_table


# In[23]:


cosine_pivot_table.index=movie_data['userId'].unique()
cosine_pivot_table


# In[24]:


cosine_pivot_table.values


# In[25]:


cosine_ubcf= pairwise_distances(X=cosine_pivot_table.values,metric='cosine')


# In[26]:


cosine_ubcf


# In[29]:


cosine_ubcf = pd.DataFrame(data =cosine_ubcf)
cosine_ubcf.index=movie_data['userId'].unique()
cosine_ubcf.columns=movie_data['userId'].unique()
cosine_ubcf.round(2)


# ##  filter the data top 50 users

# In[30]:


first_50_user= cosine_ubcf.iloc[:50,:50]
first_50_user


# In[31]:


import numpy as np


# In[34]:


np.fill_diagonal(a = first_50_user.to_numpy(), val=0)


# In[35]:


first_50_user= cosine_ubcf.iloc[:50,:50]


# In[36]:


first_50_user


# In[37]:


first_50_user.round(2)


# In[43]:


first_50_user.idxmax()


# In[42]:


movie_data[(movie_data['userId']==140)|(movie_data['userId']==90)]


# ###  Inference

# ####  90 th user and 140 user highley correlated

# In[ ]:





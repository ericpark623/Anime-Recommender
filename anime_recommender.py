#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import ast
import scipy
from collections import defaultdict
from implicit import bpr


# In[12]:


anime = pd.read_csv('/Users/ericpark/Desktop/Anime_data/animes.csv')
anime = anime.drop_duplicates(subset=['uid'])


# In[13]:


reviews = pd.read_csv('/Users/ericpark/Desktop/Anime_data/reviews.csv')
reviews = reviews.drop_duplicates(subset = ['uid','anime_uid'])
profile_id_df = pd.DataFrame(reviews['profile'].unique()).reset_index().rename(columns={'index':'p_uid',0:'profile'})
full_reviews = reviews.merge(profile_id_df, how='left',on='profile')
data = full_reviews.to_dict('records')


# In[14]:


userIDs, itemIDs = {},{}
for d in data:
    u, i  = d['p_uid'],d['anime_uid']
    if not u in userIDs:
        userIDs[u] = len(userIDs)
    if not i in itemIDs:
        itemIDs[i] = len(itemIDs)

nUsers,nItems = len(userIDs),len(itemIDs)


# In[15]:


Xui = scipy.sparse.lil_matrix((nUsers, nItems))
for d in data:
    Xui[userIDs[d['p_uid']],itemIDs[d['anime_uid']]] = 1
    
Xui_csr = scipy.sparse.csr_matrix(Xui)


# In[16]:


data = full_reviews.to_dict('records')


# In[17]:


model = bpr.BayesianPersonalizedRanking(factors = 5)


# In[18]:


model.fit(Xui_csr)


# In[19]:


recommended = model.recommend(0,Xui_csr[0])


# In[20]:


recommended[0]


# In[21]:


anime[anime['uid'].isin(recommended[0])]['title']


# In[22]:


def BPR_recommendation(username,model,anime):
    name = profile_id_df[profile_id_df['profile']==username]
    if len(name) == 0:
        return "Username not found"
    else:
        user_id = name.index[0]
        recommended = model.recommend(user_id,Xui_csr[user_id])
        rec_ids = recommended[0]
        anime_rec = anime[anime['uid'].isin(rec_ids)]
        return anime_rec['title']
        
    


# In[24]:


BPR_recommendation('skrn',model,anime)


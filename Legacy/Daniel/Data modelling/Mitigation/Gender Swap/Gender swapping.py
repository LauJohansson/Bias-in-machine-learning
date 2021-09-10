#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
import math 


# In[2]:


fall_data = pd.read_csv('/restricted/s161749/G2020-57-Aalborg-bias/Data_air/Fall_count_clusterOHE_std.csv')


# In[12]:


fall_data['Original'] = 1


# In[13]:


fall_data_swap = fall_data.copy()


# In[14]:


fall_data_swap['Original'] = 0


# In[15]:


fall_data


# In[16]:


fall_data_swap['Gender']=(fall_data_swap['Gender']-1)*(-1)


# In[17]:


fall_data_swap


# In[18]:


fall_data_gender_swap = pd.concat([fall_data,fall_data_swap])


# In[19]:


fall_data_gender_swap


# In[22]:


fall_data_gender_swap['Gender'].value_counts()


# In[23]:


fall_data_gender_swap.to_csv(('/restricted/s161749/G2020-57-Aalborg-bias/Data_air/Fall_count_swap.csv'))


# In[ ]:





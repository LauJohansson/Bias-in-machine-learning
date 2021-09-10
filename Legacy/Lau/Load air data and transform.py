#!/usr/bin/env python
# coding: utf-8

# # Load original fall data

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:


pathRoot="../Data_air/"


# In[5]:


pathFall=pathRoot+"Fall.csv"
fallData=pd.read_csv(pathFall)


# In[6]:


fallData


# # Load embeddings

# In[43]:


#fallData.count()


# In[44]:


pathFallEmbd=pathRoot+"Fall_emb.csv"
fallDataEmbd=pd.read_csv(pathFallEmbd)


# In[45]:


fallDataEmbd


# # Load fall count
# 

# In[123]:


pathFallOHE=pathRoot+"Fall_count.csv"
fallDataOHE=pd.read_csv(pathFallOHE)


# In[124]:


fallDataOHE.shape


# In[125]:


fallDataOHE.head(1)


# In[126]:


fallDataOHE[fallDataOHE["Ats_Toiletarmstøtter"]>1]["Ats_Toiletarmstøtter"]


# In[127]:


fallDataOHE.max().sort_values(ascending=False).head(20)


# In[128]:


not_these=["Gender","BirthYear","Cluster","LoanPeriod","NumberAts","Fall"]
these=[col for col in fallDataOHE.columns if col not in not_these]
fallDataOHE[fallDataOHE[these]>1]= (fallDataOHE[these] > 1) * 1


# In[129]:


fallDataOHE.max().sort_values(ascending=False).head(10)


# # Cluster dummy

# In[130]:


just_dummies=pd.get_dummies(fallDataOHE["Cluster"],prefix="Cluster")
fallDataOHE = pd.concat([fallDataOHE, just_dummies], axis=1)
fallDataOHE=fallDataOHE.drop(["Cluster"] ,axis=1)
fallDataOHE.head(1)


# In[131]:


fallDataOHE=fallDataOHE.drop(columns="Ats_0")
fallDataOHE.to_csv(pathRoot+"Fall_count_clusterOHE.csv")


# ## Standardize + cluster dummy

# In[132]:


from sklearn import preprocessing
fallDataOHE_std=fallDataOHE.copy()
X_col_names_to_std=["BirthYear","LoanPeriod","NumberAts"]
fallDataOHE_std[X_col_names_to_std] = pd.DataFrame(preprocessing.scale(fallDataOHE_std[X_col_names_to_std]),columns=X_col_names_to_std)


# In[133]:


fallDataOHE_std.head(1)


# In[134]:



fallDataOHE_std.to_csv(pathRoot+"Fall_count_clusterOHE_std.csv")


# In[135]:


#for col in fallDataOHE_std:
#    print("\'"+col+"\',")


# In[136]:


fallDataOHE_std


# In[ ]:





# In[ ]:





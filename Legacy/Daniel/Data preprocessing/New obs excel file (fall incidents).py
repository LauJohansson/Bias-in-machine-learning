#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # Getting data

# In[2]:


fall_data = pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/Observationer_Rasmus_BorgerIdOgGenderMinusCPR.xlsx",index_col=None,sheet_name=0)


# In[19]:


ID_list = pd.read_csv('/restricted/s161749/G2020-57-Aalborg-bias/Transformed_data/ID_birth_list.csv',index_col=0)


# In[20]:


fall_data.head(5)


# In[21]:


ID_list.head(5)


# # Preprocessing fall data (obs)

# ### There might be many Nans in BorgerId

# In[26]:


fall_data.shape


# In[25]:


fall_data.dropna(subset=['BorgerId'])


# ### Yes, only 14.114 rows with BorgerId not Nan

# In[27]:


fall_data = fall_data.dropna(subset=['BorgerId'])


# # Changing names and data types

# In[40]:


ID_list = ID_list.rename(columns={"BorgerID":"BorgerId"})
ID_list.shape


# In[42]:


fall_data = fall_data.astype({"BorgerId":int})
fall_data.shape


# In[41]:


fall_data.dtypes


# In[39]:


ID_list.dtypes


# # Merging fall_data with ID_list as to add birth year to obs data (fall)

# In[31]:


obs_data = pd.merge(
    left=fall_data,
    right=ID_list,
    how='left',
    on='BorgerId'
    
)


# In[32]:


obs_data


# In[34]:


obs_data['Birth Year'].isna().sum()


# In[36]:


obs_data.dropna(subset=['Birth Year']).drop_duplicates(subset=['BorgerId'])


# In[ ]:





# In[ ]:





# In[ ]:





# # NEW TRY WITH STRING BORGERIDs

# ## Getting data

# In[97]:


fall_data = pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/Observationer_Rasmus_BorgerIdOgGenderMinusCPR.xlsx",index_col=None)#,dtype={"BorgerId":str})


# In[98]:


ID_list = pd.read_csv('/restricted/s161749/G2020-57-Aalborg-bias/Transformed_data/ID_birth_list.csv',index_col=0)#,dtype={"BorgerID":str})


# In[99]:


fall_data.dtypes


# In[100]:


fall_data.shape


# In[101]:


ID_list.dtypes


# In[102]:


ID_list.shape


# In[103]:


ID_list = ID_list.rename(columns={"BorgerID":"BorgerId"})


# In[104]:


obs_data = pd.merge(
    left=fall_data,
    right=ID_list,
    how='left',
    on='BorgerId'
    
)


# In[109]:


obs_data=obs_data.rename(columns={"Birth Year":"BirthYear"})
obs_data


# In[ ]:





# In[ ]:





# In[112]:


obs_data_no_na=obs_data.dropna(subset=['BirthYear'])


# In[114]:


obs_data_no_na


# In[115]:


obs_data_no_na.to_excel('/restricted/s161749/G2020-57-Aalborg-bias/air/data/raw/2020/Observationer_Rasmus_BorgerId_Gender_BirthYear.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np


# In[65]:


#aid_data = []
#aid_data = pd.DataFrame(aid_data)

for i in range(0,6):
    if i==0:
        aid_data=pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/borgere_hmi_Rasmus_BorgerId_Gender_BirthYear.xlsx",index_col=0,sheet_name=i,dtype={"BorgerID":str})
    else:
        aid_data.append(pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/borgere_hmi_Rasmus_BorgerId_Gender_BirthYear.xlsx",index_col=0,sheet_name=i,dtype={"BorgerID":str}))
    


# In[100]:


aid_data_0 = pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/borgere_hmi_Rasmus_BorgerId_Gender_BirthYear.xlsx",index_col=0,sheet_name=0)#,dtype={"BorgerID":int})
aid_data_1 = pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/borgere_hmi_Rasmus_BorgerId_Gender_BirthYear.xlsx",index_col=0,sheet_name=1)#,dtype={"BorgerID":int})
aid_data_2 = pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/borgere_hmi_Rasmus_BorgerId_Gender_BirthYear.xlsx",index_col=0,sheet_name=2)#,dtype={"BorgerID":int})
aid_data_3 = pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/borgere_hmi_Rasmus_BorgerId_Gender_BirthYear.xlsx",index_col=0,sheet_name=3)#,dtype={"BorgerID":int})
aid_data_4 = pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/borgere_hmi_Rasmus_BorgerId_Gender_BirthYear.xlsx",index_col=0,sheet_name=4)#,dtype={"BorgerID":int})
aid_data_5 = pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/borgere_hmi_Rasmus_BorgerId_Gender_BirthYear.xlsx",index_col=0,sheet_name=5)#,dtype={"BorgerID":int})
aid_data   = pd.concat([aid_data_0,aid_data_1,aid_data_2,aid_data_3,aid_data_4,aid_data_5],sort=False,axis=0)


# In[101]:


aid_data.shape


# In[102]:


aid_data = aid_data.reset_index()


# In[ ]:





# In[103]:


#aid_data = pd.read_excel("/restricted/s161749/G2020-57-Aalborg-bias/Data/borgere_hmi_Rasmus_BorgerId_Gender_BirthYear.xlsx",index_col=0,sheet_name=5,dtype={"BorgerID":str})


# In[ ]:





# In[104]:


aid_data


# In[105]:


#aid_data=aid_data.drop_duplicates(subset=['BorgerID','Birth Year'])
aid_data_no_nan=aid_data.dropna(subset=['BorgerID'])
aid_data_no_nan=aid_data_no_nan.dropna(subset=['Birth Year'])
aid_data_no_nan=aid_data_no_nan.drop_duplicates(subset=['BorgerID'])


# In[106]:


ID_birth = aid_data_no_nan[['BorgerID','Birth Year']]
ID_birth = ID_birth.reset_index(drop=True)
ID_birth


# In[107]:


ID_birth['BorgerID'].value_counts()


# In[108]:


ID_birth.dropna()


# In[109]:


ID_birth.to_csv('../Transformed_data/ID_birth_list.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





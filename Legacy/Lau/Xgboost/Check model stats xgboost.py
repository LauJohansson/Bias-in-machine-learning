#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')


# In[10]:


from utils_copy import *


# # All stats (chose yourself)

# In[11]:




### USE ALL DATA
#file_name="fall.csv"
#full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name



###USE SMALL TEST
titel_mitigation="test23may"
PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/Xgboost/models/"+titel_mitigation+"/"

file_name="all_test_data.csv"



full_file_path=PATH_orig+file_name


print("PATH_orig:",PATH_orig)
print("PATH to file:",full_file_path)


# In[12]:


y_col_name="Fall"



X_col_names=['Gender', 'BirthYear', 'Cluster', 'LoanPeriod', 'NumberAts', '1Ats',
       '2Ats', '3Ats', '4Ats', '5Ats', '6Ats', '7Ats', '8Ats', '9Ats', '10Ats',
       '11Ats', '12Ats', '13Ats', '14Ats', '15Ats', '16Ats', '17Ats', '18Ats',
       '19Ats', '20Ats', '21Ats', '22Ats', '23Ats', '24Ats', '25Ats', '26Ats',
       '27Ats', '28Ats', '29Ats', '30Ats', '31Ats', '32Ats', '33Ats', '34Ats',
       '35Ats', '36Ats', '37Ats', '38Ats', '39Ats', '40Ats', '41Ats', '42Ats',
       '43Ats', '44Ats', '45Ats', '46Ats', '47Ats', '48Ats', '49Ats', '50Ats',]



procted_col_name="Gender"


output_col_name="output"


unfavourable_name=0 #women=0
favourable_name=1 #men=1


# In[13]:


df2 = pd.read_csv(full_file_path)


# In[18]:


#get_df_w_metrics(df2,procted_col_name,y_col_name,output_col_name)


# In[15]:


df2_copy=df2.copy()

mean_age=df2_copy["BirthYear"].mean()

df2_copy["age_cat_custom"]=df2_copy["BirthYear"].apply(lambda x: "Above mean" if x<mean_age else "Under mean")


# In[20]:


get_df_w_metrics(df2,procted_col_name,y_col_name,output_col_name).sort_values("Gender")[["Gender","TPR","FPR","TNR","FNR","ACC"]]*100


# In[22]:


get_df_w_metrics(df2_copy,"age_cat_custom",y_col_name,output_col_name).sort_values("age_cat_custom")[["age_cat_custom","TPR","FPR","TNR","FNR","ACC"]]*100


# In[12]:


compare_bias_metrics(df2_copy,"Gender",y_col_name,output_col_name,0,1,True)


# In[11]:


compare_bias_metrics(df2_copy,"age_cat_custom",y_col_name,output_col_name,"Under mean","Above mean",True)


# In[ ]:





# In[ ]:





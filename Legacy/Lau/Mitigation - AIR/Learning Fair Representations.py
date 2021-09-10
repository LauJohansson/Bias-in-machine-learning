#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.datasets as datasets
from torch.utils.data import DataLoader
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from IPython.display import clear_output


import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import numpy as np
from sklearn.model_selection import train_test_split

#from google.colab import drive
from sklearn.model_selection import KFold

from datetime import datetime

import pytz
import random

import os



#FAIRNESS
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric


# # FIle path

# In[81]:


file_name="Fall_count_clusterOHE_std.csv"
#file_name="Fall_count_clusterOHE.csv"
full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name


print("PATH to file:",full_file_path)


# # Specify attributes

# In[82]:


y_col_name="Fall"



X_col_names=[ 'BirthYear', 'LoanPeriod', 'NumberAts']



procted_col_name="Gender"


output_col_name="output"


unfavourable_name=0 #women=0
favourable_name=1 #men=1


# # Read the data

# In[83]:


full_file_path


# In[84]:


df2 = pd.read_csv(full_file_path)

df2_all=df2.drop(columns=["Unnamed: 0"]+X_col_names+[y_col_name]).copy()


# In[85]:


#Keep only numerical features
df2=df2[X_col_names+[y_col_name]+[procted_col_name]]


# # Creating the binary dataset

# See methods from: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.statistical_parity_difference

# Rename trainingset TRUE falls to be named "selected_col". <br>
# Rename trainingset PREDICTED falls to be named "selected_col".

# In[86]:


trainset_renamed=df2.rename(columns={y_col_name: "selected_col"})
#predicted_trainset_renamed=df2.rename(columns={output_col_name: "selected_col"}).drop([y_col_name], axis=1)


# Needs all columns when using BinaryLabelDataset-class:

# In[87]:


all_cols=X_col_names
all_cols=all_cols.append("selected_col")


# ### Train binary dataset:

# In[88]:


#Train TRUE
train_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=trainset_renamed,
                                label_names=['selected_col'],
                                protected_attribute_names=[procted_col_name],
                                unprivileged_protected_attributes=['0'])
#TRAIN PREDICTED
#pred_train_BLD = BinaryLabelDataset(favorable_label='1',
#                                unfavorable_label='0',
#                                df=predicted_trainset_renamed,
#                                label_names=['selected_col'], #label_names=['preds'],
#                                protected_attribute_names=[procted_col_name],
#                                unprivileged_protected_attributes=['0'])


# ## Bias in TRUE Y's

# In[89]:


#In 
bldm1 = BinaryLabelDatasetMetric(train_BLD, 
                                    privileged_groups=[{procted_col_name: 1}], 
                                    unprivileged_groups=[{procted_col_name: 0}])


# In[90]:


print(f"Disparate impact: {bldm1.disparate_impact()}")
print(f"Individual fairness: {bldm1.consistency()}") #Learning Fair Representations
print(f"Statistical parity difference (𝑃𝑟(𝑌=1|𝐷=unprivileged)−𝑃𝑟(𝑌=1|𝐷=privileged)): {bldm1.statistical_parity_difference()}")


# ## Bias in PREDICTED Y's

# In[91]:


#In 
#bldm2 = BinaryLabelDatasetMetric(pred_train_BLD, 
#                                    privileged_groups=[{procted_col_name: favourable_name}], 
#                                    unprivileged_groups=[{procted_col_name: unfavourable_name}])


# In[92]:


#print(f"Disparate impact: {bldm2.disparate_impact()}")
#print(f"Individual fairness: {bldm2.consistency()}") #Learning Fair Representations
#print(f"Statistical parity difference (𝑃𝑟(𝑌=1|𝐷=unprivileged)−𝑃𝑟(𝑌=1|𝐷=privileged)): {bldm2.statistical_parity_difference()}")


# # Disparate impact remove

# In[191]:


from aif360.algorithms.preprocessing import LFR
di = LFR(privileged_groups=[{procted_col_name: 1}], 
                                    unprivileged_groups=[{procted_col_name: 0}],
        
        #k=10,
        Az=100.0
        
        )


# ## Create new data set with repaired (level=1)

# Save column names

# In[192]:


#train_BLD.feature_names


# In[193]:


train_BLD.label_names


# In[194]:


all_col_names=train_BLD.feature_names+train_BLD.label_names
print(all_col_names)


# ### Train data:

# In[195]:


rp_train = di.fit_transform(train_BLD) #Using the repaired data
rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=all_col_names) #save as pandas
rp_train_pd = rp_train_pd.drop(columns=["Gender"])
rp_train_pd = pd.concat([rp_train_pd,df2["Gender"]],axis=1)


rp_train_pd["Gender_string"]=rp_train_pd["Gender"].apply(lambda x: "Female" if x==0 else "Male")
rp_train_pd=rp_train_pd.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})


# In[196]:



df2_for_plot=df2.copy()
df2_for_plot["Gender_string"]=df2_for_plot["Gender"].apply(lambda x: "Female" if x==0 else "Male")
df2_for_plot=df2_for_plot.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})


# In[197]:


palette_custom ={"Female": "C0", "Male": "C1"}


# In[198]:



fig, ax = plt.subplots(1,2,figsize=(20,8),sharey=True,tight_layout=True)

ax=ax.ravel()


sns.histplot(data=df2_for_plot,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,ax=ax[0],palette=palette_custom,bins=20)
ax[0].set_title("Original standardized data")

sns.histplot(data=rp_train_pd,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,ax=ax[1],palette=palette_custom,bins=20)
ax[1].set_title("LFR")
plt.show()


# In[ ]:





# In[199]:



fig, ax = plt.subplots(3,2,figsize=(20,15),sharey=True,tight_layout=True)

ax=ax.ravel()


#Loan period
sns.histplot(data=df2_for_plot,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,ax=ax[0],palette=palette_custom,bins=20)
ax[0].set_title("Original data",fontsize=20)

sns.histplot(data=rp_train_pd,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,ax=ax[1],palette=palette_custom,bins=20)
ax[1].set_title("LFR",fontsize=20)



#NumberAts
sns.histplot(data=df2_for_plot,x="NumberAts",hue="Gender",stat="probability",common_norm=False,ax=ax[2],palette=palette_custom,bins=20)
#ax[2].set_title("Original standardized data")
#ax[i].set_ylabel("Rate",fontsize=20)

sns.histplot(data=rp_train_pd,x="NumberAts",hue="Gender",stat="probability",common_norm=False,ax=ax[3],palette=palette_custom,bins=20)
#ax[3].set_title("LFR")


#Birthyear
sns.histplot(data=df2_for_plot,x="BirthYear",hue="Gender",stat="probability",common_norm=False,ax=ax[4],palette=palette_custom,bins=20)
#ax[4].set_title("Original standardized data")

sns.histplot(data=rp_train_pd,x="BirthYear",hue="Gender",stat="probability",common_norm=False,ax=ax[5],palette=palette_custom,bins=20)
#ax[5].set_title("LFR")


plt.show()


# In[ ]:





# In[ ]:





# In[163]:



fig, ax = plt.subplots(3,2,figsize=(20,15),sharey=True,tight_layout=True)

ax=ax.ravel()


#Loan period
sns.histplot(data=df2_for_plot,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,ax=ax[0],palette=palette_custom,bins=20)
ax[0].set_title("Original data",fontsize=20)

sns.histplot(data=rp_train_pd,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,ax=ax[1],palette=palette_custom,bins=20)
ax[1].set_title("LFR",fontsize=20)



#NumberAts
sns.histplot(data=df2_for_plot,x="NumberAts",hue="Gender",stat="probability",common_norm=False,ax=ax[2],palette=palette_custom,bins=20)
#ax[2].set_title("Original standardized data")
#ax[i].set_ylabel("Rate",fontsize=20)

sns.histplot(data=rp_train_pd,x="NumberAts",hue="Gender",stat="probability",common_norm=False,ax=ax[3],palette=palette_custom,bins=20)
#ax[3].set_title("LFR")


#Birthyear
sns.histplot(data=df2_for_plot,x="BirthYear",hue="Gender",stat="probability",common_norm=False,ax=ax[4],palette=palette_custom,bins=20)
#ax[4].set_title("Original standardized data")

sns.histplot(data=rp_train_pd,x="BirthYear",hue="Gender",stat="probability",common_norm=False,ax=ax[5],palette=palette_custom,bins=20)
#ax[5].set_title("LFR")


plt.show()


# In[ ]:





# # Save the LFR data

# In[76]:


#rp_to_saving = pd.concat ([rp_train_pd.rename(columns={"selected_col":"Fall"}).drop(columns=["Gender"]),df2_all], axis=1)
    
#rp_to_saving.to_csv("/restricted/s164512/G2020-57-Aalborg-bias/data_air/Learning Fair Representations/"+file_name[:-4]+"_LFR"+".csv")
    


# In[ ]:





# In[ ]:





# In[50]:


def LFR_custom(df_train,lfr=None):
    from aif360.algorithms.preprocessing import LFR
    from aif360.datasets import BinaryLabelDataset
    
    
    X_col_names_f=['Gender', 'BirthYear', 'LoanPeriod', 'NumberAts']
    df2_all=df_train.drop(columns=X_col_names_f).copy() #Gemmer alle kolonner, undtagen numerical og gender
    df2=df_train[X_col_names_f+["Fall"]].copy() #Gem kun numerical features
    df2_gender=df_train["Gender"].copy() #Gemmer bare gender
    
    
    #Create the binarylabeldataset
    df_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=df2,
                                label_names=['Fall'],
                                protected_attribute_names=["Gender"],
                                unprivileged_protected_attributes=['0'])
    #Define the DI remover
    if lfr is None:
        lfr = LFR(privileged_groups=[{"Gender": 1}], 
                                    unprivileged_groups=[{"Gender": 0}])
        rp_df = lfr.fit_transform(df_BLD)
    else:
        rp_df = lfr.transform(df_BLD)
        

    #Save the columnnames
    all_col_names=df_BLD.feature_names+df_BLD.label_names
        
        
    
    #Save repaired data as pandas DF
    rp_df_pd = pd.DataFrame(np.hstack([rp_df.features,rp_df.labels]),columns=all_col_names) 
    
    #Somehow gender is also transformed! So we drop it! DETTE SKAL VI NOK LIGE HOLDE ØJE MED
    ###OBS!#####
    rp_df_pd = rp_df_pd.drop(columns=["Gender"])
    rp_df_pd = pd.concat([rp_df_pd,df2_gender],axis=1)

    ##########
    
    
    #Concatenate the non-numerical columns
    transformed_data = pd.concat ([rp_df_pd,df2_all], axis=1)
    
    

    
    return transformed_data,lfr


# In[51]:


df_new = pd.read_csv(full_file_path)


# In[52]:


kf=KFold(n_splits=5,random_state=1,shuffle=True)

for train_index, test_index in kf.split(df_new):
    X_train, X_test = df_new.iloc[train_index],df_new.iloc[test_index]
    
    
    X_train_rp,lfr=LFR_custom(X_train,lfr=None)
    X_test_rp,lfr=LFR_custom(X_test,lfr)
    


# In[56]:


X_train["Gender_string"]=X_train["Gender"].apply(lambda x: "Female" if x==0.0 else "Male")
X_train=X_train.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})
sns.histplot(data=X_train,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,bins=20)


# In[57]:


X_train_rp["Gender_string"]=X_train_rp["Gender"].apply(lambda x: "Female" if x==0.0 else "Male")
X_train_rp=X_train_rp.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})
sns.histplot(data=X_train_rp,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,bins=20)


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[144]:


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


# In[145]:


plt.plot([0,0])


# # FIle path

# In[146]:


#file_name="Fall_count_clusterOHE_std.csv"
file_name="Fall_count_clusterOHE.csv"
full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name


print("PATH to file:",full_file_path)


# # Specify attributes

# In[147]:


y_col_name="Fall"



X_col_names=['Gender', 'BirthYear', 'LoanPeriod', 'NumberAts']



procted_col_name="Gender"


output_col_name="output"


unfavourable_name=0 #women=0
favourable_name=1 #men=1


# # Read the data

# In[148]:


full_file_path


# In[149]:


df2 = pd.read_csv(full_file_path)

df2_all=df2.drop(columns=["Unnamed: 0"]+X_col_names+[y_col_name]).copy()


# In[150]:


#Keep only numerical features
df2=df2[X_col_names+[y_col_name]]


# In[151]:


df2.head(1)


# # Creating the binary dataset

# See methods from: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.statistical_parity_difference

# Rename trainingset TRUE falls to be named "selected_col". <br>
# Rename trainingset PREDICTED falls to be named "selected_col".

# In[152]:


trainset_renamed=df2.rename(columns={y_col_name: "selected_col"})
#predicted_trainset_renamed=df2.rename(columns={output_col_name: "selected_col"}).drop([y_col_name], axis=1)


# Needs all columns when using BinaryLabelDataset-class:

# In[153]:


all_cols=X_col_names
all_cols=all_cols.append("selected_col")


# ### Train binary dataset:

# In[154]:


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


# In[155]:


trainset_renamed


# ## Bias in TRUE Y's

# In[156]:


#In 
bldm1 = BinaryLabelDatasetMetric(train_BLD, 
                                    privileged_groups=[{procted_col_name: 1}], 
                                    unprivileged_groups=[{procted_col_name: 0}])


# In[157]:


print(f"Disparate impact: {bldm1.disparate_impact()}")
print(f"Individual fairness: {bldm1.consistency()}") #Learning Fair Representations
print(f"Statistical parity difference (𝑃𝑟(𝑌=1|𝐷=unprivileged)−𝑃𝑟(𝑌=1|𝐷=privileged)): {bldm1.statistical_parity_difference()}")


# ## Bias in PREDICTED Y's

# In[158]:


#In 
#bldm2 = BinaryLabelDatasetMetric(pred_train_BLD, 
#                                    privileged_groups=[{procted_col_name: favourable_name}], 
#                                    unprivileged_groups=[{procted_col_name: unfavourable_name}])


# In[159]:


#print(f"Disparate impact: {bldm2.disparate_impact()}")
#print(f"Individual fairness: {bldm2.consistency()}") #Learning Fair Representations
#print(f"Statistical parity difference (𝑃𝑟(𝑌=1|𝐷=unprivileged)−𝑃𝑟(𝑌=1|𝐷=privileged)): {bldm2.statistical_parity_difference()}")


# # Disparate impact remove

# In[160]:


from aif360.algorithms.preprocessing import DisparateImpactRemover
di = DisparateImpactRemover(repair_level=1.0)


# ## Create new data set with repaired (level=1)

# Save column names

# In[161]:


#train_BLD.feature_names


# In[162]:


train_BLD.label_names


# In[163]:


all_col_names=train_BLD.feature_names+train_BLD.label_names
print(all_col_names)


# ### Train data:

# In[164]:


rp_train = di.fit_transform(train_BLD) #Using the repaired data
rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=all_col_names) #save as pandas


# In[165]:


sns.histplot(data=df2,x="BirthYear",hue="Gender",stat="density",common_norm=False)
plt.title("Original data (level=1)")
plt.show()


# In[166]:


sns.histplot(data=rp_train_pd,x="BirthYear",hue="Gender",stat="density",common_norm=False)
plt.title("Repaired data")
plt.show()


# In[167]:


palette_custom ={"Female": "C0", "Male": "C1"}


# In[168]:




#level_list=[0.0,0.25,0.5,0.75,1.0]
level_list=[0.0,1.0]


all_col_names=train_BLD.feature_names+rp_train.label_names

fig,ax = plt.subplots(len(level_list),1, figsize=(15, 15), facecolor='w', edgecolor='k',sharex=True)

ax =ax.ravel()


rows= len(level_list)
cols=1
i=0


for level in level_list:
    
    di = DisparateImpactRemover(repair_level=level)
    
    #Training data
    rp_train = di.fit_transform(train_BLD) #Using the 
    rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=all_col_names)
    
    
    rp_to_saving = pd.concat ([rp_train_pd.rename(columns={"selected_col":"Fall"}),df2_all], axis=1)
    
    rp_to_saving.to_csv("/restricted/s164512/G2020-57-Aalborg-bias/data_air/DI_removed/"+file_name[:-4]+"_RPlevel"+str(level)+".csv")
    
    
    #test data
    #rp_test = di.fit_transform(test_BLD)
    #rp_test_pd = pd.DataFrame(np.hstack([rp_test.features,rp_test.labels]),columns=all_col_names)

    #plt.subplot(rows,cols,i)
    #sns.histplot(data=rp_train_pd,x="LoanPeriod",hue="Gender",stat="density",common_norm=False,bins=20)
    #plt.title(f"Repaired data (Level={level}).")
    
    rp_train_pd["Gender_string"]=rp_train_pd["Gender"].apply(lambda x: "Female" if x==0.0 else "Male")
    rp_train_pd=rp_train_pd.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})
    sns.histplot(data=rp_train_pd,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,bins=20,ax=ax[i],palette=palette_custom)
    ax[i].set_title(f"Repaired data (Level={level}).")
    #ax[i].legend()
    
    
    i=i+1
    
plt.show()


# In[ ]:





# # Testing custom DI function

# In[104]:


def DI_remove_custom(df_train,RP_level=1.0):
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    from aif360.datasets import BinaryLabelDataset
    X_col_names_f=['Gender', 'BirthYear', 'LoanPeriod', 'NumberAts']
    df2_all=df_train.drop(columns=["Unnamed: 0"]+X_col_names_f).copy() #Gemmer alle kolonner, undtagen numerical og gender
    df2=df_train[X_col_names_f].copy() #Gem kun numerical features
    
    df2["dummy"]=1 # this is a dummy variable, since DI remover dont use y. 
    
    #Create the binarylabeldataset
    df_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=df2,
                                label_names=['dummy'],
                                protected_attribute_names=["Gender"],
                                unprivileged_protected_attributes=['0'])
    #Define the DI remover
    di = DisparateImpactRemover(repair_level=RP_level)
    #Save the columnnames
    all_col_names=df_BLD.feature_names+df_BLD.label_names
    #Reparing the data
    rp_df = di.fit_transform(df_BLD)  
    #Save repaired data as pandas DF
    rp_df_pd = pd.DataFrame(np.hstack([rp_df.features,rp_df.labels]),columns=all_col_names) 
    #Concatenate the non-numerical columns
    transformed_data = pd.concat ([rp_df_pd,df2_all], axis=1)
    
    
    transformed_data_train=transformed_data.drop(columns=["dummy"])

    
    return transformed_data_train


# In[105]:


df_new = pd.read_csv(full_file_path)


# In[103]:


kf=KFold(n_splits=5,random_state=1,shuffle=True)
df_new#=df_new.drop(columns=["Fall"])
for train_index, test_index in kf.split(df_new):
    X_train, X_test = df_new.iloc[train_index],df_new.iloc[test_index]
    
    
    X_train_rp=DI_remove_custom(X_train)
    


# In[106]:


X_train


# In[107]:


X_train_rp


# In[ ]:





# In[100]:


X_train["Gender_string"]=X_train["Gender"].apply(lambda x: "Female" if x==0.0 else "Male")
X_train=X_train.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})
sns.histplot(data=X_train,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,bins=20)


# In[98]:


X_train_rp["Gender_string"]=X_train_rp["Gender"].apply(lambda x: "Female" if x==0.0 else "Male")
X_train_rp=X_train_rp.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})
sns.histplot(data=X_train_rp,x="LoanPeriod",hue="Gender",stat="probability",common_norm=False,bins=20)


# In[99]:


X_train_rp["Gender"].value_counts()


# In[131]:


np.hstack([rp_train.features,rp_train.labels])


# In[135]:


np.hstack([rp_train.features])


# In[139]:


len(rp_train.features)


# In[140]:


train_BLD


# In[ ]:





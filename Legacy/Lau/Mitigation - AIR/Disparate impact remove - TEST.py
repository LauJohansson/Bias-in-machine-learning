#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[5]:


plt.plot([2],[2])
plt.show()


# # FIle path

# In[6]:




### USE ALL DATA
#file_name="fall.csv"
#full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name



###USE SMALL TEST
titel_mitigation="testAIR"
PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/FFNN/models/"+titel_mitigation+"/"

file_name="all_test_data_localmodel.csv"
#file_name="all_test_data_globalmodel.csv"


full_file_path=PATH_orig+file_name


print("PATH_orig:",PATH_orig)
print("PATH to file:",PATH_orig+"SOMEMODEL/"+file_name)


# # Specify attributes

# In[8]:


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


# # Read the data

# In[10]:


df2 = pd.read_csv(full_file_path)


# # Creating the binary dataset

# See methods from: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.statistical_parity_difference

# Rename trainingset TRUE falls to be named "selected_col". <br>
# Rename trainingset PREDICTED falls to be named "selected_col".

# In[12]:


trainset_renamed=df2.rename(columns={y_col_name: "selected_col"}).drop([output_col_name], axis=1)
predicted_trainset_renamed=df2.rename(columns={output_col_name: "selected_col"}).drop([y_col_name], axis=1)


# Needs all columns when using BinaryLabelDataset-class:

# In[14]:


all_cols=X_col_names
all_cols=all_cols.append("selected_col")


# ### Train binary dataset:

# In[16]:


#Train TRUE
train_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=trainset_renamed,
                                label_names=['selected_col'],
                                protected_attribute_names=[procted_col_name],
                                unprivileged_protected_attributes=['0'])
#TRAIN PREDICTED
pred_train_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=predicted_trainset_renamed,
                                label_names=['selected_col'], #label_names=['preds'],
                                protected_attribute_names=[procted_col_name],
                                unprivileged_protected_attributes=['0'])


# ## Bias in TRUE Y's

# In[18]:


#In 
bldm1 = BinaryLabelDatasetMetric(train_BLD, 
                                    privileged_groups=[{procted_col_name: 1}], 
                                    unprivileged_groups=[{procted_col_name: 0}])


# In[20]:


print(f"Disparate impact: {bldm1.disparate_impact()}")
print(f"Individual fairness: {bldm1.consistency()}") #Learning Fair Representations
print(f"Statistical parity difference (𝑃𝑟(𝑌=1|𝐷=unprivileged)−𝑃𝑟(𝑌=1|𝐷=privileged)): {bldm1.statistical_parity_difference()}")


# ## Bias in PREDICTED Y's

# In[10]:


#In 
#bldm2 = BinaryLabelDatasetMetric(pred_train_BLD, 
#                                    privileged_groups=[{procted_col_name: favourable_name}], 
#                                    unprivileged_groups=[{procted_col_name: unfavourable_name}])


# In[11]:


#print(f"Disparate impact: {bldm2.disparate_impact()}")
#print(f"Individual fairness: {bldm2.consistency()}") #Learning Fair Representations
#print(f"Statistical parity difference (𝑃𝑟(𝑌=1|𝐷=unprivileged)−𝑃𝑟(𝑌=1|𝐷=privileged)): {bldm2.statistical_parity_difference()}")


# # Disparate impact remove

# In[22]:


from aif360.algorithms.preprocessing import DisparateImpactRemover
di = DisparateImpactRemover(repair_level=1.0)


# ## Create new data set with repaired (level=1)

# Save column names

# In[24]:


#train_BLD.feature_names


# In[25]:


train_BLD.label_names


# In[26]:


all_col_names=train_BLD.feature_names+train_BLD.label_names
print(all_col_names)


# ### Train data:

# In[28]:


rp_train = di.fit_transform(train_BLD) #Using the repaired data
rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=all_col_names) #save as pandas


# In[36]:


sns.histplot(data=df2,x="BirthYear",hue="Gender",stat="density",common_norm=False)
plt.title("Original data (level=1)")
plt.show()


# In[31]:


sns.histplot(data=rp_train_pd,x="BirthYear",hue="Gender",stat="density",common_norm=False)
plt.title("Repaired data")
plt.show()


# In[33]:


all_col_names=train_BLD.feature_names+rp_train.label_names

plt.subplots(2,5, figsize=(20, 20), facecolor='w', edgecolor='k')

level_list=[0.0,0.3,0.5,0.8,1.0]

rows= len(level_list)
cols=1
i=1


for level in level_list:
    
    di = DisparateImpactRemover(repair_level=level)
    
    #Training data
    rp_train = di.fit_transform(train_BLD) #Using the 
    rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=all_col_names)

    #test data
    #rp_test = di.fit_transform(test_BLD)
    #rp_test_pd = pd.DataFrame(np.hstack([rp_test.features,rp_test.labels]),columns=all_col_names)

    plt.subplot(rows,cols,i)
    sns.histplot(data=rp_train_pd,x="BirthYear",hue="Gender",stat="density",common_norm=False,bins=20)
    plt.title(f"Repaired data (Level={level})")
    
    i=i+1
    
plt.show()


# In[ ]:





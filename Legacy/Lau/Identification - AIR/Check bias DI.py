#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[4]:


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
from aif360.metrics import ClassificationMetric


# In[14]:


from utils_copy import *


# # FILE PATH

# In[50]:




### USE ALL DATA
#file_name="fall.csv"
#full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name



###USE SMALL TEST

PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"

file_name="Fall_count_clusterOHE_std.csv"
#file_name="Fall_count_clusterOHE.csv"


full_file_path=PATH_orig+file_name


print("PATH_orig:",PATH_orig)
print("PATH to file:",full_file_path)


# # Specify the y, X and protected variable

# In[51]:


y_col_name="Fall"


X_col_names=['Gender',
'BirthYear',
'LoanPeriod',
'NumberAts',
'Ats_Polstring',
'Ats_Mobilitystokke',
'Ats_Belysning',
'Ats_Underlag',
'Ats_ToiletforhøjereStativ',
'Ats_Signalgivere',
'Ats_EldrevneKørestole',
'Ats_Forstørrelsesglas',
'Ats_Nødalarmsystemer',
'Ats_MobilePersonløftere',
'Ats_TrappelifteMedPlatforme',
'Ats_Badekarsbrætter',
'Ats_Albuestokke',
'Ats_MaterialerOgRedskaberTilAfmærkning',
'Ats_Ryglæn',
'Ats_0',
'Ats_GanghjælpemidlerStøtteTilbehør',
'Ats_Støttebøjler',
'Ats_Lejringspuder',
'Ats_Strømpepåtagere',
'Ats_Dørtrin',
'Ats_Spil',
'Ats_BordePåStole',
'Ats_Drejeskiver',
'Ats_Toiletstole',
'Ats_LøftereStationære',
'Ats_Madmålingshjælpemidler',
'Ats_Fodbeskyttelse',
'Ats_Ståløftere',
'Ats_Stole',
'Ats_Sengeborde',
'Ats_Toiletter',
'Ats_ToiletforhøjereFaste',
'Ats_Påklædning',
'Ats_Brusere',
'Ats_VævsskadeLiggende',
'Ats_Døråbnere',
'Ats_ServeringAfMad',
'Ats_TrappelifteMedSæder',
'Ats_SæderTilMotorkøretøjer',
'Ats_KørestoleManuelleHjælper',
'Ats_Gangbukke',
'Ats_Rollatorer',
'Ats_TryksårsforebyggendeSidde',
'Ats_Fastnettelefoner',
'Ats_Bækkener',
'Ats_Vendehjælpemidler',
'Ats_Sanseintegration',
'Ats_Kørestolsbeskyttere',
'Ats_Arbejdsstole',
'Ats_Løftesejl',
'Ats_KørestoleForbrændingsmotor',
'Ats_Løftestropper',
'Ats_Stiger',
'Ats_TransportTrapper',
'Ats_DrivaggregaterKørestole',
'Ats_Emballageåbnere',
'Ats_ToiletforhøjereLøse',
'Ats_Hårvask',
'Ats_PersonløftereStationære',
'Ats_Madrasser',
'Ats_Vinduesåbnere',
'Ats_Læsestativer',
'Ats_KørestoleManuelleDrivringe',
'Ats_Sædepuder',
'Ats_UdstyrCykler',
'Ats_Karkludsvridere',
'Ats_Vaskeklude',
'Ats_Sengeudstyr',
'Ats_Madlavningshjælpemidler',
'Ats_Skohorn',
'Ats_GribetængerManuelle',
'Ats_Hvilestole',
'Ats_EldrevneKørestoleStyring',
'Ats_BærehjælpemidlerTilKørestole',
'Ats_LøftegalgerSeng',
'Ats_Høreforstærkere',
'Ats_Kalendere',
'Ats_Stokke',
'Ats_Løftegalger',
'Ats_Ure',
'Ats_StøttegrebFlytbare',
'Ats_Forflytningsplatforme',
'Ats_RamperFaste',
'Ats_Rygehjælpemidler',
'Ats_Personvægte',
'Ats_Manøvreringshjælpemidler',
'Ats_Overtøj',
'Ats_Lydoptagelse',
'Ats_Gangborde',
'Ats_Ståstøttestole',
'Ats_RamperMobile',
'Ats_Bærehjælpemidler',
'Ats_Badekarssæder',
'Ats_Siddemodulsystemer',
'Ats_Videosystemer',
'Ats_Siddepuder',
'Ats_Sengeheste',
'Ats_Stolerygge',
'Ats_Rulleborde',
'Ats_Sengeforlængere',
'Ats_Madningsudstyr',
'Ats_Brusestole',
'Ats_Flerpunktsstokke',
'Ats_SengebundeMedMotor',
'Ats_Cykler',
'Ats_CykelenhederKørestole',
'Ats_Stokkeholdere',
'Ats_Toiletarmstøtter',
'Ats_Coxitstole',
'Ats_Toiletsæder',
'Ats_Rebstiger',
'Ats_Forhøjerklodser',
'Cluster_0',
'Cluster_1',
'Cluster_2',
'Cluster_3',
'Cluster_4',
'Cluster_5',
'Cluster_6',
'Cluster_7',
'Cluster_8',
'Cluster_9',
'Cluster_10',
'Cluster_11',
'Cluster_12',
'Cluster_13',
'Cluster_14',
'Cluster_15',
'Cluster_16',
'Cluster_17',
'Cluster_18',
'Cluster_19']


procted_col_name="Gender"


output_col_name="output"


unfavourable_name=0 #women=0
favourable_name=1 #men=1


# In[ ]:





# # Read the data

# In[52]:



df2 = pd.read_csv(full_file_path)


# In[53]:


#sns.histplot(data=df2,x="LoanPeriod",stat="probability",hue="Gender")


# In[54]:


#df_X=df2[X_col_names]
#df_y=df2[[y_col_name]]


# In[55]:


#X=np.array(df_X)
#y=np.array(df_y)


# In[ ]:





#  

#  

# # Identify bias

# In[31]:


print(f"The dataset has {df2.shape[0]} rows")
print(f"The dataset has {df2.shape[1]} cols")

print(f"The dataset to test has the protected attribute: {procted_col_name}")
print(f"The protected variable can be assigned: {df2[procted_col_name].unique()}")

print(f"The ratio of records with y=1: {df2[df2[y_col_name]==1][y_col_name].count()/df2[y_col_name].count()}")


# ## Creating the binary dataset

# See methods from: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.statistical_parity_difference

# Rename trainingset TRUE falls to be named "selected_col". <br>
# Rename trainingset PREDICTED falls to be named "selected_col".

# In[32]:


trainset_renamed=df2.rename(columns={y_col_name: "selected_col"})#.drop([output_col_name], axis=1)
#predicted_trainset_renamed=df2.rename(columns={output_col_name: "selected_col"}).drop([y_col_name], axis=1)


# Needs all columns when using BinaryLabelDataset-class:

# In[33]:


all_cols=X_col_names
all_cols=all_cols.append("selected_col")


# ### Train binary dataset:

# In[35]:


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

# In[36]:


#In 
bldm1 = BinaryLabelDatasetMetric(train_BLD, 
                                    privileged_groups=[{procted_col_name: 1}], 
                                    unprivileged_groups=[{procted_col_name: 0}])


# In[37]:


print(f"Disparate impact: {bldm1.disparate_impact()}")
print(f"Individual fairness: {bldm1.consistency()}") #Learning Fair Representations
print(f"Statistical parity difference (𝑃𝑟(𝑌=1|𝐷=unprivileged)−𝑃𝑟(𝑌=1|𝐷=privileged)): {bldm1.statistical_parity_difference()}")


# ## Bias in PREDICTED Y's

# In[39]:


##In 
#bldm2 = BinaryLabelDatasetMetric(pred_train_BLD, 
#                                    privileged_groups=[{procted_col_name: favourable_name}], 
#                                    unprivileged_groups=[{procted_col_name: unfavourable_name}])


# In[40]:


#print(f"Disparate impact: {bldm2.disparate_impact()}")
#print(f"Individual fairness: {bldm2.consistency()}") #Learning Fair Representations
#print(f"Statistical parity difference (𝑃𝑟(𝑌=1|𝐷=unprivileged)−𝑃𝑟(𝑌=1|𝐷=privileged)): {bldm2.statistical_parity_difference()}")


# # Classification metrics

# A lot of different metrics to use: 

# https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html#aif360.metrics.ClassificationMetric

# In[ ]:





# In[ ]:


CM=ClassificationMetric(train_BLD,pred_train_BLD,privileged_groups=[{procted_col_name: favourable_name}], 
                                    unprivileged_groups=[{procted_col_name: unfavourable_name}])


# In[ ]:


CM.binary_confusion_matrix()


# In[ ]:


print(f"Equal of opportunity diffence: {CM.equal_opportunity_difference()}")


# In[ ]:


CM.disparate_impact()


# In[ ]:





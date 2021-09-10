#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


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


# In[2]:


from utils import *


# # Device

# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # FILE PATH

# In[4]:




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
print("PATH to file:",full_file_path)


# # Specify the y, X and protected variable

# In[5]:


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

# In[6]:



df2 = pd.read_csv(full_file_path)


# In[ ]:





# In[7]:


df_X=df2[X_col_names]
df_y=df2[[y_col_name]]


# In[8]:


X=np.array(df_X)
y=np.array(df_y)


# In[ ]:





# # Specify the model

# In[9]:


#Hyperparameters

n_feat=X.shape[1]
output_dim=1 #binary

n_nodes=500
batch_size=40
epochs=100
p_drop=0.4

optim_type="Adam" #SGD
lr=0.001 #0.001 er godt
wd=0.003 #0.001 er godt


# In[10]:


#which_model="model0/"


# In[11]:


#PATH=PATH_orig+which_model


# In[ ]:





# # Define y's

# In[12]:


#model1 = Network().to(device)
#model1.load_state_dict(torch.load(PATH+'_FFNN_model_local.pth'))
#model1.eval()

#X_numpy=np.array(X)
#X_torch=torch.tensor(X_numpy)
#y_pred = model1(X_torch.float().to(device))
#list_of_output=[round(a.item(),0) for a in y_pred.detach().cpu()]
#y_pred=list_of_output
#y_true=y.flatten()
#gender_list=df_X[procted_col_name]

#y_pred=df2["output"]
#y_true=df2["Fall"]
#gender_list=df2[procted_col_name]
#d = {'y_true': y_true, 'y_pred': y_pred,'Gender':gender_list,}
#data=pd.DataFrame(d
#)


#  

#  

#  

# # Identify bias

# In[13]:


print(f"The dataset has {df2.shape[0]} rows")
print(f"The dataset has {df2.shape[1]} cols")

print(f"The dataset to test has the protected attribute: {procted_col_name}")
print(f"The protected variable can be assigned: {df2[procted_col_name].unique()}")

print(f"The ratio of records with y=1: {df2[df2[y_col_name]==1][y_col_name].count()/df2[y_col_name].count()}")


# In[14]:


metrics=compare_bias_metrics(data=df2,
                        protected_variable_name=procted_col_name,
                        y_target_name=y_col_name,
                        y_pred_name=output_col_name,
                        unfavourable_name=unfavourable_name,
                        favourable_name=favourable_name,
                        print_var=True
                        
                       )


# ## Creating the binary dataset

# See methods from: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.statistical_parity_difference

# Rename trainingset TRUE falls to be named "selected_col". <br>
# Rename trainingset PREDICTED falls to be named "selected_col".

# In[ ]:


trainset_renamed=df2.rename(columns={y_col_name: "selected_col"}).drop([output_col_name], axis=1)
predicted_trainset_renamed=df2.rename(columns={output_col_name: "selected_col"}).drop([y_col_name], axis=1)


# Needs all columns when using BinaryLabelDataset-class:

# In[ ]:


all_cols=X_col_names
all_cols=all_cols.append("selected_col")


# ### Train binary dataset:

# In[ ]:


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

# In[ ]:


#In 
bldm1 = BinaryLabelDatasetMetric(train_BLD, 
                                    privileged_groups=[{procted_col_name: 1}], 
                                    unprivileged_groups=[{procted_col_name: 0}])


# In[ ]:


print(f"Disparate impact: {bldm1.disparate_impact()}")
print(f"Individual fairness: {bldm1.consistency()}") #Learning Fair Representations
print(f"Statistical parity difference (𝑃𝑟(𝑌=1|𝐷=unprivileged)−𝑃𝑟(𝑌=1|𝐷=privileged)): {bldm1.statistical_parity_difference()}")


# ## Bias in PREDICTED Y's

# In[ ]:


#In 
bldm2 = BinaryLabelDatasetMetric(pred_train_BLD, 
                                    privileged_groups=[{procted_col_name: favourable_name}], 
                                    unprivileged_groups=[{procted_col_name: unfavourable_name}])


# In[ ]:


print(f"Disparate impact: {bldm2.disparate_impact()}")
print(f"Individual fairness: {bldm2.consistency()}") #Learning Fair Representations
print(f"Statistical parity difference (𝑃𝑟(𝑌=1|𝐷=unprivileged)−𝑃𝑟(𝑌=1|𝐷=privileged)): {bldm2.statistical_parity_difference()}")


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





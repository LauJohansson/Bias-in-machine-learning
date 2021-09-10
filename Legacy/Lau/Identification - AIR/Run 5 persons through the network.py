#!/usr/bin/env python
# coding: utf-8

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
from sklearn import preprocessing

#from google.colab import drive
from sklearn.model_selection import KFold

from datetime import datetime

import pytz
import random

import os
from sklearn.model_selection import StratifiedKFold


# In[13]:


#PATH="/content/drive/My Drive/Kandidat speciale/500 - Notebooks/models/all races_CV50/"


#### AIR ### 

AIR=True
file_name="Fall_count_clusterOHE_std.csv"
full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name

#titel_mitigation="testAIR"
titel_mitigation="original"
PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/FFNN_ohe/models/"+titel_mitigation+"/"


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
#'Ats_0',
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
#X_col_names = [col for col in X_col_names if col not in leave_out ]

procted_col_name="Gender"


###### COMPASS ####

#AIR=False

#titel_mitigation="testCOMPASS"
#PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/FFNN/models/"+titel_mitigation+"/"

#full_file_path = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'

#y_col_name="is_recid"
#X_col_names=['remember_index','sex','age','race', 'juv_fel_count','juv_misd_count','juv_other_count','priors_count',"c_charge_desc","c_charge_degree"]

#procted_col_name="race"


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


n_nodes=500


batch_size=40
epochs=400
p_drop=0.4

optim_type="Adam" #SGD
lr=0.001 #0.001 er godt
wd=0.003 #0.001 er godt

n_feat=len(X_col_names)
output_dim=1 #binary


# In[15]:




class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fully_connected1 = nn.Sequential(
            nn.Linear(n_feat,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )

        self.fully_connectednew = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )

        self.fully_connectednew1 = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )

        self.fully_connectednew2 = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )
        self.fully_connectednew3 = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )


        self.fully_connected2 = nn.Sequential(
            nn.Linear(n_nodes,output_dim),
            #nn.Softmax(dim = 1)
            nn.Sigmoid()

            )

    def forward(self, x):
      #reshaping x so it becomes flat, except for the first dimension (which is the minibatch)
        #x = x.view(x.size(0),-1)
        x = self.fully_connected1(x)
        x = self.fully_connectednew(x)
        x = self.fully_connectednew1(x)
        x = self.fully_connectednew2(x)
        x = self.fully_connectednew3(x)
        x = self.fully_connected2(x)
        return x


# In[16]:


#from utils_copy import *


# # Get the five persons

# In[17]:


#df=pd.read_csv(full_file_path).drop(columns=["Unnamed: 0"])
#df=df[X_col_names+[y_col_name]]
#df_5_persons=df.sample(6,random_state=7)
#df_5_persons=df_5_persons[X_col_names+[y_col_name]]
#df_5_persons.to_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/6_cit_data.csv")


# In[ ]:





# In[ ]:





# In[18]:


df_5_persons=pd.read_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/6_cit_data.csv")


# In[19]:


df_5_persons_nonstd=pd.read_csv("/restricted/s164512/G2020-57-Aalborg-bias/data_air/Fall_count_clusterOHE.csv").drop(columns=["Unnamed: 0"])
df_5_persons_nonstd=df_5_persons_nonstd.iloc[list(df_5_persons["Unnamed: 0"])]
df_5_persons_nonstd=df_5_persons_nonstd.reset_index(drop=True)
df_5_persons_nonstd.to_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/6_cit_data_nonstd.csv")


# In[20]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[21]:


all_cols=X_col_names+[y_col_name]
all_cols=all_cols+["output"]
all_cols=all_cols+["output_prob"]
all_cols=all_cols+["Model"]


# In[22]:


df_predicted=pd.DataFrame([],columns=all_cols)


# In[23]:




for i in range(50):
    seedName="model"+str(i)
    PATH=PATH_orig+seedName+"/"
    model1 = Network().to(device)
    model1.load_state_dict(torch.load(PATH+'_FFNN_model_local.pth'))
    
    df_predicted_subset=df_5_persons.copy()
    
    X_numpy=np.array(df_predicted_subset[X_col_names])
    X_torch=torch.tensor(X_numpy)
    y_pred = model1(X_torch.float().to(device))
    
    list_of_output=[round(a.item(),0) for a in y_pred.detach().cpu()]
    list_of_output_prob=[a.item() for a in y_pred.detach().cpu()]

    df_predicted_subset["output"]=list_of_output
    df_predicted_subset["output_prob"]=list_of_output_prob
    
    df_predicted_subset["Model"]=seedName
    
    df_predicted_subset=df_predicted_subset.reset_index(drop=True)
    
    df_predicted=pd.concat([df_predicted,df_predicted_subset],axis=0,sort=False)
    
    
#df_predicted=df_predicted.reset_index(drop=True)


# In[24]:


df_predicted.to_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/FFNN_predictions.csv")


# In[160]:





# In[ ]:





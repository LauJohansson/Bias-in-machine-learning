#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PATH="/content/drive/My Drive/Kandidat speciale/500 - Notebooks/models/all races_CV50/"


#### AIR ### 

AIR=True

file_name="Fall_count_clusterOHE_std.csv"

full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name



#titel_mitigation="original"
#titel_mitigation="DroppingD"
#titel_mitigation="Gender Swap"
#titel_mitigation="DI remove"
titel_mitigation="LFR"



PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/FFNN_ohe/models/"+titel_mitigation+"/"

dropping_D=True
gender_swap=False
DI_remove=False
LFR_mitigation=True #Sæt droppingD=True, men ikke fjern den fra X

y_col_name="Fall"
X_col_names=[
'Gender',
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

#titel_mitigation="testCOMPAS"
#PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/FFNN/models/"+titel_mitigation+"/"

#full_file_path = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'

#y_col_name="is_recid"
#X_col_names=['remember_index','sex','age','race', 'juv_fel_count','juv_misd_count','juv_other_count','priors_count',"c_charge_desc","c_charge_degree"]

#procted_col_name="race"


# In[2]:


def LFR_custom(df_train,y_train,lfr=None):
    from aif360.algorithms.preprocessing import LFR
    from aif360.datasets import BinaryLabelDataset
    
    df_train=pd.concat([df_train,y_train],axis=1)
    
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
    #rp_df_pd = pd.concat([rp_df_pd,df2_gender],axis=1)

    ##########
    
    
    #Concatenate the non-numerical columns
    transformed_data = pd.concat ([rp_df_pd,df2_all], axis=1)
    
    
    transformed_data=transformed_data.drop(columns=["Fall"])
    
    return transformed_data,lfr


# In[3]:


def DI_remove_custom(df_train,RP_level=1.0):
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    from aif360.datasets import BinaryLabelDataset
    X_col_names_f=['Gender', 'BirthYear', 'LoanPeriod', 'NumberAts']
    df2_all=df_train.drop(columns=X_col_names_f).copy() #Gemmer alle kolonner, undtagen numerical og gender
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


# In[4]:


n_nodes=500


batch_size=40
epochs=400
p_drop=0.4

optim_type="Adam" #SGD
lr=0.001 #0.001 er godt
wd=0.05       




#0.01 er godt til AIR ny!!


# In[5]:


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


# In[6]:


plt.plot([0,0])


# In[7]:


from utils_Copy import *


# In[8]:


def loss_fn(target,predictions):
    criterion = nn.BCELoss()
    loss_out = criterion(predictions, target)
    return loss_out


# In[9]:



def accuracy(true,pred):
    acc = (true.float().round() == pred.float().round()).float().detach().cpu().numpy()
    return float(100 * acc.sum() / len(acc))

def get_test():
    avg_loss_ts = 0
    avg_acc_ts=0
    model.eval()  # train mode
    for X_batch, Y_batch in data_ts:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)


        # forward
        Y_pred = model(X_batch.float()) 
        loss = loss_fn(Y_batch.float(), Y_pred.squeeze()) 

        # calculate metrics to show the user
        avg_loss_ts += loss / len(data_ts)
        avg_acc_ts+=accuracy(Y_batch,Y_pred.squeeze()) / len(data_ts)
    #toc = time()

    return avg_loss_ts, avg_acc_ts

def get_all_time_low(all_time,new_val):
    if all_time>new_val:
        return new_val
    else:
        return all_time
    
def get_all_time_high(all_time,new_val):
    if all_time<new_val:
        return new_val
    else:
        return all_time


# In[10]:




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


# In[11]:


def custom_create_indexes(df,n,seed,strat=False,y_col=None):
    list_of_index=[]
    
    
    
    if strat==False:
        kf=KFold(n_splits=n, random_state=seed, shuffle=True)
        
        
        for train_index, test_index in kf.split(df):
            list_of_index.append(test_index)

        tr_val_ts_indexes=[
        #[[train],[validate],[test]]

        [[*list_of_index[0],*list_of_index[1],*list_of_index[2]],list_of_index[3],list_of_index[4]],
        [[*list_of_index[4],*list_of_index[0],*list_of_index[1]],list_of_index[2],list_of_index[3]],
        [[*list_of_index[3],*list_of_index[4],*list_of_index[0]],list_of_index[1],list_of_index[2]],
        [[*list_of_index[2],*list_of_index[3],*list_of_index[4]],list_of_index[0],list_of_index[1]],
        [[*list_of_index[1],*list_of_index[2],*list_of_index[3]],list_of_index[4],list_of_index[0]],

        ]
    
    
    
    
    else:
        kf = StratifiedKFold(n_splits=n, random_state=seed, shuffle=True)
    
    
        for train_index, test_index in kf.split(df,df[y_col]):
            list_of_index.append(test_index)

        tr_val_ts_indexes=[
        #[[train],[validate],[test]]

        [[*list_of_index[0],*list_of_index[1],*list_of_index[2]],list_of_index[3],list_of_index[4]],
        [[*list_of_index[4],*list_of_index[0],*list_of_index[1]],list_of_index[2],list_of_index[3]],
        [[*list_of_index[3],*list_of_index[4],*list_of_index[0]],list_of_index[1],list_of_index[2]],
        [[*list_of_index[2],*list_of_index[3],*list_of_index[4]],list_of_index[0],list_of_index[1]],
        [[*list_of_index[1],*list_of_index[2],*list_of_index[3]],list_of_index[4],list_of_index[0]],

        ]
    
    return tr_val_ts_indexes


# In[30]:


modelcounter=0
for custom_seed in range(1,11):

    

    torch.manual_seed(custom_seed)
    random.seed(custom_seed)
    np.random.seed(custom_seed)
    

    df2 = pd.read_csv(full_file_path)



   
    X=df2[X_col_names]

    if dropping_D==True:
        y=df2[[y_col_name,procted_col_name]]
    else:
        y=df2[[y_col_name]]

    #https://stackoverflow.com/questions/11587782/creating-dummy-variables-in-pandas-for-python

    if AIR==False:
        just_dummies=pd.get_dummies(X[['sex',"race","c_charge_desc","c_charge_degree"]])
        X = pd.concat([X, just_dummies], axis=1) 
        X=X.drop(['sex',"race","c_charge_desc","c_charge_degree"] ,axis=1)


    
    tr_val_ts_indexes= custom_create_indexes(X,5,custom_seed)
    #tr_val_ts_indexes= custom_create_indexes(df2,5,custom_seed,True,y_col_name) #stratify

    #i=0
    for mini_loop in range(len(tr_val_ts_indexes)):
        print("Running overall number "+str(modelcounter))
         
        
        X_train_pd, y_train_pd = X.iloc[tr_val_ts_indexes[mini_loop][0]], y.iloc[tr_val_ts_indexes[mini_loop][0]]
        X_val_pd, y_val_pd = X.iloc[tr_val_ts_indexes[mini_loop][1]], y.iloc[tr_val_ts_indexes[mini_loop][1]]
        X_test_pd, y_test_pd = X.iloc[tr_val_ts_indexes[mini_loop][2]], y.iloc[tr_val_ts_indexes[mini_loop][2]]
        
        
        seedName="model"+str(modelcounter)

        PATH=PATH_orig+seedName+"/"
        print(PATH)

        #Make dir to files
        if not os.path.exists(PATH):
            os.makedirs(PATH)
            print("Created new path!: ",PATH)
        
        
        if gender_swap==True:
            X_train_pd_copy=X_train_pd.copy()
            y_train_pd_copy=y_train_pd.copy()
            
            X_train_pd_copy["Gender"]=(X_train_pd_copy["Gender"]-1)*(-1)
            
            X_train_pd=pd.concat([X_train_pd,X_train_pd_copy])
            
            y_train_pd=pd.concat([y_train_pd,y_train_pd_copy])
            
            
            
            
        if DI_remove==True:
            X_train_pd=DI_remove_custom(X_train_pd.reset_index(drop=True))
            X_val_pd=DI_remove_custom(X_val_pd.reset_index(drop=True))
            X_test_pd=DI_remove_custom(X_test_pd.reset_index(drop=True))
            
            y_train_pd=y_train_pd.reset_index(drop=True)
            
            y_val_pd=y_val_pd.reset_index(drop=True)
            
            y_test_pd=y_test_pd.reset_index(drop=True)
            
            
        if LFR_mitigation==True:
            
            
            X_train_pd,lfr=LFR_custom(X_train_pd.reset_index(drop=True),
                                  y_train_pd.reset_index(drop=True).drop(columns=["Gender"]),
                                 lfr=None
                                 )
            X_val_pd,lfr=LFR_custom(X_val_pd.reset_index(drop=True),
                                  y_val_pd.reset_index(drop=True).drop(columns=["Gender"]),
                                 lfr=lfr
                                 )
            X_test_pd,lfr=LFR_custom(X_test_pd.reset_index(drop=True),
                                  y_test_pd.reset_index(drop=True).drop(columns=["Gender"]),
                                 lfr=lfr
                                 )
            
            y_train_pd=y_train_pd.reset_index(drop=True)
            
            y_val_pd=y_val_pd.reset_index(drop=True)
            
            y_test_pd=y_test_pd.reset_index(drop=True)
            
            
            
        X_train, y_train = X_train_pd, y_train_pd
        X_val, y_val = X_val_pd, y_val_pd
        X_test, y_test = X_test_pd, y_test_pd


        #Save as numpy array for the DATALOADER (PyTorch)
        
        if LFR_mitigation==True:
            temp_col_name_LFR=[name for name in X_col_names if name not in ["Gender"]]
            X_train=np.array(X_train[temp_col_name_LFR])
            y_train=np.array(y_train[y_col_name])

            X_val=np.array(X_val[temp_col_name_LFR])
            y_val=np.array(y_val[y_col_name])

            X_test=np.array(X_test[temp_col_name_LFR])
            y_test=np.array(y_test[y_col_name])
        
        
        else:
            
            X_train=np.array(X_train[X_col_names])
            y_train=np.array(y_train[y_col_name])

            X_val=np.array(X_val[X_col_names])
            y_val=np.array(y_val[y_col_name])

            X_test=np.array(X_test[X_col_names])
            y_test=np.array(y_test[y_col_name])


       # print("X_train shape: {}".format(X_train.shape))
       # print("y_train shape: {}".format(y_train.shape))

        #print("X_val shape: {}".format(X_val.shape))
        #print("y_val shape: {}".format(y_val.shape))

        #print("X_test shape: {}".format(X_test.shape))
        #print("y_test shape: {}".format(y_test.shape))




        #n_feat=X_train.shape[1]
        
        if LFR_mitigation==True:
            n_feat=len(X_col_names)-1#minus gender
        
        else:
            n_feat=len(X_col_names)
        
        
        output_dim=1 #binary


        data_tr = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=False)
        data_val = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False)
        data_ts = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)




        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device="cpu"
        print(device)


        model = Network().to(device)



        opt=optim.Adam(model.parameters(),lr=lr, weight_decay = wd)
        

        epochnumber = []
        all_train_losses = []
        all_val_losses = []
        all_ts_losses = []

        all_train_acc=[]
        all_val_acc=[]
        all_ts_acc=[]

        all_time_low_train_loss=1000
        all_time_low_val_loss=1000

        all_time_high_train_acc=0
        all_time_high_val_acc=0


        for epoch in range(epochs):
            if (epoch)%20==0:
                print('* Epoch %d/%d' % (epoch+1, epochs))

            epochnumber.append(epoch)

            avg_loss_train = 0
            avg_acc=0
            model.train()  # train mode
            for X_batch, Y_batch in data_tr:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)



                # set parameter gradients to zero
                opt.zero_grad()

                # forward
                Y_pred = model(X_batch.float()) #oprdindeligt havde vi 3 lag (RGB), nu har vi kun 1 (greyscale) -> 
                loss = loss_fn(Y_batch.float(), Y_pred.squeeze())  # forward-pass
                loss.backward()  # backward-pass
                opt.step()  # update weights

                # calculate metrics to show the user
                avg_loss_train += loss / len(data_tr)

                avg_acc+=accuracy(Y_batch,Y_pred.squeeze()) / len(data_tr)


            all_time_low_train_loss=get_all_time_low(all_time_low_train_loss,avg_loss_train)
            all_time_low_train_acc=get_all_time_high(all_time_high_train_acc,avg_acc)
              #print(' - train loss: %f' % avg_loss_train)
              #print(' - train acc: {} %'.format(round(avg_acc,2)))

            all_train_losses.append(avg_loss_train)
            all_train_acc.append(avg_acc)



            with torch.no_grad():
                avg_loss_val = 0
                avg_acc_val=0
                model.eval()  # eval mode
                for X_batch, Y_batch in data_val:
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)


                    # forward
                    Y_pred = model(X_batch.float()) 
                    loss = loss_fn(Y_batch.float(), Y_pred.squeeze())  # forward-pass

                    # calculate metrics to show the user
                    avg_loss_val += loss / len(data_val)
                    avg_acc_val+=accuracy(Y_batch,Y_pred.squeeze()) / len(data_val)
                #toc = time()
                all_time_low_val_loss=get_all_time_low(all_time_low_val_loss,avg_loss_val)
                all_time_low_val_acc=get_all_time_high(all_time_high_val_acc,avg_acc_val)
                #print(' - val loss: %f' % avg_loss_val)
                #print(' - val acc: {} %'.format(round(avg_acc_val,2)))


                ########Save model####

            if  epoch == 0 or avg_loss_val <= min(all_val_losses) :
                torch.save(model.state_dict(), PATH+'_FFNN_model_local.pth')
                print('####Saved model####')

            all_val_losses.append(avg_loss_val)
            all_val_acc.append(avg_acc_val)




          ###PLOT########

        if epoch==epochs-1:
            #Save the last epoch
            torch.save(model.state_dict(), PATH+'_FFNN_model_global.pth')

            #take the best model (with lowest validation loss)
            model.load_state_dict(torch.load(PATH+'_FFNN_model_local.pth'))
            model.eval()

            all_ts_losses=[get_test()[0]] * (epoch+1)
            all_ts_acc=[get_test()[1]] * (epoch+1)

            plt.figure(1)
            plt.plot(epochnumber, all_train_losses, 'r', epochnumber, all_val_losses, 'b',epochnumber, all_ts_losses, '--')
            plt.xlabel('Epochs'), plt.ylabel('Loss')
            plt.legend(['Train Loss', 'Val Loss','Test loss'])
            plt.savefig(PATH+'_loss.png')
            plt.show()

            plt.figure(2)
            plt.plot(epochnumber, all_train_acc, 'black', epochnumber, all_val_acc, 'grey',epochnumber, all_ts_acc, '--')
            plt.xlabel('Epochs'), plt.ylabel('Accuracy')
            plt.legend(['Train acc', 'Val acc','Test acc'])
            plt.savefig(PATH+'_acc.png')
            plt.show()



            metrics=pd.DataFrame({"all_time_low_train_loss":[all_time_low_train_loss.item()],
                                  "all_time_low_train_acc":[all_time_low_train_acc],
                              "all_time_low_val_loss":[all_time_low_val_loss.item()],
                                  "all_time_val_train_acc":[all_time_low_val_acc],
                              "test_acc":[all_ts_acc[0]],
                              "test_loss":[all_ts_losses[0].item()]
                                                })
            metrics.to_csv( PATH+'_metrics.csv')





        for local_best in [0,1]:
            #local_best=0

            model1 = Network().to(device)
            if local_best==1:
                model1.load_state_dict(torch.load(PATH+'_FFNN_model_local.pth'))
            else:
                model1.load_state_dict(torch.load(PATH+'_FFNN_model_global.pth'))

            model1.eval()


            df_evaluate = X_test_pd.copy()
            df_evaluate[y_col_name]=y_test_pd[y_col_name]
            
            if dropping_D==True:
                df_evaluate[procted_col_name]=y_test_pd[procted_col_name]
            
            else:
                df_evaluate[procted_col_name]=X_test_pd[procted_col_name]

            if AIR==False:
                cols= [col for col in list(df_evaluate.columns) if col not in [y_col_name,"sex","age_cat","race","c_charge_desc","c_charge_degree"]]
            else:
                if dropping_D==True:
                    cols= [col for col in list(df_evaluate.columns) if col not in [y_col_name,procted_col_name]]
                    
                elif gender_swap==True:
                    cols= [col for col in list(df_evaluate.columns) if col not in [y_col_name,"Original"]]
                else:
                    cols= [col for col in list(df_evaluate.columns) if col not in [y_col_name]]
                    



            X_numpy=np.array(df_evaluate[cols])
            X_torch=torch.tensor(X_numpy)
            y_pred = model1(X_torch.float().to(device))


            list_of_output=[round(a.item(),0) for a in y_pred.detach().cpu()]
            list_of_output_prob=[a.item() for a in y_pred.detach().cpu()]

            df_evaluate["output"]=list_of_output
            df_evaluate["output_prob"]=list_of_output_prob
            df_evaluate["Model"]=seedName


            ##SAVING THE TEST DATA
            if local_best==1:
                df_evaluate.to_csv(PATH+"test_data_localmodel.csv")
            else:
                df_evaluate.to_csv(PATH+"test_data_globalmodel.csv")
            

        modelcounter=modelcounter+1


# # Save all test data (and output)

# In[32]:




for file_name in ["localmodel","globalmodel"]:
    first_time=True
    
    for j in range(modelcounter):
        
        if first_time==True:
            test_data_all = pd.read_csv(PATH_orig+"model"+str(j)+"/test_data_"+file_name+".csv")
            first_time=False
        else:
            test_subset=pd.read_csv(PATH_orig+"model"+str(j)+"/test_data_"+file_name+".csv")
            test_data_all=pd.concat([test_data_all,test_subset],sort=False,axis=0)
       

       

    test_data_all.to_csv(PATH_orig+"all_test_data_"+file_name+".csv")
    print(f"the shape of {file_name} is {test_data_all.shape}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#  # GLOBAL ALL

# In[ ]:


#column_names = ["Group", "ML", "Measure","Value"]

#df_out = pd.DataFrame(columns = column_names)

#for i in range(50):
  
#    PATH_loop=PATH_orig+"model"+str(i)+"/_all_stats_global.csv"
  
#    data=pd.read_csv(PATH_loop)
#    for group in ["all"]:
#        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP']:
#            value=float(data[data[procted_col_name]==group][measure])

#            df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

#df_out.to_csv(PATH_orig+"/FFNN_metrics_crossvalidated_global_all.csv")


# In[ ]:


#global_all_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
#global_all_bar.set_title('Global all')
#global_all_bar.get_figure().savefig(PATH_orig+"/barplot_global_all.png")


# # LOCAL ALL

# In[ ]:


#column_names = ["Group", "ML", "Measure","Value"]

#df_out = pd.DataFrame(columns = column_names)

#for i in range(50):
#
#    PATH_loop=PATH_orig+"model"+str(i)+"/_all_stats_local.csv"
#  
#    data=pd.read_csv(PATH_loop)
#    for group in ["all"]:
#        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP',"y_hat_mean","y_target_mean"]:
#            value=float(data[data[procted_col_name]==group][measure])

#            df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

#df_out.to_csv(PATH_orig+"FFNN_metrics_crossvalidated_local_all.csv")


# In[ ]:


#local_all_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
#local_all_bar.set_title('Global all')
#local_all_bar.get_figure().savefig(PATH_orig+"/barplot_local_all.png")


# # Global protected

# In[ ]:


#column_names = ["Group", "ML", "Measure","Value"]

#df_out = pd.DataFrame(columns = column_names)

#for i in range(50):

#    PATH_loop=PATH_orig+"model"+str(i)+"/_"+procted_col_name+"_stats_global.csv"
  
#    data=pd.read_csv(PATH_loop)
#    for group in list(data[procted_col_name].unique()):
#        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP',"y_hat_mean","y_target_mean"]:
#            value=float(data[data[procted_col_name]==group][measure])

 #           df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

#df_out.to_csv(PATH_orig+"FFNN_metrics_crossvalidated_global_"+procted_col_name+".csv")


# In[ ]:


#global_proc_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
#global_proc_bar.set_title('Global proctected: '+procted_col_name)
#global_proc_bar.get_figure().savefig(PATH_orig+"/barplot_global_proc.png")


# # Local protected

# In[ ]:


#column_names = ["Group", "ML", "Measure","Value"]

#df_out = pd.DataFrame(columns = column_names)

#for i in range(50):
#    PATH_loop=PATH_orig+"model"+str(i)+"/_"+procted_col_name+"_stats_local.csv"
  
#    data=pd.read_csv(PATH_loop)
#    for group in list(data[procted_col_name].unique()):
#        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP',"y_hat_mean","y_target_mean"]:
#            value=float(data[data[procted_col_name]==group][measure])

#            df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

#df_out.to_csv(PATH_orig+"FFNN_metrics_crossvalidated_local_"+procted_col_name+".csv")


# In[ ]:


#local_proc_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
#local_proc_bar.set_title('Local protected: '+procted_col_name)
#local_proc_bar.get_figure().savefig(PATH_orig+"/barplot_local_proc.png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



'''
for file_name in ["localmodel","globalmodel"]:
    
    for j in range(modelcounter):
    
        test_data_0 = pd.read_csv(PATH_orig+"model0/test_data_"+file_name+".csv")
        test_data_1 = pd.read_csv(PATH_orig+"model1/test_data_"+file_name+".csv")
        test_data_2 = pd.read_csv(PATH_orig+"model2/test_data_"+file_name+".csv")
        test_data_3 = pd.read_csv(PATH_orig+"model3/test_data_"+file_name+".csv")
        test_data_4 = pd.read_csv(PATH_orig+"model4/test_data_"+file_name+".csv")
        test_data_5 = pd.read_csv(PATH_orig+"model5/test_data_"+file_name+".csv")
        test_data_6 = pd.read_csv(PATH_orig+"model6/test_data_"+file_name+".csv")
        test_data_7 = pd.read_csv(PATH_orig+"model7/test_data_"+file_name+".csv")
        test_data_8 = pd.read_csv(PATH_orig+"model8/test_data_"+file_name+".csv")
        test_data_9 = pd.read_csv(PATH_orig+"model9/test_data_"+file_name+".csv")

        df2=    pd.concat([test_data_0,
                            test_data_1,
                            test_data_2,
                            test_data_3,
                            test_data_4,
                            test_data_5,
                            test_data_6,
                            test_data_7,
                            test_data_8,
                            test_data_9
                           ],sort=False,axis=0)

    df2.to_csv(PATH_orig+"all_test_data_"+file_name+".csv")
'''


# In[ ]:





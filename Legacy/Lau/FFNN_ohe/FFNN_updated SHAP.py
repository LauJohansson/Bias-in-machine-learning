#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PATH="/content/drive/My Drive/Kandidat speciale/500 - Notebooks/models/all races_CV50/"


#### AIR ### 

AIR=True

file_name="Fall_count_clusterOHE_std.csv"

full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name



#titel_mitigation="SHAPoriginal"
titel_mitigation="original"




PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/FFNN_ohe/models/"+titel_mitigation+"/"

dropping_D=False
gender_swap=False
DI_remove=False
LFR_mitigation=False #Sæt droppingD=True, men ikke fjern den fra X

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


def create_shape(modelname,model,xcolnames,X_train,X_test,directory,modelnr,plot_type="bar"):
    import shap
   
    if modelname.lower()=="xgboost":
        print("Using treeexplainer")
        shap_explainer = shap.TreeExplainer(model, data=X_train)
        shap_values = shap_explainer.shap_values(X_test)
        
    elif modelname.lower()=="ffnn":
        print("Using deepexplainer")
        shap_explainer = shap.DeepExplainer(model, data=X_train)
        shap_values = shap_explainer.shap_values(X_test)
    else:
        raise Exception("Lau says: Sorry, cant find the model")

    
    feature_names=xcolnames

    
    #Dette er Christians måde at hente values fra SHAP
    importance_df  = pd.DataFrame()
    importance_df['feature'] = feature_names
    importance_df['shap_values'] = np.around(np.array(shap_values)[:,:].mean(0), decimals=3)
    importance_df['shap_values_abs'] = np.around(abs(np.array(shap_values)[:,:]).mean(0), decimals=3)
    
    
    if modelname.lower()=="xgboost":
        importance_df['feat_imp'] = np.around(model.feature_importances_, decimals=3)
    feat_importance_df_shap = importance_df.groupby('feature').mean().sort_values('shap_values',
                                                                                   ascending=False)
    feat_importance_df_shap = feat_importance_df_shap.reset_index()
    

   

    feat_importance_df_shap.to_csv(directory+modelname+f"best features model "+str(modelnr)+".csv")
    
    
    ##ALL VALUES###
    
    importance_df_all  = pd.DataFrame(shap_values,columns=feature_names)
    importance_df_all.to_csv(directory+modelname+f"best features model "+str(modelnr)+"_all.csv")
    
    
    

    file_name_sum = "shap_summary"
    file_name_exp = "shap_row_0"
  
    
    plt.close()
    shap.summary_plot(shap_values,
                      X_test,
                      feature_names=feature_names,
                      plot_type=plot_type,
                      show=False)
    
    plt.savefig(directory+"/barplots/"+modelname+"shap_plot model"+str(modelnr)+".png",
                bbox_inches = "tight")
    
    plt.close()
    shap.summary_plot(shap_values,
                      X_test,
                      feature_names=feature_names,
                      
                      show=False)
    
    plt.savefig(directory+"/beeplots/"+modelname+"shap_plot beeswarm model"+str(modelnr)+".png",
                bbox_inches = "tight")

    
    


# In[5]:


n_nodes=500


batch_size=40
epochs=10
p_drop=0.4

optim_type="Adam" #SGD
lr=0.001 #0.001 er godt
wd=0.05       




#0.01 er godt til AIR ny!!


# In[6]:


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


# In[12]:


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




 
        model1 = Network().to(device)

        model1.load_state_dict(torch.load(PATH+'_FFNN_model_local.pth'))


        model1.eval()


        X_train_shap=np.array(X_train)
        X_train_shap=torch.tensor(X_train_shap).to(device).float()

        X_test_shap=np.array(X_test)
        X_test_shap=torch.tensor(X_test_shap).to(device).float()

        create_shape("FFNN",model1,X_col_names,X_train_shap,X_test_shap,"/restricted/s164512/G2020-57-Aalborg-bias/SHAP/",modelcounter)
        print(f"Shap nr {modelcounter} is created ")

        modelcounter=modelcounter+1


# In[ ]:





# In[ ]:





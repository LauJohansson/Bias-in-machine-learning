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


# In[2]:


def create_shape(modelname,model,xcolnames,X_train,X_test,directory,modelnr,plot_type="bar"):
    import shap
   
    if modelname.lower()=="xgboost":
        shap_explainer = shap.TreeExplainer(model, data=X_train)
        shap_values = shap_explainer.shap_values(X_test)
        
    elif modelname.lower()=="ffnn":
        shap_explainer = shap.DeepExplainer(model, data=X_train)
        shap_values = shap_explainer.shap_values(X_test)
    else:
        raise Exception("Lau says: Sorry, cant find the model")

    
    feature_names=xcolnames

    
    #Dette er Christians måde at hente values fra SHAP
    importance_df  = pd.DataFrame()
    importance_df['feature'] = feature_names
    importance_df['shap_values'] = np.around(abs(np.array(shap_values)[:,:]).mean(0), decimals=3)
    
    if modelname.lower()=="xgboost":
        importance_df['feat_imp'] = np.around(model.feature_importances_, decimals=3)
    feat_importance_df_shap = importance_df.groupby('feature').mean().sort_values('shap_values',
                                                                                   ascending=False)
    feat_importance_df_shap = feat_importance_df_shap.reset_index()
   

    feat_importance_df_shap.to_csv(directory+modelname+f"best features model "+str(modelnr)+".csv")

    file_name_sum = "shap_summary"
    file_name_exp = "shap_row_0"
  
    
    plt.close()
    shap.summary_plot(shap_values,
                      X_test,
                      feature_names=feature_names,
                      plot_type=plot_type,
                      show=False)
    
    plt.savefig(directory+modelname+"shap_plot model"+str(modelnr)+".png",
                bbox_inches = "tight")
    
    #plt.close()
    #shap.plots.beeswarm(shap_values,
     #                )
    
    #plt.savefig(directory+modelname+"shap_plot beeswarm model"+str(modelnr)+".png",
    #            bbox_inches = "tight")

    
    


# In[3]:


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


# # GET SHAP mean aggregated

# In[4]:


SHAP_path="/restricted/s164512/G2020-57-Aalborg-bias/SHAP/"


# In[5]:


shap_frame=pd.DataFrame([],columns=["model","feature","shap_values","shap_values_abs"])


# In[6]:



for mname in ["svm","lr","rf","ffnn","xgboost"]:
    for i in range(50):
        PATH_loop=SHAP_path+mname+"best features model "+str(i)+".csv"

        data=pd.read_csv(PATH_loop)

        data=data[["feature","shap_values","shap_values_abs"]]
        data["model"]=mname
        
        shap_frame=shap_frame.append(data,ignore_index=True)


# In[ ]:





# In[7]:


shap_frame.loc[shap_frame.feature == "BirthYear", "feature"] = "Age"
shap_frame.loc[(shap_frame.feature == "BirthYear"), "shap_values"] = shap_frame.loc[(shap_frame.feature == "BirthYear"), "shap_values"]*(-1)


# In[14]:


model_list=["svm","lr","rf","ffnn","xgboost"]
fig,ax=plt.subplots(len(model_list),1,figsize=(10,10))
ax = ax.ravel()

top_n=10
i=0
for mm in model_list:
    shap_frame_sub=shap_frame[shap_frame["model"]==mm]

    means=pd.DataFrame(shap_frame_sub[["feature","shap_values_abs"]].groupby(["feature"]).mean()).reset_index()
    top20names_abs=list(means.sort_values("shap_values_abs",ascending=False).head(top_n)["feature"])
    #top20names_abs=list(np.sort(top20names_abs))

    means=pd.DataFrame(shap_frame_sub[["feature","shap_values"]].groupby(["feature"]).mean()).reset_index()
    top20names=list(means.sort_values("shap_values",ascending=False).head(top_n)["feature"])
    #top20names=list(np.sort(top20names))

    data=shap_frame_sub[shap_frame_sub["feature"].isin(top20names_abs)]
    sns.barplot(data=data,ax=ax[i],x="shap_values_abs",y="feature",color="blue",order=top20names_abs)
    ax[i].set_title(mm)
    i=i+1
plt.show()
 


# In[21]:


top_n=20
mm="lr"
shap_frame_sub=shap_frame[shap_frame["model"]==mm]

means=pd.DataFrame(shap_frame_sub[["feature","shap_values_abs"]].groupby(["feature"]).mean()).reset_index()
top20names_abs=list(means.sort_values("shap_values_abs",ascending=False).head(top_n)["feature"])
#top20names_abs=list(np.sort(top20names_abs))

means=pd.DataFrame(shap_frame_sub[["feature","shap_values"]].groupby(["feature"]).mean()).reset_index()
top20names=list(means.sort_values("shap_values",ascending=False).head(top_n)["feature"])
#top20names=list(np.sort(top20names))

data=shap_frame_sub[shap_frame_sub["feature"].isin(top20names_abs)]
sns.barplot(data=data,x="shap_values_abs",y="feature",color="blue",order=top20names_abs)


# In[22]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(15,15))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

top_n=15

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


model_list=["svm","lr","rf","ffnn","xgboost"]


#for i,v in enumerate(all_data_gender["Model"].unique()):
for i,mm in enumerate(model_list):
    
    shap_frame_sub=shap_frame[shap_frame["model"]==mm]

    means=pd.DataFrame(shap_frame_sub[["feature","shap_values_abs"]].groupby(["feature"]).mean()).reset_index()
    top20names_abs=list(means.sort_values("shap_values_abs",ascending=False).head(top_n)["feature"])
    #top20names_abs=list(np.sort(top20names_abs))

    means=pd.DataFrame(shap_frame_sub[["feature","shap_values"]].groupby(["feature"]).mean()).reset_index()
    top20names=list(means.sort_values("shap_values",ascending=False).head(top_n)["feature"])
    #top20names=list(np.sort(top20names))

    data=shap_frame_sub[shap_frame_sub["feature"].isin(top20names_abs)]
    sns.barplot(data=data,ax=ax[i],x="shap_values_abs",y="feature",color="cornflowerblue",order=top20names_abs)
    
    
    
    
    
    
    if mm=="svm":
        ax[i].set_title("SVM")
    elif mm=="lr":
        ax[i].set_title("LR")
    elif mm=="rf":
        ax[i].set_title("RF")
    elif mm=="ffnn":
        ax[i].set_title("FFNN")
    elif mm=="xgboost":
        ax[i].set_title("XGBoost")
    
    
    
    #ax[i].legend(title="Gender")
    #ax[i].legend( loc="upper right")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set_ylabel("Feature",fontsize=20)
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        #ax[i].tick_params( labelleft=False)
    if i==2:
        ax[i].set_ylabel("Feature",fontsize=20)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Absolute Shap Value",fontsize=20)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Absolute Shap Value",fontsize=20)
        #ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('Feature',fontsize=20)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Absolute Shap value",fontsize=20)
    

#fig.delaxes(gs[0, 1])<
#plt.savefig(f"/restricted/s164512/G2020-57-Aalborg-bias/Plots/{folder_name}/{folder_name}_gender_metrics_allmodels", bbox_inches = 'tight')
plt.show()


# In[10]:


'''
fig,ax=plt.subplots(2,1,figsize=(10,10))
ax = ax.ravel()

i=0
for mm in ["xgboost","ffnn"]:
    shap_frame_sub=shap_frame[shap_frame["model"]==mm]

    means=pd.DataFrame(shap_frame_sub[["feature","shap_values_abs"]].groupby(["feature"]).mean()).reset_index()
    top20names_abs=list(means.sort_values("shap_values_abs",ascending=False).head(10)["feature"])
    #top20names_abs=list(np.sort(top20names_abs))

    means=pd.DataFrame(shap_frame_sub[["feature","shap_values"]].groupby(["feature"]).mean()).reset_index()
    top20names=list(means.sort_values("shap_values",ascending=False).head(10)["feature"])
    #top20names=list(np.sort(top20names))

    
 
    data=shap_frame_sub[shap_frame_sub["feature"].isin(top20names_abs)].copy()
    data['sign'] = data['shap_values'].apply(lambda x: "negative" if x<0 else "positive")
    sns.barplot(data=data,ax=ax[i],x="shap_values",y="feature",hue="sign",order=top20names_abs,palette={"negative":"red","positive":"green"})
    ax[i].set_title(mm)
    i=i+1
plt.show()
    
'''


# In[ ]:





# # ALL data

# ## This section takes all the SHAP data. Not the mean aggregated!

# In[11]:


shap_frame_all=pd.DataFrame([],columns=X_col_names+["model"])


# In[12]:


#
model_list=["svm","lr","rf","ffnn","xgboost"]
for mname in model_list:
    for i in range(50):
        PATH_loop=SHAP_path+mname+"best features model "+str(i)+"_all.csv"

        data=pd.read_csv(PATH_loop).drop(columns=["Unnamed: 0"])

        data=data[X_col_names]
        data["model"]=mname
        
        shap_frame_all=shap_frame_all.append(data,ignore_index=True)


# In[13]:


shap_frame_all=shap_frame_all.melt(id_vars=["model"],var_name="feature",value_name="shap_values")


# In[14]:


shap_frame_all.loc[shap_frame_all.feature == "BirthYear", "feature"] = "Age"
shap_frame_all.loc[(shap_frame_all.feature == "BirthYear"), "shap_values"] = shap_frame_all.loc[(shap_frame_all.feature == "BirthYear"), "shap_values"]*(-1)


# ### Best plot

# In[15]:


'''
model_list=["svm","lr","rf","ffnn","xgboost"]
fig,ax=plt.subplots(len(model_list),1,figsize=(10,30))
ax = ax.ravel()

i=0
for mm in model_list:
    shap_frame_sub_all=shap_frame_all[shap_frame_all["model"]==mm].copy()
    
    shap_frame_sub_all["shap_values_abs"]=shap_frame_sub_all["shap_values"].apply(lambda x: abs(x))

    means=pd.DataFrame(shap_frame_sub_all[["feature","shap_values"]].groupby(["feature"]).mean()).reset_index()
    
    top5names_positive=list(means.sort_values("shap_values",ascending=False).head(10)["feature"])

    top5names_negatives=list(means.sort_values("shap_values",ascending=True).head(10)["feature"])
    
    all_features_names=top5names_positive+top5names_negatives
    
    color_list=["lightgreen"]*10+["salmon"]*10

    data=shap_frame_sub_all[shap_frame_sub_all["feature"].isin(all_features_names)].copy()
    sns.barplot(data=data,ax=ax[i],x="shap_values",y="feature",order=all_features_names,palette=color_list)
    ax[i].set_title(mm)
    i=i+1
plt.show()
    
'''


# In[16]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(15,15))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


list_of_models=["svm","lr","rf","ffnn","xgboost"]


#for i,v in enumerate(all_data_gender["Model"].unique()):
for i,mm in enumerate(list_of_models):
    
    shap_frame_sub_all=shap_frame_all[shap_frame_all["model"]==mm].copy()
    
    shap_frame_sub_all["shap_values_abs"]=shap_frame_sub_all["shap_values"].apply(lambda x: abs(x))

    means=pd.DataFrame(shap_frame_sub_all[["feature","shap_values"]].groupby(["feature"]).mean()).reset_index()
    
    top5names_positive=list(means.sort_values("shap_values",ascending=False).head(10)["feature"])

    top5names_negatives=list(means.sort_values("shap_values",ascending=True).head(10)["feature"])
    
    all_features_names=top5names_positive+top5names_negatives
    
    color_list=["lightgreen"]*10+["salmon"]*10

    data=shap_frame_sub_all[shap_frame_sub_all["feature"].isin(all_features_names)].copy()
    sns.barplot(data=data,ax=ax[i],x="shap_values",y="feature",order=all_features_names,palette=color_list)
    
    if mm=="svm":
        ax[i].set_title("SVM")
    elif mm=="lr":
        ax[i].set_title("LR")
    elif mm=="rf":
        ax[i].set_title("RF")
    elif mm=="ffnn":
        ax[i].set_title("FFNN")
    elif mm=="xgboost":
        ax[i].set_title("XGBoost")
    
    
    
    #ax[i].legend(title="Gender")
    #ax[i].legend( loc="upper right")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set_ylabel("Feature",fontsize=20)
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        #ax[i].tick_params( labelleft=False)
    if i==2:
        ax[i].set_ylabel("Feature",fontsize=20)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Shap Value",fontsize=20)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Shap Value",fontsize=20)
        #ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('Feature',fontsize=20)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Shap value",fontsize=20)
    

#fig.delaxes(gs[0, 1])<
#plt.savefig(f"/restricted/s164512/G2020-57-Aalborg-bias/Plots/{folder_name}/{folder_name}_gender_metrics_allmodels", bbox_inches = 'tight')
plt.show()


# In[17]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(15,15))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


list_of_models=["svm","lr","rf","ffnn","xgboost"]


#for i,v in enumerate(all_data_gender["Model"].unique()):
for i,mm in enumerate(list_of_models):
    
    shap_frame_sub_all=shap_frame_all[shap_frame_all["model"]==mm].copy()
    
    shap_frame_sub_all["shap_values_abs"]=shap_frame_sub_all["shap_values"].apply(lambda x: abs(x))

   
    
    
    all_features_names=['Gender', 'Age', 'LoanPeriod', 'NumberAts']
    
    color_list=[]
    
    for n in all_features_names:
        if shap_frame_sub_all[shap_frame_sub_all["feature"]==n]["shap_values"].mean()<0:
            color_list.append("salmon")
        else:
            color_list.append("lightgreen")
    
    
    

    data=shap_frame_sub_all[shap_frame_sub_all["feature"].isin(all_features_names)].copy()
    sns.barplot(data=data,ax=ax[i],x="shap_values",y="feature",order=all_features_names,palette=color_list)
    
    
    
    if mm=="svm":
        ax[i].set_title("SVM")
    elif mm=="lr":
        ax[i].set_title("LR")
    elif mm=="rf":
        ax[i].set_title("RF")
    elif mm=="ffnn":
        ax[i].set_title("FFNN")
    elif mm=="xgboost":
        ax[i].set_title("XGBoost")
    
    
    
    #ax[i].legend(title="Gender")
    #ax[i].legend( loc="upper right")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set_ylabel("Feature",fontsize=20)
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        ax[i].tick_params( labelleft=False)
    if i==2:
        ax[i].set_ylabel("Feature",fontsize=20)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Shap Value",fontsize=20)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Shap Value",fontsize=20)
        ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('Feature',fontsize=20)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Shap value",fontsize=20)
    

#fig.delaxes(gs[0, 1])<
#plt.savefig(f"/restricted/s164512/G2020-57-Aalborg-bias/Plots/{folder_name}/{folder_name}_gender_metrics_allmodels", bbox_inches = 'tight')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Take absolute most important. And group by sign

# In[320]:


fig,ax=plt.subplots(2,1,figsize=(10,10))
ax = ax.ravel()

i=0
for mm in ["xgboost"]:
    shap_frame_sub_all=shap_frame_all[shap_frame_all["model"]==mm]
    
    shap_frame_sub_all["shap_values_abs"]=shap_frame_sub_all["shap_values"].apply(lambda x: abs(x))

    means=pd.DataFrame(shap_frame_sub_all[["feature","shap_values_abs"]].groupby(["feature"]).mean()).reset_index()
    top20names_abs=list(means.sort_values("shap_values_abs",ascending=False).head(10)["feature"])


    means=pd.DataFrame(shap_frame_sub_all[["feature","shap_values"]].groupby(["feature"]).mean()).reset_index()
    top20names=list(means.sort_values("shap_values",ascending=False).head(10)["feature"])
    

    
 
    data=shap_frame_sub_all[shap_frame_sub_all["feature"].isin(top20names_abs)].copy()
    data['sign'] = data['shap_values'].apply(lambda x: "negative" if x<0 else "positive")
    sns.barplot(data=data,ax=ax[i],x="shap_values",y="feature",hue="sign",order=top20names_abs,palette={"negative":"red","positive":"green"})
    ax[i].set_title(mm)
    i=i+1
plt.show()
    


# In[ ]:





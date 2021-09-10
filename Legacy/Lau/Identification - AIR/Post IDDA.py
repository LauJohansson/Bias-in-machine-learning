#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


#titel_mitigation="Original"
#titel_mitigation="Dropping D"
#titel_mitigation="Gender Swap"
#titel_mitigation="DI remove"
titel_mitigation="DI remove no gender"
#titel_mitigation="LFR"


# In[2]:


import warnings
warnings.filterwarnings("ignore")


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

from sklearn import metrics

#from google.colab import drive
from sklearn.model_selection import KFold

from datetime import datetime

import pytz
import random

import os


import math


#FAIRNESS
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

import scipy.stats as st


# In[4]:


from utils_copy import *


# In[5]:


def create_custom_roc_plot(df,y_true_name,y_pred_prob_name):
    female_index=df["Gender"]=="Female"
    male_index=df["Gender"]=="Male"


    fpr_all, tpr_all, thresholds_all = metrics.roc_curve(df[y_true_name], df[y_pred_prob_name],drop_intermediate=False)
    roc_auc_all = metrics.auc(fpr_all, tpr_all)

    
    fpr_female, tpr_female, thresholds_female = metrics.roc_curve(df[female_index][y_true_name], df[female_index][y_pred_prob_name],drop_intermediate=False)
    roc_auc_female = metrics.auc(fpr_female, tpr_female)

    fpr_male, tpr_male, thresholds_male = metrics.roc_curve(df[male_index][y_true_name], df[male_index][y_pred_prob_name],drop_intermediate=False)
    roc_auc_male = metrics.auc(fpr_male, tpr_male)


    plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
    plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))
    plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))

    plt.legend(loc=4)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    #plt.title("Male and females")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()
    


# In[6]:


def create_custom_roc_plot_ax(df,y_true_name,y_pred_prob_name,ax):
    female_index=df["Gender"]=="Female"
    male_index=df["Gender"]=="Male"


    fpr_all, tpr_all, thresholds_all = metrics.roc_curve(df[y_true_name], df[y_pred_prob_name],drop_intermediate=False)
    roc_auc_all = metrics.auc(fpr_all, tpr_all)

    
    fpr_female, tpr_female, thresholds_female = metrics.roc_curve(df[female_index][y_true_name], df[female_index][y_pred_prob_name],drop_intermediate=False)
    roc_auc_female = metrics.auc(fpr_female, tpr_female)

    fpr_male, tpr_male, thresholds_male = metrics.roc_curve(df[male_index][y_true_name], df[male_index][y_pred_prob_name],drop_intermediate=False)
    roc_auc_male = metrics.auc(fpr_male, tpr_male)


    ax.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
    ax.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))
    ax.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)),linestyle='--')

    ax.legend(loc=4)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    #plt.title("Male and females")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    
    


# In[7]:


def create_custom_roc_plot2(df):
    thres_list=[]
    
    female_index=df["Gender"]==0
    male_index=df["Gender"]==1

    true1=df[female_index]["Fall"]
    true2=df[male_index]["Fall"]
    probs1=df[female_index]["output_prob"]
    probs2=df[male_index]["output_prob"]


    fpr1=[]
    fpr2=[]
    tpr1=[]
    tpr2=[]

    for thres in range(0,1000,1):
        thres=thres/100
        thres_list.append(thres)


        y_pred1 = (probs1 > thres)#.float()
        y_pred2 = (probs2 > thres)#.float()
        TN1, FP1, FN1, TP1 = confusion_matrix(list(true1), list(y_pred1),labels=[0, 1]).ravel()
        TN2, FP2, FN2, TP2 = confusion_matrix(list(true2), list(y_pred2),labels=[0, 1]).ravel()

        TPR1 = TP1/(TP1+FN1)
        TPR2 = TP2/(TP2+FN2)
        FPR1 = FP1/(FP1+TN1)
        FPR2 = FP2/(FP2+TN2)

        fpr1.append(FPR1)
        fpr2.append(FPR2)
        tpr1.append(TPR1)
        tpr2.append(TPR2)



    plt.plot(fpr2,tpr2,color="green",label="Male"),# auc="+str(round(roc_auc_male,2)))
    plt.plot(fpr1,tpr1,color="blue",label="Female")#, auc="+str(round(roc_auc_female,2)))

    plt.legend(loc=4)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
    
    
    plt.title("Male and females")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()    


# In[8]:


def create_custom_roc_plot_relation(df,y_true_name,y_pred_prob_name,ptype,ax):
    thres_list=[]
    
    female_index=df["Gender"]=="Female"
    male_index=df["Gender"]=="Male"

    true1=df[female_index][y_true_name]
    true2=df[male_index][y_true_name]
    probs1=df[female_index][y_pred_prob_name]
    probs2=df[male_index][y_pred_prob_name]


    fpr1=[]
    fpr2=[]
    tpr1=[]
    tpr2=[]
    relationsTPR=[]
    relationsTNR=[]
    relationsFPR=[]
    relationsFNR=[]

    for thres in range(0,1000,1):
        thres=thres/100
        thres_list.append(thres)


        y_pred1 = (probs1 > thres)#.float()
        y_pred2 = (probs2 > thres)#.float()
        TN1, FP1, FN1, TP1 = confusion_matrix(list(true1), list(y_pred1),labels=[0, 1]).ravel()
        TN2, FP2, FN2, TP2 = confusion_matrix(list(true2), list(y_pred2),labels=[0, 1]).ravel()

        TPR1 = TP1/(TP1+FN1)
        TPR2 = TP2/(TP2+FN2)
        FPR1 = FP1/(FP1+TN1)
        FPR2 = FP2/(FP2+TN2)
        
        TNR1 =  TN1/(TN1+FP1)
        TNR2 = TN2/(TN2+FP2)
        FNR1 =  FN1/(TP1+FN1)
        FNR2 =  FN2/(TP2+FN2)

        #fpr1.append(FPR1)
        #fpr2.append(FPR2)
        #tpr1.append(TPR1)
        #tpr2.append(TPR2)
        
        #fpr1.append(FPR1)
        #fpr2.append(FPR2)
        #tpr1.append(TPR1)
        #tpr2.append(TPR2)
        
        relationsTPR.append(TPR1/TPR2)
        relationsFPR.append(FPR1/FPR2)
        relationsTNR.append(TNR1/TNR2)
        relationsFNR.append(FNR1/FNR2)



    #plt.plot(fpr2,tpr2,color="green",label="Male"),# auc="+str(round(roc_auc_male,2)))
    #plt.plot(fpr1,tpr1,color="blue",label="Female")#, auc="+str(round(roc_auc_female,2)))

    if ptype=="TPR":
        ax.plot(thres_list,relationsTPR,color="Black",label="TPR Relation"),# auc="+str(round(roc_auc_male,2)))
        ax.set_ylabel("TPR relation")
    if ptype=="TNR":
        ax.plot(thres_list,relationsTNR,color="Black",label="TNR Relation"),# auc="+str(round(roc_auc_male,2)))
        ax.set_ylabel("TNR relation")
    if ptype=="FPR":
        ax.plot(thres_list,relationsFPR,color="Black",label="FPR Relation"),# auc="+str(round(roc_auc_male,2)))
        ax.set_ylabel("FPR relation")
    if ptype=="FNR":
        ax.plot(thres_list,relationsFNR,color="Black",label="FNR Relation"),# auc="+str(round(roc_auc_male,2)))
        ax.set_ylabel("FNR relation")
    
    ax.legend(loc=4)
    ax.hlines(y=1, xmin=0, xmax=1, colors='red', linestyles='--', lw=1, label='Equal relation')
    ax.hlines(y=0.8, xmin=0, xmax=1, colors='grey', linestyles='--', lw=1, label='Relation boundary')
    ax.hlines(y=1.25, xmin=0, xmax=1, colors='grey', linestyles='--', lw=1)#, label='Relative difference=0.8')
    #ax.plot([0, 1], [1, 1], linestyle='--', lw=2, color='r',
    #            label='Chance', alpha=.8)
    #ax.plot([0, 0.8], [1, 0.8], linestyle='--', lw=2, color='grey',
    #            label='Chance', alpha=.8)
    #ax.plot([0, 1.25], [1, 1.25], linestyle='--', lw=2, color='grey',
    #            label='Chance', alpha=.8)
    #ax.title("Relation")
    ax.set_xlabel("Threshold")
    #ax.show()    


# # FILE PATH

# In[9]:




### USE ALL DATA
#file_name="fall.csv"
#full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name



###USE SMALL TEST

PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"

#file_name="Fall_count_clusterOHE_std.csv"
file_name="Fall_count_clusterOHE.csv"


full_file_path=PATH_orig+file_name


print("PATH_orig:",PATH_orig)
print("PATH to file:",full_file_path)


# # Specify the y, X and protected variable

# In[10]:


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


# # Read the data

# In[11]:



df2 = pd.read_csv(full_file_path).drop(columns=["Unnamed: 0"])
df2["Gender_string"]=df2["Gender"].apply(lambda x: "Female" if x==0 else "Male")
df2_copy=df2.copy()
df2_copy=df2_copy.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})


# In[ ]:





# # Sturge formula

# In[12]:


math.ceil(math.log2(len(df2)))+1


# In[13]:


math.ceil(math.log2(len(df2[df2["Gender"]==1])))+1


# In[14]:


math.ceil(math.log2(len(df2[df2["Gender"]==0])))+1


#  

#  

# # Identify bias

# In[15]:


print(f"The dataset has {df2.shape[0]} rows")
print(f"The dataset has {df2.shape[1]} cols")

print(f"The dataset to test has the protected attribute: {procted_col_name}")
print(f"The protected variable can be assigned: {df2[procted_col_name].unique()}")

print(f"The ratio of records with y=1: {df2[df2[y_col_name]==1][y_col_name].count()/df2[y_col_name].count()}")


# # Plotting numeric features

# In[16]:


fig,ax=plt.subplots(3,1,figsize=(10,10))

ax = ax.ravel()


sns.histplot(data=df2_copy,x="LoanPeriod",stat="probability",hue="Gender",ax=ax[0],bins=12,common_norm=False)
sns.histplot(data=df2_copy,x="BirthYear",stat="probability",hue="Gender",ax=ax[1],bins=12,common_norm=False)
sns.histplot(data=df2_copy,x="NumberAts",stat="probability",hue="Gender",ax=ax[2],bins=12,common_norm=False)


plt.plot()


# # Y depending on gender

# In[17]:


sns.histplot(data=df2_copy,x="Fall",stat="probability",hue="Gender",common_norm=False)
plt.xticks([0,1])
plt.grid(axis="x")
plt.plot()


# In[18]:


print((df2_copy.groupby("Gender")["Fall"].mean()*100).to_latex())


# In[ ]:





# In[19]:


print("Male fall")
df2_copy[(df2_copy["Gender"]=="Male")&(df2_copy["Fall"]==1)]["Fall"].count()


# In[20]:


print("Male not fall")
df2_copy[(df2_copy["Gender"]=="Male")&(df2_copy["Fall"]==0)]["Fall"].count()


# In[21]:


print("Female fall")
df2_copy[(df2_copy["Gender"]=="Female")&(df2_copy["Fall"]==1)]["Fall"].count()


# In[22]:


print("Female not fall")
df2_copy[(df2_copy["Gender"]=="Female")&(df2_copy["Fall"]==0)]["Fall"].count()


# In[ ]:





# # BirthYear

# In[23]:


fig,ax=plt.subplots(1,2,figsize=(20,8))
ax=ax.ravel()
custom_palette={"Female":"C0","Male":"C1"}
sns.histplot(data=df2_copy,x="BirthYear",stat="probability",hue="Gender",common_norm=False,bins=10,ax=ax[0],palette=custom_palette,hue_order=["Female","Male"])
sns.boxplot(data=df2_copy,y="BirthYear",x="Gender",ax=ax[1])
plt.show()


# In[24]:


sns.boxplot(data=df2_copy,y="BirthYear",x="Gender",palette=custom_palette)
plt.ylabel("BirthYear",fontsize=20)
plt.xlabel("Gender",fontsize=20)
plt.show()


# In[25]:



birth_palette={"Male (No Fall)":"C1","Male (Fall)":"C1","Female (No Fall)":"C0","Female (Fall)":"C0"}
ordr=["Female (No Fall)","Male (No Fall)","Female (Fall)","Male (Fall)"]

df2_copy_for_birthplot=df2_copy.copy()
df2_copy_for_birthplot["Age"]=df2_copy_for_birthplot["BirthYear"].apply(lambda x: 2021-(x+1900))

df2_copy_for_birthplot["Gender-fall"]=df2_copy_for_birthplot["Gender"]+" "+df2_copy_for_birthplot["Fall"].apply(lambda x: "(Fall)" if x==1 else "(No Fall)")

sns.boxplot(data=df2_copy_for_birthplot,y="Age",x="Gender-fall",palette=birth_palette,order=ordr,showfliers=False)
plt.ylabel("Age",fontsize=20)
plt.xlabel("Gender",fontsize=20)
plt.show()


# In[26]:


a_list=list(df2_copy_for_birthplot[(df2_copy_for_birthplot["Gender-fall"]=="Male (Fall)")]["Age"])
A = plt.boxplot(a_list)
print([item.get_ydata()[1] for item in A['whiskers']])
print([item.get_ydata()[1] for item in A['medians']])

#Birthyear
#Male (No Fall), 21
# Female (No Fall), 17
# Female () Fall), 17
# Male () Fall), 18


#AGE MAX
#Male (No Fall), 100
# Female (No Fall), 104
# Female () Fall), 104
# Male () Fall), 103


#AGE median
#Male (No Fall), 83
# Female (No Fall), 86
# Female () Fall), 88
# Male () Fall), 87


# In[ ]:





# In[ ]:





# In[27]:


df2_copy_for_birthplot["Fall_string"]=df2_copy_for_birthplot["Fall"].apply(lambda x: "Fall" if x==1 else "No Fall")
sns.boxplot(data=df2_copy_for_birthplot,y="BirthYear",x="Fall_string",palette={"Fall":"grey","No Fall":"grey"})
plt.ylabel("BirthYear",fontsize=20)
plt.xlabel("Fall",fontsize=20)
plt.show()


# In[28]:


df2_copy_for_birthplot.groupby(["Gender-fall"])["BirthYear"].mean()


# In[29]:


df2_copy.groupby(["Gender"])["BirthYear"].mean()


# In[30]:


filtera=df2_copy["Gender"]=="Male"
a_list=df2_copy[filtera]["BirthYear"]
a_list=list(a_list)


filterb=df2_copy["Gender"]=="Female"
b_list=df2_copy[filterb]["BirthYear"]
b_list=list(b_list)



print("Quartiles males:")
print(np.percentile(a_list, np.arange(0, 101, 25)))
print("Mean males:")
print(np.mean(a_list))


print("Quartiles females:")
print(np.percentile(b_list, np.arange(0, 101, 25)))
print("Mean females:")
print(np.mean(b_list))


# In[31]:


#A = plt.boxplot(a_list)
#print([item.get_ydata()[1] for item in A['whiskers']])


# In[32]:


#B = plt.boxplot(b_list)
#print([item.get_ydata()[1] for item in B['whiskers']])


# # LoanPeriod

# In[ ]:





# In[33]:


sns.histplot(data=df2_copy,x="LoanPeriod",stat="probability",hue="Gender",common_norm=False,bins=12)
plt.show()


# In[34]:


sns.boxplot(data=df2_copy,y="LoanPeriod",x="Gender",palette=custom_palette)
plt.ylabel("LoanPeriod",fontsize=20)
plt.xlabel("Gender",fontsize=20)


# In[35]:




sns.boxplot(data=df2_copy_for_birthplot,y="LoanPeriod",x="Gender-fall",palette=birth_palette,order=ordr,showfliers=False)
plt.ylabel("LoanPeriod",fontsize=20)
plt.xlabel("Gender",fontsize=20)
plt.show()


# In[36]:


a_list=list(df2_copy_for_birthplot[(df2_copy_for_birthplot["Gender-fall"]=="Male (No Fall)")]["LoanPeriod"])
A = plt.boxplot(a_list)
#print([item.get_ydata()[1] for item in A])#['whiskers']])
print(print([item.get_ydata()[1] for item in A['medians']]))

#Female fall: 1024
#Female no fall: 1086


#male fall: 518
#male no fall: 719


# In[ ]:





# In[ ]:





# In[37]:




sns.histplot(data=df2_copy_for_birthplot,x="LoanPeriod",hue="Gender-fall",stat="probability",common_norm=False,bins=20)#,hue_order=ordr,)
#plt.ylabel("LoanPeriod",fontsize=20)
#plt.xlabel("Gender",fontsize=20)
plt.show()


# In[38]:


df2_copy_for_birthplot.groupby(["Gender-fall"])["LoanPeriod"].mean()


# In[39]:


#df2_copy_for_birthplot[(df2_copy_for_birthplot["Gender-fall"]=="Female (Fall)")&(df2_copy_for_birthplot["LoanPeriod"]==0)]["LoanPeriod"].count()


# In[40]:


#df2_copy_for_birthplot[(df2_copy_for_birthplot["Gender-fall"]=="Female (No Fall)")&(df2_copy_for_birthplot["LoanPeriod"]==0)]["LoanPeriod"].count()


# In[41]:


#df2_copy_for_birthplot[(df2_copy_for_birthplot["Gender-fall"]=="Male (Fall)")&(df2_copy_for_birthplot["LoanPeriod"]==0)]["LoanPeriod"].count()


# In[42]:


#df2_copy_for_birthplot[(df2_copy_for_birthplot["Gender-fall"]=="Male (No Fall)")&(df2_copy_for_birthplot["LoanPeriod"]==0)]["LoanPeriod"].count()


# # NumberAts

# In[43]:


sns.histplot(data=df2_copy,x="NumberAts",stat="probability",hue="Gender",common_norm=False,bins=12)
plt.show()


# In[44]:


sns.boxplot(data=df2_copy,y="NumberAts",x="Gender",palette=custom_palette)
plt.ylabel("NumberAts",fontsize=20)
plt.xlabel("Gender",fontsize=20)


# In[45]:




sns.boxplot(data=df2_copy_for_birthplot,y="NumberAts",x="Gender-fall",palette=birth_palette,order=ordr,showfliers=False)
plt.ylabel("NumberAts",fontsize=20)
plt.xlabel("Gender",fontsize=20)
plt.show()


# In[46]:


df2_copy_for_birthplot.groupby(["Gender-fall"])["NumberAts"].mean()


# In[47]:


a_list=list(df2_copy_for_birthplot[(df2_copy_for_birthplot["Gender-fall"]=="Female (Fall)")]["NumberAts"])
A = plt.boxplot(a_list)
print("Whiskers: ",[item.get_ydata()[1] for item in A['whiskers']])
print("Median",[item.get_ydata()[1] for item in A['medians']])

print("Mean",[item.get_ydata()[1] for item in A['means']])

#Male fall: 8
#Male no fall: 9


#female no fall: 8
#female fall: 9


# In[48]:


a_list=list(df2_copy_for_birthplot[(df2_copy_for_birthplot["Gender-fall"]=="Male (Fall)")]["NumberAts"])
A = plt.boxplot(a_list)
print([item.get_ydata()[1] for item in A['whiskers']])


# In[ ]:





# In[49]:


filtera=df2_copy["Gender"]=="Male"
a_list=df2_copy[filtera]["NumberAts"]
a_list=list(a_list)


filterb=df2_copy["Gender"]=="Female"
b_list=df2_copy[filterb]["NumberAts"]
b_list=list(b_list)



print("Quartiles males:")
print(np.percentile(a_list, np.arange(0, 100, 25)))
print("Mean males:")
print(np.mean(a_list))


print("Quartiles females:")
print(np.percentile(b_list, np.arange(0, 100, 25)))
print("Mean females:")
print(np.mean(b_list))


# In[50]:


df2_copy.groupby(["Gender"])["NumberAts"].mean()


# In[ ]:





# # Correlation - females

# In[51]:


dd=df2_copy[df2_copy["Gender"]=="Female"]
dd["Age"]=dd["BirthYear"].apply(lambda x: 2021-(x+1900))
dd=dd.drop(columns="Gender")
#just_dummies=pd.get_dummies(dd)
#df2_dummy = pd.concat([df2_copy, just_dummies], axis=1) 
#df2_dummy=df2_dummy.drop(["Gender"] ,axis=1)


#https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
#f, ax = plt.subplots(figsize=(10, 8))
corr = dd[["Age","LoanPeriod","NumberAts","Fall"]].corr()
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            square=True, ax=ax)
print(round(corr,4).to_latex())


# # Correlation - males

# In[52]:


dd=df2_copy[df2_copy["Gender"]=="Male"]
dd["Age"]=dd["BirthYear"].apply(lambda x: 2021-(x+1900))
dd=dd.drop(columns="Gender")
#just_dummies=pd.get_dummies(dd)
#df2_dummy = pd.concat([df2_copy, just_dummies], axis=1) 
#df2_dummy=df2_dummy.drop(["Gender"] ,axis=1)


#https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
#f, ax = plt.subplots(figsize=(10, 8))
corr = dd[["Age","LoanPeriod","NumberAts","Fall"]].corr()
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            square=True, ax=ax)
print(round(corr,4).to_latex())


# In[53]:


dd=df2_copy[df2_copy["Gender"]=="Male"]
dd=dd.drop(columns="Gender")
#just_dummies=pd.get_dummies(dd)
#df2_dummy = pd.concat([df2_copy, just_dummies], axis=1) 
#df2_dummy=df2_dummy.drop(["Gender"] ,axis=1)


#https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
#f, ax = plt.subplots(figsize=(10, 8))
corr = dd[["BirthYear","LoanPeriod","NumberAts","Fall"]].corr()
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            square=True, ax=ax)
print(round(corr,4).to_latex())


# # Correlation

# In[54]:


just_dummies=pd.get_dummies(df2_copy[['Gender']])
df2_dummy = pd.concat([df2_copy, just_dummies], axis=1) 
df2_dummy=df2_dummy.drop(["Gender"] ,axis=1)


# In[55]:


#https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
f, ax = plt.subplots(figsize=(10, 8))
corr = df2_dummy[["Gender_Female","Gender_Male","BirthYear","LoanPeriod","NumberAts",]].corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[56]:


#https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
f, ax = plt.subplots(figsize=(10, 8))
corr = df2_dummy[["Gender_Female","Gender_Male","Cluster_0","Cluster_1","Cluster_2","Cluster_3","Cluster_4","Cluster_5",
                             "Cluster_6","Cluster_7","Cluster_8","Cluster_9","Cluster_10","Cluster_11",
                             "Cluster_12","Cluster_13","Cluster_14","Cluster_15","Cluster_16","Cluster_17",
                             "Cluster_18","Cluster_19"]].corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# # Cluster

# In[57]:


df2_copy.groupby(["Gender"])[["Cluster_0","Cluster_1","Cluster_2","Cluster_3","Cluster_4","Cluster_5",
                             "Cluster_6","Cluster_7","Cluster_8","Cluster_9","Cluster_10","Cluster_11",
                             "Cluster_12","Cluster_13","Cluster_14","Cluster_15","Cluster_16","Cluster_17",
                             "Cluster_18","Cluster_19"]].sum()


# In[58]:


ddd=df2_copy.groupby(["Fall"])[["Cluster_0","Cluster_1","Cluster_2","Cluster_3","Cluster_4","Cluster_5",
                             "Cluster_6","Cluster_7","Cluster_8","Cluster_9","Cluster_10","Cluster_11",
                             "Cluster_12","Cluster_13","Cluster_14","Cluster_15","Cluster_16","Cluster_17",
                             "Cluster_18","Cluster_19"]].sum().reset_index()
cluster_to_bar=ddd.melt(id_vars=["Fall"],var_name="Cluster",value_name="Value")

cluster_to_bar["Cluster"]=cluster_to_bar["Cluster"].apply(lambda x: x[-1] if len(x)==9 else x[-2:])
sns.barplot(data=cluster_to_bar,x="Cluster",y="Value",hue="Fall")#,ci=None)
#plt.y_label("Count",)


# In[59]:


ddd=df2_copy[["Cluster_0","Cluster_1","Cluster_2","Cluster_3","Cluster_4","Cluster_5",
                             "Cluster_6","Cluster_7","Cluster_8","Cluster_9","Cluster_10","Cluster_11",
                             "Cluster_12","Cluster_13","Cluster_14","Cluster_15","Cluster_16","Cluster_17",
                             "Cluster_18","Cluster_19"]].sum().to_frame().reset_index().rename(columns={"index": "Cluster", 0: "Value"})


# In[60]:


ddd["Cluster"]=ddd["Cluster"].apply(lambda x: x[-1] if len(x)==9 else x[-2:])


# In[61]:


ddd["dummy"]="dummy"


# In[84]:


#sns.set_palette("tab10")
sns.barplot(data=ddd,x="Cluster",y="Value",hue="dummy",palette={"dummy":"steelblue"},alpha=0.7)#,color="cornflo")#,ci=None)
plt.legend("")
plt.ylabel("Count",size=15)
plt.xlabel("Cluster",size=15)
plt.plot(legend=None)


# # SVM

# In[29]:


SVM_testdata=pd.read_csv(f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{titel_mitigation}/SVM_gender_obs.csv")

SVM_testdata["Gender_string"]=SVM_testdata["Gender"].apply(lambda x: "Female" if x==0 else "Male")
SVM_testdata=SVM_testdata.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})


# In[30]:


SVM_testdata[SVM_testdata["y_true"].isna()]


# In[31]:


SVM_testdata["Gender"].unique()


# In[32]:


print("SVM OUTPUT (Binary)")
for g in ["Female","Male"]:
    
    data=SVM_testdata[(SVM_testdata["Gender"]==g)]["y_hat_binary"]
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    print(f"Mean for {g}",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")
    


# In[33]:


#SVM_testdata.groupby("Gender")["y_hat_probs"].mean()



print("SVM OUTPUT (probs)")
for g in ["Female","Male"]:
    
    data=SVM_testdata[(SVM_testdata["Gender"]==g)]["y_hat_probs"]
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    print(f"Mean for {g}",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")


# In[34]:


#SVM_testdata.groupby("Gender")["y_hat_probs"].mean()



print("SVM OUTPUT (probs)")
str1="& "
str2="& "
for y_val in [1,0]:
    for g in ["Female","Male"]:



        data=SVM_testdata[(SVM_testdata["Gender"]==g)&((SVM_testdata["y_true"]==y_val))]["y_hat_probs"]
        m=np.mean(data)
        (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 

        print(f"Mean for {g} (Fall={y_val})",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")
        str1=str1+str(round(m*100,2))+"& "
        str2=str2+f"({round(slow*100,2)}-{round(shigh*100,2)})"+"& "
print()
print(str1)
print(str2)


# In[10]:


sns.histplot(data=SVM_testdata,x="y_hat_probs",hue="y_true")


# # LR

# In[35]:


LR_testdata=pd.read_csv(f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{titel_mitigation}/LR_gender_obs.csv")

LR_testdata["Gender_string"]=LR_testdata["Gender"].apply(lambda x: "Female" if x==0 else "Male")

LR_testdata=LR_testdata.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})


# In[36]:


LR_testdata[LR_testdata["y_true"].isna()]


# In[37]:


#LR_testdata.groupby("Gender")["y_hat_binary"].mean()



print("LR OUTPUT (Binary)")
for g in ["Female","Male"]:
    
    data=LR_testdata[(LR_testdata["Gender"]==g)]["y_hat_binary"]
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    print(f"Mean for {g}",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")


# In[38]:


print("LR OUTPUT (Prob)")
for g in ["Female","Male"]:
    
    data=LR_testdata[(LR_testdata["Gender"]==g)]["y_hat_probs"]
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    print(f"Mean for {g}",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")


# In[39]:


#SVM_testdata.groupby("Gender")["y_hat_probs"].mean()



print("LR OUTPUT (probs)")
str1="& "
str2="& "
for y_val in [1,0]:
    for g in ["Female","Male"]:



        data=LR_testdata[(LR_testdata["Gender"]==g)&((LR_testdata["y_true"]==y_val))]["y_hat_probs"]
        m=np.mean(data)
        (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 

        print(f"Mean for {g} (Fall={y_val})",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")
        str1=str1+str(round(m*100,1))+"& "
        str2=str2+f"({round(slow*100,1)}-{round(shigh*100,1)})"+"& "
print()
print(str1)
print(str2)


# In[16]:


LR_testdata.groupby("Gender")["y_hat_probs"].mean()


# In[ ]:





# # RF

# In[40]:


RF_testdata=pd.read_csv(f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{titel_mitigation}/RF_gender_obs.csv")


RF_testdata["Gender_string"]=RF_testdata["Gender"].apply(lambda x: "Female" if x==0 else "Male")

RF_testdata=RF_testdata.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})


# In[41]:


RF_testdata[RF_testdata["y_true"].isna()]


# In[42]:


print("RF OUTPUT (Binary)")
for g in ["Female","Male"]:
    
    data=RF_testdata[(RF_testdata["Gender"]==g)]["y_hat_binary"]
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    print(f"Mean for {g}",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")


# In[43]:


print("RF OUTPUT (PROBS)")
for g in ["Female","Male"]:
    
    data=RF_testdata[(RF_testdata["Gender"]==g)]["y_hat_probs"]
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    print(f"Mean for {g}",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")


# In[44]:


#SVM_testdata.groupby("Gender")["y_hat_probs"].mean()



print("RF OUTPUT (probs)")
str1="& "
str2="& "
for y_val in [1,0]:
    for g in ["Female","Male"]:



        data=RF_testdata[(RF_testdata["Gender"]==g)&((RF_testdata["y_true"]==y_val))]["y_hat_probs"]
        m=np.mean(data)
        (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 

        print(f"Mean for {g} (Fall={y_val})",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")
        str1=str1+str(round(m*100,1))+"& "
        str2=str2+f"({round(slow*100,1)}-{round(shigh*100,1)})"+"& "
print()
print(str1)
print(str2)


# # FFNN

# In[45]:


FFNN_testdata=pd.read_csv(f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{titel_mitigation}/FFNN_gender_obs.csv")

FFNN_testdata["Gender_string"]=FFNN_testdata["Gender"].apply(lambda x: "Female" if x==0 else "Male")

FFNN_testdata=FFNN_testdata.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})



# In[46]:



print("FFNN OUTPUT (BINARY)")
for g in ["Female","Male"]:
    
    data=FFNN_testdata[(FFNN_testdata["Gender"]==g)]["output"]
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    #print(f"Mean for {g}",FFNN_testdata.groupby("Gender")["output"].mean()[g])
    print(f"Mean for {g}",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")


# In[47]:



print("FFNN OUTPUT (probability)")
for g in ["Female","Male"]:
    
    data=FFNN_testdata[(FFNN_testdata["Gender"]==g)]["output_prob"]
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    #print(f"Mean for {g}",FFNN_testdata.groupby("Gender")["output"].mean()[g])
    print(f"Mean for {g}",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")


# In[48]:


#SVM_testdata.groupby("Gender")["y_hat_probs"].mean()



print("FFNN OUTPUT (probs)")
str1="& "
str2="& "
for y_val in [1,0]:
    for g in ["Female","Male"]:



        data=FFNN_testdata[(FFNN_testdata["Gender"]==g)&((FFNN_testdata["Fall"]==y_val))]["output_prob"]
        m=np.mean(data)
        (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 

        print(f"Mean for {g} (Fall={y_val})",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")
        str1=str1+str(round(m*100,1))+"& "
        str2=str2+f"({round(slow*100,1)}-{round(shigh*100,1)})"+"& "
print()
print(str1)
print(str2)


# In[26]:


#sns.histplot(data=FFNN_testdata,x="output_prob",hue="Gender",stat="probability",common_norm=False)
#plt.plot()


# In[27]:


#create_custom_roc_plot(FFNN_testdata)


# In[28]:


#create_custom_roc_plot2(FFNN_testdata)


# # XGboost

# In[49]:


Xg_boost_testdata=pd.read_csv(f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{titel_mitigation}/Xgboost_gender_obs.csv")


Xg_boost_testdata["Gender_string"]=Xg_boost_testdata["Gender"].apply(lambda x: "Female" if x==0 else "Male")

Xg_boost_testdata=Xg_boost_testdata.drop(columns=["Gender"]).rename(columns={"Gender_string":"Gender"})

#data_gender_Xgboost.loc[data_gender_Xgboost.Model == "Xgboost", "Model"] = "XGBoost"


# In[50]:


print("Xgboost OUTPUT (Binary)")
for g in ["Female","Male"]:
    
    data=Xg_boost_testdata[(Xg_boost_testdata["Gender"]==g)]["output"]
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    print(f"Mean for {g}",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")


# In[51]:


print("Xgboost OUTPUT (PROBAILITY)")
for g in ["Female","Male"]:
    
    data=Xg_boost_testdata[(Xg_boost_testdata["Gender"]==g)]["output_prob"]
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    print(f"Mean for {g}",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")


# In[52]:


#SVM_testdata.groupby("Gender")["y_hat_probs"].mean()



print("Xgboost OUTPUT (probs)")
str1="& "
str2="& "
for y_val in [1,0]:
    for g in ["Female","Male"]:



        data=Xg_boost_testdata[(Xg_boost_testdata["Gender"]==g)&((Xg_boost_testdata["Fall"]==y_val))]["output_prob"]
        m=np.mean(data)
        (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 

        print(f"Mean for {g} (Fall={y_val})",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")
        str1=str1+str(round(m*100,1))+"& "
        str2=str2+f"({round(slow*100,1)}-{round(shigh*100,1)})"+"& "
print()
print(str1)
print(str2)


# In[53]:


sns.histplot(data=Xg_boost_testdata,x="output_prob",hue="Gender",stat="probability",common_norm=False)
plt.plot()


# In[31]:


#create_custom_roc_plot(Xg_boost_testdata)


# In[ ]:





# In[ ]:





# # All models

# In[32]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(12,12))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","XGBoost"]
n_bins=50

palette_custom={"Female":"C0","Male":"C1"}


for i,v in enumerate(list_of_models):
    
    ax[i].set_title(v,size=15)
    ax[i].set_xlim([0, 1])
    #ax[i].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ])
    ax[i].set_xticks([0,0.5,1.0])
    ax[i].grid(axis='x')
    
    
    
    if v=="SVM":
        sns.histplot(data=SVM_testdata,x="y_hat_probs",hue="Gender",stat="probability",ax=ax[i],bins=n_bins,common_norm=False,palette=palette_custom,hue_order = ['Female', 'Male'])
    if v=="LR":
        sns.histplot(data=LR_testdata,x="y_hat_probs",hue="Gender",stat="probability",ax=ax[i],bins=n_bins,common_norm=False,palette=palette_custom,hue_order = ['Female', 'Male'])
    if v=="RF":
        sns.histplot(data=RF_testdata,x="y_hat_probs",hue="Gender",stat="probability",ax=ax[i],bins=n_bins,common_norm=False,palette=palette_custom,hue_order = ['Female', 'Male'])
    
    if v=="FFNN":
        sns.histplot(data=FFNN_testdata,x="output_prob",hue="Gender",stat="probability",ax=ax[i],bins=n_bins,common_norm=False,palette=palette_custom,hue_order = ['Female', 'Male'])
    if v=="XGBoost":
        sns.histplot(data=Xg_boost_testdata,x="output_prob",hue="Gender",stat="probability",ax=ax[i],bins=n_bins,common_norm=False,palette=palette_custom,hue_order = ['Female', 'Male'])
    
    
    #ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set_ylabel("Probability",fontsize=10)
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        #ax[i].tick_params( labelleft=False)
    if i==2:
        ax[i].set_ylabel("Probability",fontsize=10)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Predicted risk of falling",fontsize=20)
        #ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('Probability',fontsize=10)
        ax[i].set_xlabel("Predicted risk of falling",fontsize=20)
    ax[i].set_xlabel("Predicted risk of falling",fontsize=10)

#fig.delaxes(gs[0, 1])
plt.show()


# In[33]:


sns.histplot(data=LR_testdata[LR_testdata["Gender"]=="Male"],x="y_hat_probs",hue="Gender",stat="probability",bins=100,common_norm=False,palette=palette_custom)
    


# In[34]:


lst=list(LR_testdata[LR_testdata["Gender"]=="Male"]["y_hat_probs"])
max(set(lst), key=lst.count)


# In[35]:


filternew=LR_testdata["y_hat_probs"]>0.5

lst=list(LR_testdata[(LR_testdata["Gender"]=="Male")&(filternew)]["y_hat_probs"])
max(set(lst), key=lst.count)


# In[36]:


sns.histplot(data=LR_testdata[LR_testdata["Gender"]=="Female"],x="y_hat_probs",hue="Gender",stat="probability",bins=100,common_norm=False,palette=palette_custom)
    


# In[37]:


lst=list(LR_testdata[LR_testdata["Gender"]=="Female"]["y_hat_probs"])
max(set(lst), key=lst.count)


# In[38]:


filternew=LR_testdata["y_hat_probs"]>0.5

lst=list(LR_testdata[(LR_testdata["Gender"]=="Female")&(filternew)]["y_hat_probs"])
max(set(lst), key=lst.count)


# In[39]:


print("For the xgboost:")
lst=list(Xg_boost_testdata[Xg_boost_testdata["Gender"]=="Male"]["output_prob"])
max(set(lst), key=lst.count)


# In[40]:


filternew=Xg_boost_testdata["output_prob"]>0.7

lst=list(Xg_boost_testdata[(Xg_boost_testdata["Gender"]=="Male")&(filternew)]["output_prob"])
max(set(lst), key=lst.count)


# In[41]:


custom_palette_new={1:"C3",0:"C4"}




SVM_testdata["Gender-falltrue"]=SVM_testdata["Gender"]+" "+SVM_testdata["y_true"].apply(lambda x: "(Fall)" if x==1 else "(No Fall)")
LR_testdata["Gender-falltrue"]=LR_testdata["Gender"]+" "+LR_testdata["y_true"].apply(lambda x: "(Fall)" if x==1 else "(No Fall)")
RF_testdata["Gender-falltrue"]=RF_testdata["Gender"]+" "+RF_testdata["y_true"].apply(lambda x: "(Fall)" if x==1 else "(No Fall)")

FFNN_testdata["Gender-falltrue"]=FFNN_testdata["Gender"]+" "+FFNN_testdata["Fall"].apply(lambda x: "(Fall)" if x==1 else "(No Fall)")
Xg_boost_testdata["Gender-falltrue"]=Xg_boost_testdata["Gender"]+" "+Xg_boost_testdata["Fall"].apply(lambda x: "(Fall)" if x==1 else "(No Fall)")


# In[42]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(12,12))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","XGBoost"]
n_bins=50

palette_custom={"Female":"C0","Male":"C1"}


for i,v in enumerate(list_of_models):
    
    ax[i].set_title(v,size=15)
    ax[i].set_xlim([0, 1])
    #ax[i].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ])
    ax[i].set_xticks([0,0.5,1.0])
    ax[i].grid(axis='x')
    
    
    
    if v=="SVM":
        sns.histplot(data=SVM_testdata,x="y_hat_probs",hue="Gender-falltrue",stat="probability",ax=ax[i],bins=n_bins,common_norm=False)#,palette=birth_palette)#,hue_order = ['Female', 'Male'])
    if v=="LR":
        sns.histplot(data=LR_testdata,x="y_hat_probs",hue="Gender-falltrue",stat="probability",ax=ax[i],bins=n_bins,common_norm=False)#,palette=birth_palette)#,hue_order = ['Female', 'Male'])
    if v=="RF":
        sns.histplot(data=RF_testdata,x="y_hat_probs",hue="Gender-falltrue",stat="probability",ax=ax[i],bins=n_bins,common_norm=False)#,palette=birth_palette)#,hue_order = ['Female', 'Male'])
    
    if v=="FFNN":
        sns.histplot(data=FFNN_testdata,x="output_prob",hue="Gender-falltrue",stat="probability",ax=ax[i],bins=n_bins,common_norm=False)#,palette=birth_palette)#,hue_order = ['Female', 'Male'])
    if v=="XGBoost":
        sns.histplot(data=Xg_boost_testdata,x="output_prob",hue="Gender-falltrue",stat="probability",ax=ax[i],bins=n_bins,common_norm=False)#,palette=birth_palette)#,hue_order = ['Female', 'Male'])
    
    
    #ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set_ylabel("Probability",fontsize=10)
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        #ax[i].tick_params( labelleft=False)
    if i==2:
        ax[i].set_ylabel("Probability",fontsize=10)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Predicted risk of falling",fontsize=20)
        #ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('Probability',fontsize=10)
        ax[i].set_xlabel("Predicted risk of falling",fontsize=20)
    ax[i].set_xlabel("Predicted risk of falling",fontsize=10)

#fig.delaxes(gs[0, 1])
plt.show()


# In[43]:


pred_fall_palette={"Fall":"C8","No Fall":"C3"}
SVM_testdata["y pred binary"]=SVM_testdata["y_hat_binary"].apply(lambda x: "Fall" if x==1 else "No Fall")
LR_testdata["y pred binary"]=LR_testdata["y_hat_binary"].apply(lambda x: "Fall" if x==1 else "No Fall")
RF_testdata["y pred binary"]=RF_testdata["y_hat_binary"].apply(lambda x: "Fall" if x==1 else "No Fall")
FFNN_testdata["y pred binary"]=FFNN_testdata["output"].apply(lambda x: "Fall" if x==1 else "No Fall")
Xg_boost_testdata["y pred binary"]=Xg_boost_testdata["output"].apply(lambda x: "Fall" if x==1 else "No Fall")


# In[44]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(12,12))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","XGBoost"]
n_bins=50

#palette_custom={"Female":"C0","Male":"C1"}


for i,v in enumerate(list_of_models):
    
    ax[i].set_title(v,size=15)
    ax[i].set_xlim([0, 1])
    #ax[i].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ])
    ax[i].set_xticks([0,0.5,1.0])
    ax[i].grid(axis='x')
    
    
    
    if v=="SVM":
        sns.histplot(data=SVM_testdata,x="y_hat_probs",hue="y pred binary",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=pred_fall_palette,hue_order = ['Fall', 'No Fall'])
    if v=="LR":
        sns.histplot(data=LR_testdata,x="y_hat_probs",hue="y pred binary",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=pred_fall_palette,hue_order = ['Fall', 'No Fall'])
    if v=="RF":
        sns.histplot(data=RF_testdata,x="y_hat_probs",hue="y pred binary",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=pred_fall_palette,hue_order = ['Fall', 'No Fall'])
    
    if v=="FFNN":
        sns.histplot(data=FFNN_testdata,x="output_prob",hue="y pred binary",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=pred_fall_palette,hue_order = ['Fall', 'No Fall'])
    if v=="XGBoost":
        sns.histplot(data=Xg_boost_testdata,x="output_prob",hue="y pred binary",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=pred_fall_palette,hue_order = ['Fall', 'No Fall'])
    
    
    #ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set_ylabel("Probability",fontsize=10)
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        #ax[i].tick_params( labelleft=False)
    if i==2:
        ax[i].set_ylabel("Probability",fontsize=10)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Predicted risk of falling",fontsize=20)
        #ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('Probability',fontsize=10)
        ax[i].set_xlabel("Predicted risk of falling",fontsize=20)
    ax[i].set_xlabel("Predicted risk of falling",fontsize=10)

#fig.delaxes(gs[0, 1])
plt.show()


# In[45]:


true_fall_palette={"Fall":"C8","No Fall":"C3"}
SVM_testdata["y true"]=SVM_testdata["y_true"].apply(lambda x: "Fall" if x==1 else "No Fall")
LR_testdata["y true"]=LR_testdata["y_true"].apply(lambda x: "Fall" if x==1 else "No Fall")
RF_testdata["y true"]=RF_testdata["y_true"].apply(lambda x: "Fall" if x==1 else "No Fall")
FFNN_testdata["y true"]=FFNN_testdata["Fall"].apply(lambda x: "Fall" if x==1 else "No Fall")
Xg_boost_testdata["y true"]=Xg_boost_testdata["Fall"].apply(lambda x: "Fall" if x==1 else "No Fall")


# In[46]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(12,12))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","XGBoost"]
n_bins=50

#palette_custom={"Female":"C0","Male":"C1"}


for i,v in enumerate(list_of_models):
    
    ax[i].set_title(v,size=15)
    ax[i].set_xlim([0, 1])
    #ax[i].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ])
    ax[i].set_xticks([0,0.5,1.0])
    ax[i].grid(axis='x')
    
    
    
    if v=="SVM":
        sns.histplot(data=SVM_testdata,x="y_hat_probs",hue="y true",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=true_fall_palette,hue_order = ['Fall', 'No Fall'])
    if v=="LR":
        sns.histplot(data=LR_testdata,x="y_hat_probs",hue="y true",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=true_fall_palette,hue_order = ['Fall', 'No Fall'])
    if v=="RF":
        sns.histplot(data=RF_testdata,x="y_hat_probs",hue="y true",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=true_fall_palette,hue_order = ['Fall', 'No Fall'])
    
    if v=="FFNN":
        sns.histplot(data=FFNN_testdata,x="output_prob",hue="y true",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=true_fall_palette,hue_order = ['Fall', 'No Fall'])
    if v=="XGBoost":
        sns.histplot(data=Xg_boost_testdata,x="output_prob",hue="y true",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=true_fall_palette,hue_order = ['Fall', 'No Fall'])
    
    
    #ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set_ylabel("Probability",fontsize=10)
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        #ax[i].tick_params( labelleft=False)
    if i==2:
        ax[i].set_ylabel("Probability",fontsize=10)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Predicted risk of falling",fontsize=20)
        #ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('Probability',fontsize=10)
        ax[i].set_xlabel("Predicted risk of falling",fontsize=20)
    ax[i].set_xlabel("Predicted risk of falling",fontsize=10)

#fig.delaxes(gs[0, 1])
plt.show()


# In[47]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig,ax = plt.subplots(5,2,figsize=(15,25),sharey=True)#,sharey='row')

ax=ax.ravel()

#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)


plt.subplots_adjust(
            hspace = 0.2, wspace = 0.1)


list_of_models=["SVM","LR","RF","FFNN","XGBoost"]
n_bins=50

#palette_custom={"Female":"C0","Male":"C1"}

i=0

for j,v in enumerate(list_of_models):
    
    for g in ["Female", "Male"]:
        ax[i].set_title(v+" "+g+"s",size=15)
        ax[i].set_xlim([0, 1])
        #ax[i].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ])
        ax[i].set_xticks([0,0.5,1.0])
        ax[i].grid(axis='x')



        if v=="SVM":
            sns.histplot(data=SVM_testdata[SVM_testdata["Gender"]==g],x="y_hat_probs",hue="y true",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=true_fall_palette,hue_order = ['Fall', 'No Fall'])
        if v=="LR":
            sns.histplot(data=LR_testdata[LR_testdata["Gender"]==g],x="y_hat_probs",hue="y true",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=true_fall_palette,hue_order = ['Fall', 'No Fall'])
        if v=="RF":
            sns.histplot(data=RF_testdata[RF_testdata["Gender"]==g],x="y_hat_probs",hue="y true",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=true_fall_palette,hue_order = ['Fall', 'No Fall'])

        if v=="FFNN":
            sns.histplot(data=FFNN_testdata[FFNN_testdata["Gender"]==g],x="output_prob",hue="y true",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=true_fall_palette,hue_order = ['Fall', 'No Fall'])
        if v=="XGBoost":
            sns.histplot(data=Xg_boost_testdata[Xg_boost_testdata["Gender"]==g],x="output_prob",hue="y true",stat="probability",ax=ax[i],bins=n_bins,common_norm=True,palette=true_fall_palette,hue_order = ['Fall', 'No Fall'])

      
            
        if i%2==1:   
            ax[i].set_ylabel('',fontsize=10)
        if i!=8 and i!=9:   
            ax[i].set_xlabel("",fontsize=20)
        if i==8 or i==9:   
            ax[i].set_xlabel("Predicted prob",fontsize=10)
            
        
        
        
        i=i+1

#fig.delaxes(gs[0, 1])
plt.show()


# In[101]:


i


# # DI identification

# In[102]:


#SVM
calc_prop(SVM_testdata, "Gender", "Female", "y_hat_binary", 1)/calc_prop(SVM_testdata, "Gender", "Male", "y_hat_binary", 1)


# In[103]:


#LR
calc_prop(LR_testdata, "Gender", "Female", "y_hat_binary", 1)/calc_prop(LR_testdata, "Gender", "Male", "y_hat_binary", 1)


# In[104]:


#RF
calc_prop(RF_testdata, "Gender", "Female", "y_hat_binary", 1)/calc_prop(RF_testdata, "Gender", "Male", "y_hat_binary", 1)


# In[105]:


#FFNN
calc_prop(FFNN_testdata, "Gender", "Female", "output", 1)/calc_prop(FFNN_testdata, "Gender", "Male", "output", 1)


# In[106]:


#FFNN
calc_prop(Xg_boost_testdata, "Gender", "Female", "output", 1)/calc_prop(Xg_boost_testdata, "Gender", "Male", "output", 1)


# In[ ]:





# In[ ]:





# In[ ]:





# # 6 citizens

# In[9]:


cit6_nonstd=pd.read_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/6_cit_data_nonstd.csv").drop(columns=["Unnamed: 0"])


# In[10]:


data_6cit_FFNN=pd.read_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/FFNN_predictions.csv",index_col=0)
data_6cit_Xgboost=pd.read_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/Xgboost_predictions.csv",index_col=0)
data_6cit_SVM=pd.read_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/SVM_predictions.csv",index_col=0)
data_6cit_LR=pd.read_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/LR_predictions.csv",index_col=0)
data_6cit_RF=pd.read_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/RF_predictions.csv",index_col=0)


# ## Citizen 1

# In[11]:


cit6_nonstd.iloc[cit6_nonstd.index==0][["Gender","BirthYear","LoanPeriod","NumberAts","Fall"]]#["Cluster_1"]


# In[12]:


citnr=0

for mtype in ["SVM","LR","RF","FFNN","Xgboost"]:
    print("Model: ",mtype)
    
    if mtype=="LR":
        data=data_6cit_LR.iloc[data_6cit_LR.index==citnr]["y_hat_probs"]
    if mtype=="SVM":
        data=data_6cit_SVM.iloc[data_6cit_SVM.index==citnr]["y_hat_probs"]
    
    if mtype=="RF":
        data=data_6cit_RF.iloc[data_6cit_RF.index==citnr]["y_hat_probs"]
    
    if mtype=="FFNN":
        data=data_6cit_FFNN.iloc[data_6cit_FFNN.index==citnr]["output_prob"]
        
    if mtype=="Xgboost":
        data=data_6cit_Xgboost.iloc[data_6cit_Xgboost.index==citnr]["output_prob"]
    
        
    
    m=np.mean(data)
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 


    print(round(m*100,1))
    print("("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")")


# In[111]:


data_6cit_Xgboost.iloc[data_6cit_Xgboost.index==5]["output"].mean()


# # Citizen 2

# In[13]:


cit6_nonstd.iloc[cit6_nonstd.index==1][["Gender","BirthYear","LoanPeriod","NumberAts","Fall"]]


# In[14]:


citnr=1
str1="& "
str2="& "

for mtype in ["SVM","LR","RF","FFNN","Xgboost"]:
    print("Model: ",mtype)
    if mtype=="LR":
        data=data_6cit_LR.iloc[data_6cit_LR.index==citnr]["y_hat_probs"]
    if mtype=="SVM":
        data=data_6cit_SVM.iloc[data_6cit_SVM.index==citnr]["y_hat_probs"]
    
    if mtype=="RF":
        data=data_6cit_RF.iloc[data_6cit_RF.index==citnr]["y_hat_probs"]
    
    if mtype=="FFNN":
        data=data_6cit_FFNN.iloc[data_6cit_FFNN.index==citnr]["output_prob"]
        
    if mtype=="Xgboost":
        data=data_6cit_Xgboost.iloc[data_6cit_Xgboost.index==citnr]["output_prob"]
    
        
    m=np.mean(data)
    str1=str1+f"{round(m*100,1)}"+" &"
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    str2=str2+"("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")"+" &"


    print(round(m*100,1))
    print("("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")")


# In[114]:


str1


# In[115]:


str2


# ## Citizen 3

# In[15]:


cit6_nonstd.iloc[cit6_nonstd.index==2][["Gender","BirthYear","LoanPeriod","NumberAts","Fall"]]


# In[16]:


citnr=2
str1="& "
str2="& "
for mtype in ["SVM","LR","RF","FFNN","Xgboost"]:
    print("Model: ",mtype)
    if mtype=="LR":
        data=data_6cit_LR.iloc[data_6cit_LR.index==citnr]["y_hat_probs"]
    if mtype=="SVM":
        data=data_6cit_SVM.iloc[data_6cit_SVM.index==citnr]["y_hat_probs"]
    
    if mtype=="RF":
        data=data_6cit_RF.iloc[data_6cit_RF.index==citnr]["y_hat_probs"]
    
    if mtype=="FFNN":
        data=data_6cit_FFNN.iloc[data_6cit_FFNN.index==citnr]["output_prob"]
        
    if mtype=="Xgboost":
        data=data_6cit_Xgboost.iloc[data_6cit_Xgboost.index==citnr]["output_prob"]
    
        
    m=np.mean(data)
    str1=str1+f"{round(m*100,1)}"+" &"
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    str2=str2+"("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")"+" &"


    #print(round(m*100,1))
    #print("("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")")
print(str1)
print(str2)


# ## Citizen 4

# In[17]:


cit6_nonstd.iloc[cit6_nonstd.index==3][["Gender","BirthYear","LoanPeriod","NumberAts","Fall"]]


# In[18]:


citnr=3
str1="& "
str2="& "
for mtype in ["SVM","LR","RF","FFNN","Xgboost"]:
    print("Model: ",mtype)
    if mtype=="LR":
        data=data_6cit_LR.iloc[data_6cit_LR.index==citnr]["y_hat_probs"]
    if mtype=="SVM":
        data=data_6cit_SVM.iloc[data_6cit_SVM.index==citnr]["y_hat_probs"]
    
    if mtype=="RF":
        data=data_6cit_RF.iloc[data_6cit_RF.index==citnr]["y_hat_probs"]
    
    if mtype=="FFNN":
        data=data_6cit_FFNN.iloc[data_6cit_FFNN.index==citnr]["output_prob"]
        
    if mtype=="Xgboost":
        data=data_6cit_Xgboost.iloc[data_6cit_Xgboost.index==citnr]["output_prob"]
    
        
    m=np.mean(data)
    str1=str1+f"{round(m*100,1)}"+" &"
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    str2=str2+"("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")"+" &"


    #print(round(m*100,1))
    #print("("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")")
print(str1)
print(str2)


# In[120]:


data_6cit_LR.iloc[data_6cit_LR.index==3]["y_hat_binary"].mean()


# ## Citizen 5

# In[19]:


cit6_nonstd.iloc[cit6_nonstd.index==4][["Gender","BirthYear","LoanPeriod","NumberAts","Fall"]]


# In[20]:


citnr=4
str1="& "
str2="& "
for mtype in ["SVM","LR","RF","FFNN","Xgboost"]:
    print("Model: ",mtype)
    if mtype=="LR":
        data=data_6cit_LR.iloc[data_6cit_LR.index==citnr]["y_hat_probs"]
    if mtype=="SVM":
        data=data_6cit_SVM.iloc[data_6cit_SVM.index==citnr]["y_hat_probs"]
    
    if mtype=="RF":
        data=data_6cit_RF.iloc[data_6cit_RF.index==citnr]["y_hat_probs"]
    
    if mtype=="FFNN":
        data=data_6cit_FFNN.iloc[data_6cit_FFNN.index==citnr]["output_prob"]
        
    if mtype=="Xgboost":
        data=data_6cit_Xgboost.iloc[data_6cit_Xgboost.index==citnr]["output_prob"]
    
        
    m=np.mean(data)
    str1=str1+f"{round(m*100,1)}"+" &"
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    str2=str2+"("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")"+" &"


    #print(round(m*100,1))
    #print("("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")")
print(str1)
print(str2)


# ## Citizen 6

# In[123]:


cit6_nonstd.iloc[cit6_nonstd.index==5][["Gender","BirthYear","LoanPeriod","NumberAts","Fall"]]


# In[21]:


citnr=5
str1="& "
str2="& "
for mtype in ["SVM","LR","RF","FFNN","Xgboost"]:
    print("Model: ",mtype)
    if mtype=="LR":
        data=data_6cit_LR.iloc[data_6cit_LR.index==citnr]["y_hat_probs"]
    if mtype=="SVM":
        data=data_6cit_SVM.iloc[data_6cit_SVM.index==citnr]["y_hat_probs"]
    
    if mtype=="RF":
        data=data_6cit_RF.iloc[data_6cit_RF.index==citnr]["y_hat_probs"]
    
    if mtype=="FFNN":
        data=data_6cit_FFNN.iloc[data_6cit_FFNN.index==citnr]["output_prob"]
        
    if mtype=="Xgboost":
        data=data_6cit_Xgboost.iloc[data_6cit_Xgboost.index==citnr]["output_prob"]
    
        
    m=np.mean(data)
    str1=str1+f"{round(m*100,1)}"+" &"
    (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    
    str2=str2+"("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")"+" &"


    #print(round(m*100,1))
    #print("("+str(round(slow*100,1))+"-"+str(round(shigh*100,1))+")")
print(str1)
print(str2)


# # ROC curves LR

# In[125]:


female_index=LR_testdata["Gender"]=="Female"
male_index=LR_testdata["Gender"]=="Male"



fpr_female, tpr_female, thresholds_female = metrics.roc_curve(LR_testdata[female_index]["y_true"], LR_testdata[female_index]["y_hat_probs"],drop_intermediate=False)
roc_auc_female = metrics.auc(fpr_female, tpr_female)

fpr_male, tpr_male, thresholds_male = metrics.roc_curve(LR_testdata[male_index]["y_true"], LR_testdata[male_index]["y_hat_probs"],drop_intermediate=False)
roc_auc_male = metrics.auc(fpr_male, tpr_male)


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))

plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("Male and females")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[126]:


fpr_all, tpr_all, thresholds_all = metrics.roc_curve(LR_testdata["y_true"], LR_testdata["y_hat_probs"],drop_intermediate=False)
roc_auc_all = metrics.auc(fpr_all, tpr_all)


plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))


plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("All")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[127]:


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))
plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))


plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("LR ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # ROC curves SVM

# In[128]:


female_index=SVM_testdata["Gender"]=="Female"
male_index=SVM_testdata["Gender"]=="Male"



fpr_female, tpr_female, thresholds_female = metrics.roc_curve(SVM_testdata[female_index]["y_true"], SVM_testdata[female_index]["y_hat_probs"],drop_intermediate=False)
roc_auc_female = metrics.auc(fpr_female, tpr_female)

fpr_male, tpr_male, thresholds_male = metrics.roc_curve(SVM_testdata[male_index]["y_true"], SVM_testdata[male_index]["y_hat_probs"],drop_intermediate=False)
roc_auc_male = metrics.auc(fpr_male, tpr_male)


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))

plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("Male and females")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[129]:


fpr_all, tpr_all, thresholds_all = metrics.roc_curve(SVM_testdata["y_true"], SVM_testdata["y_hat_probs"],drop_intermediate=False)
roc_auc_all = metrics.auc(fpr_all, tpr_all)


plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))


plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("All")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[130]:


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))
plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))


plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("SVM ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # ROC curves RF

# In[131]:


female_index=RF_testdata["Gender"]=="Female"
male_index=RF_testdata["Gender"]=="Male"



fpr_female, tpr_female, thresholds_female = metrics.roc_curve(RF_testdata[female_index]["y_true"], RF_testdata[female_index]["y_hat_probs"],drop_intermediate=False)
roc_auc_female = metrics.auc(fpr_female, tpr_female)

fpr_male, tpr_male, thresholds_male = metrics.roc_curve(RF_testdata[male_index]["y_true"], RF_testdata[male_index]["y_hat_probs"],drop_intermediate=False)
roc_auc_male = metrics.auc(fpr_male, tpr_male)


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))

plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("Male and females")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[132]:


fpr_all, tpr_all, thresholds_all = metrics.roc_curve(RF_testdata["y_true"], RF_testdata["y_hat_probs"],drop_intermediate=False)
roc_auc_all = metrics.auc(fpr_all, tpr_all)


plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))


plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("All")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[133]:


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))
plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))


plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("RF ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # ROC curves FFNN

# In[134]:


female_index=FFNN_testdata["Gender"]=="Female"
male_index=FFNN_testdata["Gender"]=="Male"



fpr_female, tpr_female, thresholds_female = metrics.roc_curve(FFNN_testdata[female_index]["Fall"], FFNN_testdata[female_index]["output_prob"],drop_intermediate=False)
roc_auc_female = metrics.auc(fpr_female, tpr_female)

fpr_male, tpr_male, thresholds_male = metrics.roc_curve(FFNN_testdata[male_index]["Fall"], FFNN_testdata[male_index]["output_prob"],drop_intermediate=False)
roc_auc_male = metrics.auc(fpr_male, tpr_male)


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))

plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("Male and females")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[135]:


fpr_all, tpr_all, thresholds_all = metrics.roc_curve(FFNN_testdata["Fall"], FFNN_testdata["output_prob"],drop_intermediate=False)
roc_auc_all = metrics.auc(fpr_all, tpr_all)


plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))


plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("All")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[136]:



plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))
plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))


plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("FFNN ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # ROC curves xgboost

# In[137]:


female_index=Xg_boost_testdata["Gender"]=="Female"
male_index=Xg_boost_testdata["Gender"]=="Male"



fpr_female, tpr_female, thresholds_female = metrics.roc_curve(Xg_boost_testdata[female_index]["Fall"], Xg_boost_testdata[female_index]["output_prob"],drop_intermediate=False)
roc_auc_female = metrics.auc(fpr_female, tpr_female)

fpr_male, tpr_male, thresholds_male = metrics.roc_curve(Xg_boost_testdata[male_index]["Fall"], Xg_boost_testdata[male_index]["output_prob"],drop_intermediate=False)
roc_auc_male = metrics.auc(fpr_male, tpr_male)


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))

plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("Male and females")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[138]:


fpr_all, tpr_all, thresholds_all = metrics.roc_curve(Xg_boost_testdata["Fall"], Xg_boost_testdata["output_prob"],drop_intermediate=False)
roc_auc_all = metrics.auc(fpr_all, tpr_all)


plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))


plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("All")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[139]:




plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(round(roc_auc_male,2)))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(round(roc_auc_female,2)))
plt.plot(fpr_all,tpr_all,color="Black",label="All, auc="+str(round(roc_auc_all,2)))


plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("Xgboost ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[ ]:





# In[14]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(12,12))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","XGBoost"]
n_bins=50

#palette_custom={"Female":"C0","Male":"C1"}


for i,v in enumerate(list_of_models):
    
    ax[i].set_title(v,size=15)
    ax[i].set_xlim([0, 1])
    #ax[i].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ])
    ax[i].set_xticks([0,0.2,0.4,0.6,0.8,1.0 ])
    #ax[i].set_xticks([0,0.5,1.0])
    ax[i].grid(axis='x')
    
    
    
    if v=="SVM":
        create_custom_roc_plot_ax(SVM_testdata,"y_true","y_hat_probs",ax[i])
    if v=="LR":
        create_custom_roc_plot_ax(LR_testdata,"y_true","y_hat_probs",ax[i])
    if v=="RF":
        create_custom_roc_plot_ax(RF_testdata,"y_true","y_hat_probs",ax[i])
    
    if v=="FFNN":
        create_custom_roc_plot_ax(FFNN_testdata,"Fall","output_prob",ax[i])
    if v=="XGBoost":
        create_custom_roc_plot_ax(Xg_boost_testdata,"Fall","output_prob",ax[i])
    
    
    #ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set(ylabel='TPR')
        ax[i].tick_params( labelbottom=False)
        
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        ax[i].set_ylabel("",fontsize=10)
        ax[i].tick_params( labelleft=False)
        ax[i].tick_params( labelbottom=False)
    if i==2:
        ax[i].set_ylabel("TPR",fontsize=10)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("FPR",fontsize=10)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("FPR",fontsize=10)
        ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('TPR',fontsize=10)
        ax[i].set_xlabel("FPR",fontsize=10)

#fig.delaxes(gs[0, 1])
plt.show()


# In[ ]:





# # Relation + threshold

# In[ ]:





# In[47]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(12,12))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","Xgboost"]
n_bins=50

#palette_custom={"Female":"C0","Male":"C1"}


for i,v in enumerate(list_of_models):
    
    ax[i].set_title(v,size=15)
    ax[i].set_xlim([0, 1])
    #ax[i].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ])
    ax[i].set_xticks([0,0.2,0.4,0.6,0.8,1.0 ])
    #ax[i].set_xticks([0,0.5,1.0])
    ax[i].grid(axis='x')
    
    
    
    if v=="SVM":
        create_custom_roc_plot_relation(SVM_testdata,"y_true","y_hat_probs","TPR",ax[i])
    if v=="LR":
        create_custom_roc_plot_relation(LR_testdata,"y_true","y_hat_probs","TPR",ax[i])
    if v=="RF":
        create_custom_roc_plot_relation(RF_testdata,"y_true","y_hat_probs","TPR",ax[i])
    
    if v=="FFNN":
        create_custom_roc_plot_relation(FFNN_testdata,"Fall","output_prob","TPR",ax[i])
    if v=="Xgboost":
        create_custom_roc_plot_relation(Xg_boost_testdata,"Fall","output_prob","TPR",ax[i])
    
    ax[i].get_legend().remove()
    #ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set(ylabel='TPR relation')
        ax[i].tick_params( labelbottom=False)
        
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        ax[i].set_ylabel("",fontsize=10)
        #ax[i].tick_params( labelleft=False)
        ax[i].tick_params( labelbottom=False)
    if i==2:
        ax[i].set_ylabel("TPR relation",fontsize=10)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Threshold",fontsize=10)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Threshold",fontsize=10)
        #ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('TPR relation',fontsize=10)
        ax[i].set_xlabel("Threshold",fontsize=10)

#fig.delaxes(gs[0, 1])
plt.show()


# In[ ]:





# In[46]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(12,12))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","Xgboost"]
n_bins=50

#palette_custom={"Female":"C0","Male":"C1"}


for i,v in enumerate(list_of_models):
    
    ax[i].set_title(v,size=15)
    ax[i].set_xlim([0, 1])
    #ax[i].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ])
    ax[i].set_xticks([0,0.2,0.4,0.6,0.8,1.0 ])
    #ax[i].set_xticks([0,0.5,1.0])
    ax[i].grid(axis='x')
    
    
    
    if v=="SVM":
        create_custom_roc_plot_relation(SVM_testdata,"y_true","y_hat_probs","TNR",ax[i])
    if v=="LR":
        create_custom_roc_plot_relation(LR_testdata,"y_true","y_hat_probs","TNR",ax[i])
    if v=="RF":
        create_custom_roc_plot_relation(RF_testdata,"y_true","y_hat_probs","TNR",ax[i])
    
    if v=="FFNN":
        create_custom_roc_plot_relation(FFNN_testdata,"Fall","output_prob","TNR",ax[i])
    if v=="Xgboost":
        create_custom_roc_plot_relation(Xg_boost_testdata,"Fall","output_prob","TNR",ax[i])
    
    ax[i].get_legend().remove()
    #ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set(ylabel='TNR relation')
        ax[i].tick_params( labelbottom=False)
        
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        ax[i].set_ylabel("",fontsize=10)
        #ax[i].tick_params( labelleft=False)
        ax[i].tick_params( labelbottom=False)
    if i==2:
        ax[i].set_ylabel("TNR relation",fontsize=10)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Threshold",fontsize=10)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Threshold",fontsize=10)
        #ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('TNR relation',fontsize=10)
        ax[i].set_xlabel("Threshold",fontsize=10)

#fig.delaxes(gs[0, 1])
plt.show()


# In[45]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(12,12))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","Xgboost"]
n_bins=50

#palette_custom={"Female":"C0","Male":"C1"}


for i,v in enumerate(list_of_models):
    
    ax[i].set_title(v,size=15)
    ax[i].set_xlim([0, 1])
    #ax[i].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ])
    ax[i].set_xticks([0,0.2,0.4,0.6,0.8,1.0 ])
    #ax[i].set_xticks([0,0.5,1.0])
    ax[i].grid(axis='x')
    
    
    
    if v=="SVM":
        create_custom_roc_plot_relation(SVM_testdata,"y_true","y_hat_probs","FPR",ax[i])
    if v=="LR":
        create_custom_roc_plot_relation(LR_testdata,"y_true","y_hat_probs","FPR",ax[i])
    if v=="RF":
        create_custom_roc_plot_relation(RF_testdata,"y_true","y_hat_probs","FPR",ax[i])
    
    if v=="FFNN":
        create_custom_roc_plot_relation(FFNN_testdata,"Fall","output_prob","FPR",ax[i])
    if v=="Xgboost":
        create_custom_roc_plot_relation(Xg_boost_testdata,"Fall","output_prob","FPR",ax[i])
    
    ax[i].get_legend().remove()
    #ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set(ylabel='FPR relation')
        ax[i].tick_params( labelbottom=False)
        
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        ax[i].set_ylabel("",fontsize=10)
        #ax[i].tick_params( labelleft=False)
        ax[i].tick_params( labelbottom=False)
    if i==2:
        ax[i].set_ylabel("FPR relation",fontsize=10)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Threshold",fontsize=10)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Threshold",fontsize=10)
        #ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('FPR relation',fontsize=10)
        ax[i].set_xlabel("Threshold",fontsize=10)

#fig.delaxes(gs[0, 1])
plt.show()


# In[44]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(12,12))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","Xgboost"]
n_bins=50

#palette_custom={"Female":"C0","Male":"C1"}


for i,v in enumerate(list_of_models):
    
    ax[i].set_title(v,size=15)
    ax[i].set_xlim([0, 1])
    #ax[i].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ])
    ax[i].set_xticks([0,0.2,0.4,0.6,0.8,1.0 ])
    #ax[i].set_xticks([0,0.5,1.0])
    ax[i].grid(axis='x')
    
    
    
    if v=="SVM":
        create_custom_roc_plot_relation(SVM_testdata,"y_true","y_hat_probs","FNR",ax[i])
    if v=="LR":
        create_custom_roc_plot_relation(LR_testdata,"y_true","y_hat_probs","FNR",ax[i])
    if v=="RF":
        create_custom_roc_plot_relation(RF_testdata,"y_true","y_hat_probs","FNR",ax[i])
    
    if v=="FFNN":
        create_custom_roc_plot_relation(FFNN_testdata,"Fall","output_prob","FNR",ax[i])
    if v=="Xgboost":
        create_custom_roc_plot_relation(Xg_boost_testdata,"Fall","output_prob","FNR",ax[i])
    
    ax[i].get_legend().remove()
    #ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set(ylabel='FNR relation')
        ax[i].tick_params( labelbottom=False)
        
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        ax[i].set_ylabel("",fontsize=10)
        #ax[i].tick_params( labelleft=False)
        ax[i].tick_params( labelbottom=False)
    if i==2:
        ax[i].set_ylabel("FNR relation",fontsize=10)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Threshold",fontsize=10)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("Threshold",fontsize=10)
        #ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('FNR relation',fontsize=10)
        ax[i].set_xlabel("Threshold",fontsize=10)

#fig.delaxes(gs[0, 1])
plt.show()


# In[ ]:





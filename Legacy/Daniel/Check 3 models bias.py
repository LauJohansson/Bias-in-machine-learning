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


# # FILE PATH

# In[4]:




### USE ALL DATA
#file_name="fall.csv"
#full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name



###USE SMALL TEST

PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/Daniel/"

file_name="something.csv"
#file_name="all_test_data_globalmodel.csv"


full_file_path=PATH_orig+file_name


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


# # Read the data

# In[6]:



df2 = pd.read_csv(full_file_path)

#test_data_0 = pd.read_csv(PATH_orig+"model0/"+file_name)
#test_data_1 = pd.read_csv(PATH_orig+"model1/"+file_name)
#test_data_2 = pd.read_csv(PATH_orig+"model2/"+file_name)
#test_data_3 = pd.read_csv(PATH_orig+"model3/"+file_name)
#test_data_4 = pd.read_csv(PATH_orig+"model4/"+file_name)
#test_data_5 = pd.read_csv(PATH_orig+"model5/"+file_name)
#test_data_6 = pd.read_csv(PATH_orig+"model6/"+file_name)
#test_data_7 = pd.read_csv(PATH_orig+"model7/"+file_name)
#test_data_8 = pd.read_csv(PATH_orig+"model8/"+file_name)
#test_data_9 = pd.read_csv(PATH_orig+"model9/"+file_name)

#df2=    pd.concat([test_data_0,
#                    test_data_1,
#                    test_data_2,
#                    test_data_3,
#                    test_data_4,
#                    test_data_5,
#                    test_data_6,
#                    test_data_7,
#                    test_data_8,
#                    test_data_9
#                   ],sort=False,axis=0)

#df2 = pd.read_csv(full_file_path)


# In[ ]:





# In[7]:


df_X=df2[X_col_names]
df_y=df2[[y_col_name]]


# In[8]:


X=np.array(df_X)
y=np.array(df_y)


# In[ ]:





# In[ ]:





# # Functions

# In[12]:


# @title get_df_w_metric(double click to expand)

def get_df_w_metrics(df,protected_variable_name,y_target_name,y_pred_name):
    """
    This function takes a dataframe (df), and returns FPR/FNR for each value in the protected variable

    Input: 
        df:                         a dataframe
        protected_variable_name:    the name of the protected variable in the df
        y_target_name:              the name of the target variable in the df
        y_pred_name:                the name of the predticted variable in the df
    """

    #Create empty DataFrame
    confusion_df=pd.DataFrame(columns=[protected_variable_name,"FPR","FNR"])
    


    #For each value of the protected variable, calculated FPR/FNR and insert into the empty DataFrame
    for name in list(df[protected_variable_name].unique()):
        a=df[df[protected_variable_name]==name][y_target_name]
        b=df[df[protected_variable_name]==name][y_pred_name]#.apply(lambda x: 0 if x<t else 1 )

    
        TN, FP, FN, TP = confusion_matrix(list(a), list(b),labels=[0, 1]).ravel()
        
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)

        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        LRplus=TPR/FPR
        LRminus=FNR/TNR


        #F1-score
        F1=2*(PPV*TPR)/(PPV+TPR)

        confusion_df=confusion_df.append({protected_variable_name:name,
                                          "TPR":TPR,
                                          "TNR":TNR,
                                          "FPR":FPR,
                                          "FNR":FNR,
                                          "PPV":PPV,
                                          "NPV":NPV,
                                          "FDR":FDR,
                                          "ACC":ACC,
                                          "F1":F1,
                                          "LRplus":LRplus,
                                          "TN":TN,
                                          "FP":FP,
                                          "FN":FN,
                                          "TP":TP
                                          },ignore_index=True)

    return confusion_df


# In[13]:


#from: https://nbviewer.jupyter.org/github/srnghn/bias-mitigation-examples/blob/master/Bias%20Mitigation%20with%20Disparate%20Impact%20Remover.ipynb
def calc_prop(data, group_col, group, output_col, output_val):
    '''
    data:       The dataframe
    group_col:  The protected atrtibute column (e.g Gender)
    group:      The chosen group (e.g Male or Female)
    output_col: The column holding the y-value (either y_hat or y   - could be Fall)
    output_val: The value of the y  (e.g.   all y=1 )
    
    
    Example:
    
    Find p(y=0 | G="Female")
    
    calc_prop(data,"Gender","Female","y_true",0)
    
    
    
    '''
    new = data[data[group_col] == group]
    return len(new[new[output_col] == output_val])/len(new)


# In[14]:


def calc_prop_no_group(data, output_col, output_val):
    return len(data[data[output_col] == output_val])/len(data)


# In[15]:


def compare_bias_metrics(data,protected_variable_name,y_target_name,y_pred_name,unfavourable_name,favourable_name,print_var=False,fav_value=1,unfav_value=0):
    
    
    df = get_df_w_metrics(data,protected_variable_name,y_target_name,y_pred_name)
    
    
    
    #==================== DISPARATE IMPACT ======================#
    #Feldman et al
    
    #a1=data[y_target_name]
    #b1=data[protected_variable_name]#.apply(lambda x: 0 if x<t else 1 )
    #d, b, c, a = confusion_matrix(list(a1), list(b1),labels=[0, 1]).ravel()
    
    a=calc_prop(data,protected_variable_name,unfavourable_name,y_target_name,unfav_value) #prop of unfav group, recieve unfav value
    b=calc_prop(data,protected_variable_name,favourable_name,y_target_name,unfav_value) #prop of ufav group, recieve unfav value
    c=calc_prop(data,protected_variable_name,unfavourable_name,y_target_name,fav_value) #prop of ufav group, recieve fav value
    d=calc_prop(data,protected_variable_name,favourable_name,y_target_name,fav_value) #prop of ufav group, recieve fav value
    
    
    #a=df[df[protected_variable_name]==unfavourable_name]["TN"].item()
    #b=df[df[protected_variable_name]==favourable_name]["TN"].item()
    #c=df[df[protected_variable_name]==unfavourable_name]["TP"].item()
    #d=df[df[protected_variable_name]==favourable_name]["TP"].item()
        
    Feldman_Disparate_impact=(c/(a + c)) / (d/(b + d))
    
    if print_var:
        print("==================================== Feldman et al ====================================")
        
        print(f"If DATASET has no disparate impact then DI>=0.8.")
        print(f"DI={Feldman_Disparate_impact}")
        
        if Feldman_Disparate_impact>=0.8:
            print("The DATASET has no disparate impact")
        else:
            print("The DATASET has disparate impact")
        print("======================================================================================= \n")
            
    #===========================================================#
    
    
    #==================== Learning Fair representations ======================#
    #Zafar et al
    
    ###Disparate TREATMENT ####
    
    p_yhat1_z1=calc_prop(data,protected_variable_name,favourable_name,y_pred_name,fav_value)
    p_yhat1_z0=calc_prop(data,protected_variable_name,unfavourable_name,y_pred_name,fav_value)
    p_yhat1=calc_prop_no_group(data, y_pred_name, fav_value)
    
    p_yhat0_z1=calc_prop(data,protected_variable_name,favourable_name,y_pred_name,unfav_value)
    p_yhat0_z0=calc_prop(data,protected_variable_name,unfavourable_name,y_pred_name,unfav_value)
    p_yhat0=calc_prop_no_group(data, y_pred_name, unfav_value)
    
    
    ###Disparate IMPACT ####
    
    if print_var:
        print("==================================== Zafar et al ====================================")
        
        print(f"If the CLASSIFIER has no DISPARATE TREATMENT, these equations should hold:")
        print(f"P(y_hat=1|z={favourable_name},x) = P(y_hat=1,x) <=> {round(p_yhat1_z1,2)} = {round(p_yhat1,2)}")
        print(f"P(y_hat=1|z={unfavourable_name},x) = P(y_hat=1,x) <=> {round(p_yhat1_z0,2)} = {round(p_yhat1,2)}")
        print(f"P(y_hat=0|z={favourable_name},x) = P(y_hat=0,x) <=> {round(p_yhat0_z1,2)} = {round(p_yhat0,2)}")
        print(f"P(y_hat=0|z={unfavourable_name},x) = P(y_hat=0,x) <=> {round(p_yhat0_z0,2)} = {round(p_yhat0,2)}")
        print("\n")
        
        print(f"If the CLASSIFIER has no DISPARATE IMPACT, these equations should hold:")
        print(f"P(y_hat=1|z={unfavourable_name}) = P(y_hat=1,z={favourable_name}) <=> {round(p_yhat1_z0,2)} = {round(p_yhat1_z1,2)}")
        print("\n")
        
        print(f"If the CLASSIFIER has no DISPARATE MISTREATMENT, these equations should hold:")
        
        FPR_z0=df[df[protected_variable_name]==unfavourable_name]["FPR"].item()
        FPR_z1=df[df[protected_variable_name]==favourable_name]["FPR"].item()
        
        FNR_z0=df[df[protected_variable_name]==unfavourable_name]["FNR"].item()
        FNR_z1=df[df[protected_variable_name]==favourable_name]["FNR"].item()
        
        
        
        print(f"FPR: P(y_hat!=y|z={unfavourable_name},y=0) = P(y_hat!=y|z={favourable_name},y=0) <=> {round(FPR_z0,2)} = {round(FPR_z1,2)}")
        print(f"FNR: P(y_hat!=y|z={unfavourable_name},y=1) = P(y_hat!=y|z={favourable_name},y=1) <=> {round(FNR_z0,2)} = {round(FNR_z1,2)}")
        
        print("======================================================================================= \n")
    
    
    
    
    
    #==================== Equality of Opportunity in Supervised Learning ======================#
    #Hardt et al
    
    ### Equalized odds ####
    
    TPR_z0=df[df[protected_variable_name]==unfavourable_name]["TPR"].item()
    TPR_z1=df[df[protected_variable_name]==favourable_name]["TPR"].item()
    
    TNR_z0=df[df[protected_variable_name]==unfavourable_name]["TNR"].item()
    TNR_z1=df[df[protected_variable_name]==favourable_name]["TNR"].item()
    
    if print_var:
        print("==================================== Hardt et al ====================================")
        
        print(f"If the CLASSIFIER has EQUALIZED ODDS, these equations should hold:")
        print(f"P(y_hat=1|z={unfavourable_name},y=1) = P(y_hat=1|z={favourable_name},y=1) <=> {round(TPR_z0,2)} = {round(TPR_z1,2)}")
        print(f"P(y_hat=0|z={unfavourable_name},y=0) = P(y_hat=0|z={favourable_name},y=0) <=> {round(TNR_z0,2)} = {round(TNR_z1,2)}")
        print("\n")
        
        print(f"If the CLASSIFIER has EQUAL OPPORTUNITY, these equations should hold:")
        print(f"P(y_hat=1|z={unfavourable_name},y=1) = P(y_hat=1|z={favourable_name},y=1) <=> {round(TPR_z0,2)} = {round(TPR_z1,2)}")
        print("\n")
    
        print("======================================================================================= \n")
    
    
    ############Measuring racial discrimination in algorithms####
    #Arnold et al. 
    
    
    
    if print_var:
        print("==================================== Arnold et al ====================================")
        
        my=data[y_target_name].mean()
        delta=(TNR_z1-TNR_z0)*(1-my)+(FNR_z1-FNR_z0)*my
        
        
        print(f"The racial discrimination paramenter (delta) = {delta}")
        print("\n")
    
        print("======================================================================================= \n")
    
        
        
        
        
    ############GENERAL CLASSIFICATION METRICS####
    
    
    
    if print_var:
        print("==================================== GENERAL CLASSIFICATION METRICS ====================================")
        
        print(f"TPR for {unfavourable_name}: {TPR_z0}")
        print(f"TPR for {favourable_name}: {TPR_z1}")
        print("\n")
        
        print(f"TNR for {unfavourable_name}: {TNR_z0}")
        print(f"TNR for {favourable_name}: {TNR_z1}")
        print("\n")
        
        print(f"FPR for {unfavourable_name}: {FPR_z0}")
        print(f"FPR for {favourable_name}:  {FPR_z1}")
        print("\n")
        
        print(f"FNR for {unfavourable_name}:  {FNR_z0}")
        print(f"FNR for {favourable_name}:  {FNR_z1}")
        print("\n")
    
        print("======================================================================================= \n")
    
        
        
        
    
    
    
    return Feldman_Disparate_impact
    


#  

#  

#  

# # Identify bias

# In[25]:


print(f"The dataset has {df2.shape[0]} rows")
print(f"The dataset has {df2.shape[1]} cols")

print(f"The dataset to test has the protected attribute: {procted_col_name}")
print(f"The protected variable can be assigned: {df2[procted_col_name].unique()}")

print(f"The ratio of records with y=1: {df2[df2[y_col_name]==1][y_col_name].count()/df2[y_col_name].count()}")


# In[26]:


metrics=compare_bias_metrics(data=df2,
                        protected_variable_name=procted_col_name,
                        y_target_name=y_col_name,
                        y_pred_name=output_col_name,
                        unfavourable_name=0,
                        favourable_name=1,
                        print_var=True
                        
                       )


# In[ ]:





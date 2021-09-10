#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install fairlearn


# In[2]:


#pip install BlackBoxAuditing


# In[3]:


import pandas as pd
from sklearn import preprocessing
import math 
import numpy as np
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# In[4]:


plt.plot([0,0])


# In[5]:


fall_data = pd.read_csv('/restricted/s164512/G2020-57-Aalborg-bias/Data_air/Fall_count_clusterOHE_std.csv')


# # Gender bias

# In[6]:


X = fall_data.drop(columns=['Fall',"Unnamed: 0"]) # using all covariates in the dataset. ,'Ats_0'
#X1 = pd.DataFrame(preprocessing.scale(X1),columns=X1.columns)
#X2 = pd.DataFrame(fall_data['Gender'])
#X = pd.concat([X1],axis=1)
y = fall_data[['Gender','Fall']]


# In[7]:


def DI_remove_custom(df_train,RP_level=1.0):

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


# In[8]:


kf=KFold(n_splits=3, random_state=2, shuffle=True)

for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    X_train_temp = X_train.reset_index(drop=True)
    X_test_temp = X_test.reset_index(drop=True)
    
    X_train_rp = DI_remove_custom(X_train_temp)  
    X_test_rp  = DI_remove_custom(X_test_temp)


# In[ ]:





# In[9]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

## Creating empty lists for:
# TP/TN/FP/FN
TP_list_W=[]
TN_list_W=[]
FP_list_W=[]
FN_list_W=[]
F1_list_W=[]
ACC_list_W=[]
TPR_list_W=[]
TNR_list_W=[]
FPR_list_W=[]
FNR_list_W=[]
yhat_list_W=[]
yhat_prob_list_W=[]

TP_list_M=[]
TN_list_M=[]
FP_list_M=[]
FN_list_M=[]
F1_list_M=[]
ACC_list_M=[]
TPR_list_M=[]
TNR_list_M=[]
FPR_list_M=[]
FNR_list_M=[]
yhat_list_M=[]
yhat_prob_list_M=[]

ACC_list_total=[]

class_names = ['No fall','Fall']

model_counter=0


classified_df_M = pd.DataFrame([],columns=X.columns)
classified_df_M['y_true'] = []
classified_df_M['y_hat_binary'] = []
classified_df_M['y_hat_probs'] = []

classified_df_W = pd.DataFrame([],columns=X.columns)
classified_df_W['y_true'] = []
classified_df_W['y_hat_binary'] = []
classified_df_W['y_hat_probs'] = []



for i in range(1,11):
    
    kf=KFold(n_splits=5, random_state=i, shuffle=True)
    
    for train_index, test_index in kf.split(X):
        print("Running model ",model_counter)
        model_counter=model_counter+1
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train_temp = X_train.reset_index(drop=True)
        X_test_temp = X_test.reset_index(drop=True)
        
        #y_train = y_train.reset_index(drop=True)
        #y_test = y_test.reset_index(drop=True)
        
        X_train_rp = DI_remove_custom(X_train_temp)  
        X_test_rp  = DI_remove_custom(X_test_temp)
        
    #### Flip fra den ene til den anden for at skifte mellem SVM, LR og RF ###
    
        classifier = svm.SVC(kernel='rbf', C=1, random_state=0,class_weight='balanced',probability=True).fit(X_train_rp.drop(columns=["Gender"]), y_train['Fall'])
        #classifier = LogisticRegression(max_iter=1000,class_weight='balanced').fit(X_train_rp.drop(columns=["Gender"]), y_train['Fall'])
        #classifier = RandomForestClassifier(random_state=1).fit(X_train_rp.drop(columns=["Gender"]), y_train['Fall'])
    
    
    
   # class weight balanced?  
   # 
        np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None)]
    
    
        
     ############ FOR WOMEN ################   
        
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, X_test_rp[X_test_rp['Gender']==0].drop(columns=['Gender']),
                                         y_test[y_test['Gender']==0]['Fall'], 
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues, normalize=normalize)
            
            
            
        # getting data on observation level
        classified_df_W_local = X_test_rp[X_test_rp['Gender']==0].drop(columns=['Gender']).reset_index(drop=True)
        bin_W=classifier.predict(classified_df_W_local)
        prob_W=classifier.predict_proba(classified_df_W_local)[:,1]
        classified_df_W_local['y_true'] = y_test[y_test['Gender']==0]['Fall'].reset_index(drop=True)
        classified_df_W_local['y_hat_binary']=bin_W
        classified_df_W_local['y_hat_probs']=prob_W
        classified_df_W_local['Gender']=X_test[X_test['Gender']==0]['Gender'].reset_index(drop=True)
        classified_df_W = pd.concat([classified_df_W,classified_df_W_local])
    
    
        # getting TP/TN/FP/FN
        TP=disp.confusion_matrix[1][1]
        TN=disp.confusion_matrix[0][0]
        FP=disp.confusion_matrix[0][1]
        FN=disp.confusion_matrix[1][0]
        F1=2*TP/(2*TP+FP+FN)
        ACC=classifier.score(X_test_rp[X_test_rp['Gender']==0].drop(columns=['Gender']), y_test[y_test['Gender']==0]['Fall']) # mark gender
        yhat=np.mean(classifier.predict(X_test_rp[X_test_rp['Gender']==0].drop(columns=['Gender'])))
        yhat_prob=pd.DataFrame(classifier.predict_proba(X_test_rp[X_test_rp['Gender']==0].drop(columns=['Gender'])))[1].mean()
        
        # rates
        FNR = FN/(FN+TP)
        FPR = FP/(FP+TN)
        TNR = TN/(TN+FP)
        TPR = TP/(TP+FN)
        
        # appending to lists
        TP_list_W.append(TP)
        TN_list_W.append(TN)
        FP_list_W.append(FP)
        FN_list_W.append(FN)
        F1_list_W.append(F1)
        ACC_list_W.append(ACC)
        TPR_list_W.append(TPR)
        TNR_list_W.append(TNR)
        FPR_list_W.append(FPR)
        FNR_list_W.append(FNR)
        yhat_list_W.append(yhat)
        yhat_prob_list_W.append(yhat_prob)
        
        
    ############ FOR MEN ################    
        
        
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, X_test_rp[X_test_rp['Gender']==1].drop(columns=['Gender']),
                                         y_test[y_test['Gender']==1]['Fall'], 
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues, normalize=normalize)
            
            
            
        # getting data on observation level
        classified_df_M_local = X_test_rp[X_test_rp['Gender']==1].drop(columns=['Gender']).reset_index(drop=True)
        bin_M=classifier.predict(classified_df_M_local)
        prob_M=classifier.predict_proba(classified_df_M_local)[:,1]
        classified_df_M_local['y_true'] = y_test[y_test['Gender']==1]['Fall'].reset_index(drop=True)
        classified_df_M_local['y_hat_binary']=bin_M
        classified_df_M_local['y_hat_probs']=prob_M
        classified_df_M_local['Gender']=X_test[X_test['Gender']==1]['Gender'].reset_index(drop=True)
        classified_df_M = pd.concat([classified_df_M,classified_df_M_local])
    
        # getting TP/TN/FP/FN
        TP=disp.confusion_matrix[1][1]
        TN=disp.confusion_matrix[0][0]
        FP=disp.confusion_matrix[0][1]
        FN=disp.confusion_matrix[1][0]
        F1=2*TP/(2*TP+FP+FN)
        ACC=classifier.score(X_test_rp[X_test_rp['Gender']==1].drop(columns=['Gender']), y_test[y_test['Gender']==1]['Fall']) # mark gender
        yhat=np.mean(classifier.predict(X_test_rp[X_test_rp['Gender']==1].drop(columns=['Gender'])))
        yhat_prob=pd.DataFrame(classifier.predict_proba(X_test_rp[X_test_rp['Gender']==1].drop(columns=['Gender'])))[1].mean()
        
        # rates
        FNR = FN/(FN+TP)
        FPR = FP/(FP+TN)
        TNR = TN/(TN+FP)
        TPR = TP/(TP+FN)
        
        # appending to lists
        TP_list_M.append(TP)
        TN_list_M.append(TN)
        FP_list_M.append(FP)
        FN_list_M.append(FN)
        F1_list_M.append(F1)
        ACC_list_M.append(ACC)
        TPR_list_M.append(TPR)
        TNR_list_M.append(TNR)
        FPR_list_M.append(FPR)
        FNR_list_M.append(FNR)
        yhat_list_M.append(yhat)
        yhat_prob_list_M.append(yhat_prob)
        
                                
    ############ FOR All ################   
        ACC=classifier.score(X_test_rp.drop(columns=['Gender']), y_test['Fall']) # mark gender
        ACC_list_total.append(ACC)
        
        


# In[10]:


classified_df_M_local["Gender"].unique()


# In[11]:


X_test


# In[12]:


classified_df_W_local["Gender"]


# In[13]:


model_name="SVM"


# In[14]:


model=model_name
classified_df_W['Model']=model
classified_df_M['Model']=model
classified_df = pd.concat([classified_df_W,classified_df_M])
classified_df
classified_df.to_csv(f'/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/DI remove no gender/{model_name}_gender_obs.csv')


# In[15]:


classified_df_W


# In[16]:


metricW_df=np.round(pd.DataFrame(TPR_list_W,columns=['TPR']),4)
metricW_df['FPR']=np.round(pd.DataFrame(FPR_list_W),4)
metricW_df['TNR']=np.round(pd.DataFrame(TNR_list_W),4)
metricW_df['FNR']=np.round(pd.DataFrame(FNR_list_W),4)
metricW_df['ACC']=np.round(pd.DataFrame(ACC_list_W),4)
metricW_df['Mean_y_hat']=np.round(pd.DataFrame(yhat_list_W),4)
metricW_df['Mean_y_hat_prob']=np.round(pd.DataFrame(yhat_prob_list_W),4)
metricW_df['Gender']=0
metricW_df['Model']=model_name #change to correct model
colsW = list(metricW_df.columns.values)
metricW_df = metricW_df[['Gender','TPR', 'FPR', 'TNR', 'FNR', 'ACC', 'Mean_y_hat', 'Mean_y_hat_prob', 'Model']]
metricW_df.mean()


# In[17]:


metricM_df=np.round(pd.DataFrame(TPR_list_M,columns=['TPR']),4)
metricM_df['FPR']=np.round(pd.DataFrame(FPR_list_M),4)
metricM_df['TNR']=np.round(pd.DataFrame(TNR_list_M),4)
metricM_df['FNR']=np.round(pd.DataFrame(FNR_list_M),4)
metricM_df['ACC']=np.round(pd.DataFrame(ACC_list_M),4)
metricM_df['Mean_y_hat']=np.round(pd.DataFrame(yhat_list_M),4)
metricM_df['Mean_y_hat_prob']=np.round(pd.DataFrame(yhat_prob_list_M),4)
metricM_df['Gender']=1
metricM_df['Model']=model_name #change to correct model
colsM = list(metricM_df.columns.values)
metricM_df = metricM_df[['Gender','TPR', 'FPR', 'TNR', 'FNR', 'ACC', 'Mean_y_hat', 'Mean_y_hat_prob', 'Model']]
metricM_df.mean()



# In[18]:


metric_df = pd.concat([metricM_df,metricW_df],axis=0)
metric_df


# In[19]:


metric_df.to_csv(f'/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/DI remove no gender/{model_name}_gender.csv')


# ### Accuracy

# In[20]:


ACC_df=np.round(pd.DataFrame(ACC_list_total,columns=['ACC']),4)


# In[21]:


ACC_df.to_csv(f'/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/DI remove no gender/{model_name}_all.csv')


# ## prints for latex tables

# In[22]:


#metrics for women
TPRW_mean=np.round((metricW_df['TPR']*100).mean(),1)
TPRW_low=np.round((metricW_df['TPR']*100).mean()-((1.96*(metricW_df['TPR']*100).std()/math.sqrt(len(metricW_df)))),1)
TPRW_high=np.round((metricW_df['TPR']*100).mean()+((1.96*(metricW_df['TPR']*100).std()/math.sqrt(len(metricW_df)))),1)

FPRW_mean=np.round((metricW_df['FPR']*100).mean(),1)
FPRW_low=np.round((metricW_df['FPR']*100).mean()-((1.96*(metricW_df['FPR']*100).std()/math.sqrt(len(metricW_df)))),1)
FPRW_high=np.round((metricW_df['FPR']*100).mean()+((1.96*(metricW_df['FPR']*100).std()/math.sqrt(len(metricW_df)))),1)

TNRW_mean=np.round((metricW_df['TNR']*100).mean(),1)
TNRW_low=np.round((metricW_df['TNR']*100).mean()-((1.96*(metricW_df['TNR']*100).std()/math.sqrt(len(metricW_df)))),1)
TNRW_high=np.round(metricW_df['TNR'].mean()+((1.96*metricW_df['TNR'].std()/math.sqrt(len(metricW_df)))),1)

FNRW_mean=np.round((metricW_df['FNR']*100).mean(),1)
FNRW_low=np.round((metricW_df['FNR']*100).mean()-((1.96*(metricW_df['FNR']*100).std()/math.sqrt(len(metricW_df)))),1)
FNRW_high=np.round((metricW_df['FNR']*100).mean()+((1.96*(metricW_df['FNR']*100).std()/math.sqrt(len(metricW_df)))),1)



# In[23]:


#metrics for men
TPRM_mean=np.round((metricM_df['TPR']*100).mean(),1)
TPRM_low=np.round((metricM_df['TPR']*100).mean()-((1.96*(metricM_df['TPR']*100).std()/math.sqrt(len(metricM_df)))),1)
TPRM_high=np.round((metricM_df['TPR']*100).mean()+((1.96*(metricM_df['TPR']*100).std()/math.sqrt(len(metricM_df)))),1)

FPRM_mean=np.round((metricM_df['FPR']*100).mean(),1)
FPRM_low=np.round((metricM_df['FPR']*100).mean()-((1.96*(metricM_df['FPR']*100).std()/math.sqrt(len(metricM_df)))),1)
FPRM_high=np.round((metricM_df['FPR']*100).mean()+((1.96*(metricM_df['FPR']*100).std()/math.sqrt(len(metricM_df)))),1)

TNRM_mean=np.round((metricM_df['TNR']*100).mean(),2)
TNRM_low=np.round((metricM_df['TNR']*100).mean()-((1.96*(metricM_df['TNR']*100).std()/math.sqrt(len(metricM_df)))),1)
TNRM_high=np.round((metricM_df['TNR']*100).mean()+((1.96*(metricM_df['TNR']*100).std()/math.sqrt(len(metricM_df)))),1)

FNRM_mean=np.round((metricM_df['FNR']*100).mean(),1)
FNRM_low=np.round((metricM_df['FNR']*100).mean()-((1.96*(metricM_df['FNR']*100).std()/math.sqrt(len(metricM_df)))),1)
FNRM_high=np.round((metricM_df['FNR']*100).mean()+((1.96*(metricM_df['FNR']*100).std()/math.sqrt(len(metricM_df)))),1)



# In[24]:


print("\\textbf{Female}: & \\textbf{",TPRW_mean,"} & \\textbf{",FPRW_mean,"} & \\textbf{",TNRW_mean,"} & \\textbf{",FNRW_mean,"}  \\\ ")
print(f"& ({TPRW_low}-{TPRW_high}) & ({FPRW_low}-{FPRW_high}) & ({TNRW_low}-{TNRW_high}) & ({FNRW_low}-{FNRW_high})\\\ ")
print("\\textbf{Male}: & \\textbf{",TPRM_mean,"} & \\textbf{",FPRM_mean,"} & \\textbf{",TNRM_mean,"} & \\textbf{",FNRM_mean,"}  \\\ ")
print(f"& ({TPRM_low}-{TPRM_high}) & ({FPRM_low}-{FPRM_high}) & ({TNRM_low}-{TNRM_high}) & ({FNRM_low}-{FNRM_high}) \\\ ")


# ### ACC print

# In[25]:


#women
ACCW_mean=np.round((metricW_df['ACC']*100).mean(),1)
ACCW_low=np.round((metricW_df['ACC']*100).mean()-((1.96*(metricW_df['ACC']*100).std()/math.sqrt(len(metricW_df)))),1)
ACCW_high=np.round((metricW_df['ACC']*100).mean()+((1.96*(metricW_df['ACC']*100).std()/math.sqrt(len(metricW_df)))),1)

#men
ACCM_mean=np.round((metricM_df['ACC']*100).mean(),1)
ACCM_low=np.round((metricM_df['ACC']*100).mean()-((1.96*(metricM_df['ACC']*100).std()/math.sqrt(len(metricM_df)))),1)
ACCM_high=np.round((metricM_df['ACC']*100).mean()+((1.96*(metricM_df['ACC']*100).std()/math.sqrt(len(metricM_df)))),1)

#total
ACCT_mean=np.round((ACC_df['ACC']*100).mean(),1)
ACCT_low=np.round((ACC_df['ACC']*100).mean()-((1.96*(ACC_df['ACC']*100).std()/math.sqrt(len(ACC_df)))),1)
ACCT_high=np.round((ACC_df['ACC']*100).mean()+((1.96*(ACC_df['ACC']*100).std()/math.sqrt(len(ACC_df)))),1)


# In[26]:


print("\\textbf{Female}: & \\textbf{",ACCW_mean,"} \\\ ")
print(f"& ({ACCW_low}-{ACCW_high}) \\\ ")
print("\\textbf{Male}: & \\textbf{",ACCM_mean,"} \\\ ")
print(f"& ({ACCM_low}-{ACCM_high}) \\\ ")
print("\\textbf{Total}: &\\textbf{",ACCT_mean,"} \\\ ")
print(f"& ({ACCT_low}-{ACCT_high}) \\\ ")


# In[ ]:





# In[ ]:





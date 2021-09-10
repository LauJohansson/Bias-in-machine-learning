#!/usr/bin/env python
# coding: utf-8

# In[34]:


#pip install fairlearn


# In[35]:


#pip install BlackBoxAuditing


# In[15]:


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


fall_data = pd.read_csv('/restricted/s161749/G2020-57-Aalborg-bias/Data_air/Fall_count_clusterOHE_std.csv')


# In[6]:


fall_data['Gender'].value_counts()


# # Gender bias

# In[7]:


fall_data.shape


# In[8]:


fall_data['Fall'].value_counts()


# In[9]:


fall_data


# In[10]:


X = fall_data.drop(columns=['Fall']) # using all covariates in the dataset. ,'Ats_0'
#X1 = pd.DataFrame(preprocessing.scale(X1),columns=X1.columns)
#X2 = pd.DataFrame(fall_data['Gender'])
#X = pd.concat([X1],axis=1)
y = fall_data[['Gender','Fall']]


# In[11]:


X.columns


# In[12]:


X.max().sort_values(ascending=False).head(10)


# In[68]:


def DI_remove_custom(df_train,RP_level=1.0):

    X_col_names_f=['Gender', 'BirthYear', 'LoanPeriod', 'NumberAts']
    df2_all=df_train.drop(columns=["Unnamed: 0"]+X_col_names_f).copy() #Gemmer alle kolonner, undtagen numerical og gender
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


# In[69]:


kf=KFold(n_splits=3, random_state=2, shuffle=True)

for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    X_train_temp = X_train.reset_index(drop=True)
    X_test_temp = X_test.reset_index(drop=True)
    
    X_train_rp = DI_remove_custom(X_train_temp)  
    X_test_rp  = DI_remove_custom(X_test_temp)


# In[70]:


X_train.shape,X_train_rp.shape


# In[71]:


X_train.drop(columns=['Unnamed: 0']).groupby('Gender').mean()


# In[72]:


X_train_rp.groupby('Gender').mean()


# In[73]:


fig,ax=plt.subplots(2,1,sharex=True)

ax=ax.ravel()

sns.histplot(x='BirthYear',data=X_train,hue='Gender',kde=True,bins=30,ax=ax[0])
sns.histplot(x='BirthYear',data=X_train_rp,hue='Gender',kde=True,bins=30,ax=ax[1])


# In[74]:


fig,ax=plt.subplots(2,1,sharex=True)

ax=ax.ravel()

sns.histplot(x='LoanPeriod',data=X_train,hue='Gender',kde=True,bins=30,ax=ax[0])
sns.histplot(x='LoanPeriod',data=X_train_rp,hue='Gender',kde=True,bins=30,ax=ax[1])


# In[75]:


fig,ax=plt.subplots(2,1,sharex=True)

ax=ax.ravel()

sns.histplot(x='NumberAts',data=X_train,hue='Gender',kde=True,bins=30,ax=ax[0])
sns.histplot(x='NumberAts',data=X_train_rp,hue='Gender',kde=True,bins=30,ax=ax[1])


# In[ ]:





# In[108]:


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

for i in range(1,11):
    
    kf=KFold(n_splits=5, random_state=i, shuffle=True)
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train_temp = X_train.reset_index(drop=True)
        X_test_temp = X_test.reset_index(drop=True)
        
        X_train_rp = DI_remove_custom(X_train_temp)  
        X_test_rp  = DI_remove_custom(X_test_temp)
        
    #### Flip fra den ene til den anden for at skifte mellem SVM, LR og RF ###
    
        #classifier = svm.SVC(kernel='rbf', C=1, random_state=0,class_weight='balanced',probability=True).fit(X_train_rp, y_train['Fall'])
        #classifier = LogisticRegression(max_iter=1000,class_weight='balanced').fit(X_train_rp, y_train['Fall'])
        classifier = RandomForestClassifier(random_state=1).fit(X_train_rp, y_train['Fall'])
    
    
    
   # class weight balanced?  
   # 
        np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None)]
    
    
        
     ############ FOR WOMEN ################   
        
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, X_test_rp[X_test_rp['Gender']==0],
                                         y_test[y_test['Gender']==0]['Fall'], 
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues, normalize=normalize)
    
        # getting TP/TN/FP/FN
        TP=disp.confusion_matrix[1][1]
        TN=disp.confusion_matrix[0][0]
        FP=disp.confusion_matrix[0][1]
        FN=disp.confusion_matrix[1][0]
        F1=2*TP/(2*TP+FP+FN)
        ACC=classifier.score(X_test_rp[X_test_rp['Gender']==0], y_test[y_test['Gender']==0]['Fall']) # mark gender
        yhat=np.mean(classifier.predict(X_test_rp[X_test_rp['Gender']==0]))
        yhat_prob=pd.DataFrame(classifier.predict_proba(X_test_rp[X_test_rp['Gender']==0]))[1].mean()
        
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
            disp = plot_confusion_matrix(classifier, X_test_rp[X_test_rp['Gender']==1],
                                         y_test[y_test['Gender']==1]['Fall'], 
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues, normalize=normalize)
    
        # getting TP/TN/FP/FN
        TP=disp.confusion_matrix[1][1]
        TN=disp.confusion_matrix[0][0]
        FP=disp.confusion_matrix[0][1]
        FN=disp.confusion_matrix[1][0]
        F1=2*TP/(2*TP+FP+FN)
        ACC=classifier.score(X_test_rp[X_test_rp['Gender']==1], y_test[y_test['Gender']==1]['Fall']) # mark gender
        yhat=np.mean(classifier.predict(X_test_rp[X_test_rp['Gender']==1]))
        yhat_prob=pd.DataFrame(classifier.predict_proba(X_test_rp[X_test_rp['Gender']==1]))[1].mean()
        
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
        ACC=classifier.score(X_test_rp, y_test['Fall']) # mark gender
        ACC_list_total.append(ACC)
        
        


# In[111]:


metricW_df=np.round(pd.DataFrame(TPR_list_W,columns=['TPR']),4)
metricW_df['FPR']=np.round(pd.DataFrame(FPR_list_W),4)
metricW_df['TNR']=np.round(pd.DataFrame(TNR_list_W),4)
metricW_df['FNR']=np.round(pd.DataFrame(FNR_list_W),4)
metricW_df['ACC']=np.round(pd.DataFrame(ACC_list_W),4)
metricW_df['Mean_y_hat']=np.round(pd.DataFrame(yhat_list_W),4)
metricW_df['Mean_y_hat_prob']=np.round(pd.DataFrame(yhat_prob_list_W),4)
metricW_df['Gender']=0
metricW_df['Model']='RF' #change to correct model
colsW = list(metricW_df.columns.values)
metricW_df = metricW_df[['Gender','TPR', 'FPR', 'TNR', 'FNR', 'ACC', 'Mean_y_hat', 'Mean_y_hat_prob', 'Model']]
metricW_df.mean()


# In[112]:


metricM_df=np.round(pd.DataFrame(TPR_list_M,columns=['TPR']),4)
metricM_df['FPR']=np.round(pd.DataFrame(FPR_list_M),4)
metricM_df['TNR']=np.round(pd.DataFrame(TNR_list_M),4)
metricM_df['FNR']=np.round(pd.DataFrame(FNR_list_M),4)
metricM_df['ACC']=np.round(pd.DataFrame(ACC_list_M),4)
metricM_df['Mean_y_hat']=np.round(pd.DataFrame(yhat_list_M),4)
metricM_df['Mean_y_hat_prob']=np.round(pd.DataFrame(yhat_prob_list_M),4)
metricM_df['Gender']=1
metricM_df['Model']='RF' #change to correct model
colsM = list(metricM_df.columns.values)
metricM_df = metricM_df[['Gender','TPR', 'FPR', 'TNR', 'FNR', 'ACC', 'Mean_y_hat', 'Mean_y_hat_prob', 'Model']]
metricM_df.mean()



# In[113]:


metric_df = pd.concat([metricM_df,metricW_df],axis=0)
metric_df


# In[114]:


metric_df.to_csv('/restricted/s161749/G2020-57-Aalborg-bias/Plot_metrics/DI remove/RF_gender.csv')


# ### Accuracy

# In[115]:


ACC_df=np.round(pd.DataFrame(ACC_list_total,columns=['ACC']),4)


# In[116]:


ACC_df.to_csv('/restricted/s161749/G2020-57-Aalborg-bias/Plot_metrics/DI remove/RF_all.csv')


# ## prints for latex tables

# In[117]:


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



# In[118]:


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



# In[119]:


print("\\textbf{Female}: & \\textbf{",TPRW_mean,"} & \\textbf{",FPRW_mean,"} & \\textbf{",TNRW_mean,"} & \\textbf{",FNRW_mean,"}  \\\ ")
print(f"& ({TPRW_low}-{TPRW_high}) & ({FPRW_low}-{FPRW_high}) & ({TNRW_low}-{TNRW_high}) & ({FNRW_low}-{FNRW_high})\\\ ")
print("\\textbf{Male}: & \\textbf{",TPRM_mean,"} & \\textbf{",FPRM_mean,"} & \\textbf{",TNRM_mean,"} & \\textbf{",FNRM_mean,"}  \\\ ")
print(f"& ({TPRM_low}-{TPRM_high}) & ({FPRM_low}-{FPRM_high}) & ({TNRM_low}-{TNRM_high}) & ({FNRM_low}-{FNRM_high}) \\\ ")


# ### ACC print

# In[120]:


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


# In[121]:


print("\\textbf{Female}: & \\textbf{",ACCW_mean,"} \\\ ")
print(f"& ({ACCW_low}-{ACCW_high}) \\\ ")
print("\\textbf{Male}: & \\textbf{",ACCM_mean,"} \\\ ")
print(f"& ({ACCM_low}-{ACCM_high}) \\\ ")
print("\\textbf{Total}: &\\textbf{",ACCT_mean,"} \\\ ")
print(f"& ({ACCT_low}-{ACCT_high}) \\\ ")


# In[ ]:





# # 80 pct. rule?

# In[492]:


np.mean(FNR_list_M)/np.mean(FNR_list_W)


# ## There doesn't seem to be bias between genders regarding classifications using either algorithm (LR, RF, SVM)

# # Age bias

# In[493]:


age_group=pd.DataFrame((fall_data['BirthYear']>fall_data['BirthYear'].mean()).astype(int)) # the young ones are = 1
age_group=age_group.rename(columns={"BirthYear":"Age Group"})


# In[494]:


age_group.sum()


# In[495]:


X1 = fall_data.drop(columns=['Unnamed: 0','Fall']) # using all covariates in the dataset (no gender). 
#X1 = pd.DataFrame(preprocessing.scale(X1),columns=X1.columns)
#X2 = pd.DataFrame(fall_data['Gender']) # gender in (not standardized)
X3 = age_group
X = pd.concat([X1,X3],axis=1)
y1 = fall_data['Fall']
y2 = age_group
y = pd.concat([y1,y2],axis=1)


# In[496]:


X.groupby('Age Group')['BirthYear'].mean()


# In[497]:


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
TP_list_old=[]
TN_list_old=[]
FP_list_old=[]
FN_list_old=[]
F1_list_old=[]
ACC_list_old=[]
TPR_list_old=[]
TNR_list_old=[]
FPR_list_old=[]
FNR_list_old=[]

TP_list_young=[]
TN_list_young=[]
FP_list_young=[]
FN_list_young=[]
F1_list_young=[]
ACC_list_young=[]
TPR_list_young=[]
TNR_list_young=[]
FPR_list_young=[]
FNR_list_young=[]

class_names = ['No fall','Fall']

for i in range(1,11):
    
    
    kf=KFold(n_splits=5, random_state=i, shuffle=True)
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    #### Flip fra den ene til den anden for at skifte mellem SVM, LR og RF ###
    
        #classifier = svm.SVC(kernel='rbf', C=1, random_state=0,class_weight='balanced').fit(X_train, y_train['Fall']) # not trained on Age Group var
        classifier = LogisticRegression(max_iter=1000,class_weight='balanced').fit(X_train, y_train['Fall'])
        #classifier = RandomForestClassifier(random_state=1).fit(X_train, y_train['Fall'])
    
        np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None)]
    
    
        
     ############ FOR WOMEN ################   
        
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, X_test[X_test['Age Group']==0],
                                         y_test[y_test['Age Group']==0]['Fall'], 
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues, normalize=normalize)
    
        # getting TP/TN/FP/FN
        TP=disp.confusion_matrix[1][1]
        TN=disp.confusion_matrix[0][0]
        FP=disp.confusion_matrix[0][1]
        FN=disp.confusion_matrix[1][0]
        F1=2*TP/(2*TP+FP+FN)
        ACC=classifier.score(X_test[X_test['Age Group']==0], y_test[y_test['Age Group']==0]['Fall']) # mark gender
        
        # rates
        FNR = FN/(FN+TP)
        FPR = FP/(FP+TN)
        TNR = TN/(TN+FP)
        TPR = TP/(TP+FN)
        
        # appending to lists
        TP_list_old.append(TP)
        TN_list_old.append(TN)
        FP_list_old.append(FP)
        FN_list_old.append(FN)
        F1_list_old.append(F1)
        ACC_list_old.append(ACC)
        TPR_list_old.append(TPR)
        TNR_list_old.append(TNR)
        FPR_list_old.append(FPR)
        FNR_list_old.append(FNR)
        
        
    ############ FOR MEN ################    
        
        
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, X_test[X_test['Age Group']==1],
                                         y_test[y_test['Age Group']==1]['Fall'], 
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues, normalize=normalize)
    
        # getting TP/TN/FP/FN
        TP=disp.confusion_matrix[1][1]
        TN=disp.confusion_matrix[0][0]
        FP=disp.confusion_matrix[0][1]
        FN=disp.confusion_matrix[1][0]
        F1=2*TP/(2*TP+FP+FN)
        ACC=classifier.score(X_test[X_test['Age Group']==1], y_test[y_test['Age Group']==1]['Fall']) # mark gender
        
        # rates
        FNR = FN/(FN+TP)
        FPR = FP/(FP+TN)
        TNR = TN/(TN+FP)
        TPR = TP/(TP+FN)
        
        # appending to lists
        TP_list_young.append(TP)
        TN_list_young.append(TN)
        FP_list_young.append(FP)
        FN_list_young.append(FN)
        F1_list_young.append(F1)
        ACC_list_young.append(ACC)
        TPR_list_young.append(TPR)
        TNR_list_young.append(TNR)
        FPR_list_young.append(FPR)
        FNR_list_young.append(FNR)


# In[498]:


print(f"Mean of TPR for old:{round(np.mean(TPR_list_old)*100,1)}{round((np.mean(TPR_list_old)-(np.std(TPR_list_old)*1.96))*100,1),round((np.mean(TPR_list_old)+(np.std(TPR_list_old)*1.96))*100,1)}")
print(f"Mean of FPR for old:{round(np.mean(FPR_list_old)*100,1)}{round((np.mean(FPR_list_old)-(np.std(FPR_list_old)*1.96))*100,1),round((np.mean(FPR_list_old)+(np.std(FPR_list_old)*1.96))*100,1)}")
print(f"Mean of TNR for old:{round(np.mean(TNR_list_old)*100,1)}{round((np.mean(TNR_list_old)-(np.std(TNR_list_old)*1.96))*100,1),round((np.mean(TNR_list_old)+(np.std(TNR_list_old)*1.96))*100,1)}")
print(f"Mean of FNR for old:{round(np.mean(FNR_list_old)*100,1)}{round((np.mean(FNR_list_old)-(np.std(FNR_list_old)*1.96))*100,1),round((np.mean(FNR_list_old)+(np.std(FNR_list_old)*1.96))*100,1)}")
print(f"Mean of ACC for old:{round(np.mean(ACC_list_old)*100,1)}{round((np.mean(ACC_list_old)-(np.std(ACC_list_old)*1.96))*100,1),round((np.mean(ACC_list_old)+(np.std(ACC_list_old)*1.96))*100,1)}")
print("--------------------------------------------")
print(f"Mean of TPR for young:{round(np.mean(TPR_list_young)*100,1)}{round((np.mean(TPR_list_young)-(np.std(TPR_list_young)*1.96))*100,1),round((np.mean(TPR_list_young)+(np.std(TPR_list_young)*1.96))*100,1)}")
print(f"Mean of FPR for young:{round(np.mean(FPR_list_young)*100,1)}{round((np.mean(FPR_list_young)-(np.std(FPR_list_young)*1.96))*100,1),round((np.mean(FPR_list_young)+(np.std(FPR_list_young)*1.96))*100,1)}")
print(f"Mean of TNR for young:{round(np.mean(TNR_list_young)*100,1)}{round((np.mean(TNR_list_young)-(np.std(TNR_list_young)*1.96))*100,1),round((np.mean(TNR_list_young)+(np.std(TNR_list_young)*1.96))*100,1)}")
print(f"Mean of FNR for young:{round(np.mean(FNR_list_young)*100,1)}{round((np.mean(FNR_list_young)-(np.std(FNR_list_young)*1.96))*100,1),round((np.mean(FNR_list_young)+(np.std(FNR_list_young)*1.96))*100,1)}")
print(f"Mean of ACC for young:{round(np.mean(ACC_list_young)*100,1)}{round((np.mean(ACC_list_young)-(np.std(ACC_list_young)*1.96))*100,1),round((np.mean(ACC_list_young)+(np.std(ACC_list_young)*1.96))*100,1)}")


# In[499]:


#women
TPR_final_old=[round(np.mean(TPR_list_old)*100,1),round((np.mean(TPR_list_old)-(np.std(TPR_list_old)*1.96))*100,1),round((np.mean(TPR_list_old)+(np.std(TPR_list_old)*1.96))*100,1)]
FPR_final_old=[round(np.mean(FPR_list_old)*100,1),round((np.mean(FPR_list_old)-(np.std(FPR_list_old)*1.96))*100,1),round((np.mean(FPR_list_old)+(np.std(FPR_list_old)*1.96))*100,1)]
TNR_final_old=[round(np.mean(TNR_list_old)*100,1),round((np.mean(TNR_list_old)-(np.std(TNR_list_old)*1.96))*100,1),round((np.mean(TNR_list_old)+(np.std(TNR_list_old)*1.96))*100,1)]
FNR_final_old=[round(np.mean(FNR_list_old)*100,1),round((np.mean(FNR_list_old)-(np.std(FNR_list_old)*1.96))*100,1),round((np.mean(FNR_list_old)+(np.std(FNR_list_old)*1.96))*100,1)]
ACC_final_old=[round(np.mean(ACC_list_old)*100,1),round((np.mean(ACC_list_old)-(np.std(ACC_list_old)*1.96))*100,1),round((np.mean(ACC_list_old)+(np.std(ACC_list_old)*1.96))*100,1)]

#men
TPR_final_young=[round(np.mean(TPR_list_young)*100,1),round((np.mean(TPR_list_young)-(np.std(TPR_list_young)*1.96))*100,1),round((np.mean(TPR_list_young)+(np.std(TPR_list_young)*1.96))*100,1)]
FPR_final_young=[round(np.mean(FPR_list_young)*100,1),round((np.mean(FPR_list_young)-(np.std(FPR_list_young)*1.96))*100,1),round((np.mean(FPR_list_young)+(np.std(FPR_list_young)*1.96))*100,1)]
TNR_final_young=[round(np.mean(TNR_list_young)*100,1),round((np.mean(TNR_list_young)-(np.std(TNR_list_young)*1.96))*100,1),round((np.mean(TNR_list_young)+(np.std(TNR_list_young)*1.96))*100,1)]
FNR_final_young=[round(np.mean(FNR_list_young)*100,1),round((np.mean(FNR_list_young)-(np.std(FNR_list_young)*1.96))*100,1),round((np.mean(FNR_list_young)+(np.std(FNR_list_young)*1.96))*100,1)]
ACC_final_young=[round(np.mean(ACC_list_young)*100,1),round((np.mean(ACC_list_young)-(np.std(ACC_list_young)*1.96))*100,1),round((np.mean(ACC_list_young)+(np.std(ACC_list_young)*1.96))*100,1)]


# In[500]:


print("old: \\textbf{",TPR_final_old[0],"} & \\textbf{",FPR_final_old[0],"} & \\textbf{",TNR_final_old[0],"} & \\textbf{",FNR_final_old[0],"} & \\textbf{",ACC_final_old[0],"} \\\ ")
print(f"({TPR_final_old[1]}-{TPR_final_old[2]}) & ({FPR_final_old[1]}-{FPR_final_old[2]}) & ({TNR_final_old[1]}-{TNR_final_old[2]}) & ({FNR_final_old[1]}-{FNR_final_old[2]}) & ({ACC_final_old[1]}-{ACC_final_old[2]}) \\\ ")
print("young: \\textbf{",TPR_final_young[0],"} & \\textbf{",FPR_final_young[0],"} & \\textbf{",TNR_final_young[0],"} & \\textbf{",FNR_final_young[0],"} & \\textbf{",ACC_final_young[0],"} \\\ ")
print(f"({TPR_final_young[1]}-{TPR_final_young[2]}) & ({FPR_final_young[1]}-{FPR_final_young[2]}) & ({TNR_final_young[1]}-{TNR_final_young[2]}) & ({FNR_final_young[1]}-{FNR_final_young[2]}) & ({ACC_final_young[1]}-{ACC_final_young[2]}) \\\ ")


# # 80 pct. rule

# In[166]:


np.mean(FNR_list_old)/np.mean(FNR_list_young)


# ## Result: as the citizens get older, the models does a worse job on them. For citizens above the mean age, there are more false negatives. In other words, the algorithm in a higher rate misclassifies the older part of the distribution by predicting that a fall would not occur when - in fact - the citizen did fall. 
# 
# ## Although this is only really the case for SVM, a little bit for LR and not at all for RF.
# 
# ## But, nonetheless, it is problematic that the algorithms is worse for older citizens, since it is more of a problem if they fall!

# ### Checking mean of fall pr. Birth Year to see if elderly fall more (as expected), could explain some of the discrepancy in rates. 
# ### (MOVE ALL OF THIS IDDA TO ANOTHER NOTEBOOK)

# In[21]:


fall_data


# In[22]:


import seaborn as sns
sns.barplot(x=fall_data['BirthYear'],y=fall_data['Fall'],ci=None)


# In[23]:


age_fall_scat=pd.DataFrame(fall_data.groupby('BirthYear')['Fall'].mean()).reset_index()
age_fall_scat.head(5)


# In[24]:


sns.scatterplot(x=age_fall_scat.BirthYear,y=age_fall_scat.Fall)


# In[25]:


age_fall_scat.corr(method='pearson') # with all ages


# In[26]:


age_fall_scat_sub60 = age_fall_scat[age_fall_scat['BirthYear']<60]


# In[27]:


sns.scatterplot(x=age_fall_scat_sub60.BirthYear,y=age_fall_scat_sub60.Fall)


# In[28]:


age_fall_scat_sub60.corr(method='pearson') # with all ages


# In[29]:


fall_data['agegroup']=(fall_data['BirthYear']>36.88).astype(int) # young is one!


# In[30]:


fall_data.groupby('agegroup')['Fall'].mean()


# In[31]:


sns.histplot(x=fall_data.Fall,hue=fall_data.agegroup,stat='density',common_norm=False)


# ### All this points towards that the older citizens actually do fall more than the younger ones. Although this does not mean that the mapping from actual falls to registered falls is not worse for the older citizens, since some of their falls lead to hospitalizations, and this might not be registered by DigiRehab. 

# ### Next up could be to do IDDA on number of Ats of the like. Older should have more Ats'.

# In[32]:


fall_data


# In[33]:


fall_data.groupby('agegroup')['NumberAts'].mean()


# In[34]:


sns.histplot(x=fall_data.NumberAts,hue=fall_data.agegroup,stat='density',common_norm=False)


# ### Aha! The young ones have more Ats, which does not really make sense.... 

# In[35]:


fall_data[['Gender','BirthYear','NumberAts','Fall']].corr(method='pearson') # with all ages


# ### But, NumberAts is not positively correlated with falling...

# In[36]:


fall_data[['1Ats','2Ats','3Ats','4Ats','5Ats','Fall']].corr(method='pearson')


# In[37]:


classifier_Ats = RandomForestClassifier(random_state=1).fit(X_train[['1Ats','2Ats','3Ats','4Ats','5Ats']], y_train['Fall'])


# In[38]:


titles_options = [("Confusion matrix, without normalization", None)]
    
    
        
     ############ FOR WOMEN ################   
    
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier_Ats, X_test[['1Ats','2Ats','3Ats','4Ats','5Ats']],
                                 y_test['Fall'], 
                             display_labels=class_names,
                             cmap=plt.cm.Blues, normalize=normalize)

ACC=classifier_Ats.score(X_test[['1Ats','2Ats','3Ats','4Ats','5Ats']],y_test['Fall'])
ACC


# ### The classifier using only the first five Ats is just as good as the one using all covariates

# In[ ]:





# ## Looking at the difference between the non-embedded and the embedded fall data

# In[39]:


fall_data_ori = pd.read_csv('/restricted/s161749/G2020-57-Aalborg-bias/Data_air/fall.csv')


# In[75]:


fall_data_ori['1Ats'].value_counts().head(5), fall_data_ori['1Ats'].value_counts().head(5).sum()


# In[74]:


fall_data_ori['2Ats'].value_counts().head(5), fall_data_ori['2Ats'].value_counts().head(5).sum()


# In[73]:


fall_data_ori['3Ats'].value_counts().head(5), fall_data_ori['3Ats'].value_counts().head(5).sum()


# In[72]:


fall_data_ori['4Ats'].value_counts().head(5), fall_data_ori['4Ats'].value_counts().head(5).sum()


# In[71]:


fall_data_ori['5Ats'].value_counts().head(5), fall_data_ori['5Ats'].value_counts().head(5).sum()


# ### If we use these six Ats numbers OHE for on the first six places we will match a lot of the observations: 120606, 93307, 0, 222718, 91218, 215103

# In[244]:


80/100


# In[245]:


100/80


# In[ ]:





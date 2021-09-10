#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
import math 


# In[2]:


#modelname="LR"
modelname="SVM"
#modelname="RF"


# In[3]:


def create_shape(modelname,model,xcolnames,X_train,X_test,directory,modelnr,plot_type="bar"):
    import shap
   
    if modelname.lower()=="xgboost" or modelname.lower()=="rf":
        print("Using treeexplainer")
        shap_explainer = shap.TreeExplainer(model, data=X_train)
        
        if modelname.lower()=="rf":
            shap_values = shap_explainer.shap_values(X_test,check_additivity=False)
            shap_values=shap_values[1]
        
    elif modelname.lower()=="ffnn":
        print("Using deepexplainer")
        shap_explainer = shap.DeepExplainer(model, data=X_train)
        shap_values = shap_explainer.shap_values(X_test)
    elif modelname.lower()=="svm":
        print("Using kernelshap")
        shap_explainer = shap.KernelExplainer(model.predict_proba, data=shap.kmeans(X_train, 10))
        shap_values = shap_explainer.shap_values(shap.sample(X_test, 100))
        shap_values=shap_values[1]
        
        #https://slundberg.github.io/shap/notebooks/linear_explainer/Sentiment%20Analysis%20with%20Logistic%20Regression.html
    elif modelname.lower()=="lr":
        print("Using explin")
        background = shap.maskers.Independent(X_train, max_samples=100)
        shap_explainer = shap.LinearExplainer(model=model, data=X_train,masker=background)
        shap_values = shap_explainer.shap_values(X_test)
        
        
    else:
        raise Exception("Lau says: Sorry, cant find the model")

    
    feature_names=xcolnames

    
    #Dette er Christians måde at hente values fra SHAP
    importance_df  = pd.DataFrame()
    importance_df['feature'] = feature_names
    
    importance_df['shap_values'] = np.around(np.array(shap_values)[:,:].mean(0), decimals=3)
    importance_df['shap_values_abs'] = np.around(abs(np.array(shap_values)[:,:]).mean(0), decimals=3)
    
    
    #if modelname.lower()=="xgboost":
    #    importance_df['feat_imp'] = np.around(model.feature_importances_, decimals=3)
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
    
    if modelname.lower()!="svm":
        plt.close()
        shap.summary_plot(shap_values,
                          X_test,
                          feature_names=feature_names,

                          show=False)

        plt.savefig(directory+"/beeplots/"+modelname+"shap_plot beeswarm model"+str(modelnr)+".png",
                    bbox_inches = "tight")


    


# In[4]:


fall_data = pd.read_csv('/restricted/s164512/G2020-57-Aalborg-bias/Data_air/Fall_count_clusterOHE_std.csv')


# # Gender bias

# In[5]:


X = fall_data.drop(columns=['Unnamed: 0','Fall']) # using all covariates in the dataset. ,'Ats_0'
y = fall_data[['Gender','Fall']]


# In[6]:


X_col_names=list(X.columns)


# In[7]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


modelcounter=0
for i in range(1,11):
    
    kf=KFold(n_splits=5, random_state=i, shuffle=True)
    
    for train_index, test_index in kf.split(X):
        print("Running ",modelname," nr ",modelcounter)
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    #### Flip fra den ene til den anden for at skifte mellem SVM, LR og RF ###
    
        if modelname=="SVM":
            classifier = svm.SVC(kernel='rbf', C=1, random_state=0,class_weight='balanced',probability=True).fit(X_train, y_train['Fall'])
        elif modelname=="LR":
            classifier = LogisticRegression(max_iter=1000,class_weight='balanced').fit(X_train, y_train['Fall'])
        elif modelname=="RF":
            classifier = RandomForestClassifier(random_state=1).fit(X_train, y_train['Fall'])
    
    
        #### Creating shap
        create_shape(modelname,classifier,X_col_names,X_train,X_test,"/restricted/s164512/G2020-57-Aalborg-bias/SHAP/",modelcounter)
        modelcounter=modelcounter+1
    
   


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
import numpy as np
import config as cfg
import pandas as pd
#from tools import file_reader, file_writer, explainer
from utility import metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold


# In[ ]:





# In[2]:


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


# In[3]:


from utils_Copy import *


# In[4]:


procted_col_name="Gender"
y_col_name="Fall"


# In[5]:


pathRoot="../../Data_air/"

pathFall=pathRoot+"Fall_count_clusterOHE_std.csv"

df=pd.read_csv(pathFall)


# In[6]:


#titel_mitigation="nostratify"
#titel_mitigation="DroppingD"
#titel_mitigation="Gender Swap"
#titel_mitigation="DI remove"


titel_mitigation="original"

dropping_D=False
gender_swap=False
DI_remove=False

PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/Xgboost_ohe/models/"+titel_mitigation+"/"

PATH=PATH_orig#+seedName+"/"


# In[7]:


model_dir = cfg.FALL_XGB_DIR
target_name = "Fall"
y_col_name=target_name
        
    

        

#DATA_DIR = cfg.PROCESSED_DATA_DIR
#CASES = ["Complete", "Compliance", "Fall", "Fall_test"]      
#df = file_reader.read_csv(DATA_DIR, 'fall_emb.csv')
 


# In[ ]:





# In[8]:


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


# In[9]:


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

    
    


# In[10]:


#df_5_persons=pd.read_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/6_cit_data.csv")

#all_cols=X_col_names+[y_col_name]
#all_cols=all_cols+["output"]
#all_cols=all_cols+["output_prob"]
#all_cols=all_cols+["Model"]
#df_predicted=pd.DataFrame([],columns=all_cols)


# In[11]:


modelcounter=0
df_test=pd.DataFrame([],columns=list(X_col_names)+["Fall"]+["output"]+["output_prob"])

for new_seed in range(1,11):
    
    
    df = df.sample(frac=1, random_state=new_seed).reset_index(drop=True)
    
    
    
  
  
    X = df[X_col_names]
    y = df[target_name].to_frame()
        
        
        

    
    neg, pos = np.bincount(y[target_name])
    scale_pos_weight = neg / pos

    params = {"n_estimators": 400,
            "objective": "binary:logistic",
            "scale_pos_weight": scale_pos_weight,
            "use_label_encoder": False,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "seed": 0
    }
    
    
    
    model = xgb.XGBClassifier(**params)
    #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=new_seed)
    skf=KFold(n_splits=5, random_state=new_seed, shuffle=True)

    
    y_valid_pred = 0*y[target_name]
    valid_acc, valid_pre, valid_recall, valid_roc_auc = list(), list(), list(), list()
    #for train_index, valid_index in skf.split(X_train, y_train):
    for train_index, valid_index in skf.split(X):
        print(f"Running model {modelcounter}")
        #X_train_split, X_valid_split = X_train.iloc[train_index,:], X_train.iloc[valid_index,:]
        #y_train_split, y_valid_split = y_train.iloc[train_index], y_train.iloc[valid_index]
        
        
        X_train_split, X_valid_split = X.iloc[train_index,:], X.iloc[valid_index,:]
        y_train_split, y_valid_split = y.iloc[train_index], y.iloc[valid_index]
        
        if gender_swap==True:
            X_train_split_copy=X_train_split.copy()
            y_train_split_copy=y_train_split.copy()
            
            X_train_split_copy["Gender"]=(X_train_split_copy["Gender"]-1)*(-1)
            
            X_train_split=pd.concat([X_train_split,X_train_split_copy])
            
            y_train_split=pd.concat([y_train_split,y_train_split_copy])
            
        if DI_remove==True:
            X_train_split=DI_remove_custom(X_train_split.reset_index(drop=True))
            X_valid_split=DI_remove_custom(X_valid_split.reset_index(drop=True))
        

        optimize_rounds = True
        early_stopping_rounds = 50
        if optimize_rounds:
            eval_set=[(X_valid_split, y_valid_split)]
            fit_model = model.fit(X_train_split, y_train_split, 
                                    eval_set=eval_set,
                                    eval_metric=metrics.gini_xgb,
                                    early_stopping_rounds=early_stopping_rounds,
                                    verbose=False)

        else:
            fit_model = model.fit(X_train_split, y_train_split)

        pred = fit_model.predict_proba(X_valid_split)[:,1]
        y_valid_pred.iloc[valid_index] = pred

        y_valid_scores = (y_valid_pred.iloc[valid_index] > 0.5)


        #### SAVE DATA####
        y_true_pd=y_valid_split.reset_index(drop=True)
        #y_true_pd=y_valid_split.to_frame().reset_index(drop=True)
        y_pred_pd=y_valid_scores.apply(lambda x: 1 if x==True else 0).to_frame().reset_index(drop=True).rename(columns={"Fall":"output"})
        y_pred_prob_pd=pd.DataFrame(pred, columns = ["output_prob"])
        
        
        


        df_subset=pd.concat([X_valid_split.reset_index(drop=True),y_true_pd,y_pred_pd,y_pred_prob_pd],axis=1)
        
        
        if dropping_D==True:
            df_subset[procted_col_name]=list(df[procted_col_name].iloc[valid_index])
        
        df_subset["Model"]="Model"+str(modelcounter)
        
        
        df_test=df_test.append(df_subset, ignore_index=True)
        
        
        
        create_shape("XGBoost",model,X_col_names,X_train_split,X_valid_split,"/restricted/s164512/G2020-57-Aalborg-bias/SHAP/",modelcounter)
        
        ######
        
        
        
        ##SAVE 6 persons#
  
        #df_predicted_subset=df_5_persons.copy().drop(columns=["Unnamed: 0"])
    
        #X_numpy=np.array(df_predicted_subset[X_col_names])
        #pred_6cit = fit_model.predict_proba(X_numpy)[:,1]

        
        #y_valid_scores_6cit = pred_6cit>0.5
        #y_valid_scores_6cit=pd.DataFrame(y_valid_scores_6cit,columns=["prob"])
        
        
        #y_pred_pd_6cit=y_valid_scores_6cit.rename(columns={"prob":"output"})["output"].apply(lambda x: 1 if x==True else 0).to_frame()#.rename(columns={"prob":"output"})
        #y_pred_prob_pd_6cit=pd.DataFrame(pred_6cit, columns = ["output_prob"])
                
        

        #df_predicted_subset=pd.concat([df_predicted_subset[X_col_names+[y_col_name]].reset_index(drop=True),y_pred_pd_6cit,y_pred_prob_pd_6cit],axis=1)
        #df_predicted_subset["Model"]="Model"+str(modelcounter)

        #df_predicted_subset=df_predicted_subset.reset_index(drop=True)

        #df_predicted=pd.concat([df_predicted,df_predicted_subset],axis=0,sort=False)

        
        ##########################


        valid_acc.append(accuracy_score(y_valid_split, y_valid_scores))
        valid_pre.append(precision_score(y_valid_split, y_valid_scores))
        valid_recall.append(recall_score(y_valid_split, y_valid_scores))
        valid_roc_auc.append(roc_auc_score(y_valid_split, y_valid_pred.iloc[valid_index]))

        modelcounter=modelcounter+1



# # Save all data

# In[ ]:


#df_test.to_csv(PATH+"all_test_data.csv")
#print("The full test data lies here:",PATH+"all_test_data.csv")


# # Save 6 persons

# In[ ]:



#df_predicted.to_csv("/restricted/s164512/G2020-57-Aalborg-bias/6 citizens/Xgboost_predictions.csv")


# # Evaluate

# In[ ]:


#y_pred = model.predict(X_test)
#y_proba = model.predict_proba(X_test)[:,1]


# In[ ]:



#file_writer.write_cm_plot(y_test, y_pred, cfg.REPORTS_PLOTS_DIR,
                       # f'{case.lower()}_xgb_cm.pdf', case)
#file_writer.write_joblib(model, model_dir, f'{case.lower()}_xgboost.joblib')

#print(f"Scores for XGBoost model:")
#print(f"Accuracy: {np.around(accuracy_score(y_test, y_pred), decimals=3)}")
#print(f"Precision: {np.around(precision_score(y_test, y_pred), decimals=3)}")
#print(f"Recall: {np.around(recall_score(y_test, y_pred), decimals=3)}")
#print(f"ROC AUC: {np.around(roc_auc_score(y_test, y_proba), decimals=3)}\n")


# # Save the confusion data

# In[ ]:


#column_names = ["Group", "ML", "Measure","Value"]

#df_out = pd.DataFrame(columns = column_names)

#for i in range(50):
  
#    PATH_loop=PATH+"model"+str(i)+"_all.csv"
  
#    data=pd.read_csv(PATH_loop)
#    for group in ["all"]:
#        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP',"y_hat_mean","y_target_mean","y_hat_prob"]:
#            value=float(data[data[procted_col_name]==group][measure])

#            df_out=df_out.append({'Group': group,"ML":"Xgboost"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

#df_out.to_csv(PATH+"/Xgboost_metrics_crossvalidated_all.csv")


# In[ ]:


#global_all_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
#global_all_bar.set_title('All')
#global_all_bar.get_figure().savefig(PATH_orig+"/barplot_all.png")


# In[ ]:


#column_names = ["Group", "ML", "Measure","Value"]

#df_out = pd.DataFrame(columns = column_names)

#for i in range(50):

#    PATH_loop=PATH+"model"+str(i)+"_"+procted_col_name+".csv"
  
#    data=pd.read_csv(PATH_loop)
#    for group in list(data[procted_col_name].unique()):
#        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP',"y_hat_mean","y_target_mean","y_hat_prob"]:
#            value=float(data[data[procted_col_name]==group][measure])

#            df_out=df_out.append({'Group': group,"ML":"Xgboost"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

#df_out.to_csv(PATH+"Xgboost_metrics_crossvalidated_"+procted_col_name+".csv")


# In[ ]:


#global_proc_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
#global_proc_bar.set_title('Proctected: '+procted_col_name)#
#global_proc_bar.get_figure().savefig(PATH_orig+"/barplot_proc.png")


# In[ ]:





# In[ ]:


list(df[procted_col_name].iloc[valid_index])


# In[ ]:





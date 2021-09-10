#!/usr/bin/env python
# coding: utf-8

# In[23]:


#!/usr/bin/env python
import numpy as np
import config as cfg
import pandas as pd
from tools import file_reader, file_writer, explainer
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


# In[24]:


from utils_copy import *


# In[25]:


procted_col_name="Gender"
y_col_name="Fall"


# In[26]:


pathRoot="../../Data_air/"
pathFall=pathRoot+"Fall.csv"
df=pd.read_csv(pathFall)


# In[27]:


titel_mitigation="test23may"
PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/Xgboost/models/"+titel_mitigation+"/"

PATH=PATH_orig#+seedName+"/"
print(PATH)

#Make dir to files
if not os.path.exists(PATH):
    os.makedirs(PATH)
    print("Created new path!: ",PATH)


# In[28]:


df.shape


# In[30]:


model_dir = cfg.FALL_XGB_DIR
target_name = "Fall"
        
    
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    
X = df.drop([target_name], axis=1)

X_col_names_to_std = [name for name in X.columns if not name in [procted_col_name]]
X[X_col_names_to_std] = pd.DataFrame(preprocessing.scale(X[X_col_names_to_std]),columns=X_col_names_to_std)
y = df[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            stratify=y, random_state=0)
        

#DATA_DIR = cfg.PROCESSED_DATA_DIR
#CASES = ["Complete", "Compliance", "Fall", "Fall_test"]      
#df = file_reader.read_csv(DATA_DIR, 'fall_emb.csv')
 


# In[31]:


neg, pos = np.bincount(y)
scale_pos_weight = neg / pos

params = {"n_estimators": 400,
        "objective": "binary:logistic",
        "scale_pos_weight": scale_pos_weight,
        "use_label_encoder": False,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "seed": 0
}


# In[32]:


model = xgb.XGBClassifier(**params)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)


# In[33]:


df_test=pd.DataFrame([],columns=list(X.columns)+["Fall"]+["output"]+["output_prob"])
df_test


# In[ ]:





# In[34]:


i=0
y_valid_pred = 0*y
valid_acc, valid_pre, valid_recall, valid_roc_auc = list(), list(), list(), list()
for train_index, valid_index in skf.split(X_train, y_train):
    X_train_split, X_valid_split = X_train.iloc[train_index,:], X_train.iloc[valid_index,:]
    y_train_split, y_valid_split = y_train.iloc[train_index], y_train.iloc[valid_index]

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
    y_true_pd=y_valid_split.to_frame().reset_index(drop=True)
    y_pred_pd=y_valid_scores.apply(lambda x: 1 if x==True else 0).to_frame().reset_index(drop=True).rename(columns={"Fall":"output"})
    y_pred_prob_pd=pd.DataFrame(pred, columns = ["output_prob"])
    
    df_subset=pd.concat([X_valid_split.reset_index(drop=True),y_true_pd,y_pred_pd,y_pred_prob_pd],axis=1)
    
    df_test=df_test.append(df_subset, ignore_index=True)
    ######
    
    ###### Save the metrics ####
    
    df_evaluate_proc=get_df_w_metrics(df_subset,procted_col_name,y_col_name,"output")
    df_evaluate_proc.to_csv(PATH+"model"+str(i)+"_"+procted_col_name+".csv")
    
    
    df_evaluate_together=df_subset.copy()
    df_evaluate_together[procted_col_name]="all"
    df_evaluate_all=get_df_w_metrics(df_evaluate_together,procted_col_name,y_col_name,"output")
    df_evaluate_all.to_csv(PATH+"model"+str(i)+"_all.csv")
    
    #############################
    
    
    valid_acc.append(accuracy_score(y_valid_split, y_valid_scores))
    valid_pre.append(precision_score(y_valid_split, y_valid_scores))
    valid_recall.append(recall_score(y_valid_split, y_valid_scores))
    valid_roc_auc.append(roc_auc_score(y_valid_split, y_valid_pred.iloc[valid_index]))
    
    i=i+1



# # Save all data

# In[35]:


df_test.to_csv(PATH+"all_test_data.csv")
print("The full test data lies here:",PATH+"all_test_data.csv")


# # Evaluate

# In[36]:


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]


# In[37]:



#file_writer.write_cm_plot(y_test, y_pred, cfg.REPORTS_PLOTS_DIR,
                       # f'{case.lower()}_xgb_cm.pdf', case)
#file_writer.write_joblib(model, model_dir, f'{case.lower()}_xgboost.joblib')

print(f"Scores for XGBoost model:")
print(f"Accuracy: {np.around(accuracy_score(y_test, y_pred), decimals=3)}")
print(f"Precision: {np.around(precision_score(y_test, y_pred), decimals=3)}")
print(f"Recall: {np.around(recall_score(y_test, y_pred), decimals=3)}")
print(f"ROC AUC: {np.around(roc_auc_score(y_test, y_proba), decimals=3)}\n")


# # Save the confusion data

# In[38]:


column_names = ["Group", "ML", "Measure","Value"]

df_out = pd.DataFrame(columns = column_names)

for i in [0,1,2,3,4,5,6,7,8,9]:
  
    PATH_loop=PATH+"model"+str(i)+"_all.csv"
  
    data=pd.read_csv(PATH_loop)
    for group in ["all"]:
        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP']:
            value=float(data[data[procted_col_name]==group][measure])

            df_out=df_out.append({'Group': group,"ML":"Xgboost"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

df_out.to_csv(PATH+"/Xgboost_metrics_crossvalidated_all.csv")


# In[39]:


global_all_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
global_all_bar.set_title('All')
global_all_bar.get_figure().savefig(PATH_orig+"/barplot_all.png")


# In[ ]:





# In[40]:


column_names = ["Group", "ML", "Measure","Value"]

df_out = pd.DataFrame(columns = column_names)

for i in [0,1,2,3,4,5,6,7,8,9]:

    PATH_loop=PATH+"model"+str(i)+"_"+procted_col_name+".csv"
  
    data=pd.read_csv(PATH_loop)
    for group in list(data[procted_col_name].unique()):
        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP']:
            value=float(data[data[procted_col_name]==group][measure])

            df_out=df_out.append({'Group': group,"ML":"Xgboost"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

df_out.to_csv(PATH+"Xgboost_metrics_crossvalidated_"+procted_col_name+".csv")


# In[41]:


global_proc_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
global_proc_bar.set_title('Proctected: '+procted_col_name)
global_proc_bar.get_figure().savefig(PATH_orig+"/barplot_proc.png")


# In[ ]:





# In[ ]:





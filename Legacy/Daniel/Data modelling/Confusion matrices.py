#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn import preprocessing
import math 
import numpy as np
from sklearn.metrics import confusion_matrix


# In[40]:


#model = "SVM"
#model = "LR"
#model = "RF"
#model = "FFNN"
model = "XGBoost"
method = "DI remove no gender"


# In[41]:


matrix_data = pd.read_csv(f'/restricted/s161749/G2020-57-Aalborg-bias/Plot_metrics/{method}/{model}_gender_obs.csv')
matrix_data = matrix_data.drop(columns=['Unnamed: 0'])


# In[42]:


matrix_data


# In[43]:



if model=="FFNN" or model=="XGBoost":    
y_true = matrix_data['Fall']
y_pred = matrix_data['output']
else:    
y_true = matrix_data['y_true']
y_pred = matrix_data['y_hat_binary']


# In[44]:


tn_list=[]
fp_list=[]
fn_list=[]
tp_list=[]


# In[45]:


for i in range(0,10):
    
    y_true_local=y_true[(i*2144):2144+(i*2144)]
    y_pred_local=y_pred[(i*2144):2144+(i*2144)]
    
    tn, fp, fn, tp = confusion_matrix(y_true_local,y_pred_local,normalize='all').ravel()
    
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)


# In[46]:


#tn
tn_mean=np.round((pd.DataFrame(tn_list)[0]*100).mean(),1)
tn_low=np.round((pd.DataFrame(tn_list)[0]*100).mean()-((1.96*(pd.DataFrame(tn_list)[0]*100).std()/math.sqrt(len(tn_list)))),1)
tn_high=np.round((pd.DataFrame(tn_list)[0]*100).mean()+((1.96*(pd.DataFrame(tn_list)[0]*100).std()/math.sqrt(len(tn_list)))),1)

#fp
fp_mean=np.round((pd.DataFrame(fp_list)[0]*100).mean(),1)
fp_low=np.round((pd.DataFrame(fp_list)[0]*100).mean()-((1.96*(pd.DataFrame(fp_list)[0]*100).std()/math.sqrt(len(fp_list)))),1)
fp_high=np.round((pd.DataFrame(fp_list)[0]*100).mean()+((1.96*(pd.DataFrame(fp_list)[0]*100).std()/math.sqrt(len(fp_list)))),1)

#fn
fn_mean=np.round((pd.DataFrame(fn_list)[0]*100).mean(),1)
fn_low=np.round((pd.DataFrame(fn_list)[0]*100).mean()-((1.96*(pd.DataFrame(fn_list)[0]*100).std()/math.sqrt(len(fn_list)))),1)
fn_high=np.round((pd.DataFrame(fn_list)[0]*100).mean()+((1.96*(pd.DataFrame(fn_list)[0]*100).std()/math.sqrt(len(fn_list)))),1)

#tp
tp_mean=np.round((pd.DataFrame(tp_list)[0]*100).mean(),1)
tp_low=np.round((pd.DataFrame(tp_list)[0]*100).mean()-((1.96*(pd.DataFrame(tp_list)[0]*100).std()/math.sqrt(len(tp_list)))),1)
tp_high=np.round((pd.DataFrame(tp_list)[0]*100).mean()+((1.96*(pd.DataFrame(tp_list)[0]*100).std()/math.sqrt(len(tp_list)))),1)




# In[47]:


print("\\textbf{",model,"} & \\textbf{",tp_mean,"} & \\textbf{",fp_mean,"} & \\textbf{",tn_mean,"} & \\textbf{",fn_mean,"} \\\ ")
print(f"& ({tp_low}-{tp_high}) & ({fp_low}-{fp_high}) & ({tn_low}-{tn_high}) & ({fn_low}-{fn_high})\\\ ")


# In[495]:


print(tn_high)


# In[ ]:





# In[ ]:





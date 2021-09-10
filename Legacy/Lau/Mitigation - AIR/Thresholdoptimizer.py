#!/usr/bin/env python
# coding: utf-8

# In[467]:


#pip install raiwidgets
#pip install fairlearn
#pip install ipywidgets
import pandas as pd
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

import matplotlib.pyplot as plt
plt.style.use('seaborn') 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier

from raiwidgets import FairnessDashboard
from fairlearn.widget import FairlearnDashboard



import matplotlib.pyplot as plt  
import numpy as np
from sklearn import metrics


import scikitplot as skplt
import matplotlib.pyplot as plt


from fairlearn.postprocessing import plot_threshold_optimizer


from sklearn.metrics import confusion_matrix


# # Load data

# In[465]:


file_name="Fall_count_clusterOHE_std.csv"
full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name


merged=pd.read_csv(full_file_path)


# In[323]:


merged=merged.drop(columns=["Unnamed: 0"])


# In[324]:


merged=merged[["BirthYear","Fall","Gender","NumberAts","LoanPeriod"]]


# # Logistic regression

# In[326]:


df_train, df_test = train_test_split(merged, test_size=0.5,stratify=merged["Fall"])

df_test=df_test.reset_index(drop=True)

X_train = df_train.drop(columns=['Fall'], axis=1) #drop Sex (protected) and paymant (y)
X_test = df_test.drop(columns=['Fall'], axis=1)  #drop Sex (protected) and paymant (y)

y_train = df_train['Fall']
y_test = df_test['Fall']



model = RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[327]:


y_pred_model=model.predict(X_test)


# ## Custommade ROC curve

# In[351]:


#fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_model)
#roc_auc = metrics.auc(fpr, tpr)
#display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
#display.plot()  
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#        label='Chance', alpha=.8)
#plt.show()


# # Plotting both gender
# 

# In[406]:


y_pred_model=model.predict(X_test)


#SAVING PROBABILITIES
y_pred_model_proba=model.predict_proba(X_test)
y_pred_model_proba = np.array([x[1] for x in y_pred_model_proba])


# In[407]:


female_index=df_test[df_test["Gender"]==0].index
male_index=df_test[df_test["Gender"]==1].index

fpr_female, tpr_female, thresholds_female = metrics.roc_curve(y_test[female_index], y_pred_model[female_index])
roc_auc_female = metrics.auc(fpr_female, tpr_female)
fpr_male, tpr_male, thresholds_male = metrics.roc_curve(y_test[male_index], y_pred_model[male_index])
roc_auc_male = metrics.auc(fpr_male, tpr_male)


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(roc_auc_male))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(roc_auc_female))

plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("Male and females")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[408]:


print(f"TPR_male: {tpr_male[1]}, FPR_male: {fpr_male[1]} ")
print(f"TPR_female: {tpr_female[1]}, FPR_female: {fpr_female[1]} ")


# In[485]:


female_index=df_test[df_test["Gender"]==0].index
male_index=df_test[df_test["Gender"]==1].index

fpr_female, tpr_female, thresholds_female = metrics.roc_curve(y_test[female_index], y_pred_model_proba[female_index],drop_intermediate=False)
roc_auc_female = metrics.auc(fpr_female, tpr_female)
fpr_male, tpr_male, thresholds_male = metrics.roc_curve(y_test[male_index], y_pred_model_proba[male_index],drop_intermediate=False)
roc_auc_male = metrics.auc(fpr_male, tpr_male)


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(roc_auc_male))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(roc_auc_female))

plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.title("Male and females")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # Optimized by Equal Opportunity

# In[ ]:





# In[492]:


#a = tpr_female[(tpr_female!=0.0)&(tpr_female!=1.0)]
#b = tpr_male[(tpr_male!=0.0)&(tpr_male!=1.0)]
#high_fem=
#fem_index=
#high_mal=
#mal_index=
#AUC=1/2 - FPR/2 + TPR/2
#for i,tpr_valfemale in enumerate(a):
#    for j,tpr_valfemale in enumerate(b):
        
    
#a = tpr_female[(tpr_female!=0.0)&(tpr_female!=1.0)]
#b = tpr_male[(tpr_male!=0.0)&(tpr_male!=1.0)]
#ans = list(map(lambda y:min(a, key=lambda x:abs(x-y)),b))
#np.argmin((ans-b))
#thresholds_male[56]    
        


# In[493]:


tprs1=tpr_male
tprs2=tpr_female

fprs1=fpr_male
fprs2=fpr_female


thresholds1=thresholds_female
thresholds2=thresholds_male

probs1=y_pred_model_proba[female_index]
probs2=y_pred_model_proba[male_index]

true1=y_test[female_index]
true2=y_test[male_index]


# In[495]:


accs = []
new_thres1 = []
new_thres2 = []
x1s = []
x2s = []
y1s = []
y2s = []
for point in range(min(len(tprs1),len(tprs2))):

    y2 = tprs2[point]
    ydiff = abs(y2 - tprs1)
    intersection_idx = np.argmin(ydiff)
    y1 = tprs1[intersection_idx]
    x2 = fprs2[point]
    x1 = fprs1[intersection_idx]

    x1s.append(x1)
    x2s.append(x2)
    y1s.append(y1)
    y2s.append(y2)

    thres1 = thresholds1[intersection_idx]
    thres2 = thresholds2[point]
    new_thres1.append(thres1)
    new_thres2.append(thres2)

    y_pred1 = (probs1 > thres1)#.float()
    y_pred2 = (probs2 > thres2)#.float()
    conf1 = confusion_matrix(true1, y_pred1)
    conf2 = confusion_matrix(true2, y_pred2)
    acc = (conf1[0,0] + conf1[1,1] + conf2[0,0] + conf2[1,1]) / (len(y_pred1) + len(y_pred2))
    accs.append(acc)
best_idx = np.argmax(accs)
best_acc = accs[best_idx]
best_thres1 = new_thres1[best_idx]
best_thres2 = new_thres2[best_idx]

## Plotting the ROC-curves on their own
#plt.plot(fprs1,tprs1,color="blue", label = "Female")
#plt.plot(fprs2,tprs2,color="green", label = "Male")
#x_values = [0, 1] 
#y_values = [0, 1]
#plt.plot(x_values, y_values,'--',color="red")
#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.title("Female and Male ROC-Curves")
#plt.ylim(0, 1.1)
#plt.legend(loc = "lower right")
#plt.show()

# Plotting of the points with equal tpr and finding accuracy for these thresholds
plt.plot(fprs1,tprs1,color="blue", label = "Female")
plt.plot(fprs2,tprs2,color="green", label = "Male")
x_values = [0, 1] 
y_values = [0, 1]
plt.plot(x_values, y_values,'--',color="red")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.ylim(0, 1.1)
plt.legend(loc = "lower right")


x1 = x1s[best_idx]
x2 = x2s[best_idx]
y1 = y1s[best_idx]
y2 = y2s[best_idx]
plt.scatter(np.array([x1,x2]), np.array([y1,y2]), color = ["blue", "green"])
if x1 > x2:
    xline = np.linspace(x2, x1, num = 2)
else:
    xline = np.linspace(x1, x2, num = 2)
yline = np.asarray([y2] * len(xline))
plt.plot(xline, yline, "--", color = "red")

plt.title("Best Equal Opportunity Threshold")
plt.text(-0.02, 0.93, f"Accuracy = {best_acc:.3f}\nThreshold 1 = {best_thres1:.2f}\nThreshold 2 = {best_thres2:.3f}")
plt.show()


# In[ ]:





# In[449]:





# In[451]:





# ## ROC curve with module

# In[353]:


#plot_roc_curve(model, X_test, y_test)
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#        label='Chance', alpha=.8)
#plt.show()


# In[ ]:





# # Threshold optimizer

# In[354]:


from fairlearn.postprocessing import ThresholdOptimizer
#https://fairlearn.org/v0.5.0/api_reference/fairlearn.postprocessing.html


# In[382]:


#optimizer = ThresholdOptimizer(estimator=model, constraints='demographic_parity',objective="accuracy_score") #
optimizer = ThresholdOptimizer(estimator=model, constraints="equalized_odds")
#optimizer = ThresholdOptimizer(estimator=model, constraints="true_positive_rate_parity")
#optimizer = ThresholdOptimizer(estimator=model, constraints="true_negative_rate_parity")
#optimizer = ThresholdOptimizer(estimator=model, constraints="false_positive_rate_parity")
#optimizer = ThresholdOptimizer(estimator=model, constraints="false_negative_rate_parity")

optimizer.fit(X_train, y_train, sensitive_features=df_train['Gender'])
from fairlearn.widget import FairlearnDashboard


# In[383]:


y_pred_optim=optimizer.predict(X_test, sensitive_features=df_test['Gender'])


# In[ ]:


optimizer.


# In[384]:


plot_threshold_optimizer(optimizer)


# In[385]:


#fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_optim)
#roc_auc = metrics.auc(fpr, tpr)
#display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
#display.plot()  
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#        label='Chance', alpha=.8)
#plt.show()


# In[ ]:





# In[390]:


female_index=df_test[df_test["Gender"]==0].index
male_index=df_test[df_test["Gender"]==1].index

fpr_female, tpr_female, thresholds_female = metrics.roc_curve(y_test[female_index], y_pred_optim[female_index])
roc_auc_female = metrics.auc(fpr_female, tpr_female)
fpr_male, tpr_male, thresholds_male = metrics.roc_curve(y_test[male_index], y_pred_optim[male_index])
roc_auc_male = metrics.auc(fpr_male, tpr_male)


plt.plot(fpr_male,tpr_male,color="green",label="Male, auc="+str(roc_auc_male))
plt.plot(fpr_female,tpr_female,color="blue",label="Female, auc="+str(roc_auc_female))

plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[387]:


print(f"TPR_male: {tpr_male[1]}, FPR_male: {fpr_male[1]} ")
print(f"TPR_female: {tpr_female[1]}, FPR_female: {fpr_female[1]} ")


# In[388]:


#LÆG MÆRKE TIL, at TPR ER DET tættere!


# In[ ]:





# In[392]:


optimizer.inter


# # Another threshold optimizer

# https://pypi.org/project/threshold-optimizer/

# https://pycaret.org/optimize-threshold/

# In[ ]:





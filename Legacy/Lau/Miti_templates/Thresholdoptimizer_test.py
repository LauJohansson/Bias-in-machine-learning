#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from sklearn import svm
from sklearn.metrics import plot_roc_curve

from raiwidgets import FairnessDashboard
from fairlearn.widget import FairlearnDashboard


# # Load data

# In[5]:


merged=pd.read_csv("../../Data_air/dummy_data.csv")


# In[ ]:


merged=merged.drop(columns=["BorgerId","Unnamed: 0"])


# In[ ]:


merged=merged[["Birth Year","Fall","Gender"]]


# In[3]:


import random
merged["Corr_var"]=merged["Fall"].apply(lambda x: random.randint(30,60) if x==1 else random.randint(25,40))


# In[34]:


merged["Gender_string"]=merged["Gender"].apply(lambda x: "Male" if x==1 else "Female")


# In[19]:


merged["Fall"].mean()


# In[20]:


merged.shape


# In[21]:


#merged=merged.head(500)


# # Logistic regression

# In[22]:


#X_cols=[name for name in merged.columns.tolist() if name not in ["Fall","BorgerId"]]
#X=merged[X_cols]
#y=merged["Fall"]



df_train, df_test = train_test_split(merged, test_size=0.25)

X_train = df_train.drop(columns=['Fall','Gender_string'], axis=1) #drop Sex (protected) and paymant (y)
X_test = df_test.drop(columns=['Fall','Gender_string'], axis=1)  #drop Sex (protected) and paymant (y)

y_train = df_train['Fall']
y_test = df_test['Fall']



model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[23]:


plot_roc_curve(model, X_test, y_test)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.show()


# In[35]:


'''
FairnessDashboard(
    sensitive_features=df_test['Gender'],
    #sensitive_feature_names=df_test['Gender_string'],
    y_true=df_test['Fall'],
    y_pred=model.predict(X_test)
)
'''


# In[36]:


FairlearnDashboard(sensitive_features=df_test['Gender'],
                   sensitive_feature_names=['BinaryGender'],
                   y_true=df_test['Fall'].tolist(),
                   y_pred=model.predict(X_test))


# # Threshold optimizer

# In[37]:


from fairlearn.postprocessing import ThresholdOptimizer
#https://fairlearn.org/v0.5.0/api_reference/fairlearn.postprocessing.html


# In[38]:


optimizer = ThresholdOptimizer(estimator=model, constraints='demographic_parity') #
#optimizer = ThresholdOptimizer(estimator=model, constraints=’equalized_odds')
#optimizer = ThresholdOptimizer(estimator=model, constraints=’true_positive_rate_parity’)
#optimizer = ThresholdOptimizer(estimator=model, constraints=’true_negative_rate_parity’)
#optimizer = ThresholdOptimizer(estimator=model, constraints=’false_positive_rate_parity’)
#optimizer = ThresholdOptimizer(estimator=model, constraints=’false_negative_rate_parity’)

optimizer.fit(X_train, y_train, sensitive_features=df_train['Gender'])
from fairlearn.widget import FairlearnDashboard


# In[39]:


y_pred_optim=optimizer.predict(X_test, sensitive_features=df_test['Gender'])


# In[ ]:





# In[ ]:


from fairlearn.postprocessing import plot_threshold_optimizer


# In[ ]:


plot_threshold_optimizer(optimizer)


# In[ ]:





# In[ ]:





# In[40]:


from raiwidgets import FairnessDashboard


# In[ ]:





# In[41]:


'''

comparison = {
    'Original model': model.predict(X_test),
    'TresholdOptimizer': optimizer.predict(X_test, sensitive_features=df_test['Gender'])
}



FairnessDashboard(
    sensitive_features=df_test['Gender'],
    #sensitive_feature_names=df_test['Gender_string'],
    y_true=df_test['Fall'],
    y_pred=comparison
)
'''


# ## Fairness dashboard
# https://fairlearn.org/v0.5.0/user_guide/assessment.html#fairlearn-dashboard

# In[42]:


from fairlearn.widget import FairlearnDashboard


# In[43]:



orig_model_ys=model.predict(X_test)
optimizer_ys=optimizer.predict(X_test, sensitive_features=df_test['Gender'])


comparison = {
    'Original model': orig_model_ys,
    'TresholdOptimizer': optimizer_ys
}

FairlearnDashboard(sensitive_features=df_test['Gender'],
                   sensitive_feature_names=['BinaryGender'],
                   y_true=df_test['Fall'].tolist(),
                   y_pred=comparison)


# ![image.png](attachment:image.png)

# In[ ]:





# In[ ]:





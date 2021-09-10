#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

from sklearn.linear_model import LogisticRegression

from sklearn import svm


# In[68]:


merged=pd.read_csv("../Data/dummy_data.csv")


# In[69]:


#making a correlated variable

import random

merged=merged[["Birth Year","Fall","Gender","BorgerId"]]
merged["Corr_var"]=merged["Fall"].apply(lambda x: random.randint(30,60) if x==1 else random.randint(25,40))


# # Logistic regression

# In[70]:


X_cols=[name for name in merged.columns.tolist() if name not in ["Fall","BorgerId"]]
X=merged[X_cols]
y=merged["Fall"]


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.33, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

preds=clf.predict(X_test)


# In[71]:


preds.mean()


# In[ ]:





# # Equalized odds

# ## Training the classifier (SVM)

# In[72]:


X_cols=[name for name in merged.columns.tolist() if name not in ["Fall","BorgerId"]]


# In[73]:


train, test =     train_test_split(merged.drop(["BorgerId"],axis=1), test_size = 0.2, random_state = 123)


# In[74]:


train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)


# In[75]:


clf = svm.SVC()
clf.fit(train[X_cols], train["Fall"])


# In[76]:


clf.score(test[X_cols], test["Fall"])


# Making predictions:

# In[77]:


train["output"]=clf.predict(train[X_cols])


# In[78]:


test["output"]=clf.predict(test[X_cols])


# In[79]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test["Fall"].ravel(), test["output"].ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
disp.plot() 


# ### FEMALE

# In[93]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test[test["Gender"]==0]["Fall"].ravel(), test[test["Gender"]==0]["output"].ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
disp.plot() 


# ### MALE

# In[94]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test[test["Gender"]==1]["Fall"].ravel(), test[test["Gender"]==1]["output"].ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
disp.plot() 


# ## Creating the binary dataset

# Rename trainingset TRUE falls to be named "selected_col". <br>
# Rename trainingset PREDICTED falls to be named "selected_col".

# In[80]:


trainset_renamed=train.rename(columns={"Fall": "selected_col"}).drop(['output'], axis=1)
predicted_trainset_renamed=train.rename(columns={"output": "selected_col"}).drop(['Fall'], axis=1)


# Rename testset TRUE falls to be named "selected_col". <br>
# Rename testset PREDICTED falls to be named "selected_col".

# In[81]:


testset_renamed=test.rename(columns={"Fall": "selected_col"}).drop(['output'], axis=1)
predicted_testset_renamed=test.rename(columns={"output": "selected_col"}).drop(['Fall'], axis=1)


# Needs all columns when using BinaryLabelDataset-class:

# In[82]:


all_cols=X_cols
all_cols=all_cols.append("selected_col")


# ### Train binary dataset:

# In[83]:


#Train TRUE
train_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=trainset_renamed,
                                label_names=['selected_col'],
                                protected_attribute_names=['Gender'],
                                unprivileged_protected_attributes=['0'])
#TRAIN PREDICTED
pred_train_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=predicted_trainset_renamed,
                                label_names=['selected_col'], #label_names=['preds'],
                                protected_attribute_names=['Gender'],
                                unprivileged_protected_attributes=['0'])


# ### Test binary dataset:

# In[84]:


#Test TRUE
test_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=testset_renamed,
                                label_names=['selected_col'],
                                protected_attribute_names=['Gender'],
                                unprivileged_protected_attributes=['0'])
#test PREDICTED
pred_test_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=predicted_testset_renamed,
                                label_names=['selected_col'], #label_names=['preds'],
                                protected_attribute_names=['Gender'],
                                unprivileged_protected_attributes=['0'])


# ## Training the EqOdds model

# In[85]:


#https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.EqOddsPostprocessing.html#aif360.algorithms.postprocessing.EqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
new_eq=EqOddsPostprocessing(unprivileged_groups= [{'Gender': 0}],privileged_groups=[{'Gender': 1}])


# In[86]:


#fitting
new_eq.fit(train_BLD,pred_train_BLD)


# In[87]:


#create predictions
prediction_test = new_eq.predict(pred_test_BLD)


# In[88]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test["Fall"].ravel(), prediction_test.labels.ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
disp.plot() 


# ## Training the Calibrated model

# In[89]:


#https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.EqOddsPostprocessing.html#aif360.algorithms.postprocessing.EqOddsPostprocessing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
new_ca=CalibratedEqOddsPostprocessing(unprivileged_groups= [{'Gender': 0}],privileged_groups=[{'Gender': 1}])


# In[90]:


#fitting
new_ca.fit(train_BLD,pred_train_BLD)


# In[91]:


#predict
prediction_test = new_ca.predict(pred_test_BLD)


# In[92]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test["Fall"].ravel(), prediction_test.labels.ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
disp.plot() 


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Load data

# In[1]:


#pip install BlackBoxAuditing
import pandas as pd
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn import svm


# In[2]:


merged=pd.read_csv("../../Data/dummy_data.csv")


# In[3]:


#making a correlated variable

import random

merged=merged[["Birth Year","Fall","Gender","BorgerId"]]
merged["Corr_var"]=merged["Fall"].apply(lambda x: random.randint(30,60) if x==1 else random.randint(25,40))

mean_female=10
std_female=2

mean_male=18
std_male=1


merged["Biased_var"]=merged["Gender"].apply(lambda x: np.random.normal(mean_male, std_male, 1)[0] if x==1 else np.random.normal(mean_female, std_female, 1)[0])


# # Functions

# In[4]:


def calc_prop(data, group_col, group, output_col, output_val):
    new = data[data[group_col] == group]
    return len(new[new[output_col] == output_val])/len(new)


# # Train model

# In[5]:


X_cols=[name for name in merged.columns.tolist() if name not in ["Fall","BorgerId"]]
train, test =     train_test_split(merged.drop(["BorgerId"],axis=1), test_size = 0.2, random_state = 123)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
clf = svm.SVC()
clf.fit(train[X_cols], train["Fall"])
clf.score(test[X_cols], test["Fall"])


# In[6]:


train["output"]=clf.predict(train[X_cols])
test["output"]=clf.predict(test[X_cols])


# In[7]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test["Fall"].ravel(), test["output"].ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
disp.plot() 


# ## Creating the binary dataset

# Rename trainingset TRUE falls to be named "selected_col". <br>
# Rename trainingset PREDICTED falls to be named "selected_col".

# In[8]:


trainset_renamed=train.rename(columns={"Fall": "selected_col"}).drop(['output'], axis=1)
predicted_trainset_renamed=train.rename(columns={"output": "selected_col"}).drop(['Fall'], axis=1)


# Rename testset TRUE falls to be named "selected_col". <br>
# Rename testset PREDICTED falls to be named "selected_col".

# In[9]:


testset_renamed=test.rename(columns={"Fall": "selected_col"}).drop(['output'], axis=1)
predicted_testset_renamed=test.rename(columns={"output": "selected_col"}).drop(['Fall'], axis=1)


# Needs all columns when using BinaryLabelDataset-class:

# In[10]:


all_cols=X_cols
all_cols=all_cols.append("selected_col")


# ### Train binary dataset:

# In[35]:


#Train TRUE
train_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=trainset_renamed,
                                label_names=['selected_col'],
                                protected_attribute_names=['Gender'],
                                unprivileged_protected_attributes=["0"])
#TRAIN PREDICTED
pred_train_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=predicted_trainset_renamed,
                                label_names=['selected_col'], #label_names=['preds'],
                                protected_attribute_names=['Gender'],
                                unprivileged_protected_attributes=['0'])


# ### Test binary dataset:

# In[36]:


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


# # Disparate impact remove

# In[37]:


from aif360.algorithms.preprocessing import DisparateImpactRemover
di = DisparateImpactRemover(repair_level=1.0)


# ## Create new data set with repaired (level=1)

# Save column names

# In[38]:


train_BLD.feature_names


# In[39]:


train_BLD.label_names


# In[40]:


all_col_names=train_BLD.feature_names+train_BLD.label_names
print(all_col_names)


# ### Train data:

# In[41]:


rp_train = di.fit_transform(train_BLD) #Using the 
rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=all_col_names)


# ### Test data:

# In[42]:


rp_test = di.fit_transform(test_BLD)
rp_test_pd = pd.DataFrame(np.hstack([rp_test.features,rp_test.labels]),columns=all_col_names)


# In[43]:


sns.histplot(data=train,x="Birth Year",hue="Gender",stat="density",common_norm=False)
plt.title("Original data (level=1)")
plt.show()


# In[44]:


sns.histplot(data=rp_train_pd,x="Birth Year",hue="Gender",stat="density",common_norm=False)
plt.title("Repaired data")
plt.show()


# In[ ]:





# # Looping through different repairs

# In[45]:


all_col_names=train_BLD.feature_names+rp_train.label_names

plt.subplots(2,5, figsize=(20, 20), facecolor='w', edgecolor='k')

level_list=[0.0,0.3,0.5,0.8,1.0]

rows= len(level_list)
cols=1
i=1


for level in level_list:
    
    di = DisparateImpactRemover(repair_level=level)
    
    #Training data
    rp_train = di.fit_transform(train_BLD) #Using the 
    rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=all_col_names)

    #test data
    #rp_test = di.fit_transform(test_BLD)
    #rp_test_pd = pd.DataFrame(np.hstack([rp_test.features,rp_test.labels]),columns=all_col_names)

    plt.subplot(rows,cols,i)
    sns.histplot(data=rp_train_pd,x="Birth Year",hue="Gender",stat="density",common_norm=False,bins=50)
    plt.title(f"Repaired data (Level={level})")
    
    i=i+1
    
plt.show()


# In[ ]:





# In[48]:


all_col_names=train_BLD.feature_names+rp_train.label_names

plt.subplots(2,5, figsize=(20, 20), facecolor='w', edgecolor='k')

level_list=[0.0,0.3,0.5,0.8,1.0]

rows= len(level_list)
cols=1
i=1


for level in level_list:
    
    di = DisparateImpactRemover(repair_level=level)
    
    #Training data
    rp_train = di.fit_transform(train_BLD) #Using the 
    rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=all_col_names)

    #test data
    #rp_test = di.fit_transform(test_BLD)
    #rp_test_pd = pd.DataFrame(np.hstack([rp_test.features,rp_test.labels]),columns=all_col_names)

    plt.subplot(rows,cols,i)
    sns.histplot(data=rp_train_pd,x="Biased_var",hue="Gender",stat="density",common_norm=False,bins=50)
    plt.title(f"Repaired data (Level={level})")
    
    i=i+1
    
plt.show()


# In[ ]:





# In[ ]:


fig,ax=plt.subplotw(2,1,sharex=True)

ax=ax.ravel()

sns. (ax=ax[0])

sns. (ax=ax[1])


# In[ ]:





# In[ ]:





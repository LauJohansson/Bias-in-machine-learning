#!/usr/bin/env python
# coding: utf-8

# # Load data

# In[ ]:





# In[28]:


import pandas as pd
import seaborn as sns
import numpy as np

#from aif360.datasets import BinaryLabelDataset
#from aif360.metrics import BinaryLabelDatasetMetric


# Citizens:

# In[29]:


pathRoot="../Data/"


# In[30]:


pathCitizens=pathRoot+"borgere_hmi_Rasmus_BorgerId_Gender_BirthYear.xlsx"
citizensData=pd.read_excel(pathCitizens,sheet_name=5)


# In[31]:


citizensData["INDLEVERINGSDATO"]=pd.to_datetime(citizensData["INDLEVERINGSDATO"], errors='coerce')


# Patientdata

# In[32]:


pathPatientData=pathRoot+"DrPatientData_RasmusPlusBorgerIdMinusCPR.xlsx"
patientData=pd.read_excel(pathPatientData)


# Observations

# In[33]:


pathObservationsData=pathRoot+"Observationer_Rasmus_BorgerIdOgGenderMinusCPR.xlsx"
observationsData=pd.read_excel(pathObservationsData)


# Training:

# In[34]:


pathTrainingData=pathRoot+"Træning_Rasmus_PlusBorgerIdOgGenderMinusCPR.xlsx"
trainingData=pd.read_excel(pathTrainingData)


# # Citizens

# In[8]:


citizensData.dtypes


# In[9]:


citizensData.head(1)


# In[10]:


citizensData.columns


# In[11]:


citizensData["HJAELPEMIDDELHMINAVN"].nunique()


# The fraction of male females:

# In[12]:


citizensData[["Gender","BorgerID"]].groupby(["Gender"]).nunique()/citizensData["BorgerID"].nunique()


# In[13]:


sns.barplot(data=citizensData[["Gender","BorgerID"]].astype({'BorgerID': 'int32'}),x="Gender",y="BorgerID")


# # Patient data

# In[14]:


patientData.dtypes


# In[15]:


patientData["Screening Content"][0].split("#")[0].split(";")


# In[16]:


patientData["Screening Content"][0].split("#")[1].split(";")


# In[17]:


patientData["Screening Content"][1].split("#")[0].split(";")


# In[18]:


for i in range(len(patientData["Screening Content"][0].split("#"))):
    for j in range(len(patientData["Screening Content"][0].split("#")[i].split(";"))):
        print(patientData["Screening Content"][0].split("#")[i].split(";")[j])


# ### Renaming BorgerId

# In[48]:


citizensData.head(5)


# In[49]:


citizensData=citizensData.rename(columns={"BorgerID":"BorgerId"})
citizensData.head(5)


# # Join the data

# Hvordan bør det gøres?
# 
# 
# select *, case exists(falddata) then 1 else 0 end as "Faldet" <br>
# from hjælpemiddel<br>
#      left join generelt data <br>
#      left join falddata <br>
# 

# In[50]:


observationsData


# In[51]:


citizensData


# In[54]:


#https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
one_hot = pd.get_dummies(citizensData['HJAELPEMIDDELHMINAVN'])
# Drop column B as it is now encoded
citizensData_updated = citizensData[["BorgerId","Gender","Birth Year"]]
# Join the encoded df
citizensData_updated = citizensData_updated.join(one_hot)
citizensData_updated


# In[55]:


citizensData_updated=citizensData_updated.groupby('BorgerId').max()


# In[66]:


citizensData_updated.shape


# In[67]:


citizensData_updated.dtypes


# In[57]:


citizensData_updated = citizensData_updated.reset_index()


# In[58]:


citizensData_updated.drop_duplicates(subset=['BorgerId'])


# In[59]:


citizensData_updated.dropna(subset=['BorgerId'])


# In[ ]:





# In[70]:


merged_new=pd.merge(
    left=observationsData,
    right=citizensData_updated, 
    how="left",
    on="BorgerId",
    #left_on="BorgerId",
    #right_on="BorgerId",
    left_index=False,
    right_index=False,
    #sort=True,
    suffixes=("_obs", "_cit"),
   # copy=True,
    indicator=False,
   # validate=None,
)


# In[71]:


merged_new


# In[63]:


merged_new_clean = merged_new.dropna(subset=['BorgerId'])
merged_new_clean = merged_new_clean.dropna(subset=['Birth Year'])
merged_new_clean


# In[65]:


merged_new_clean.drop_duplicates(subset=['BorgerId'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


merged=pd.merge(
    left=citizensData_updated,
    right=observationsData,
    how="left",
    on="BorgerId",
    #left_on="BorgerId",
    #right_on="BorgerId",
    left_index=False,
    right_index=False,
    #sort=True,
    suffixes=("_cit", "_obs"),
   # copy=True,
    indicator=False,
   # validate=None,
)


# In[28]:



merged["Fall"]=merged["Dato"].isnull().apply(lambda x: 1 if x is False else 0)

#merged=merged[merged["Svar lang"].notna()]


# In[29]:


merged.head(5)


# In[30]:


merged=merged.drop(['Gender_obs','Dato','Spørgmål lang','Svar lang'], axis=1)


# In[31]:


merged["Gender"]=merged["Gender_cit"].apply(lambda x: 0 if x=="FEMALE" else 1)
merged=merged.drop(["Gender_cit"],axis=1)


# In[32]:


merged=merged.drop_duplicates()


# In[33]:


merged.BorgerId.value_counts()


# In[34]:


merged.Fall.mean()


# ## Sanity check

# In[35]:


observationsData.head(5)


# In[38]:


citizensData_updated=citizensData_updated.reset_index()


# In[39]:


citizensData_updated.head(5)


# In[40]:


observationsData.BorgerId.value_counts()


# In[41]:


citizensData_updated.BorgerId.value_counts()


# In[42]:


merged.BorgerId.value_counts()


# In[43]:


merged.Fall.mean()


# In[44]:


merged.head(5)


# In[46]:


merged.to_csv("../Data/dummy_data.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Support vector machine

# In[127]:


X_cols=[name for name in merged.columns.tolist() if name not in ["Fall","BorgerId"]]


# In[128]:


X=merged[X_cols]


# In[129]:


y=merged["Fall"]


# In[130]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.33, random_state=42)


# In[131]:


from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)


# In[132]:


clf.score(X_test, y_test)


# In[133]:


preds=clf.predict(X_test)


# In[135]:


preds


# ## Equalized odds

# In[136]:


X_cols=[name for name in merged.columns.tolist() if name not in ["Fall","BorgerId"]]


# In[137]:


train, test =     train_test_split(merged.drop(["BorgerId"],axis=1), test_size = 0.2, random_state = 123)


# In[138]:


train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)


# In[139]:


clf = svm.SVC()
clf.fit(train[X_cols], train["Fall"])


# In[140]:


clf.score(test[X_cols], test["Fall"])


# In[141]:


y_pred_f1=clf.predict(test[X_cols])


# In[142]:


from sklearn.metrics import f1_score

f1_score(test["Fall"], y_pred_f1)


# In[143]:


y_pred_f1[y_pred_f1==0]


# Making predictions:

# In[144]:


train["output"]=clf.predict(train[X_cols])


# In[145]:


test["output"]=clf.predict(test[X_cols])


# In[146]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test["Fall"].ravel(), test["output"].ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
disp.plot() 


# In[ ]:





# In[ ]:





# # Equalized odds

# Rename trainingset TRUE falls to be named "selected_col". <br>
# Rename trainingset PREDICTED falls to be named "selected_col".

# In[147]:


trainset_renamed=train.rename(columns={"Fall": "selected_col"}).drop(['output'], axis=1)
predicted_trainset_renamed=train.rename(columns={"output": "selected_col"}).drop(['Fall'], axis=1)


# Rename testset TRUE falls to be named "selected_col". <br>
# Rename testset PREDICTED falls to be named "selected_col".

# In[148]:


testset_renamed=test.rename(columns={"Fall": "selected_col"}).drop(['output'], axis=1)
predicted_testset_renamed=test.rename(columns={"output": "selected_col"}).drop(['Fall'], axis=1)


# Needs all columns when using BinaryLabelDataset-class:

# In[149]:


all_cols=X_cols
all_cols=all_cols.append("selected_col")


# Create training binary dataset:

# In[171]:


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


# Create test binary dataset:

# In[172]:


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


# In[ ]:





# In[194]:


#https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.EqOddsPostprocessing.html#aif360.algorithms.postprocessing.EqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
new_eq=EqOddsPostprocessing(unprivileged_groups= [{'Gender': 0}],privileged_groups=[{'Gender': 1}])


# In[195]:


new_eq.fit(train_BLD,pred_train_BLD)


# In[196]:


prediction_test = new_eq.predict(pred_test_BLD)


# In[197]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test["Fall"].ravel(), prediction_test.labels.ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
disp.plot() 


# In[188]:


#https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.EqOddsPostprocessing.html#aif360.algorithms.postprocessing.EqOddsPostprocessing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
new_ca=CalibratedEqOddsPostprocessing(unprivileged_groups= [{'Gender': 0}],privileged_groups=[{'Gender': 1}])


# In[189]:


new_ca.fit(train_BLD,pred_train_BLD)


# In[192]:


prediction_test = new_ca.predict(pred_test_BLD)


# In[193]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test["Fall"].ravel(), prediction_test.labels.ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
disp.plot() 


# In[ ]:


test["Fall"].ravel()==


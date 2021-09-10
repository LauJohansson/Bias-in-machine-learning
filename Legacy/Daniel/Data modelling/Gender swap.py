#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn import preprocessing


# ## Getting the data

# In[6]:


dummy_data = pd.read_csv('../Data/dummy_data.csv')


# In[5]:


fall_data = pd.read_csv('../Data_air/fall_emb.csv')


# In[3]:


from sklearn.utils import shuffle
dummy_data = shuffle(dummy_data)


# In[4]:


dummy_data.shape


# In[5]:


dummy_data_hold_out = dummy_data[0:2000]
dummy_data_for_train = dummy_data[2001:]


# In[6]:


dummy_data_simple = dummy_data_for_train[['Birth Year','Gender','Fall']]


# In[7]:


dummy_simple_no = dummy_data_simple[dummy_data_simple['Fall']==0]
dummy_simple_yes = dummy_data_simple[dummy_data_simple['Fall']==1]
len(dummy_simple_yes)


# In[8]:


dummy_simple_no = dummy_simple_no.sample(n=len(dummy_simple_yes))
len(dummy_simple_no)


# In[9]:


dummy_simple_bal = pd.concat([dummy_simple_yes,dummy_simple_no])


# In[10]:


dummy_simple_bal.Fall.mean()


# In[11]:


dummy_simple_bal=dummy_simple_bal.reset_index()


# # GENDER SWAP

# In[12]:


dummy_simple_bal


# In[13]:


dummy_simple_swap = dummy_simple_bal.copy()
dummy_simple_swap['Gender'] = (dummy_simple_swap['Gender']-1)*(-1)


# In[14]:


dummy_simple_swap


# ## Sanity check

# In[15]:


dummy_simple_bal.Gender.value_counts()


# In[16]:


dummy_simple_swap.Gender.value_counts()


# In[17]:


dummy_simple_bal_SWAP = pd.concat([dummy_simple_bal,dummy_simple_swap])
dummy_simple_bal_SWAP = dummy_simple_bal_SWAP.reset_index()
dummy_simple_bal_SWAP


# In[18]:


X1 = pd.DataFrame(dummy_simple_bal_SWAP['Gender'])
X2 = pd.DataFrame(dummy_simple_bal_SWAP['Birth Year'])
X2 = pd.DataFrame(preprocessing.scale(X2),columns=X2.columns)
X = pd.concat([X1,X2],axis=1)
y = dummy_simple_bal_SWAP['Fall']


# In[19]:


y.mean()


# In[20]:


X.shape, y.shape


# ### Sanity check (should be equal for genders)

# In[21]:


dummy_simple_bal_SWAP.groupby('Gender').Fall.mean()


# In[22]:


dummy_simple_bal_SWAP.groupby('Gender')['Birth Year'].mean()


# In[23]:


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
TP_list=[]
TN_list=[]
FP_list=[]
FN_list=[]
F1_list=[]
ACC_list=[]

class_names = ['No fall','Fall']

kf=KFold(n_splits=10, random_state=None, shuffle=True)

for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#### Flip fra den ene til den anden for at skifte mellem SVM, LR og RF ###

    #classifier = svm.SVC(kernel='rbf', C=1, random_state=2).fit(X_train, y_train)
    classifier = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    #classifier = RandomForestClassifier(random_state=1).fit(X_train, y_train)

    np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None)]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test,
                                     y_test, 
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, normalize=normalize)

    # getting TP/TN/FP/FN
    TP=disp.confusion_matrix[1][1]
    TN=disp.confusion_matrix[0][0]
    FP=disp.confusion_matrix[0][1]
    FN=disp.confusion_matrix[1][0]
    F1=2*TP/(2*TP+FP+FN)
    ACC=classifier.score(X_test, y_test) # mark race
    
    # appending to lists
    TP_list.append(TP)
    TN_list.append(TN)
    FP_list.append(FP)
    FN_list.append(FN)
    F1_list.append(F1)
    ACC_list.append(ACC)
    
    


# In[24]:


ACC_list


# In[25]:


F1_list


# ### Test on a non-balanced test set

# In[26]:


ddho_simple = dummy_data_hold_out[['Birth Year','Gender','Fall']]
ddho_simple = ddho_simple.reset_index()


# In[27]:


ddho_simple.shape


# In[28]:


ddho_simple.Fall.value_counts()


# In[29]:


X1 = pd.DataFrame(ddho_simple['Gender'])
X2 = pd.DataFrame(ddho_simple['Birth Year'])
X2 = pd.DataFrame(preprocessing.scale(X2),columns=X2.columns)
X = pd.concat([X1,X2],axis=1)
y = ddho_simple['Fall']


# In[30]:


for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X,
                                     y, 
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, normalize=normalize)


# In[31]:


TP=disp.confusion_matrix[1][1]
TN=disp.confusion_matrix[0][0]
FP=disp.confusion_matrix[0][1]
FN=disp.confusion_matrix[1][0]
F1=2*TP/(2*TP+FP+FN)
ACC=classifier.score(ddho_simple[['Gender','Birth Year',]], ddho_simple['Fall'])
F1


# In[32]:


# rates
FNR = FN/(FN+TP)
FPR = FP/(FP+TN)
TNR = TN/(TN+FP)
TPR = TP/(TP+FN)
print(f"FNR:{FNR}")
print(f"FPR:{FPR}")
print(f"TNR:{TNR}")
print(f"TPR:{TPR}")


# In[33]:


all_metrics = pd.DataFrame([[FNR,FPR,TNR,TPR,ACC,F1,'All']],columns=['FNR','FPR','TNR','TPR','ACC','F1','Group'])


# ## Gender specific TP/TN/FP/FN

# #### Women

# In[34]:


ddho_simple_women = ddho_simple[ddho_simple['Gender']==0]
ddho_simple_women = ddho_simple_women.reset_index()


# In[35]:


X1 = pd.DataFrame(ddho_simple_women['Gender'])
X2 = pd.DataFrame(ddho_simple_women['Birth Year'])
X2 = pd.DataFrame(preprocessing.scale(X2),columns=X2.columns)
X = pd.concat([X1,X2],axis=1)
y = ddho_simple_women['Fall']


# In[36]:


for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X,
                                     y, 
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, normalize=normalize)


# In[37]:


TP=disp.confusion_matrix[1][1]
TN=disp.confusion_matrix[0][0]
FP=disp.confusion_matrix[0][1]
FN=disp.confusion_matrix[1][0]
F1=2*TP/(2*TP+FP+FN)
ACC=classifier.score(ddho_simple_women[['Gender','Birth Year',]], ddho_simple_women['Fall'])
F1


# In[38]:


# rates
FNR = FN/(FN+TP)
FPR = FP/(FP+TN)
TNR = TN/(TN+FP)
TPR = TP/(TP+FN)
print(f"FNR:{FNR}")
print(f"FPR:{FPR}")
print(f"TNR:{TNR}")
print(f"TPR:{TPR}")


# In[39]:


women_metrics = pd.DataFrame([[FNR,FPR,TNR,TPR,ACC,F1,'Women']],columns=['FNR','FPR','TNR','TPR','ACC','F1','Group'])


# #### Men

# In[40]:


ddho_simple_men = ddho_simple[ddho_simple['Gender']==1]
ddho_simple_men = ddho_simple_men.reset_index()


# In[41]:


X1 = pd.DataFrame(ddho_simple_men['Gender'])
X2 = pd.DataFrame(ddho_simple_men['Birth Year'])
X2 = pd.DataFrame(preprocessing.scale(X2),columns=X2.columns)
X = pd.concat([X1,X2],axis=1)
y = ddho_simple_men['Fall']


# In[42]:


for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X,
                                     y, 
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, normalize=normalize)


# In[43]:


TP=disp.confusion_matrix[1][1]
TN=disp.confusion_matrix[0][0]
FP=disp.confusion_matrix[0][1]
FN=disp.confusion_matrix[1][0]
F1=2*TP/(2*TP+FP+FN)
ACC=classifier.score(ddho_simple_men[['Gender','Birth Year',]], ddho_simple_men['Fall'])
F1


# In[44]:


# rates
FNR = FN/(FN+TP)
FPR = FP/(FP+TN)
TNR = TN/(TN+FP)
TPR = TP/(TP+FN)
print(f"FNR:{FNR}")
print(f"FPR:{FPR}")
print(f"TNR:{TNR}")
print(f"TPR:{TPR}")


# In[45]:


men_metrics = pd.DataFrame([[FNR,FPR,TNR,TPR,ACC,F1,'Men']],columns=['FNR','FPR','TNR','TPR','ACC','F1','Group'])


# ## Combining and plot

# In[48]:


metrics = pd.concat([all_metrics,women_metrics,men_metrics])
metrics


# In[49]:


metrics['FNR']+metrics['TPR']


# In[50]:


metrics['FPR']+metrics['TNR']


# # NEXT: put hold out data into cv-loop, as to generate 10 metrics pr. group

# In[47]:


sns.set_theme(style="whitegrid")

g2 = sns.catplot(x="ML",y="value",col="GROUP",hue="measure",data=all_bias_sub, kind="bar",legend_out=True,ci=95)
g2.set_axis_labels("Models", "Bias measure value")
g2.legend.set_title("Measure")


# In[ ]:





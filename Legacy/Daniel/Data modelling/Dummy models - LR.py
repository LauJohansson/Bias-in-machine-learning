#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns


# ## Getting data

# In[2]:


citizen = pd.read_csv('../Data/dummy_data.csv')


# In[3]:


dummy_data = pd.read_csv('../Data/dummy_data.csv')


# In[4]:


dummy_data


# In[15]:


dummy_data=dummy_data.dropna(subset=['BorgerId'])
dummy_data=dummy_data.dropna(subset=['Birth Year'])
dummy_data[dummy_data['Fall']==1]


# ## Logistic regression as data analysis tool

# In[90]:


#pip install statsmodels


# ### Only gender

# In[91]:


# standardizing 
from sklearn import preprocessing


# In[92]:


#X = dummy_data[['Birth Year','Gender']]
X = pd.DataFrame(dummy_data['Gender'])
y = dummy_data['Fall']


# In[93]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[94]:


log_reg2=smf.logit(formula = 'Fall ~ C(Gender)', data = dummy_data).fit()


# In[95]:


print(log_reg2.summary())


# ### Gender and age

# In[107]:


X1 = pd.DataFrame(dummy_data['Gender'])
X2 = pd.DataFrame(dummy_data['Birth Year'])
X2 = pd.DataFrame(preprocessing.scale(X2),columns=['Birth_Year'])
X = pd.concat([X1,X2],axis=1)
X


# In[97]:


log_reg3=smf.logit(formula = 'Fall ~ C(Gender) + Birth_Year', data = X).fit()
print(log_reg3.summary())


# In[98]:


sns.boxplot(x=y,y=X['Birth_Year'])


# Those who fall are born slightly earlier, thus older.

# In[99]:


dummy_data.groupby('Gender').Fall.mean()


# Men (1) fall slightly more often than women

# In[100]:


dummy_data.groupby('Fall').Gender.mean()


# In[101]:


dummy_data.groupby('Gender')['Birth Year'].mean()


# Of those who fall, there are slightly more men than women vis a vis those who do not not fall. But, overall there are more women. 

# In[102]:


dummy_data.Gender.mean()


# In[103]:


dummy_data.Fall.mean()


# In[104]:


import seaborn as sns


# In[105]:


sns.histplot(data=dummy_data,x=dummy_data['Birth Year'],hue=dummy_data['Fall'],stat='density',bins=100,common_norm=False);


# ## Logistic regression as a classifier

# In[109]:


X1 = pd.DataFrame(dummy_data['Gender'])
X2 = pd.DataFrame(dummy_data['Birth Year'])
X2 = pd.DataFrame(preprocessing.scale(X2),columns=['Birth_Year'])
X = pd.concat([X1,X2],axis=1)
y = dummy_data['Fall']


# In[ ]:





# In[110]:


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
ACC_list=[]

class_names = ['No fall','Fall']

kf=KFold(n_splits=10, random_state=None, shuffle=True)

for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#### Flip fra den ene til den anden for at skifte mellem SVM, LR og RF ###

    #classifier = svm.SVC(kernel='rbf', C=1, random_state=2).fit(X_train, y_train['is_recid'])
    classifier = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    #classifier = RandomForestClassifier(random_state=1).fit(X_train, y_train['is_recid'])

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
    ACC=classifier.score(X_test, y_test) # mark race
    
    # appending to lists
    TP_list.append(TP)
    TN_list.append(TN)
    FP_list.append(FP)
    FN_list.append(FN)
    ACC_list.append(ACC)


# ### Pokkers: Only classifies as 'not fallen'

# ## Trying with a balanced dataset - on fall

# #### Slicing away chunk of the data to "final" test on unbalanced reality. Then run only test part of loop below on K-folded test test data set

# In[111]:


from sklearn.utils import shuffle
dummy_data = shuffle(dummy_data)


# In[112]:


dummy_data.shape


# In[113]:


dummy_data_hold_out = dummy_data[0:2000]
dummy_data_for_train = dummy_data[2001:]


# In[114]:


dummy_data_simple = dummy_data_for_train[['Birth Year','Gender','Fall']]


# In[115]:


dummy_simple_no = dummy_data_simple[dummy_data_simple['Fall']==0]
dummy_simple_yes = dummy_data_simple[dummy_data_simple['Fall']==1]
len(dummy_simple_yes)


# In[116]:


dummy_simple_no = dummy_simple_no.sample(n=len(dummy_simple_yes))
len(dummy_simple_no)


# In[117]:


dummy_simple_bal = pd.concat([dummy_simple_yes,dummy_simple_no])


# In[118]:


dummy_simple_bal.Fall.mean()


# In[119]:


dummy_simple_bal=dummy_simple_bal.reset_index()


# Creating classifier

# In[120]:


X1 = pd.DataFrame(dummy_simple_bal['Gender'])
X2 = pd.DataFrame(dummy_simple_bal['Birth Year'])
X2 = pd.DataFrame(preprocessing.scale(X2),columns=X2.columns)
X = pd.concat([X1,X2],axis=1)
y = dummy_simple_bal['Fall']


# In[121]:


y.mean()


# In[122]:


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
    
    


# In[123]:


ACC_list


# In[124]:


F1_list


# ### Test on a non-balanced test set

# In[125]:


ddho_simple = dummy_data_hold_out[['Birth Year','Gender','Fall']]
ddho_simple = ddho_simple.reset_index()


# In[126]:


ddho_simple.shape


# In[127]:


ddho_simple.Fall.value_counts()


# In[128]:


X1 = pd.DataFrame(ddho_simple['Gender'])
X2 = pd.DataFrame(ddho_simple['Birth Year'])
X2 = pd.DataFrame(preprocessing.scale(X2),columns=X2.columns)
X = pd.concat([X1,X2],axis=1)
y = ddho_simple['Fall']


# In[129]:


for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X,
                                     y, 
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, normalize=normalize)


# In[130]:


TP=disp.confusion_matrix[1][1]
TN=disp.confusion_matrix[0][0]
FP=disp.confusion_matrix[0][1]
FN=disp.confusion_matrix[1][0]
F1=2*TP/(2*TP+FP+FN)
ACC=classifier.score(ddho_simple[['Gender','Birth Year',]], ddho_simple['Fall'])
F1


# In[131]:


# rates
FNR = FN/(FN+TP)
FPR = FP/(FP+TN)
TNR = TN/(TN+FP)
TPR = TP/(TP+FN)
print(f"FNR:{FNR}")
print(f"FPR:{FPR}")
print(f"TNR:{TNR}")
print(f"TPR:{TPR}")


# In[132]:


all_metrics = pd.DataFrame([[FNR,FPR,TNR,TPR,ACC,F1,'All']],columns=['FNR','FPR','TNR','TPR','ACC','F1','Group'])


# In[ ]:





# ## Gender specific TP/TN/FP/FN

# #### Women

# In[146]:


ddho_simple_women = ddho_simple[ddho_simple['Gender']==0]
ddho_simple_women = ddho_simple_women.reset_index()


# In[147]:


X1 = pd.DataFrame(ddho_simple_women['Gender'])
X2 = pd.DataFrame(ddho_simple_women['Birth Year'])
X2 = pd.DataFrame(preprocessing.scale(X2),columns=X2.columns)
X = pd.concat([X1,X2],axis=1)
y = ddho_simple_women['Fall']


# In[148]:


for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X,
                                     y, 
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, normalize=normalize)


# In[149]:


TP=disp.confusion_matrix[1][1]
TN=disp.confusion_matrix[0][0]
FP=disp.confusion_matrix[0][1]
FN=disp.confusion_matrix[1][0]
F1=2*TP/(2*TP+FP+FN)
ACC=classifier.score(ddho_simple_women[['Gender','Birth Year',]], ddho_simple_women['Fall'])
F1


# In[150]:


# rates
FNR = FN/(FN+TP)
FPR = FP/(FP+TN)
TNR = TN/(TN+FP)
TPR = TP/(TP+FN)
print(f"FNR:{FNR}")
print(f"FPR:{FPR}")
print(f"TNR:{TNR}")
print(f"TPR:{TPR}")


# In[151]:


women_metrics = pd.DataFrame([[FNR,FPR,TNR,TPR,ACC,F1,'Women']],columns=['FNR','FPR','TNR','TPR','ACC','F1','Group'])


# In[ ]:





# #### Men

# In[152]:


ddho_simple_men = ddho_simple[ddho_simple['Gender']==1]
ddho_simple_men = ddho_simple_men.reset_index()


# In[153]:


X1 = pd.DataFrame(ddho_simple_men['Gender'])
X2 = pd.DataFrame(ddho_simple_men['Birth Year'])
X2 = pd.DataFrame(preprocessing.scale(X2),columns=X2.columns)
X = pd.concat([X1,X2],axis=1)
y = ddho_simple_men['Fall']


# In[154]:


for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X,
                                     y, 
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, normalize=normalize)


# In[155]:


TP=disp.confusion_matrix[1][1]
TN=disp.confusion_matrix[0][0]
FP=disp.confusion_matrix[0][1]
FN=disp.confusion_matrix[1][0]
F1=2*TP/(2*TP+FP+FN)
ACC=classifier.score(ddho_simple_men[['Gender','Birth Year',]], ddho_simple_men['Fall'])
F1


# In[156]:


# rates
FNR = FN/(FN+TP)
FPR = FP/(FP+TN)
TNR = TN/(TN+FP)
TPR = TP/(TP+FN)
print(f"FNR:{FNR}")
print(f"FPR:{FPR}")
print(f"TNR:{TNR}")
print(f"TPR:{TPR}")


# In[157]:


men_metrics = pd.DataFrame([[FNR,FPR,TNR,TPR,ACC,F1,'Men']],columns=['FNR','FPR','TNR','TPR','ACC','F1','Group'])


# ## Combining and plot

# In[158]:


metrics = pd.concat([all_metrics,women_metrics,men_metrics])
metrics


# # NEXT: put hold out data into cv-loop, as to generate 10 metrics pr. group

# ### RESULTS:
# #### Men have higher FPR than women
# #### Women have higher FNR than men

# ### Comment: We see that half of those women who are offered fall-preventive training would not have fallen anyway, while 62 pct. of men who are offered fall-preventive training would not have fallen anyway. Now, these "biased" prediction might not be that problematic. We are simply "wasting" more training on men than on women. If ressources are scarce, than maybe we should do something about it, but if not, who cares! We waste a little training on men, no biggie!

# In[ ]:





# In[ ]:





# In[ ]:





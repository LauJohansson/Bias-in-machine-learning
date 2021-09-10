#!/usr/bin/env python
# coding: utf-8

# In[24]:


#PATH="/content/drive/My Drive/Kandidat speciale/500 - Notebooks/models/all races_CV50/"




#### AIR ### 

AIR=True
file_name="fall_emb.csv"
full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/data_air/"+file_name

#titel_mitigation="testAIR"
titel_mitigation="test23may"
PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/FFNN/models/"+titel_mitigation+"/"


y_col_name="Fall"
X_col_names=['Gender', 'BirthYear', 'Cluster', 'LoanPeriod', 'NumberAts', '1Ats',
       '2Ats', '3Ats', '4Ats', '5Ats', '6Ats', '7Ats', '8Ats', '9Ats', '10Ats',
       '11Ats', '12Ats', '13Ats', '14Ats', '15Ats', '16Ats', '17Ats', '18Ats',
       '19Ats', '20Ats', '21Ats', '22Ats', '23Ats', '24Ats', '25Ats', '26Ats',
       '27Ats', '28Ats', '29Ats', '30Ats', '31Ats', '32Ats', '33Ats', '34Ats',
       '35Ats', '36Ats', '37Ats', '38Ats', '39Ats', '40Ats', '41Ats', '42Ats',
       '43Ats', '44Ats', '45Ats', '46Ats', '47Ats', '48Ats', '49Ats', '50Ats',]



procted_col_name="Gender"


###### COMPASS ####

#AIR=False

#titel_mitigation="testCOMPASS"
#PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/FFNN/models/"+titel_mitigation+"/"

#full_file_path = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'

#y_col_name="is_recid"
#X_col_names=['remember_index','sex','age','race', 'juv_fel_count','juv_misd_count','juv_other_count','priors_count',"c_charge_desc","c_charge_degree"]

#procted_col_name="race"


# In[ ]:





# In[25]:


n_nodes=500


batch_size=40
epochs=100
p_drop=0.4

optim_type="Adam" #SGD
lr=0.001 #0.001 er godt
wd=0.003 #0.001 er godt


# In[26]:


import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.datasets as datasets
from torch.utils.data import DataLoader
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from IPython.display import clear_output


import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#from google.colab import drive
from sklearn.model_selection import KFold

from datetime import datetime

import pytz
import random

import os
from sklearn.model_selection import StratifiedKFold


# In[27]:


from utils import *


# In[28]:


def loss_fn(target,predictions):
    criterion = nn.BCELoss()
    loss_out = criterion(predictions, target)
    return loss_out


# In[29]:



def accuracy(true,pred):
    acc = (true.float().round() == pred.float().round()).float().detach().cpu().numpy()
    return float(100 * acc.sum() / len(acc))

def get_test():
    avg_loss_ts = 0
    avg_acc_ts=0
    model.eval()  # train mode
    for X_batch, Y_batch in data_ts:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)


        # forward
        Y_pred = model(X_batch.float()) 
        loss = loss_fn(Y_batch.float(), Y_pred.squeeze()) 

        # calculate metrics to show the user
        avg_loss_ts += loss / len(data_ts)
        avg_acc_ts+=accuracy(Y_batch,Y_pred.squeeze()) / len(data_ts)
    #toc = time()

    return avg_loss_ts, avg_acc_ts

def get_all_time_low(all_time,new_val):
    if all_time>new_val:
        return new_val
    else:
        return all_time


# In[30]:




class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fully_connected1 = nn.Sequential(
            nn.Linear(n_feat,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )

        self.fully_connectednew = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )

        self.fully_connectednew1 = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )

        self.fully_connectednew2 = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )
        self.fully_connectednew3 = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )


        self.fully_connected2 = nn.Sequential(
            nn.Linear(n_nodes,output_dim),
            #nn.Softmax(dim = 1)
            nn.Sigmoid()

            )

    def forward(self, x):
      #reshaping x so it becomes flat, except for the first dimension (which is the minibatch)
        #x = x.view(x.size(0),-1)
        x = self.fully_connected1(x)
        x = self.fully_connectednew(x)
        x = self.fully_connectednew1(x)
        x = self.fully_connectednew2(x)
        x = self.fully_connectednew3(x)
        x = self.fully_connected2(x)
        return x


# In[36]:




for overall_loop in [0,1,2,3,4,5,6,7,8,9]:


    print("Running overall number "+str(overall_loop))



    custom_seed=4
    k_split=overall_loop
    torch.manual_seed(custom_seed)
    random.seed(custom_seed)
    np.random.seed(custom_seed)


    seedName="model"+str(k_split)#

    PATH=PATH_orig+seedName+"/"
    print(PATH)

    #Make dir to files
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        print("Created new path!: ",PATH)

    df2 = pd.read_csv(full_file_path)

    df2["remember_index"]=list(df2.index)

    #df2.to_csv(PATH+"/COMPASS_dataset.csv")
    
    #Standarize
    

    X=df2[X_col_names+["remember_index"]]
    X_col_names_to_std = [name for name in X_col_names if not name in [procted_col_name]]
    X[X_col_names_to_std] = pd.DataFrame(preprocessing.scale(X[X_col_names_to_std]),columns=X_col_names_to_std)
    y=df2[[y_col_name,"remember_index"]]

    #https://stackoverflow.com/questions/11587782/creating-dummy-variables-in-pandas-for-python

    if AIR==False:
        just_dummies=pd.get_dummies(X[['sex',"race","c_charge_desc","c_charge_degree"]])
        X = pd.concat([X, just_dummies], axis=1) 
        X=X.drop(['sex',"race","c_charge_desc","c_charge_degree"] ,axis=1)

    kf=KFold(n_splits=10, random_state=1, shuffle=True)
    
    
    i=1
    for train_index, val_index in kf.split(X):
        if i==k_split:
            break 
        i=i+1


        

    

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    X_val, X_test, y_val, y_test = train_test_split(
    X_val, y_val, test_size=0.5, random_state=1,shuffle=True)#,stratify=y_val[y_col_name]) #FJERN DENNE ,stratify=y[y_col_name])




    #Save indexes of train, val, test
    X_train_remember_index=X_train['remember_index']
    X_val_remember_index=X_val['remember_index']
    X_test_remember_index=X_test['remember_index']

    X_r=X #these are used last in the notebook to test the data. It hold the true indeksing of the original data
    y_r=y #these are used last in the notebook to test the data. It hold the true indeksing of the original data

    #Remove the helper-column for remembering indexes
    X=X.drop(columns=["remember_index"])
    X_train=X_train.drop(columns=["remember_index"])
    X_val=X_val.drop(columns=["remember_index"])
    X_test=X_test.drop(columns=["remember_index"])
    y=y.drop(columns=["remember_index"])
    y_train=y_train.drop(columns=["remember_index"])
    y_val=y_val.drop(columns=["remember_index"])
    y_test=y_test.drop(columns=["remember_index"])


    #Save as numpy array for the DATALOADER (PyTorch)
    X_train=np.array(X_train[X.columns])
    y_train=np.array(y_train[y_col_name])

    X_val=np.array(X_val[X.columns])
    y_val=np.array(y_val[y_col_name])

    X_test=np.array(X_test[X.columns])
    y_test=np.array(y_test[y_col_name])


    #print("X_train shape: {}".format(X_train.shape))
    #print("y_train shape: {}".format(y_train.shape))

    #print("X_val shape: {}".format(X_val.shape))
    #print("y_val shape: {}".format(y_val.shape))

    #print("X_test shape: {}".format(X_test.shape))
    #print("y_test shape: {}".format(y_test.shape))






    n_feat=X.shape[1]
    output_dim=1 #binary


    data_tr = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=False)
    data_val = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False)
    data_ts = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device="cpu"
    print(device)


    model = Network().to(device)



    opt=optim.Adam(model.parameters(),lr=lr, weight_decay = wd)



    #X_ts, Y_ts = next(iter(data_val))
    #X_ts, Y_ts = X_ts.to(device), Y_ts.to(device)

    epochnumber = []
    all_train_losses = []
    all_val_losses = []
    all_ts_losses = []

    all_train_acc=[]
    all_val_acc=[]
    all_ts_acc=[]

    all_time_low_train_loss=1000
    all_time_low_val_loss=1000

    all_time_low_train_acc=1000
    all_time_low_val_acc=1000


    for epoch in range(epochs):
        #tic = time()
        if (epoch)%20==0:
            print('* Epoch %d/%d' % (epoch+1, epochs))

        epochnumber.append(epoch)

        avg_loss_train = 0
        avg_acc=0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)



            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch.float()) #oprdindeligt havde vi 3 lag (RGB), nu har vi kun 1 (greyscale) -> 
            loss = loss_fn(Y_batch.float(), Y_pred.squeeze())  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss_train += loss / len(data_tr)

            avg_acc+=accuracy(Y_batch,Y_pred.squeeze()) / len(data_tr)


        #toc = time()
        all_time_low_train_loss=get_all_time_low(all_time_low_train_loss,avg_loss_train)
        all_time_low_train_acc=get_all_time_low(all_time_low_train_acc,avg_acc)
          #print(' - train loss: %f' % avg_loss_train)
          #print(' - train acc: {} %'.format(round(avg_acc,2)))

        all_train_losses.append(avg_loss_train)
        all_train_acc.append(avg_acc)



        with torch.no_grad():
            avg_loss_val = 0
            avg_acc_val=0
            model.eval()  # train mode
            for X_batch, Y_batch in data_val:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)


                # forward
                Y_pred = model(X_batch.float()) #oprdindeligt havde vi 3 lag (RGB), nu har vi kun 1 (greyscale) -> 
                loss = loss_fn(Y_batch.float(), Y_pred.squeeze())  # forward-pass

                # calculate metrics to show the user
                avg_loss_val += loss / len(data_val)
                avg_acc_val+=accuracy(Y_batch,Y_pred.squeeze()) / len(data_val)
            #toc = time()
            all_time_low_val_loss=get_all_time_low(all_time_low_val_loss,avg_loss_val)
            all_time_low_val_acc=get_all_time_low(all_time_low_val_acc,avg_acc_val)
            #print(' - val loss: %f' % avg_loss_val)
            #print(' - val acc: {} %'.format(round(avg_acc_val,2)))


            ########Save model####
            
        if  epoch == 0 or avg_loss_val <= min(all_val_losses) :
            torch.save(model.state_dict(), PATH+'_FFNN_model_local.pth')
            print('####Saved model####')

        all_val_losses.append(avg_loss_val)
        all_val_acc.append(avg_acc_val)




      ###PLOT########

    if epoch==epochs-1:
        #Save the last epoch
        torch.save(model.state_dict(), PATH+'_FFNN_model_global.pth')

        #take the best model (with lowest validation loss)
        #model.load_state_dict(torch.load(PATH+seedName+'_FFNN_model_local.pth'))
        model.eval()

        all_ts_losses=[get_test()[0]] * (epoch+1)
        all_ts_acc=[get_test()[1]] * (epoch+1)

        plt.figure(1)
        plt.plot(epochnumber, all_train_losses, 'r', epochnumber, all_val_losses, 'b',epochnumber, all_ts_losses, '--')
        plt.xlabel('Epochs'), plt.ylabel('Loss')
        plt.legend(['Train Loss', 'Val Loss','Test loss'])
        plt.savefig(PATH+'_loss.png')
        plt.show()

        plt.figure(2)
        plt.plot(epochnumber, all_train_acc, 'black', epochnumber, all_val_acc, 'grey',epochnumber, all_ts_acc, '--')
        plt.xlabel('Epochs'), plt.ylabel('Accuracy')
        plt.legend(['Train acc', 'Val acc','Test acc'])
        plt.savefig(PATH+'_acc.png')
        #plt.show()


        #print('####Saved model####')

        metrics=pd.DataFrame({"all_time_low_train_loss":[all_time_low_train_loss.item()],
                              "all_time_low_train_acc":[all_time_low_train_acc],
                          "all_time_low_val_loss":[all_time_low_val_loss.item()],
                              "all_time_val_train_acc":[all_time_low_val_acc],
                          "test_acc":[all_ts_acc[0]],
                          "test_loss":[all_ts_losses[0].item()]
                                            })
        metrics.to_csv( PATH+'_metrics.csv')

      #clear_output(wait=True)








    for local_best in [0,1]:
        #local_best=0

        model1 = Network().to(device)
        if local_best==1:
            model1.load_state_dict(torch.load(PATH+'_FFNN_model_local.pth'))
        else:
            model1.load_state_dict(torch.load(PATH+'_FFNN_model_global.pth'))

        model1.eval()


        X_to_test=X_r[X_r["remember_index"].isin(X_test_remember_index)]
        y_to_test=y_r[X_r["remember_index"].isin(X_test_remember_index)]


        X_to_test=X_to_test.drop(columns="remember_index")
        y_to_test=y_to_test.drop(columns="remember_index")

        df_evaluate = X_to_test
        df_evaluate[y_col_name]=y_to_test
        df_evaluate[procted_col_name]=df2[X_r["remember_index"].isin(X_test_remember_index)][procted_col_name]

        if AIR==False:
            cols= [col for col in list(df_evaluate.columns) if col not in [y_col_name,"sex","age_cat","race","c_charge_desc","c_charge_degree"]]
        else:
            cols= [col for col in list(df_evaluate.columns) if col not in [y_col_name]]



        X_numpy=np.array(df_evaluate[cols])
        X_torch=torch.tensor(X_numpy)
        y_pred = model1(X_torch.float().to(device))
        

        list_of_output=[round(a.item(),0) for a in y_pred.detach().cpu()]
        
        df_evaluate["output"]=list_of_output
        
        
        ##SAVING THE TEST DATA
        if local_best==1:
            df_evaluate.to_csv(PATH+"test_data_localmodel.csv")
        else:
            df_evaluate.to_csv(PATH+"test_data_globalmodel.csv")
        ######################

        df_for_plot=get_df_w_metrics(df_evaluate,procted_col_name,y_col_name,"output")
        

        if local_best==1:
            df_for_plot.to_csv(PATH+"_"+procted_col_name+"_stats_local.csv")
        else:
            df_for_plot.to_csv(PATH+"_"+procted_col_name+"_stats_global.csv")

        df_evaluate_together=df_evaluate
        df_evaluate_together[procted_col_name]="all"
        df_for_plot_all=get_df_w_metrics(df_evaluate_together,procted_col_name,y_col_name,"output")

        if local_best==1:
            df_for_plot_all.to_csv(PATH+"_all_stats_local.csv")
        else:
            df_for_plot_all.to_csv(PATH+"_all_stats_global.csv")
        #%reset -f


# In[ ]:





#  # GLOBAL ALL

# In[46]:


column_names = ["Group", "ML", "Measure","Value"]

df_out = pd.DataFrame(columns = column_names)

for i in [0,1,2,3,4,5,6,7,8,9]:
  
    PATH_loop=PATH_orig+"model"+str(i)+"/_all_stats_global.csv"
  
    data=pd.read_csv(PATH_loop)
    for group in ["all"]:
        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP']:
            value=float(data[data[procted_col_name]==group][measure])

            df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

df_out.to_csv(PATH_orig+"/FFNN_metrics_crossvalidated_global_all.csv")


# In[47]:


global_all_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
global_all_bar.set_title('Global all')
global_all_bar.get_figure().savefig(PATH_orig+"/barplot_global_all.png")


# # LOCAL ALL

# In[48]:


column_names = ["Group", "ML", "Measure","Value"]

df_out = pd.DataFrame(columns = column_names)

for i in [0,1,2,3,4,5,6,7,8,9]:

    PATH_loop=PATH_orig+"model"+str(i)+"/_all_stats_local.csv"
  
    data=pd.read_csv(PATH_loop)
    for group in ["all"]:
        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP']:
            value=float(data[data[procted_col_name]==group][measure])

            df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

df_out.to_csv(PATH_orig+"FFNN_metrics_crossvalidated_local_all.csv")


# In[49]:


local_all_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
local_all_bar.set_title('Global all')
local_all_bar.get_figure().savefig(PATH_orig+"/barplot_local_all.png")


# # Global protected

# In[50]:


column_names = ["Group", "ML", "Measure","Value"]

df_out = pd.DataFrame(columns = column_names)

for i in [0,1,2,3,4,5,6,7,8,9]:

    PATH_loop=PATH_orig+"model"+str(i)+"/_"+procted_col_name+"_stats_global.csv"
  
    data=pd.read_csv(PATH_loop)
    for group in list(data[procted_col_name].unique()):
        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP']:
            value=float(data[data[procted_col_name]==group][measure])

            df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

df_out.to_csv(PATH_orig+"FFNN_metrics_crossvalidated_global_"+procted_col_name+".csv")


# In[51]:


global_proc_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
global_proc_bar.set_title('Global proctected: '+procted_col_name)
global_proc_bar.get_figure().savefig(PATH_orig+"/barplot_global_proc.png")


# # Local protected

# In[52]:


column_names = ["Group", "ML", "Measure","Value"]

df_out = pd.DataFrame(columns = column_names)

for i in [0,1,2,3,4,5,6,7,8,9]:
    PATH_loop=PATH_orig+"model"+str(i)+"/_"+procted_col_name+"_stats_local.csv"
  
    data=pd.read_csv(PATH_loop)
    for group in list(data[procted_col_name].unique()):
        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP']:
            value=float(data[data[procted_col_name]==group][measure])

            df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

df_out.to_csv(PATH_orig+"FFNN_metrics_crossvalidated_local_"+procted_col_name+".csv")


# In[53]:


local_proc_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
local_proc_bar.set_title('Local protected: '+procted_col_name)
local_proc_bar.get_figure().savefig(PATH_orig+"/barplot_local_proc.png")


# In[ ]:





# # Save all test data (and output)

# In[54]:



for file_name in ["localmodel","globalmodel"]:
    test_data_0 = pd.read_csv(PATH_orig+"model0/test_data_"+file_name+".csv")
    test_data_1 = pd.read_csv(PATH_orig+"model1/test_data_"+file_name+".csv")
    test_data_2 = pd.read_csv(PATH_orig+"model2/test_data_"+file_name+".csv")
    test_data_3 = pd.read_csv(PATH_orig+"model3/test_data_"+file_name+".csv")
    test_data_4 = pd.read_csv(PATH_orig+"model4/test_data_"+file_name+".csv")
    test_data_5 = pd.read_csv(PATH_orig+"model5/test_data_"+file_name+".csv")
    test_data_6 = pd.read_csv(PATH_orig+"model6/test_data_"+file_name+".csv")
    test_data_7 = pd.read_csv(PATH_orig+"model7/test_data_"+file_name+".csv")
    test_data_8 = pd.read_csv(PATH_orig+"model8/test_data_"+file_name+".csv")
    test_data_9 = pd.read_csv(PATH_orig+"model9/test_data_"+file_name+".csv")

    df2=    pd.concat([test_data_0,
                        test_data_1,
                        test_data_2,
                        test_data_3,
                        test_data_4,
                        test_data_5,
                        test_data_6,
                        test_data_7,
                        test_data_8,
                        test_data_9
                       ],sort=False,axis=0)

    df2.to_csv(PATH_orig+"all_test_data_"+file_name+".csv")


# In[ ]:





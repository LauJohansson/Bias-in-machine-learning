#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
import numpy as np
import scipy.stats as st


# In[ ]:





# In[2]:




folder_name="original"
#folder_name="Dropping D"
#folder_name="Gender Swap"
#folder_name="DI remove" #DENNE BRUGER VI IKKE
#folder_name="DI remove no gender"
#folder_name="LFR"


# In[3]:


PATH=f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{folder_name}/"


# # Import gender data

# In[4]:


data_gender_FFNN=pd.read_csv(PATH+"FFNN_gender.csv")
data_gender_Xgboost=pd.read_csv(PATH+"Xgboost_gender.csv")
data_gender_Xgboost.loc[data_gender_Xgboost.Model == "Xgboost", "Model"] = "XGBoost"
data_gender_SVM=pd.read_csv(PATH+"SVM_gender.csv")
data_gender_LR=pd.read_csv(PATH+"LR_gender.csv")
data_gender_RF=pd.read_csv(PATH+"RF_gender.csv")


#data_age_FFNN=pd.read_csv(f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{folder_name}/FFNN_age.csv")
#data_age_Xgboost=pd.read_csv(f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{folder_name}/Xgboost_age.csv")


# In[5]:



all_data_gender=pd.concat([
    data_gender_FFNN,
    data_gender_Xgboost,
    data_gender_SVM,
    data_gender_LR,
    data_gender_RF,

    
    
    
    
],sort=False,axis=0
).drop(columns=["Unnamed: 0"])


#all_data_age=pd.concat([
#    data_age_FFNN,
#    data_age_Xgboost
#],sort=False,axis=0
#).drop(columns=["Unnamed: 0"])


# # Import total data

# In[6]:


data_total_FFNN=pd.read_csv(PATH+"FFNN_all.csv")
data_total_Xgboost=pd.read_csv(PATH+"Xgboost_all.csv")
data_total_SVM=pd.read_csv(PATH+"SVM_all.csv")
data_total_LR=pd.read_csv(PATH+"LR_all.csv")
data_total_RF=pd.read_csv(PATH+"RF_all.csv")


data_total_SVM["Model"]="SVM"
data_total_LR["Model"]="LR"
data_total_RF["Model"]="RF"
data_total_FFNN=data_total_FFNN.drop(columns=["Total"])
data_total_Xgboost=data_total_Xgboost.drop(columns=["Total"])
data_total_Xgboost.loc[data_total_Xgboost.Model == "Xgboost", "Model"] = "XGBoost"



all_data_total=pd.concat([
    data_total_FFNN,
    data_total_Xgboost,
    data_total_SVM,
    data_total_LR,
    data_total_RF,
    
],sort=False,axis=0
).drop(columns=["Unnamed: 0"])


# # Melting the data

# In[7]:


all_data_gender=all_data_gender.melt(id_vars=["Gender","Model"],var_name="Metric",value_name="Value")
all_data_total=all_data_total.melt(id_vars=["Model"],var_name="Metric",value_name="Value")


# # Convert daniels values to be between 0-1

# In[8]:


#all_data_gender["Value"]=all_data_gender["Value"].apply(lambda x: x/100 if x>1 else x)#


# # Convert binary gender to string.

# In[9]:


all_data_gender["Gender_string"]=all_data_gender["Gender"].apply(lambda x: "Female" if x==0 else "Male")


# # Add column that combines moden and gender

# In[10]:


all_data_gender["Gender-model"]=all_data_gender["Gender_string"]+all_data_gender["Model"]


# # Gender-Model metrics with CI=0.95

# In[11]:



filter1=all_data_gender["Model"]=="SVM"
sns.pointplot(data=all_data_gender[filter1],x="Gender-model", y="Value",hue="Metric",style="Gender_string",  ci=95,join=False,scale=0.1,capsize=0.0,dodge=True)#,style="Gender_string")

#lt.hlines(y=0.8, xmin=-0.5, xmax=4.5, colors='grey', linestyles='--', lw=1, label='Relation boundary')
#plt.hlines(y=1.25, xmin=-0.5, xmax=4.5, colors='grey', linestyles='--', lw=1)#, label='Relative difference=0.8')
#plt.grid(axis='x')

#10th ticks
#mintick=rounddown(newFrame["Relative Difference (%)"].min())
#maxtick=roundup(newFrame["Relative Difference (%)"].max())
#plt.yticks(np.arange(mintick,maxtick , step=10))

#0.1 ticks
#mintick=round(all_data_gender["Value"].min()-0.1,1)
#maxtick=round(all_data_gender["Value"].max()+0.1,1)
#plt.yticks(np.arange(mintick,maxtick , step=0.1))

#plt.legend(bbox_to_anchor=(1.35,1), loc="upper right")#, borderaxespad=0)
#plt.legend( loc="upper right")
#plt.savefig("Results/Difference_gender_OrigData_relation.png")
plt.title("Original Data - with CI=0.95")
plt.show()


# # All metrics with Model-Gender

# In[12]:


filter1=all_data_gender["Metric"]!="ACC"
sns.barplot(data=all_data_gender[filter1],x="Metric", y="Value",hue="Gender-model"  ,ci=95,dodge=True)#,style="Gender_string")

plt.title("Original Data - with CI=0.95")
plt.show()


# # are they normal?

# In[19]:


sns.histplot(all_data_gender[(all_data_gender["Metric"]=="FPR")&(all_data_gender["Model"]=="FFNN")],x="Value",hue="Model")


# # LR (HOV tallene er alt for store!)

# In[20]:


filter1=all_data_gender["Metric"]!="ACC"
filter2=all_data_gender["Model"]=="LR"
sns.barplot(data=all_data_gender[(filter1)&(filter2)],x="Metric", y="Value",hue="Gender"  ,ci=95,dodge=True)#,style="Gender_string")

plt.title("Original Data - with CI=0.95 - Model: FFNN")
plt.show()


# # FFNN 

# In[21]:


filter1=all_data_gender["Metric"]!="Mean_y_hat"
filter2=all_data_gender["Model"]=="FFNN"
filter3=all_data_gender["Metric"]!="Mean_y_target"
sns.barplot(data=all_data_gender[(filter1)&(filter2)&(filter3)],x="Metric", y="Value",hue="Gender"  ,ci=95,dodge=True)#,style="Gender_string")

plt.title("Original Data - with CI=0.95 - Model: FFNN")
plt.show()


# # Xgboost alone 

# In[13]:


palette_custom ={"Female": "C0", "Male": "C1"}
gender_order=["Female","Male"]


# In[34]:


plt.figure(figsize=(8,5))
filter1=all_data_gender["Metric"]!="ACC"
filter2=all_data_gender["Model"]=="XGBoost"
filter3=all_data_gender["Metric"]!="Mean_y_target"
filter4=all_data_gender["Metric"]!="Mean_y_hat"
filter5=all_data_gender["Metric"]!="p(fall)"
filter6=all_data_gender["Metric"]!="Mean_y_hat_prob"


plt.grid(axis="x")
sns.barplot(data=all_data_gender[(filter1)&(filter2)&(filter3)&(filter4)&(filter5)&(filter6)],x="Metric", y="Value",hue="Gender_string"  ,ci=95,dodge=True,errwidth=1,capsize=0.25)#,palette=palette_custom)
plt.title("XGBoost",size=15)
plt.ylim([0, 1])
plt.ylabel("Rate",size=20)
plt.xlabel("",size=20)
plt.legend(title="Gender")


#plt.title("Original Data - with CI=0.95 - Model: FFNN")
#plt.savefig("Results/OrigData_gender_metrics_xgboost.png")
#plt.savefig(f"/restricted/s164512/G2020-57-Aalborg-bias/Plots/{folder_name}/XGboost_metric_plot", bbox_inches = 'tight')

plt.show()


# In[24]:


all_data_gender


# In[ ]:





# In[ ]:





# # Each model respectively

# In[25]:



fig, ax = plt.subplots(2,2,figsize=(8, 8),sharey=True,sharex=True)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
            hspace = 0.1, wspace = 0.05)
ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN"]

#for i,v in enumerate(all_data_gender["Model"].unique()):
for i,v in enumerate(list_of_models):
    
    #filter1=frame["Metric"]==v
    #if v != "Xgboost":
    filter1=all_data_gender["Model"]==v
    filter2=all_data_gender["Metric"]!="ACC"
    #ax[i].title.set_text(v,size=15)
    ax[i].set_title(v,size=15)
    ax[i].set_ylim([0, 1])
    ax[i].grid(axis='x')
    sns.barplot(data=all_data_gender[(filter1)&(filter2)],x="Metric",y="Value",hue="Gender_string",ax=ax[i])#,palette=palette_custom)
    ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set_ylabel("Rate",fontsize=20)
    if i==1:
        ax[i].set(xlabel='',ylabel='')
    if i==2:
        ax[i].set_ylabel("Rate",fontsize=20)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)
    
plt.show()


# In[26]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(15,8))

gs = plt.GridSpec(2, 8, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 2:4]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 4:6]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[1, 5:7]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","XGBoost"]

#for i,v in enumerate(all_data_gender["Model"].unique()):
for i,v in enumerate(list_of_models):
    
    #filter1=frame["Metric"]==v
    #if v != "Xgboost":
    filter1=all_data_gender["Model"]==v
    filter2=all_data_gender["Metric"]!="ACC"
    #ax[i].title.set_text(v,size=15)
    ax[i].set_title(v,size=15)
    #plt.subplot(grid[i,0])
    ax[i].set_ylim([0, 1])
    ax[i].grid(axis='x')
    sns.barplot(data=all_data_gender[(filter1)&(filter2)],x="Metric",y="Value",hue="Gender_string",ax=ax[i])#,palette=palette_custom)
    ax[i].legend(title="Gender")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set_ylabel("Rate",fontsize=20)
    if i==1:
        ax[i].set(xlabel='',ylabel='')
    if i==2:
        ax[i].set_ylabel("Rate",fontsize=20)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)
    if i==4:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)

#fig.delaxes(gs[0, 1])
plt.show()


# In[27]:


#all_data_gender["Metric"].unique()


# In[28]:


all_data_gender.loc[all_data_gender.Metric == "Mean_y_hat", "Metric"] = "p(fall)"


# In[29]:


#grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

fig = plt.figure(constrained_layout=True,figsize=(15,15))

gs = plt.GridSpec(3, 6, figure=fig)
gs.update(wspace=0.5)

ax=[]

ax.append( fig.add_subplot(gs[0, 1:3]))
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax.append( fig.add_subplot(gs[0, 3:5]))
ax.append(fig.add_subplot(gs[1, 1:3]))
ax.append(fig.add_subplot(gs[1, 3:5]))
ax.append( fig.add_subplot(gs[2, 2:4]))


#fig, ax = plt.subplots(2,3,figsize=(8, 8),sharey=True,sharex=True)
#plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
#            hspace = 0.1, wspace = 0.05)
#ax = ax.ravel()

list_of_models=["SVM","LR","RF","FFNN","XGBoost"]


#for i,v in enumerate(all_data_gender["Model"].unique()):
for i,v in enumerate(list_of_models):
    
    #filter1=frame["Metric"]==v
    #if v != "Xgboost":
    filter1=all_data_gender["Model"]==v
    filter2=all_data_gender["Metric"]!="ACC"
    filter3=all_data_gender["Metric"]!="Mean_y_target"
    filter4=all_data_gender["Metric"]!="p(fall)"
    filter5=all_data_gender["Metric"]!="Mean_y_hat_prob"
    #ax[i].title.set_text(v,size=15)
    ax[i].set_title(v,size=15)
    #plt.subplot(grid[i,0])
    ax[i].set_ylim([0, 1])
    ax[i].grid(axis='x')
    #sns.barplot(data=all_data_gender[(filter1)&(filter2)],x="Metric",y="Value",hue="Gender_string",ax=ax[i],errwidth=1,capsize=0.25)#,palette=palette_custom)
    #sns.barplot(data=all_data_gender[(filter1)],x="Metric",y="Value",hue="Gender_string",ax=ax[i],errwidth=1,capsize=0.25)#,palette=palette_custom)
    sns.barplot(data=all_data_gender[(filter1)&(filter2)&(filter3)&(filter4)&(filter5)],x="Metric",y="Value",hue="Gender_string",ax=ax[i],errwidth=1,capsize=0.25,palette=palette_custom,hue_order=gender_order)
    
    ax[i].legend(title="Gender")
    ax[i].legend( loc="upper right")
    if i==0:
        ax[i].set(xlabel='')
        ax[i].set_ylabel("Rate",fontsize=20)
    if i==1:
        ax[i].set(xlabel='',ylabel='')
        ax[i].tick_params( labelleft=False)
    if i==2:
        ax[i].set_ylabel("Rate",fontsize=20)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)
        
    if i==3:
        ax[i].set(ylabel='')
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)
        ax[i].tick_params( labelleft=False)
    if i==4:
        ax[i].set_ylabel('Rate',fontsize=20)
        #ax[i].set_xlabel("Metric",fontsize=20)
        ax[i].set_xlabel("",fontsize=20)

#fig.delaxes(gs[0, 1])<
plt.savefig(f"/restricted/s164512/G2020-57-Aalborg-bias/Plots/{folder_name}/{folder_name}_gender_metrics_allmodels", bbox_inches = 'tight')
plt.show()


# In[ ]:





# In[36]:


fig, ax = plt.subplots(5,1,figsize=(4,20),sharey=True)
fig.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
            hspace = 0.2, wspace = 0.05)
ax = ax.ravel()

#colors=["Blue","Green","Orange","Red","Black"]
#colors=["Black"]*5

#for i,v in enumerate(["TPR","FPR","TNR","FNR","ACC"]):

list_of_models=["SVM","LR","RF","FFNN","XGBoost"]
#list_of_models=["SVM","LR","RF","Xgboost","Xgboost"]

#for i,v in enumerate(all_data_gender["Model"].unique()):
for i,v in enumerate(list_of_models):
    
    #filter1=frame["Metric"]==v
    #if v != "Xgboost":
    filter1=all_data_gender["Model"]==v
    filter2=all_data_gender["Metric"]!="ACC"
    filter3=all_data_gender["Metric"]!="Mean_y_target"
    filter4=all_data_gender["Metric"]!="p(fall)"
    filter5=all_data_gender["Metric"]!="Mean_y_hat_prob"
    
    
    #ax[i].title.set_text(v,size=15)
    
    
    
    
    if i==0:
        if folder_name=="original":
            ax[i].set_title("Original Data \n"+v,size=15)
        elif folder_name=="Dropping D":
            ax[i].set_title("Dropping Gender \n"+v,size=15)
        elif folder_name=="Gender Swap":
            ax[i].set_title("Gender Swap \n"+v,size=15)
        elif folder_name=="DI remove" or folder_name=="DI remove no gender":
            ax[i].set_title("DI removal \n"+v,size=15)
        elif folder_name=="LFR":
            ax[i].set_title("LFR \n"+v,size=15)
        else:
            ax[i].set_title("TITLE MISSING \n"+v,size=15)
    else:
        ax[i].set_title(v,size=15)
    
    
    ax[i].set_ylim([0, 1])
    
    ax[i].grid(axis='x')
    sns.barplot(data=all_data_gender[(filter1)&(filter2)&(filter3)&(filter4)&(filter5)],x="Metric",y="Value",hue="Gender_string",ax=ax[i],errwidth=1,capsize=0.25,palette=palette_custom,hue_order=gender_order)
    ax[i].set_ylabel("Rate",fontsize=20)
    if i!=4:
        ax[i].set_xlabel('',fontsize=20)
    else:
        ax[i].set_xlabel('Metric',fontsize=20)
    ax[i].legend(title="Gender")
    
   
plt.savefig(f"/restricted/s164512/G2020-57-Aalborg-bias/Plots/{folder_name}/Compare_plot_{folder_name}.png", bbox_inches = 'tight')
plt.show()


# # Calcuate mean difference (relation) between male/female in each model

# In[21]:


frame=all_data_gender
newFrame=pd.DataFrame([],columns=["Model","Metric","Abs Difference","Relation","Relative difference (%)"])

for i in list(frame["Model"].unique()):
    for j in list(frame["Metric"].unique()) :
        if j not in ["Mean_y_target","Mean_y_hat_prob"]:
            female_val=frame[(frame["Model"]==i)&(frame["Metric"]==j)&(frame["Gender_string"]=="Female")]["Value"].mean()
            male_val=frame[(frame["Model"]==i)&(frame["Metric"]==j)&(frame["Gender_string"]=="Male")]["Value"].mean()
            absdiff=abs(female_val-male_val)

            mini=min(female_val,male_val)
            maxi=max(female_val,male_val)
            reldiff=((maxi-mini)/mini)*100

            relation=female_val/male_val

            newFrame=newFrame.append({"Model":i,"Metric":j,"Abs Difference":absdiff,"Relation":relation,"Relative difference (%)":reldiff},ignore_index=True)


# In[22]:


#Rename for DI
newFrame.loc[newFrame.Metric == "p(fall)", "Metric"] = "DI"


# In[39]:


#sns.scatterplot(data=newFrame,x="Metric",y="Relation",hue="Model",jitter=0.02)#,ax=ax[i],style="Gender",legend=False,palette=[colors[i]])
#sns.stripplot(data=newFrame,x="Metric",y="Relation",hue="Model",jitter=0.1,size=20,alpha=0.8)# Big size


plt.figure(figsize=(10,10)) #old
#plt.figure(figsize=(15,8)) 


filter1=newFrame["Metric"]!="ACC"
filter2=newFrame["Metric"]!="DI"
filter3=newFrame["Metric"]!="p(fall)"
filter4=newFrame["Metric"]!="Mean_y_hat"

sns.stripplot(data=newFrame[(filter1)&(filter2)&(filter3)&(filter4)],x="Metric",y="Relation",hue="Model",jitter=0.,size=10,alpha=0.8,linewidth=1,
             palette={"FFNN":"C0","XGBoost":"C1","SVM":"C2","LR":"C3","RF":"C4"}
             
             
             )


plt.hlines(y=1, xmin=-0.5, xmax=3.5, colors='red', linestyles='--', lw=1, label='Equal relation')
plt.hlines(y=0.8, xmin=-0.5, xmax=3.5, colors='grey', linestyles='--', lw=1, label='Relation boundary')
plt.hlines(y=1.25, xmin=-0.5, xmax=3.5, colors='grey', linestyles='--', lw=1)#, label='Relative difference=0.8')
#plt.grid(axis='x')

#10th ticks
#mintick=rounddown(newFrame["Relative Difference (%)"].min())
#maxtick=roundup(newFrame["Relative Difference (%)"].max())
#plt.yticks(np.arange(mintick,maxtick , step=10))

#0.1 ticks
#mintick=round(newFrame["Relation"].min()-0.1,1)
#maxtick=round(newFrame["Relation"].max()+0.1,1)
#plt.yticks(np.arange(mintick,maxtick , step=0.1))

#plt.legend(bbox_to_anchor=(1.35,1), loc="upper right")#, borderaxespad=0)
#plt.legend(bbox_to_anchor=(-0.1,1), loc="upper right")#, borderaxespad=0)
plt.legend( loc="upper left")
plt.legend(loc=2, prop={'size': 15})
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel( "Metric",fontsize=20,labelpad=10)
plt.ylabel( "Relation",fontsize=20,labelpad=10)
plt.yticks(np.arange(0.6,1.7 , step=0.2))
plt.ylim([0.5,1.7])



if folder_name=="original":
    plt.title("Original Data",fontsize=20)
elif folder_name=="Dropping D":
    plt.title("Dropping Gender",fontsize=20)
elif folder_name=="Gender Swap":
    plt.title("Gender Swap",size=20)
elif folder_name=="DI remove" or folder_name=="DI remove no gender":
    plt.title("DI removal",size=20)
elif folder_name=="LFR":
    plt.title("LFR",size=20)
else:
    plt.title("TITLE MISSING")

#plt.savefig(f"/restricted/s164512/G2020-57-Aalborg-bias/Plots/{folder_name}/Difference_gender_{folder_name}_relation.png", bbox_inches = 'tight')


    

plt.show()


# # Plot accuracy

# In[ ]:





# In[35]:


filter2=all_data_total["Metric"]=="ACC"
plt.figure(figsize=(10,10))
#ax[i].title.set_text(v,size=15)



if folder_name=="original":
    plt.title("Accuracy - Original Data",size=20)
elif folder_name=="Dropping D":
    plt.title("Accuracy - Dropping Gender",fontsize=20)
elif folder_name=="Gender Swap":
    plt.title("Accuracy - Gender Swap",size=20)
elif folder_name=="DI remove" or folder_name=="DI remove no gender":
    plt.title("Accuracy - DI removal",size=20)
elif folder_name=="LFR":
    plt.title("Accuracy - LFR",size=20)
else:
    plt.title("TITLE MISSING")


#plt.subplot(grid[i,0])
plt.ylim([0, 1])
plt.grid(axis='x')
acc_palette = {"FFNN":"Orange",
               "XGBoost":"Orange",
               "SVM":"Orange",
               "LR":"Orange",
               "RF":"Orange",
              }
acc_order=["SVM","LR","RF","FFNN","XGBoost"]
if folder_name!="original": 
    sns.barplot(data=all_data_total[(filter2)],x="Model",y="Value",errwidth=1,capsize=0.5,color="slateblue",order=acc_order)
else:
    sns.barplot(data=all_data_total[(filter2)],x="Model",y="Value",errwidth=1,capsize=0.5,color="peru",order=acc_order)
    
#plt.legend(title="Gender")
#plt.legend( loc="upper right")

plt.xlabel('')
plt.ylabel("Rate",fontsize=15)
plt.xticks(fontsize=15)#, rotation=90)
plt.savefig(f"/restricted/s164512/G2020-57-Aalborg-bias/Plots/{folder_name}/Acc_models_{folder_name}.png", bbox_inches = 'tight')


plt.show()


# In[29]:


#SVM_testdata.groupby("Gender")["y_hat_probs"].mean()


str1="& "
str2="& "
for mtype in ["SVM","LR","RF","XGBoost","FFNN"]:

        data=all_data_total[(all_data_total["Model"]==mtype)&(all_data_total["Metric"]=="ACC")]["Value"]
        m=np.mean(data)
        (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 

        print(f"Mean for {mtype} ",round(m*100,2),f"({round(slow*100,2)}-{round(shigh*100,2)})")
        str1=str1+str(round(m*100,2))+"& "
        str2=str2+f"({round(slow*100,2)}-{round(shigh*100,2)})"+"& "
#print()
#print(str1)
#print(str2)


#  # TEST CI

# In[30]:


filter1=all_data_gender["Model"]=="SVM"
filter2=all_data_gender["Metric"]=="TPR"
filter3=all_data_gender["Gender_string"]=="Female"


data=all_data_gender[(filter1)&(filter2)&(filter3)]["Value"]
m=np.mean(data)
(slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))


# In[31]:


data.shape


# In[32]:


m


# In[33]:


(slow,shigh)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Old scripts

# In[34]:


import math

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10
def rounddown(x):
    return int(math.floor(x / 10.0)) * 10


# In[35]:


#OHE
d=[["SVM","Female","TPR",64.68],
   ["SVM","Female","FPR",26.26],
   ["SVM","Female","TNR", 73.74],
   ["SVM","Female","FNR", 35.32],
   ["SVM","Female","ACC" , 71.80],

   
   ["SVM","Male","TPR",81.19],
   ["SVM","Male","FPR",  38.75],
   ["SVM","Male","TNR"  , 61.25],
   ["SVM","Male","FNR", 18.81],
   ["SVM","Male","ACC" ,66.15],
   
   
   ["LR","Female","TPR", 54.90],
   ["LR","Female","FPR",  33.83],
   ["LR","Female","TNR", 66.17],
   ["LR","Female","FNR" , 45.1],
   ["LR","Female","ACC",63.76],
   
   
   ["LR","Male","TPR",  69.48],
   ["LR","Male","FPR",    44.39],
   ["LR","Male", "TNR",  55.61],
   ["LR","Male", "FNR", 30.52],
   ["LR","Male", "ACC",59.03],
   
   
   ["RF","Female","TPR",59.16],
   ["RF","Female","FPR",  4.17],
   ["RF","Female","TNR",   95.83],
   ["RF","Female","FNR", 40.84],
   ["RF","Female","ACC",88.14],
   
   ["RF","Male","TPR", 67.03],
   ["RF","Male","FPR",5.16],
   ["RF","Male","TNR",   94.84],
   ["RF","Male","FNR", 32.97],
   ["RF","Male","ACC",87.84],
   
   
   ["FFNN","Female","TPR", 42.25],
   ["FFNN","Female","FPR",  8.30],
   ["FFNN","Female","TNR", 91.70],
   ["FFNN","Female","FNR", 57.75],
   ["FFNN","Female","ACC",81.25],
   
   
   
   ["FFNN","Male","TPR", 40.78],
   ["FFNN","Male","FPR",11.37],
   ["FFNN","Male","TNR", 88.63],
   ["FFNN","Male","FNR", 59.22],
   ["FFNN","Male","ACC",76.37],
   
   
   
   ["Xgboost","Female","TPR", 58.30],
   ["Xgboost","Female","FPR",  16.69],
   ["Xgboost","Female","TNR",83.31],
   ["Xgboost","Female","FNR", 41.70],
   ["Xgboost","Female","ACC",77.96],
   
   
   
   ["Xgboost","Male","TPR", 69.39],
   ["Xgboost","Male","FPR",  18.94],
   ["Xgboost","Male","TNR", 81.06],
   ["Xgboost","Male","FNR", 30.61],
   ["Xgboost","Male","ACC",78.28],
   
   
  ]


# In[36]:


#EMBEDDING
'''
d=[["SVM","Female","TPR",38.20],
   ["SVM","Female","FPR",3.47],
   ["SVM","Female","TNR", 96.53],
   ["SVM","Female","FNR", 61.80],
   ["SVM","Female","ACC" ,84.38],

   
   ["SVM","Male","TPR",40.63],
   ["SVM","Male","FPR",  5.71],
   ["SVM","Male","TNR"  , 94.29],
   ["SVM","Male","FNR", 59.37],
   ["SVM","Male","ACC" ,80.90],
   
   
   ["LR","Female","TPR",45.98 ],
   ["LR","Female","FPR", 4.39 ],
   ["LR","Female","TNR", 95.61],
   ["LR","Female","FNR" , 54.02],
   ["LR","Female","ACC",85.28],
   
   
   ["LR","Male","TPR", 45.15 ],
   ["LR","Male","FPR",  7.56  ],
   ["LR","Male", "TNR", 92.44 ,],
   ["LR","Male", "FNR", 54.85],
   ["LR","Male", "ACC",80.65],
   
   
   ["RF","Female","TPR",63.98],
   ["RF","Female","FPR",  4.73],
   ["RF","Female","TNR",  95.27 ],
   ["RF","Female","FNR", 36.02],
   ["RF","Female","ACC",88.72],
   
   ["RF","Male","TPR",65.92 ],
   ["RF","Male","FPR", 5.92],
   ["RF","Male","TNR",  94.08 ],
   ["RF","Male","FNR", 34.08],
   ["RF","Male","ACC",87.05],
   
   
   ["FFNN","Female","TPR",56.21 ],
   ["FFNN","Female","FPR",6.09  ],
   ["FFNN","Female","TNR", 93.91],
   ["FFNN","Female","FNR",43.79 ],
   ["FFNN","Female","ACC",85.79],
   
   
   
   ["FFNN","Male","TPR",40.48 ],
   ["FFNN","Male","FPR", 8.96 ],
   ["FFNN","Male","TNR",91.04 ],
   ["FFNN","Male","FNR", 59.52],
   ["FFNN","Male","ACC",79.34],
   
   
   
   ["Xgboost","Female","TPR",61.80 ],
   ["Xgboost","Female","FPR",  12.39],
   ["Xgboost","Female","TNR",87.61],
   ["Xgboost","Female","FNR", 38.20],
   ["Xgboost","Female","ACC",82.17],
   
   
   
   ["Xgboost","Male","TPR",65.10 ],
   ["Xgboost","Male","FPR", 16.70 ],
   ["Xgboost","Male","TNR", 83.30],
   ["Xgboost","Male","FNR", 34.90],
   ["Xgboost","Male","ACC",78.85],
   
   
  ]

'''


# In[37]:


frame=pd.DataFrame(d,columns=["Model","Gender","Metric","Value"])


# In[38]:


frame["ModelGender"]=frame["Model"]+"-"+frame["Gender"]


# In[39]:


frame.head(1)


# In[40]:


fig, ax = plt.subplots()
sns.scatterplot(data=frame,x="Model",y="Value",hue="Metric",style="Gender")
plt.grid(axis='x')


#get the handles/labels of the plot
handles, labels = ax.get_legend_handles_labels()
# handles is a list, so append manual patch
#handles.append(red_patch) 

    # plot the legend
plt.legend(handles=handles, loc='best')
plt.show()


# In[41]:


#fig, ax = plt.subplots(3,2,figsize=(15, 15),sharey=True)
#ax = ax.ravel()

#colors=["Blue","Green","Orange","Red","Black"]
#colors=["Black"]*5

#for i,v in enumerate(["TPR","FPR","TNR","FNR","ACC"]):
    
    
#    filter1=frame["Metric"]==v
#    ax[i].title.set_text(v)
#    ax[i].set_ylim([0, 100])
#    ax[i].grid(axis='x')
#    sns.scatterplot(data=frame[filter1],x="Model",y="Value",hue="Metric",ax=ax[i],style="Gender",legend=False,palette=[colors[i]])

#plt.show()


# In[42]:


newFrame=pd.DataFrame([],columns=["Model","Metric","Abs Difference","Relation","Relative difference (%)"])

for i in list(frame["Model"].unique()):
    for j in list(frame["Metric"].unique()):
        
        female_val=frame[(frame["Model"]==i)&(frame["Metric"]==j)&(frame["Gender"]=="Female")]["Value"].item()
        male_val=frame[(frame["Model"]==i)&(frame["Metric"]==j)&(frame["Gender"]=="Male")]["Value"].item()
        absdiff=abs(female_val-male_val)
        
        mini=min(female_val,male_val)
        maxi=max(female_val,male_val)
        reldiff=((maxi-mini)/mini)*100
        
        relation=female_val/male_val
        
        
        
        
        newFrame=newFrame.append({"Model":i,"Metric":j,"Abs Difference":absdiff,"Relation":relation,"Relative difference (%)":reldiff},ignore_index=True)
    


# In[43]:


sns.scatterplot(data=newFrame,x="Metric",y="Abs Difference",hue="Model")#,ax=ax[i],style="Gender",legend=False,palette=[colors[i]])
plt.grid(axis='x')
plt.savefig("Results/Difference_gender_OrigData_absdiff.png")
plt.show()


# In[44]:


sns.scatterplot(data=newFrame,x="Metric",y="Relative difference (%)",hue="Model")#,ax=ax[i],style="Gender",legend=False,palette=[colors[i]])
#plt.hlines(y=0.8, xmin=-0.5, xmax=4.5, colors='red', linestyles='--', lw=2, label='Relative difference=0.8')
plt.grid(axis='x')

#10th ticks
mintick=rounddown(newFrame["Relative difference (%)"].min())
maxtick=roundup(newFrame["Relative difference (%)"].max())+0.1
plt.yticks(np.arange(mintick,maxtick , step=10))

#0.1 ticks
#mintick=round(newFrame["Relative difference (%)"].min()-0.1,1)
#maxtick=round(newFrame["Relative difference (%)"].max()+0.1,1)
#plt.yticks(np.arange(mintick,maxtick , step=0.1))

#plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.legend( loc="upper right")
plt.savefig("Results/Difference_gender_OrigData_relativediff.png")
plt.show()


# In[45]:


sns.scatterplot(data=newFrame,x="Metric",y="Relation",hue="Model")#,ax=ax[i],style="Gender",legend=False,palette=[colors[i]])
#plt.hlines(y=1, xmin=-0.5, xmax=4.5, colors='red', linestyles='--', lw=1, label='E')
plt.hlines(y=0.8, xmin=-0.5, xmax=4.5, colors='grey', linestyles='--', lw=1, label='Relation boundary')
plt.hlines(y=1.25, xmin=-0.5, xmax=4.5, colors='grey', linestyles='--', lw=1)#, label='Relative difference=0.8')
plt.grid(axis='x')

#10th ticks
#mintick=rounddown(newFrame["Relative Difference (%)"].min())
#maxtick=roundup(newFrame["Relative Difference (%)"].max())
#plt.yticks(np.arange(mintick,maxtick , step=10))

#0.1 ticks
mintick=round(newFrame["Relation"].min()-0.1,1)
maxtick=round(newFrame["Relation"].max()+0.1,1)
plt.yticks(np.arange(mintick,maxtick , step=0.1))

plt.legend(bbox_to_anchor=(1.35,1), loc="upper right")#, borderaxespad=0)
#plt.legend( loc="upper right")
plt.savefig("Results/Difference_gender_OrigData_relation.png")
plt.show()


# # Trying to make plot with forskydelser

# In[46]:


sns.stripplot(data=newFrame,x="Metric",y="Relative difference (%)",hue="Model",jitter=0.1,dodge=True)#,ax=ax[i],style="Gender",legend=False,palette=[colors[i]])
#plt.hlines(y=0.8, xmin=-0.5, xmax=4.5, colors='red', linestyles='--', lw=2, label='Relative difference=0.8')
plt.grid(axis='x')

#10th ticks
mintick=rounddown(newFrame["Relative difference (%)"].min())
maxtick=roundup(newFrame["Relative difference (%)"].max())+0.1
plt.yticks(np.arange(mintick,maxtick , step=10))

#0.1 ticks
#mintick=round(newFrame["Relative difference (%)"].min()-0.1,1)
#maxtick=round(newFrame["Relative difference (%)"].max()+0.1,1)
#plt.yticks(np.arange(mintick,maxtick , step=0.1))

#plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
#plt.legend( loc="upper right")
#plt.savefig("Results/Difference_gender_OrigData_relativediff.png")
#plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np


# In[22]:


from utils_Copy import *


# In[23]:


#titel_mitigation="nostratify"
#titel_mitigation="DroppingD"
#titel_mitigation="Gender Swap"
#titel_mitigation="DI remove"
#titel_mitigation="LFR"
titel_mitigation="DI remove no gender"



#save_to_folder="original"
#save_to_folder="Dropping D"
#save_to_folder="Gender Swap"
#save_to_folder="DI remove"
#save_to_folder="LFR"
save_to_folder="DI remove no gender"


PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/Xgboost_ohe/models/"+titel_mitigation+"/"

file_name="all_test_data.csv"


full_file_path=PATH_orig+file_name


print("PATH_orig:",PATH_orig)
print("PATH to file:",full_file_path)


# In[ ]:





# In[24]:


y_col_name="Fall"



X_col_names=['Gender',
'BirthYear',
'LoanPeriod',
'NumberAts',
'Ats_Polstring',
'Ats_Mobilitystokke',
'Ats_Belysning',
'Ats_Underlag',
'Ats_ToiletforhøjereStativ',
'Ats_Signalgivere',
'Ats_EldrevneKørestole',
'Ats_Forstørrelsesglas',
'Ats_Nødalarmsystemer',
'Ats_MobilePersonløftere',
'Ats_TrappelifteMedPlatforme',
'Ats_Badekarsbrætter',
'Ats_Albuestokke',
'Ats_MaterialerOgRedskaberTilAfmærkning',
'Ats_Ryglæn',
#'Ats_0',
'Ats_GanghjælpemidlerStøtteTilbehør',
'Ats_Støttebøjler',
'Ats_Lejringspuder',
'Ats_Strømpepåtagere',
'Ats_Dørtrin',
'Ats_Spil',
'Ats_BordePåStole',
'Ats_Drejeskiver',
'Ats_Toiletstole',
'Ats_LøftereStationære',
'Ats_Madmålingshjælpemidler',
'Ats_Fodbeskyttelse',
'Ats_Ståløftere',
'Ats_Stole',
'Ats_Sengeborde',
'Ats_Toiletter',
'Ats_ToiletforhøjereFaste',
'Ats_Påklædning',
'Ats_Brusere',
'Ats_VævsskadeLiggende',
'Ats_Døråbnere',
'Ats_ServeringAfMad',
'Ats_TrappelifteMedSæder',
'Ats_SæderTilMotorkøretøjer',
'Ats_KørestoleManuelleHjælper',
'Ats_Gangbukke',
'Ats_Rollatorer',
'Ats_TryksårsforebyggendeSidde',
'Ats_Fastnettelefoner',
'Ats_Bækkener',
'Ats_Vendehjælpemidler',
'Ats_Sanseintegration',
'Ats_Kørestolsbeskyttere',
'Ats_Arbejdsstole',
'Ats_Løftesejl',
'Ats_KørestoleForbrændingsmotor',
'Ats_Løftestropper',
'Ats_Stiger',
'Ats_TransportTrapper',
'Ats_DrivaggregaterKørestole',
'Ats_Emballageåbnere',
'Ats_ToiletforhøjereLøse',
'Ats_Hårvask',
'Ats_PersonløftereStationære',
'Ats_Madrasser',
'Ats_Vinduesåbnere',
'Ats_Læsestativer',
'Ats_KørestoleManuelleDrivringe',
'Ats_Sædepuder',
'Ats_UdstyrCykler',
'Ats_Karkludsvridere',
'Ats_Vaskeklude',
'Ats_Sengeudstyr',
'Ats_Madlavningshjælpemidler',
'Ats_Skohorn',
'Ats_GribetængerManuelle',
'Ats_Hvilestole',
'Ats_EldrevneKørestoleStyring',
'Ats_BærehjælpemidlerTilKørestole',
'Ats_LøftegalgerSeng',
'Ats_Høreforstærkere',
'Ats_Kalendere',
'Ats_Stokke',
'Ats_Løftegalger',
'Ats_Ure',
'Ats_StøttegrebFlytbare',
'Ats_Forflytningsplatforme',
'Ats_RamperFaste',
'Ats_Rygehjælpemidler',
'Ats_Personvægte',
'Ats_Manøvreringshjælpemidler',
'Ats_Overtøj',
'Ats_Lydoptagelse',
'Ats_Gangborde',
'Ats_Ståstøttestole',
'Ats_RamperMobile',
'Ats_Bærehjælpemidler',
'Ats_Badekarssæder',
'Ats_Siddemodulsystemer',
'Ats_Videosystemer',
'Ats_Siddepuder',
'Ats_Sengeheste',
'Ats_Stolerygge',
'Ats_Rulleborde',
'Ats_Sengeforlængere',
'Ats_Madningsudstyr',
'Ats_Brusestole',
'Ats_Flerpunktsstokke',
'Ats_SengebundeMedMotor',
'Ats_Cykler',
'Ats_CykelenhederKørestole',
'Ats_Stokkeholdere',
'Ats_Toiletarmstøtter',
'Ats_Coxitstole',
'Ats_Toiletsæder',
'Ats_Rebstiger',
'Ats_Forhøjerklodser',
'Cluster_0',
'Cluster_1',
'Cluster_2',
'Cluster_3',
'Cluster_4',
'Cluster_5',
'Cluster_6',
'Cluster_7',
'Cluster_8',
'Cluster_9',
'Cluster_10',
'Cluster_11',
'Cluster_12',
'Cluster_13',
'Cluster_14',
'Cluster_15',
'Cluster_16',
'Cluster_17',
'Cluster_18',
'Cluster_19'] #cluster out?



procted_col_name="Gender"


output_col_name="output"

output_prob_col_name="output_prob"


unfavourable_name=0 #women=0
favourable_name=1 #men=1


# # Move data to folder

# In[25]:


tdata_move=pd.read_csv(PATH_orig+"/all_test_data.csv")
tdata_move.to_csv(f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{save_to_folder}/Xgboost_gender_obs.csv")


# # Calcuate gender stats

# In[26]:


df2_copy=tdata_move.copy()


# In[27]:



metrics_frame_gender=pd.DataFrame([],columns=["Gender","TPR","FPR","TNR","FNR","ACC","Mean_y_hat","Mean_y_target","Mean_y_hat_prob"])

for modelnr in df2_copy["Model"].unique():

    metrics_frame_sub_gender=get_df_w_metrics(df2_copy[df2_copy["Model"]==modelnr],procted_col_name,y_col_name,output_col_name,output_prob_col_name).sort_values(["Gender"])[["Gender","TPR","FPR","TNR","FNR","ACC","Mean_y_hat","Mean_y_target","Mean_y_hat_prob"]]#*100
  
    metrics_frame_gender=    pd.concat([metrics_frame_gender,metrics_frame_sub_gender
                        
                       ],sort=False,axis=0)


# In[28]:


metrics_frame_gender_to_plot=metrics_frame_gender.copy()
metrics_frame_gender_to_plot["Model"]="Xgboost"
metrics_frame_gender_to_plot.to_csv(f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{save_to_folder}/Xgboost_gender.csv")


# In[29]:


print("This should be 100:",metrics_frame_gender_to_plot.shape[0])


# # Get CI intervals

# In[30]:


import scipy.stats as st

for gender in metrics_frame_gender["Gender"].unique():
    string_new2=str(gender)
    string_new="     "
    for col in ["TPR","FPR","TNR","FNR"]:#,"ACC"]:
        string_new=string_new+" & "
        string_new2=string_new2+"  & "
        
        
        data=metrics_frame_gender[(metrics_frame_gender["Gender"]==gender)][col]
        m=np.mean(data)
        (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 

        
        #print(f"{col} for {gender} is: mean {round(m*100,2)} ({round(slow*100,2)},{round(shigh*100,2)})")
        string_new2=string_new2+"\\textbf{"+f"{round(m*100,1)}"+ "}"
        string_new=string_new+f"({round(slow*100,1)}-{round(shigh*100,1)})" 
    
    print(string_new2)
    print(string_new)
   


# # Calcuate total

# In[31]:


df2_copy["Total"]="Total"


metrics_frame_all=pd.DataFrame([],columns=["Total","TPR","FPR","TNR","FNR","ACC","Mean_y_hat","Mean_y_target","Mean_y_hat_prob"])

for modelnr in df2_copy["Model"].unique():

    metrics_frame_sub_all=get_df_w_metrics(df2_copy[df2_copy["Model"]==modelnr],"Total",y_col_name,output_col_name,output_prob_col_name).sort_values(["Total"])[["Total","TPR","FPR","TNR","FNR","ACC","Mean_y_hat","Mean_y_target","Mean_y_hat_prob"]]#*100
  
    metrics_frame_all=    pd.concat([metrics_frame_all,metrics_frame_sub_all
                        
                       ],sort=False,axis=0)


# In[32]:


metrics_frame_all_to_plot=metrics_frame_all.copy()
metrics_frame_all_to_plot["Model"]="Xgboost"
metrics_frame_all_to_plot.to_csv(f"/restricted/s164512/G2020-57-Aalborg-bias/Plot_metrics/{save_to_folder}/Xgboost_all.csv")


# In[33]:


print("This should be 50:",metrics_frame_all_to_plot.shape[0])


# # Print ACC

# In[34]:


import scipy.stats as st

for gender in metrics_frame_all["Total"].unique():
    string_new2=str(gender)
    string_new="     "
    for col in ["ACC"]:#["TPR","FPR","TNR","FNR","ACC"]:
        string_new=string_new+" & "
        string_new2=string_new2+"  & "
        
        
        data=metrics_frame_all[(metrics_frame_all["Total"]==gender)][col]
        m=np.mean(data)
        (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 

        
        #print(f"{col} for {gender} is: mean {round(m*100,2)} ({round(slow*100,2)},{round(shigh*100,2)})")
        string_new2=string_new2+"\\textbf{"+f"{round(m*100,1)}"+ "}"
        string_new=string_new+f"({round(slow*100,1)}-{round(shigh*100,1)})" 
    
    print(string_new2+" \\\ ")
    print(string_new+" \\\ ")
   


# In[35]:


import scipy.stats as st

for gender in metrics_frame_gender["Gender"].unique():
    string_new2=str(gender)
    string_new="     "
    for col in ["ACC"]:
        string_new=string_new+" & "
        string_new2=string_new2+"  & "
        
        
        data=metrics_frame_gender[(metrics_frame_gender["Gender"]==gender)][col]
        m=np.mean(data)
        (slow,shigh) =st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 

        
        #print(f"{col} for {gender} is: mean {round(m*100,2)} ({round(slow*100,2)},{round(shigh*100,2)})")
        string_new2=string_new2+"\\textbf{"+f"{round(m*100,1)}"+ "}"
        string_new=string_new+f"({round(slow*100,1)}-{round(shigh*100,1)})" 
    
    print(string_new2+" \\\ ")
    print(string_new+" \\\ ")
   


# In[ ]:





# In[ ]:





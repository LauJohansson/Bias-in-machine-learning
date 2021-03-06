#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn")


# In[2]:


fall_data = pd.read_csv('/restricted/s161749/G2020-57-Aalborg-bias/Data_air/Fall_count_clusterOHE.csv')


# In[3]:


fall_data.columns


# In[4]:


fall_data


# ## Fall
# 

# In[5]:


fall_data.shape


# In[6]:


fall_data.Fall.value_counts()


# In[7]:


1-fall_data.Fall.mean()


# ## Gender

# In[8]:


fall_data.Gender.value_counts()


# In[9]:


1-fall_data.Gender.mean(),fall_data.Gender.mean()


# In[10]:


fall_data['Sex']=fall_data['Gender'].apply(lambda x: 'Female' if x ==0 else 'Male')
fall_data.Sex.value_counts()


# ## BirthYear

# In[11]:


fall_data['Age']=2020-(fall_data['BirthYear']+1900)


# In[12]:


fall_data['Age'].mean(),fall_data['Age'].min(),fall_data['Age'].max()


# In[33]:


sns.histplot(x=fall_data[fall_data['Age']>50]['Age'],bins=20,stat='probability',color='steelblue',alpha=0.7)
plt.grid(axis='x')
plt.savefig('/restricted/s161749/G2020-57-Aalborg-bias/Plots/birthyear_hist.png')


# ## plot med "Men" "Women"

# ## LoanPeriod

# In[14]:


fall_data['LoanPeriod'].mean(),fall_data['LoanPeriod'].min(),fall_data['LoanPeriod'].max()


# In[34]:


sns.histplot(x=fall_data[fall_data['LoanPeriod']<3400]['LoanPeriod'],bins=20,stat='probability',color='steelblue',alpha=0.7)
plt.grid(axis='x')
plt.savefig('/restricted/s161749/G2020-57-Aalborg-bias/Plots/loanperiod_hist.png')


# ### NumberAts

# In[16]:


fall_data.NumberAts.mean(),fall_data.NumberAts.min(),fall_data.NumberAts.max()


# In[35]:


sns.histplot(x=fall_data[fall_data['NumberAts']<40]['NumberAts'],bins=20,stat='probability',color='steelblue',alpha=0.7)
plt.grid(axis='x')
plt.savefig('/restricted/s161749/G2020-57-Aalborg-bias/Plots/numberats_hist.png')


# In[18]:


sns.histplot(x=fall_data.NumberAts,bins=fall_data.NumberAts.max(),stat='probability')
plt.grid(axis='x')


# ## Fall

# In[19]:


fall_data.Fall.value_counts()


# In[20]:


fall_data.Fall.mean()


# In[21]:


fall_data.groupby('Gender')['Fall'].mean()


# ## Ats's

# In[22]:


fall_cols=fall_data.columns


# In[23]:


fall_ats=fall_data[['Ats_Polstring', 'Ats_Mobilitystokke', 'Ats_Belysning', 'Ats_Underlag',
       'Ats_Toiletforh??jereStativ', 'Ats_Signalgivere',
       'Ats_EldrevneK??restole', 'Ats_Forst??rrelsesglas',
       'Ats_N??dalarmsystemer', 'Ats_MobilePersonl??ftere',
       'Ats_TrappelifteMedPlatforme', 'Ats_Badekarsbr??tter', 'Ats_Albuestokke',
       'Ats_MaterialerOgRedskaberTilAfm??rkning', 'Ats_Rygl??n',
       'Ats_Ganghj??lpemidlerSt??tteTilbeh??r', 'Ats_St??tteb??jler',
       'Ats_Lejringspuder', 'Ats_Str??mpep??tagere', 'Ats_D??rtrin', 'Ats_Spil',
       'Ats_BordeP??Stole', 'Ats_Drejeskiver', 'Ats_Toiletstole',
       'Ats_L??ftereStation??re', 'Ats_Madm??lingshj??lpemidler',
       'Ats_Fodbeskyttelse', 'Ats_St??l??ftere', 'Ats_Stole', 'Ats_Sengeborde',
       'Ats_Toiletter', 'Ats_Toiletforh??jereFaste', 'Ats_P??kl??dning',
       'Ats_Brusere', 'Ats_V??vsskadeLiggende', 'Ats_D??r??bnere',
       'Ats_ServeringAfMad', 'Ats_TrappelifteMedS??der',
       'Ats_S??derTilMotork??ret??jer', 'Ats_K??restoleManuelleHj??lper',
       'Ats_Gangbukke', 'Ats_Rollatorer', 'Ats_Tryks??rsforebyggendeSidde',
       'Ats_Fastnettelefoner', 'Ats_B??kkener', 'Ats_Vendehj??lpemidler',
       'Ats_Sanseintegration', 'Ats_K??restolsbeskyttere', 'Ats_Arbejdsstole',
       'Ats_L??ftesejl', 'Ats_K??restoleForbr??ndingsmotor', 'Ats_L??ftestropper',
       'Ats_Stiger', 'Ats_TransportTrapper', 'Ats_DrivaggregaterK??restole',
       'Ats_Emballage??bnere', 'Ats_Toiletforh??jereL??se', 'Ats_H??rvask',
       'Ats_Personl??ftereStation??re', 'Ats_Madrasser', 'Ats_Vindues??bnere',
       'Ats_L??sestativer', 'Ats_K??restoleManuelleDrivringe', 'Ats_S??depuder',
       'Ats_UdstyrCykler', 'Ats_Karkludsvridere', 'Ats_Vaskeklude',
       'Ats_Sengeudstyr', 'Ats_Madlavningshj??lpemidler', 'Ats_Skohorn',
       'Ats_Gribet??ngerManuelle', 'Ats_Hvilestole',
       'Ats_EldrevneK??restoleStyring', 'Ats_B??rehj??lpemidlerTilK??restole',
       'Ats_L??ftegalgerSeng', 'Ats_H??reforst??rkere', 'Ats_Kalendere',
       'Ats_Stokke', 'Ats_L??ftegalger', 'Ats_Ure', 'Ats_St??ttegrebFlytbare',
       'Ats_Forflytningsplatforme', 'Ats_RamperFaste', 'Ats_Rygehj??lpemidler',
       'Ats_Personv??gte', 'Ats_Man??vreringshj??lpemidler', 'Ats_Overt??j',
       'Ats_Lydoptagelse', 'Ats_Gangborde', 'Ats_St??st??ttestole',
       'Ats_RamperMobile', 'Ats_B??rehj??lpemidler', 'Ats_Badekarss??der',
       'Ats_Siddemodulsystemer','Ats_Siddepuder', 'Ats_Sengeheste', 'Ats_Stolerygge', 'Ats_Rulleborde',
       'Ats_Sengeforl??ngere', 'Ats_Madningsudstyr', 'Ats_Brusestole',
       'Ats_Flerpunktsstokke', 'Ats_SengebundeMedMotor', 'Ats_Cykler',
       'Ats_CykelenhederK??restole', 'Ats_Stokkeholdere',
       'Ats_Toiletarmst??tter', 'Ats_Coxitstole', 'Ats_Toilets??der',
       'Ats_Rebstiger', 'Ats_Forh??jerklodser','Fall']]


# In[19]:


len(fall_ats[fall_ats['Fall']==1])


# In[26]:


fall_ats[fall_ats['Fall']==0].sum().sort_values(ascending=False).head(10)/len(fall_ats[fall_ats['Fall']==0])


# In[27]:


fall_ats[fall_ats['Fall']==1].sum().sort_values(ascending=False).head(10)/len(fall_ats[fall_ats['Fall']==1])


# # Linear regressions
# 

# In[2]:


reg_data = pd.read_csv('/restricted/s161749/G2020-57-Aalborg-bias/Data_air/Fall_count_clusterOHE_std.csv')


# In[3]:


reg_data.columns[0:100]


# In[4]:


reg_data.columns[101:]


# In[17]:


X = reg_data[['BirthYear', 'LoanPeriod', 'NumberAts']] # using all covariates in the dataset. ,'Ats_0'
#X1 = pd.DataFrame(preprocessing.scale(X1),columns=X1.columns)
#X2 = pd.DataFrame(fall_data['Gender'])
#X = pd.concat([X1],axis=1)
y = reg_data.Fall


# In[18]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

X = sm.add_constant(X, prepend=False)

# Fit and summarize OLS model
mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())


# In[19]:


X = reg_data.drop(columns=['Unnamed: 0','Fall','Gender']) # using all covariates in the dataset. ,'Ats_0'
#X1 = pd.DataFrame(preprocessing.scale(X1),columns=X1.columns)
#X2 = pd.DataFrame(fall_data['Gender'])
#X = pd.concat([X1],axis=1)
y = reg_data.Fall


# In[20]:


import statsmodels.api as sm


X = sm.add_constant(X, prepend=False)

# Fit and summarize OLS model
mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())


# In[21]:


X = reg_data.drop(columns=['Unnamed: 0','Fall']) # using all covariates in the dataset. ,'Ats_0'
#X1 = pd.DataFrame(preprocessing.scale(X1),columns=X1.columns)
#X2 = pd.DataFrame(fall_data['Gender'])
#X = pd.concat([X1],axis=1)
y = reg_data.Fall


# In[22]:


import statsmodels.api as sm


X = sm.add_constant(X, prepend=False)

# Fit and summarize OLS model
mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())


# ### Point: when we take into account what type of aid the indivdual has, then NumberAts is significant and positively correlated, where it was negative and insignificant before! 

# ### Pointe: der er lidt noget rod med alder og loan period n??r vi kommer der op af, da der kommer gensidige p??virkninger mellem x og y. Man kunne forestille sig, at borgere der bliver ekstra gamle ogs?? er ekstra friske (m??ske ogs?? mindre tilb??jelige til at falde). Det er et selektions problem! Dem der bliver svage af at blive ??ldre d??r!

# ## interaction terms

# In[46]:


#X = reg_data[['Gender','BirthYear', 'LoanPeriod', 'NumberAts']]
X = reg_data.drop(columns=['Unnamed: 0','Fall']) # using all covariates in the dataset. ,'Ats_0'
#X1 = pd.DataFrame(preprocessing.scale(X1),columns=X1.columns)
#X2 = pd.DataFrame(fall_data['Gender'])
#X = pd.concat([X1],axis=1)
y = reg_data.Fall


# In[47]:


X['Gender*BirthYear']=X['Gender']*X['BirthYear']
X['Gender*LoanPeriod']=X['Gender']*X['LoanPeriod']
X['Gender*NumberAts']=X['Gender']*X['NumberAts']


# In[48]:


import statsmodels.api as sm


X = sm.add_constant(X, prepend=False)

# Fit and summarize OLS model
mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())


# ## Should the original covariates be kept in? 

# ## What is the interpretation? When comparing to women, the coefficient of men for BirthYear is 0.0493 lower? Meaning that the effect of being younger has a more negative effect of the probability of falling - relative to women. This makes sense, since women might be more "fit, stable" in their retirement years than men, why being young is more important (and significantly so) for mens risk of falling than for womens. The same is true for loan period.  

# ## Comment on that changing the distribution or augmentation on these variables that have gender specific interaction effects might work well. In other words, they have different returns on these variables, and if their levels are more aligned or more differentiated, it might help the classifier be less biased. Actually, if their returns are different, but their distribution is the same in the original data - then disparate impact removal will not help, while it might help to learn a representation, where the difference in returns is taken into account. For example by "making" the women older. (Remember than +1 birhtyear is -1 in age!)

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[9]:


import sys
from PIL import Image


# # Choose folders

# In[10]:


titel_mitigation1="original" #DONT TOUCH THIS
#titel_mitigation2="Dropping D"
#titel_mitigation2="Gender Swap"
#titel_mitigation2="DI remove"
titel_mitigation2="DI remove no gender"
#titel_mitigation2="LFR"


# # Metric plots

# In[11]:






PATH_orig1="/restricted/s164512/G2020-57-Aalborg-bias/plots/"+titel_mitigation1+f"/Compare_plot_{titel_mitigation1}.png"



PATH2_save="/restricted/s164512/G2020-57-Aalborg-bias/plots/"+titel_mitigation2+"/"
PATH_orig2=PATH2_save+f"Compare_plot_{titel_mitigation2}.png"


# In[12]:


images = [Image.open(x) for x in [PATH_orig1, PATH_orig2]]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

new_im.save(PATH2_save+f'Compared_bar_merge_{titel_mitigation2}.jpeg')


# # Relation plot

# In[13]:


PATH_orig1="/restricted/s164512/G2020-57-Aalborg-bias/plots/"+titel_mitigation1+f"/Difference_gender_{titel_mitigation1}_relation.png"
PATH2_save="/restricted/s164512/G2020-57-Aalborg-bias/plots/"+titel_mitigation2+"/"
PATH_orig2=PATH2_save+f"/Difference_gender_{titel_mitigation2}_relation.png"


# In[14]:


images = [Image.open(x) for x in [PATH_orig1, PATH_orig2]]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

new_im.save(PATH2_save+f'Compared_relation_merge_{titel_mitigation2}.jpeg')


# # Acc plot

# In[15]:


PATH_orig1="/restricted/s164512/G2020-57-Aalborg-bias/plots/"+titel_mitigation1+f"/Acc_models_{titel_mitigation1}.png"
PATH2_save="/restricted/s164512/G2020-57-Aalborg-bias/plots/"+titel_mitigation2+"/"
PATH_orig2=PATH2_save+f"/Acc_models_{titel_mitigation2}.png"


# In[16]:


images = [Image.open(x) for x in [PATH_orig1, PATH_orig2]]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

new_im.save(PATH2_save+f'Compared_acc_merge_{titel_mitigation2}.jpeg')


# In[ ]:





# In[ ]:





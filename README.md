# Bias-in-machine-learning

## Introduction

This repository contains code from the master thesis "Identifying and mitigating of bias in machine learning models" by Daniel Juhász Vigild and Lau Johansson.

Read our master thesis [here](https://github.com/LauJohansson/Bias-in-machine-learning/blob/main/Evaluation_of_bias_in_ML_algorithms_v31082021.pdf) <br />

In the moment, the repository is mainly focused on passing on code to the [AIR project](https://projekter.au.dk/air/). However, we hope to inspire other to investigate bias using our approach as suggested in the thesis. Here is a wrap-up:

Step 1: Choose a protected variable (we choose "gender").<br>
Step 2: Create TPR, FPR, TNR and FNR grouped by females and males.<br>
Step 3: Compare the **difference** between the rates between the two genders (use e.g. a barplot with 95% confidence intervals).<br>
Step 4: Calculate the **relation** between the classification rates between the genders.<br>
Step 5: Assess the relation using e.g. using the 80% rule. <br>
Step 6: Assess if any gender related **bias** can be identified.<br>
<br>
You are welcome to be inspired by the notebook [Plot measures.ipynb](https://github.com/LauJohansson/Bias-in-machine-learning/blob/main/Plot%20measures.ipynb) to perform step 3-6. 

## Packages

pip install fairlearn<br>
pip install BlackBoxAuditing<br>


## AIR specific: Data transformation for machine learning


### Data in correct folder
Move AIR data to the `data_air/` folder.

Should be the files:

"Fall.csv"<br>
"Fall_emb.csv"<br>
"Fall_count.csv"<br>

Actually, it is only "Fall_count.csv" that is used in our the scripts. You can choose to use one of the other AIR datasets, but then you need to manually change the code in the notebooks accordingly. 
<br>
Now use [Transform data.ipynb](https://github.com/LauJohansson/Bias-in-machine-learning/blob/main/Transform%20data.ipynb) to transform the data. You are now ready to use the scripts from `legacy/`


## AIR specific: Building XGBoost machine learning models using mitigation techniques
`XGBoost A1.ipynb` builds a XGBoost on the AIR dataset.<br>
`XGBoost A2 with SHAP plots.ipynb` builds a XGBoost on the AIR dataset and create SHAP plots.<br>
`XGBoost B: dropping gender.ipynb` as A1 where the mitigation technique "dropping gender" is applied.<br>
`XGBoost C: gender swap.ipynb` as A1 where the mitigation technique "gender swap" is applied.<br>
`XGBoost D: learning fair representations (LFR).ipynb` as A1 where the mitigation technique "learning fair representations" is applied.<br>
`XGBoost E: disparate impact removal (DI removal).ipynb` as A1 where the mitigation technique "disparate impact removal" is applied.<br>
`XGBoost F: all mitigation techniques.ipynb` as A1 where all the mitigation techniques above can be combined. <br>


## General: Plotting 

### Save plot metrics

In the folder `Plot_metrics/` create a new subfolder: `Plot_metrics/{folder_name}/`<br>
<br>
For each model you create you need to save a individual file.<br>
<br>
E.g. if your model is a XGBoost model call the file Plot_metrics/{folder_name}/"XGBoost_gender.csv"<br>
<br>
The file should have this specific columns:

| Gender | TPR | FPR | TNR | FNR | Model     |
|--------|-----|-----|-----|-----|-----------|
| Male   | 0.80 | 0.70 | 0.30 | 0.20 | "XGBoost" |
| Female | 0.80 | 0.70 | 0.30 | 0.20 | "XGBoost" |
| Male   | 0.75 | 0.65 | 0.35 | 0.25 | "XGBoost" |
| Female | 0.60 | 0.70 | 0.30 | 0.40 | "XGBoost" |
| Male   | 0.90 | 0.90 | 0.10 | 0.10 | "XGBoost" |
| Female | 0.80 | 0.70 | 0.30 | 0.20 | "XGBoost" |
| Male   | ...  | ...  | ...  | ...  | ....   |
| Female | ...  | ...  | ...  | ...  | ....   |
<br>
E.g. if your model is a FFNN model call the file Plot_metrics/{folder_name}/"FFNN_gender.csv"<br>
<br>
The file should have this specific columns:<br>

| Gender | TPR | FPR | TNR | FNR | Model     |
|--------|-----|-----|-----|-----|-----------|
| Male   | 0.80 | 0.70 | 0.30 | 0.20 | "FFNN" |
| Female | 0.80 | 0.70 | 0.30 | 0.20 | "FFNN" |
| Male   | 0.75 | 0.65 | 0.35 | 0.25 | "FFNN" |
| Female | 0.60 | 0.70 | 0.30 | 0.40 | "FFNN" |
| Male   | 0.90 | 0.90 | 0.10 | 0.10 | "FFNN" |
| Female | 0.80 | 0.70 | 0.30 | 0.20 | "FFNN" |
| Male   | ...  | ...  | ...  | ...  | ....   |
| Female | ...  | ...  | ...  | ...  | ....   |



### Create the plots
Use the notebook "2. Plot measures.ipynb" to create the plots. 


## General: Legacy
All scripts used in the master thesis is located in the `legacy/` folder.<br>
NB! All files were converted to .py files when downloading our master thesis to ensure that no personal data was extracted from the .ipynb files<br>
<br>
Almost all of the files has been converted back to .ipynb, but some of the files are still not converted to notebooks. If you know how to easily convert the files back to .ipynb, then please contact me. 







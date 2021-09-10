# Bias-in-machine-learning

## Data transformation for machine learning

### Data in correct folder
Move AIR data to the `data_air/` folder.

Should be the files:

"Fall.csv"
"Fall_emb.csv"
"Fall_count.csv"

Actually, it is only "Fall_count.csv" that is used in our the scripts. You can choose to use one of the other AIR datasets, but then you need to manually change the code in the notebooks accordingly. 

Now use "1. Transform data.ipynb" to transform the data. You are now ready to use the scripts from `legacy/`


## Plotting 

### Save plot metrics

In the folder `Plot_metrics/` create a new subfolder: `Plot_metrics/{folder_name}/`

For each model you create you need to save a individual file.

E.g. if your model is a XGBoost model call the file Plot_metrics/{folder_name}/"XGBoost_gender.csv"

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


E.g. if your model is a FFNN model call the file Plot_metrics/{folder_name}/"FFNN_gender.csv"

The file should have this specific columns:

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


### Legacy
All scripts used in the master thesis is located in the `legacy/` folder.
NB! All files are converted to .py files to ensure that no personal data was extracted from the .ipybn files

If you know how to easily convert the files back to .ipynb, then please contact me. 







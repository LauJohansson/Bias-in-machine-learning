# Bias-in-machine-learning

## Data in correct folder
Move AIR data to the `data_air/` folder.

Should be the files:

"Fall.csv"
"Fall_emb.csv"
"Fall_count.csv"

Actually, it is only "Fall_count.csv" that is used in our the scripts. You can choose to use one of the other AIR datasets, but then you need to manually change the code in the notebooks accordingly. 

## Save plot metrics

In the folder `Plot_metrics/` create a new subfolder: `Plot_metrics/{folder_name}/`

For each model you create you need to save a individual file.

E.g. if your model is a XGBoost model call the file Plot_metrics/{folder_name}/"XGBoost_gender.csv"

The file should have this specific columns:

| Gender | TPR | FPR | TNR | FNR | Model     |
|--------|-----|-----|-----|-----|-----------|
| Male   | 0.8 | 0.7 | 0.3 | 0.2 | "XGBoost" |
| Female | 0.8 | 0.7 | 0.3 | 0.2 | "XGBoost" |


E.g. if your model is a FFNN model call the file Plot_metrics/{folder_name}/"FFNN_gender.csv"

The file should have this specific columns:

| Gender | TPR | FPR | TNR | FNR | Model     |
|--------|-----|-----|-----|-----|-----------|
| Male   | 0.8 | 0.7 | 0.3 | 0.2 | "FFNN" |
| Female | 0.8 | 0.7 | 0.3 | 0.2 | "FFNN" |








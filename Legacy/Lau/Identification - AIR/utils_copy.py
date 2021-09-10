
import pandas as pd
import torch.nn as nn
from sklearn.metrics import confusion_matrix


#from: https://nbviewer.jupyter.org/github/srnghn/bias-mitigation-examples/blob/master/Bias%20Mitigation%20with%20Disparate%20Impact%20Remover.ipynb
def calc_prop(data, group_col, group, output_col, output_val):
    '''
    data:       The dataframe
    group_col:  The protected atrtibute column (e.g Gender)
    group:      The chosen group (e.g Male or Female)
    output_col: The column holding the y-value (either y_hat or y   - could be Fall)
    output_val: The value of the y  (e.g.   all y=1 )
    
    
    Example:
    
    Find p(y=0 | G="Female")
    
    calc_prop(data,"Gender","Female","y_true",0)
    
    
    
    '''
    new = data[data[group_col] == group]
    return len(new[new[output_col] == output_val])/len(new)

def calc_prop_no_group(data, output_col, output_val):
    return len(data[data[output_col] == output_val])/len(data)



def get_df_w_metrics(df,protected_variable_name,y_target_name,y_pred_name):
    
    import pandas as pd
    

    #Create empty DataFrame
    confusion_df=pd.DataFrame(columns=[protected_variable_name,"FPR","FNR"])
    


    #For each value of the protected variable, calculated FPR/FNR and insert into the empty DataFrame
    for name in list(df[protected_variable_name].unique()):
        a=df[df[protected_variable_name]==name][y_target_name]
        b=df[df[protected_variable_name]==name][y_pred_name]#.apply(lambda x: 0 if x<t else 1 )

    
        TN, FP, FN, TP = confusion_matrix(list(a), list(b),labels=[0, 1]).ravel()
        
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)

        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        LRplus=TPR/FPR
        LRminus=FNR/TNR


        #F1-score
        F1=2*(PPV*TPR)/(PPV+TPR)

        confusion_df=confusion_df.append({protected_variable_name:name,
                                          "TPR":TPR,
                                          "TNR":TNR,
                                          "FPR":FPR,
                                          "FNR":FNR,
                                          "PPV":PPV,
                                          "NPV":NPV,
                                          "FDR":FDR,
                                          "ACC":ACC,
                                          "F1":F1,
                                          "LRplus":LRplus,
                                          "LRminus":LRminus,
                                          "TN":TN,
                                          "FP":FP,
                                          "FN":FN,
                                          "TP":TP
                                          },ignore_index=True)

    return confusion_df


def compare_bias_metrics(data,protected_variable_name,y_target_name,y_pred_name,unfavourable_name,favourable_name,print_var=False,fav_value=1,unfav_value=0):
    
    
    df = get_df_w_metrics(data,protected_variable_name,y_target_name,y_pred_name)
    
    
    
    #==================== DISPARATE IMPACT ======================#
    #Feldman et al
    
    #a1=data[y_target_name]
    #b1=data[protected_variable_name]#.apply(lambda x: 0 if x<t else 1 )
    #d, b, c, a = confusion_matrix(list(a1), list(b1),labels=[0, 1]).ravel()
    
    a=calc_prop(data,protected_variable_name,unfavourable_name,y_target_name,unfav_value) #prop of unfav group, recieve unfav value
    b=calc_prop(data,protected_variable_name,favourable_name,y_target_name,unfav_value) #prop of ufav group, recieve unfav value
    c=calc_prop(data,protected_variable_name,unfavourable_name,y_target_name,fav_value) #prop of ufav group, recieve fav value
    d=calc_prop(data,protected_variable_name,favourable_name,y_target_name,fav_value) #prop of ufav group, recieve fav value
    
    
    #a=df[df[protected_variable_name]==unfavourable_name]["TN"].item()
    #b=df[df[protected_variable_name]==favourable_name]["TN"].item()
    #c=df[df[protected_variable_name]==unfavourable_name]["TP"].item()
    #d=df[df[protected_variable_name]==favourable_name]["TP"].item()
        
    Feldman_Disparate_impact=(c/(a + c)) / (d/(b + d))
    
    if print_var:
        print("==================================== Feldman et al ====================================")
        
        print(f"If DATASET has no disparate impact then DI>=0.8.")
        print(f"DI={Feldman_Disparate_impact}")
        
        if Feldman_Disparate_impact>=0.8:
            print("The DATASET has no disparate impact")
        else:
            print("The DATASET has disparate impact")
          
        
        print("======================================================================================= \n")
            
    #===========================================================#
    
    
    #==================== Learning Fair representations ======================#
    #Zafar et al
    
    ###Disparate TREATMENT ####
    
    p_yhat1_z1=calc_prop(data,protected_variable_name,favourable_name,y_pred_name,fav_value)
    p_yhat1_z0=calc_prop(data,protected_variable_name,unfavourable_name,y_pred_name,fav_value)
    p_yhat1=calc_prop_no_group(data, y_pred_name, fav_value)
    
    p_yhat0_z1=calc_prop(data,protected_variable_name,favourable_name,y_pred_name,unfav_value)
    p_yhat0_z0=calc_prop(data,protected_variable_name,unfavourable_name,y_pred_name,unfav_value)
    p_yhat0=calc_prop_no_group(data, y_pred_name, unfav_value)
    
    
    ###Disparate IMPACT ####
    
    if print_var:
        print("==================================== Zafar et al ====================================")
        
        print(f"If the CLASSIFIER has no DISPARATE TREATMENT, these equations should hold:")
        print(f"P(y_hat=1|z={favourable_name},x) = P(y_hat=1,x) <=> {round(p_yhat1_z1,2)} = {round(p_yhat1,2)}")
        print(f"P(y_hat=1|z={unfavourable_name},x) = P(y_hat=1,x) <=> {round(p_yhat1_z0,2)} = {round(p_yhat1,2)}")
        print(f"P(y_hat=0|z={favourable_name},x) = P(y_hat=0,x) <=> {round(p_yhat0_z1,2)} = {round(p_yhat0,2)}")
        print(f"P(y_hat=0|z={unfavourable_name},x) = P(y_hat=0,x) <=> {round(p_yhat0_z0,2)} = {round(p_yhat0,2)}")
        
        print("T_low: {}".format(min([min(p_yhat1_z1,p_yhat1)/max(p_yhat1_z1,p_yhat1),
                                  min(p_yhat1_z0,p_yhat1)/max(p_yhat1_z0,p_yhat1),
                                  min(p_yhat0_z1,p_yhat0)/max(p_yhat0_z1,p_yhat0),
                                  min(p_yhat0_z0,p_yhat0)/max(p_yhat0_z0,p_yhat0),
                                 ])))
        
        feld_list_values=[p_yhat1_z1/p_yhat1,
                                  p_yhat1_z0/p_yhat1,
                                  p_yhat0_z1/p_yhat0,
                                  p_yhat0_z0/p_yhat0,
                                 ]
        
        mini_feld=min(feld_list_values)
        maxi_feld=max(feld_list_values)
        
        print(f"T_low_new: min:{mini_feld}, max:{maxi_feld}")
        
        print("\n")
        
        print(f"If the CLASSIFIER has no DISPARATE IMPACT, these equations should hold:")
        print(f"P(y_hat=1|z={unfavourable_name}) = P(y_hat=1,z={favourable_name}) <=> {round(p_yhat1_z0,2)} = {round(p_yhat1_z1,2)}")
        
        print("T_low: {}".format(min([min(p_yhat1_z0,p_yhat1_z1)/max(p_yhat1_z0,p_yhat1_z1)
                                 ])))
        
        print(f"T_low_new: {p_yhat1_z0/p_yhat1_z1}")
        
        
        print("\n")
        
        print(f"If the CLASSIFIER has no DISPARATE MISTREATMENT, these equations should hold:")
        
        FPR_z0=df[df[protected_variable_name]==unfavourable_name]["FPR"].item()
        FPR_z1=df[df[protected_variable_name]==favourable_name]["FPR"].item()
        
        FNR_z0=df[df[protected_variable_name]==unfavourable_name]["FNR"].item()
        FNR_z1=df[df[protected_variable_name]==favourable_name]["FNR"].item()
        
        
        
        print(f"FPR: P(y_hat!=y|z={unfavourable_name},y=0) = P(y_hat!=y|z={favourable_name},y=0) <=> {round(FPR_z0,2)} = {round(FPR_z1,2)}")
        
        print("T_low: {}".format(min([min(FPR_z0,FPR_z1)/max(FPR_z0,FPR_z1)
                                 ])))
        print(f"T_low_new: {FPR_z0/FPR_z1}")
        
        
        
        
        
        print(f"FNR: P(y_hat!=y|z={unfavourable_name},y=1) = P(y_hat!=y|z={favourable_name},y=1) <=> {round(FNR_z0,2)} = {round(FNR_z1,2)}")
        
        print("T_low: {}".format(min([
                                      min(FNR_z0,FNR_z1)/max(FNR_z0,FNR_z1)
                                 ])))
        print(f"T_low_new: {FNR_z0/FNR_z1}")
        
        print("======================================================================================= \n")
    
    
    
    
    
    #==================== Equality of Opportunity in Supervised Learning ======================#
    #Hardt et al
    
    ### Equalized odds ####
    
    TPR_z0=df[df[protected_variable_name]==unfavourable_name]["TPR"].item()
    TPR_z1=df[df[protected_variable_name]==favourable_name]["TPR"].item()
    
    TNR_z0=df[df[protected_variable_name]==unfavourable_name]["TNR"].item()
    TNR_z1=df[df[protected_variable_name]==favourable_name]["TNR"].item()
    
    if print_var:
        print("==================================== Hardt et al ====================================")
        
        print(f"If the CLASSIFIER has EQUALIZED ODDS, these equations should hold:")
        print(f"P(y_hat=1|z={unfavourable_name},y=1) = P(y_hat=1|z={favourable_name},y=1) <=> {round(TPR_z0,2)} = {round(TPR_z1,2)}")
        
        print(f"P(y_hat=0|z={unfavourable_name},y=0) = P(y_hat=0|z={favourable_name},y=0) <=> {round(TNR_z0,2)} = {round(TNR_z1,2)}")
        
        print("T_low: {}".format(min([min(TPR_z0,TPR_z1)/max(TPR_z0,TPR_z1),
                                      min(TNR_z0,TNR_z1)/max(TNR_z0,TNR_z1)
                                 ])))
        hardt_list=[TPR_z0/TPR_z1,TNR_z0/TNR_z1]
        hardt_max=max(hardt_list)
        hardt_min=min(hardt_list)
        
        print(f"T_low_new: min:{hardt_min} , max:{hardt_max} ")
        
        print("\n")
        
        print(f"If the CLASSIFIER has EQUAL OPPORTUNITY, these equations should hold:")
        print(f"P(y_hat=1|z={unfavourable_name},y=1) = P(y_hat=1|z={favourable_name},y=1) <=> {round(TPR_z0,2)} = {round(TPR_z1,2)}")
        
        print("T_low: {}".format(min([min(TPR_z0,TPR_z1)/max(TPR_z0,TPR_z1),
                                 ])))
        print(f"T_low_new: {TPR_z0/TPR_z1} ")
        
        print("\n")
    
        print("======================================================================================= \n")
    
    
    ############Measuring racial discrimination in algorithms####
    #Arnold et al. 
    
    
    
    if print_var:
        print("==================================== Arnold et al ====================================")
        
        my=data[y_target_name].mean()
        delta=(TNR_z1-TNR_z0)*(1-my)+(FNR_z1-FNR_z0)*my
        
        
        print(f"The racial discrimination paramenter (delta) = {delta}")
        print("\n")
    
        print("======================================================================================= \n")
    
        
        
        
        
    ############GENERAL CLASSIFICATION METRICS####
    
    
    
    if print_var:
        print("==================================== GENERAL CLASSIFICATION METRICS ====================================")
        
        print(f"TPR for {unfavourable_name}: {TPR_z0}")
        print(f"TPR for {favourable_name}: {TPR_z1}")
        print("\n")
        
        print(f"TNR for {unfavourable_name}: {TNR_z0}")
        print(f"TNR for {favourable_name}: {TNR_z1}")
        print("\n")
        
        print(f"FPR for {unfavourable_name}: {FPR_z0}")
        print(f"FPR for {favourable_name}:  {FPR_z1}")
        print("\n")
        
        print(f"FNR for {unfavourable_name}:  {FNR_z0}")
        print(f"FNR for {favourable_name}:  {FNR_z1}")
        print("\n")
    
        print("======================================================================================= \n")
    
        
        
        
    
    
    
    return Feldman_Disparate_impact


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


def plot_metric_comparison(df,metric1,metric2,label_col_name):
    """
    This function plots FPR vs. FNR (including the triangle boundary)
    Input: 
        df:                 a dataframe
        metric1:            The first metric on the x-axis
        metric2:            The second metric on the y-axis
        label_col_name:     the name of the column with the labels (of the e.g. protected variable)...

        OBS: The dataframe must have a colmn with the name "FPR" and "FNR", if no input for metric1/metric2 is given

    """

    #Make scatter plot of FPR vs. FNR
    fig, ax = plt.subplots()
    sns.scatterplot(data=df,x=metric1,y=metric2,hue=label_col_name,ax=ax)
    ax.set(xlabel=metric1, ylabel=metric2)
    #ax.set_xticks(range(0,1,0.1))

    if metric1!="ACC" and metric2!="ACC":
        ax.set_xlim(0,1)


    #The red triangle
    if metric1=="FPR" and metric2=="FNR":
        red_patch=Line2D([0], [0], color='r', lw=1, label='FPR/FNR-plane')
        x = np.array([[0,0], [1,0], [0, 1]])
        t1 = plt.Polygon(x[:3,:], color="red",fill=False,linewidth=1,linestyle="-")#linestyle="--"
        plt.gca().add_patch(t1)


        #get the handles/labels of the plot
        handles, labels = ax.get_legend_handles_labels()
        # handles is a list, so append manual patch
        handles.append(red_patch) 

        # plot the legend
        plt.legend(handles=handles, loc='best')
    plt.show()

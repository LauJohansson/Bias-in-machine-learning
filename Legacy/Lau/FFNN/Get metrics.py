#!/usr/bin/env python
# coding: utf-8

# In[ ]:




for overall_loop in [0,1,2,3,4,5,6,7,8,9]:
        model1 = Network().to(device)
        model1.load_state_dict(torch.load(PATH+'_FFNN_model_local.pth'))
       
        model1.eval()


        df_evaluate.read_csv(PATH+"test_data_localmodel.csv")
        
      
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


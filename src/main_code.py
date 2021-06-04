import codes.step_1_cleaning_data
import codes.step_2_convert_timecolumns_to_standard_time
import codes.step_3_choose_features_and_define_target
import codes.step_4_EDA_1                         
import codes.step_4_EDA_2                         
import codes.step_4_EDA_of_AA_OO_WN_3
import codes.step_5_EDA_and_separate_airline_data 
airline = open("input",'r')#input('Please choose one of airlines: AA, OO, WN:')
if airline == airline.readline():
    import codes.step_6_apply_ML_models_AA_airlines.step_6_1_divide_data_into_seen_and_unseen_files 
    import codes.step_6_apply_ML_models_AA_airlines.step_6_2_predict_categories_of_target
    import codes.step_6_apply_ML_models_AA_airlines.step_6_3_predict_seen_target  
    import codes.step_6_apply_ML_models_AA_airlines.step_6_4_predict_unseen_target
else:
    if airline == 'OO':       
        import codes.step_6_apply_ML_models_OO_airlines.step_6_1_divide_data_into_seen_and_unseen_files
        import codes.step_6_apply_ML_models_OO_airlines.step_6_2_predict_categories_of_target
        import codes.step_6_apply_ML_models_OO_airlines.step_6_3_predict_seen_target
        import codes.step_6_apply_ML_models_OO_airlines.step_6_4_predict_unseen_target
    else:  
        import codes.step_6_apply_ML_models_WN_airlines.step_6_1_divide_data_into_seen_and_unseen_files
        import codes.step_6_apply_ML_models_WN_airlines.step_6_2_predict_categories_of_target
        import codes.step_6_apply_ML_models_WN_airlines.step_6_3_predict_seen_target
        import codes.step_6_apply_ML_models_WN_airlines.step_6_4_predict_unseen_target




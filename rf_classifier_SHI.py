from rf_classifier import*
from ml_helpers import *

def predict_SI_pre_treatment(df_all):
    df_all = df_all[df_all['A35'] != 0]
    features, labels = get_features_labels_all_admissions_PCL(df_all, 'all_data_admissions', 'A99')
    run_random_forest_classifier_model(features, labels, "predict_SI_pre_treatment", baseline=True)

def predict_SI_post_treatment_w_pre_SI(df_all):
    df_all = df_all[df_all['A35'] != 0]
    features, labels = get_features_labels_all_admissions_PCL(df_all, 'all_data_admissions', 'D60', includePreSI=True)
    run_random_forest_classifier_model(features, labels, "predict_SI_post_treatment_w_pre_SI", baseline=True)

def predict_SI_post_treatment_wo_pre_SI(df_all):
    # print(df_all.size / len(list(df_all.columns)))
    # print(df_all.columns)
    # df_all = df_all[df_all['A99'] != 0]
    # print(df_all.size / len(list(df_all.columns)))
    features, labels = get_features_labels_all_admissions_PCL(df_all, 'all_data_admissions', 'D60', includePreSI=False)
    run_random_forest_classifier_model(features, labels, "predict_SHI_post_treatment_wo_pre_SI", baseline=False)

def predict_SI_positivity_post_treatment_wo_pre_SI(df_all):
    features, labels = get_features_labels_all_admissions_PCL(df_all, 'all_data_admissions', 'SHI_positivity', includePreSI=False)
    run_random_forest_classifier_model(features, labels, "predict_SHI_positivity_post_treatment_wo_pre_SHI", baseline=False)

def run_SHI_rf_classifier_models(df_all):
    print('*' * 50)
    print('SELF HARM IDEATION MODELS RANDOM FOREST CLASSIFIER')
    print('*' * 50)

    # predict_SI_pre_treatment(df_all)
    # predict_SI_post_treatment_w_pre_SI(df_all)
    predict_SI_post_treatment_wo_pre_SI(df_all)
    predict_SI_positivity_post_treatment_wo_pre_SI(df_all)

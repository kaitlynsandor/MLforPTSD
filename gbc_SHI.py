from gbc import*
from ml_helpers import *

def predict_SI_pre_treatment(df_all):
    features, labels = get_features_labels_all_admissions_ALL(df_all, 'all_data_admissions', 'A99')
    run_gradient_boosting_classifier_model(features, labels, "predict_SI_pre_treatment", baseline=True)

def predict_SI_post_treatment_w_pre_SI(df_all):
    features, labels = get_features_labels_all_admissions_ALL(df_all, 'all_data_admissions', 'D60', includePreSI=True)
    run_gradient_boosting_classifier_model(features, labels, "predict_SI_post_treatment_w_pre_SI", baseline=True)

def predict_SI_post_treatment_wo_pre_SI(df_all):
    features, labels = get_features_labels_all_admissions_ALL(df_all, 'all_data_admissions', 'D60', includePreSI=False)
    run_gradient_boosting_classifier_model(features, labels, "predict_SI_post_treatment_wo_pre_SI", baseline=True)

def run_SHI_gbc_models(df_all):
    print('*' * 50)
    print('SELF HARM IDEATION MODELS GRADIENT BOOSTING CLASSIFIER')
    print('*' * 50)

    # predict_SI_pre_treatment(df_all)
    # predict_SI_post_treatment_w_pre_SI(df_all)
    predict_SI_post_treatment_wo_pre_SI(df_all)
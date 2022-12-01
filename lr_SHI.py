from lr import*
from ml_helpers import *

def predict_SI_pre_treatment(df_all):
    features, labels = get_features_labels(df_all, 'all_data_admissions', 'A99')
    run_logistical_regression_model(features, labels, "predict_SI_pre_treatment", baseline=True)

def predict_SI_post_treatment_w_pre_SI(df_all):
    features, labels = get_features_labels(df_all, 'all_data_admissions', 'D60', includePreSI=True)
    run_logistical_regression_model(features, labels, "predict_SI_post_treatment_w_pre_SI", baseline=True)

def predict_SI_post_treatment_wo_pre_SI(df_all):
    features, labels = get_features_labels(df_all, 'all_data_admissions', 'D60', includePreSI=False)
    run_logistical_regression_model(features, labels, "predict_SI_post_treatment_wo_pre_SI", baseline=True)

def run_SHI_lr_models(df_all):
    print('*' * 50)
    print('SELF HARM IDEATION MODELS LOGISTICAL REGRESSION')
    print('*' * 50)

    # predict_SI_pre_treatment(df_all)
    # predict_SI_post_treatment_w_pre_SI(df_all)
    predict_SI_post_treatment_wo_pre_SI(df_all)
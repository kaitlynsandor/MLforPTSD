from rf_classifier import *

def predict_diff_score_w_all_admissions(df_all):
    df_all = df_all[df_all['diff_score'] > 0]
    # print(df_all['diff_score'])
    features, labels = get_features_labels_all_admissions_ALL(df_all, 'all_data_admissions', 'diff_score')
    run_random_forest_classifier_model(features, labels, 'predict_diff_score_w_all_admissions')

def predict_diff_score_w_admissions_score(df_all):
    features, labels = get_features_labels_all_admissions_ALL(df_all, 'admissions_score', 'diff_score')
    run_random_forest_classifier_model(features, labels, 'predict_diff_score_w_admissions_score')

def predict_discharge_score_w_all_admissions(df_all):
    features, labels = get_features_labels_all_admissions_ALL(df_all, 'all_data_admissions', 'discharge_score')
    run_random_forest_classifier_model(features, labels, 'predict_discharge_score_w_all_admissions')

def predict_discharge_score_w_admissions_score(df_all):
    features, labels = get_features_labels_all_admissions_ALL(df_all, 'admissions_score', 'discharge_score')
    run_random_forest_classifier_model(features, labels, 'predict_discharge_score_w_admissions_score')

def predict_ptsd_healed_w_all_admissions(df_all):
    features, labels = get_features_labels_all_admissions_ALL(df_all, 'all_data_admissions', 'healed')
    run_random_forest_classifier_model(features, labels, 'predict_ptsd_healed_w_all_admissions')

def predict_ptsd_healed_w_admissions_score(df_all):
    features, labels = get_features_labels_all_admissions_ALL(df_all, 'admissions_score', 'healed')
    run_random_forest_classifier_model(features, labels, 'predict_ptsd_healed_w_admissions_score')

def run_treatment_improvement_models(df_all):
    print('*' * 50)
    print('TREATMENT IMPROVEMENT MODELS')
    print('*' * 50)
    # predict_diff_score_w_all_admissions(df_all)
    # predict_diff_score_w_admissions_score(df_all)
    #
    # predict_discharge_score_w_all_admissions(df_all)
    # predict_discharge_score_w_admissions_score(df_all)
    #
    # predict_ptsd_healed_w_all_admissions(df_all)
    # predict_discharge_score_w_admissions_score(df_all)
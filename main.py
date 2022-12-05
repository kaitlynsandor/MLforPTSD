from data_cleaning import *
from metric_outputs import get_SI_metrics
from rf_classifier_SHI import *
from lr_SHI import *
from gbc_SHI import *
from rf_classifier_treatment_improvement import *

if __name__ == "__main__":
    # First convert the sharepoint files to csvs
    # convert_sav_to_csv('../OneDrive/KS2328/FY20X')
    # convert_sav_to_csv('../OneDrive/KS2328/FY19X')

    # Remove Unnecessary Columns just admission and discharge form and SI statistics
    df_2019 = remove_unnecessary_columns('../OneDrive/KS2328/FY19X_train.csv')
    df_2020 = remove_unnecessary_columns('../OneDrive/KS2328/FY20X_train.csv')

    # Combine the two dataframes into one with an inner join
    df_merged = pd.concat([df_2019, df_2020])
    # get_SI_metrics(df_merged)
    # print(df_merged.columns)
    # update the dataframe with the new columns: admissions_score, discharge_score, inputs omitted, response, \
    # diff_score, ptsd_healed
    df_all = calculate_PTSD_scores_add_to_and_return_new_dataframe(df_merged)
    # get_admissions_discharge_score_metrics(df_all)

    run_SHI_rf_classifier_models(df_all)
    # run_SHI_lr_models(df_all)
    # run_SHI_gbc_models(df_all)
    # run_treatment_improvement_models(df_all)
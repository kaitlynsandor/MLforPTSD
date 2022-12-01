from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.utils import resample
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,\
classification_report, roc_auc_score, f1_score, mean_absolute_error, mean_squared_error, accuracy_score

def get_features_labels(df_input, input, output, includePreSI=False):
    df = df_input[:]
    if output == 'symptom_diff':
        labels = df[['D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20',
                     'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27']].copy()
    else:
        labels = df.loc[:,output]
    if input == 'all_data_admissions':
        strings = df.columns.tolist()
        strs = strings[:]
        for string in strs[:]:
            string_arr = list(string)
            # if string == 'admissions_score':
            #     pass
            if string_arr[0] != 'A':
                strs.remove(string)
            elif int(string[1:]) == 99 and not includePreSI:
                strs.remove(string)
        remove = list(set(strings) - set(strs))
        for column in remove:
            df.pop(column)
        features = df
        return features, labels
    else:
        # features = df[['admissions_score', 'inputs_omitted']].copy()
        features = df[['admissions_score']].copy()
        return features, labels

def split_training_test(features, labels):
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)
    return train_features, test_features, train_labels, test_labels

def get_confidence_interval_regression(test_features, test_labels, model=None):
    iterations = 1000
    size = int(test_features.shape[0] * 0.50)
    accuracy = list()

    for i in range(iterations):
        test_features_resample, test_labels_resample = resample(test_features, test_labels, n_samples=size)
        if model == 'baseline':
            y_pred_test = get_baseline_predictions(test_features_resample)
        else:
            y_pred_test = model.predict(test_features_resample)
        accuracy.append(mean_squared_error(test_labels_resample, y_pred_test))

    confidence_interval = 0.95

    p = ((1.0 - confidence_interval)/2.0) * 100 # space on bottom end
    acc_lower = max(0.0, np.percentile(accuracy, p))
    acc_mean = np.mean(accuracy) * 1.0

    p = (confidence_interval + ((1.0 - confidence_interval) / 2.0)) * 100 # space at top end of interval
    acc_upper = min(1.0, np.percentile(accuracy, p)) * 100

    stats_dict = {}
    print()
    print('ACCURACY mean %.3f%% with %.1f%% confidence interval %.3f%% and %.3f%%' % \
          (acc_mean, confidence_interval, acc_lower, acc_upper * 100))
    return stats_dict

def get_errors_and_accuracy_regression(predictions, test_labels, name):
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:' + name + ':', round(float(errors.mean()), 2), 'points')
    print('MAE: ', mean_absolute_error(predictions, test_labels))
    print('MSE: ', mean_squared_error(predictions, test_labels))
    # if rf is not None:
    #     get_stats_classification(test_features, test_labels, model=rf)
    # else:
    #     print('baseline')
    #     get_stats_classification(test_features, test_labels, model='baseline')

def get_baseline_predictions(test_features):
    predictions = []
    df = test_features[:]
    cluster_d_items = ['A34', 'A35', 'A36', 'A37', 'A38', 'A39' , 'A40']
    for index, row in df.iterrows():
        cluster_score = 0
        for cluster_d in cluster_d_items:
            cluster_score += row[cluster_d]
        if cluster_score < 8:
            predictions.append(0.0)
        elif cluster_score < 14:
            predictions.append(1.0)
        elif cluster_score < 21:
            predictions.append(2.0)
        else:
            predictions.append(3.0)

    df_return = np.asarray(predictions)
    return df_return

def get_stats_classification(X_test, y_test, model=None):
    n_iterations = 1000
    n_size = int(X_test.shape[0] * 0.50)
    auc = list()
    acc = list()
    sensitivity = list()
    specificity = list()
    ppv = list()
    npv = list()

    for i in range(n_iterations):
        X_test_samples, y_test_samples = resample(X_test, y_test, n_samples=n_size)
        if model =='baseline':
            y_pred_test = get_baseline_predictions(X_test_samples)
        else:
            y_pred_test = model.predict(X_test_samples)
        if model == "osvm":
            y_pred_test[y_pred_test == -1] = 0
            y_pred_test[y_pred_test == 1] = -1
            y_pred_test[y_pred_test == 0] = 1
        # if model != 'baseline':
        #     auc.append(roc_auc_score(y_test_samples, y_pred_test))
        acc.append(accuracy_score(y_test_samples, y_pred_test))
        # cr = classification_report(y_test_samples.ravel(), y_pred_test, output_dict=True,
        #                            target_names=['0', '1', '2', '3'])
        # sensitivity.append(cr['Positive Response']['recall'])
        # specificity.append(cr['Negative Response']['recall'])
        # ppv.append(cr['Positive Response']['precision'])
        # npv.append(cr['Negative Response']['precision'])

    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    # auc_lower = max(0.0, np.percentile(auc, p))
    acc_lower = max(0.0, np.percentile(acc, p))
    # sensitivity_lower = max(0.0, np.percentile(sensitivity, p))
    # specificity_lower = max(0.0, np.percentile(specificity, p))
    # ppv_lower = max(0.0, np.percentile(ppv, p))
    # npv_lower = max(0.0, np.percentile(npv, p))
    # auc_mean = np.mean(auc)
    acc_mean = np.mean(acc)
    # sensitivity_mean = np.mean(sensitivity)
    # specificity_mean = np.mean(specificity)
    # ppv_mean = np.mean(ppv)
    # npv_mean = np.mean(npv)
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    # auc_upper = min(1.0, np.percentile(auc, p))
    acc_upper = min(1.0, np.percentile(acc, p))
    # sensitivity_upper = min(1.0, np.percentile(sensitivity, p))
    # specificity_upper = min(1.0, np.percentile(specificity, p))
    # ppv_upper = min(1.0, np.percentile(ppv, p))
    # npv_upper = min(1.0, np.percentile(npv, p))

    stats_dict = {}
    print('ACC mean %.1f%% with %.1f%% confidence interval %.1f%% and %.1f%%' % \
          (acc_mean * 100, alpha * 100, acc_lower * 100, acc_upper * 100))
    # print('AUC mean %.1f with %.1f%% confidence interval %.1f and %.1f' % \
    #       (auc_mean * 100, alpha * 100, auc_lower * 100, auc_upper * 100))
    # print('Sensitivity mean %.1f with %.1f%% confidence interval %.1f and %.1f' % \
    #       (sensitivity_mean * 100, alpha * 100, sensitivity_lower * 100, sensitivity_upper * 100))
    # print('Specificity mean %.1f with %.1f%% confidence interval %.1f and %.1f' % \
    #       (specificity_mean * 100, alpha * 100, specificity_lower * 100, specificity_upper * 100))
    # print('PPV mean %.1f with %.1f%% confidence interval %.1f and %.1f' % \
    #       (ppv_mean * 100, alpha * 100, ppv_lower * 100, ppv_upper * 100))
    # print('NPV mean %.1f with %.1f%% confidence interval %.1f and %.1f' % \
    #       (npv_mean * 100, alpha * 100, npv_lower * 100, npv_upper * 100))

    # stats_dict['acc'] = (acc_mean * 100, acc_lower * 100, acc_upper * 100)
    # stats_dict['auc'] = (auc_mean * 100, auc_lower * 100, auc_upper * 100)
    # stats_dict['sensitivity'] = (sensitivity_mean * 100, sensitivity_lower * 100, \
    #                              sensitivity_upper * 100)
    # stats_dict['specificity'] = (specificity_mean * 100, specificity_lower * 100, \
    #                              specificity_upper * 100)
    # stats_dict['ppv'] = (ppv_mean * 100, ppv_lower * 100, ppv_upper * 100)
    # stats_dict['npv'] = (npv_mean * 100, npv_lower * 100, npv_upper * 100)

    return stats_dict

# def accuracy_score(y_test_samples, y_pred_test):
#     accuracy = r2_score(y_test_samples, y_pred_test)
#     # print(r2_score(y_test_samples, y_pred_test))
#     print(accuracy)
#     return accuracy
from sklearn.ensemble import RandomForestClassifier
from ml_helpers import *
from metric_outputs import *

def run_random_forest_model(features, labels, name, baseline=False, supress_debug=True):
    train_features, test_features, train_labels, test_labels = split_training_test(features, labels)

    if baseline:
        predictions = get_baseline_predictions(test_features)
        get_errors_and_accuracy_regression(predictions, test_labels, 'BASELINE_' + str(name))
        print()

    rf = RandomForestClassifier(n_estimators=1000, random_state=42, bootstrap=True)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    # print(predictions)

    get_errors_and_accuracy_regression(predictions, test_labels, name)
    generate_and_save_rf_importances(train_features, name, rf)

    if not supress_debug:
        output_reality = {0:0, 1:0, 2:0, 3:0}
        for value in test_labels:
            if value == 1:
                output_reality[1] += 1
            elif value == 2:
                output_reality[2] += 1
            elif value == 3:
                output_reality[3] += 1
            elif value == 0:
                output_reality[0] += 1
        print('output reality ' + str(output_reality) + ' ' + str(len(test_labels)))
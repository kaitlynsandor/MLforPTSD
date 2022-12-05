from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ml_helpers import *
from metric_outputs import *

def run_random_forest_classifier_model(features, labels, name, baseline=False, supress_debug=True):
    train_features, test_features, train_labels, test_labels = split_training_test(features, labels)

    if baseline:
        predictions = get_baseline_predictions(test_features)
        (predictions, test_labels, 'BASELINE_' + str(name))
        get_errors_and_accuracy_regression(predictions, test_labels, 'BASELINE_' + str(name))
        print()

    rf = RandomForestClassifier(n_estimators=3000, random_state=45, bootstrap=True)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    # print(predictions)

    get_errors_and_accuracy_regression(predictions, test_labels, name)
    # get_stats_classification(test_features, test_labels, rf)
    generate_and_save_rf_importances(train_features, name, rf)

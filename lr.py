from ml_helpers import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from metric_outputs import *

def run_logistical_regression_model(features, labels, name, baseline=False, supress_debug=True):
    train_features, test_features, train_labels, test_labels = split_training_test(features, labels)
    if baseline:
        print('BASELINE')
        # get_stats_classification(test_features, test_labels, model='baseline')
        print()

    rf = LogisticRegression(class_weight='balanced', random_state=50, max_iter=1000)
    rf.fit(train_features, train_labels)

    get_stats_classification(test_features, test_labels, model=rf)

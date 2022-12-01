from ml_helpers import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from metric_outputs import *
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier

def run_gradient_boosting_classifier_model(features, labels, name, baseline=False, supress_debug=True):
    train_features, test_features, train_labels, test_labels = split_training_test(features, labels)
    if baseline:
        print('BASELINE')
        # get_stats_classification(test_features, test_labels, model='baseline')
        print()

    rf = GradientBoostingClassifier(learning_rate=0.009, max_depth=8,
                                 n_estimators=1000, subsample=0.8, random_state=42)
    rf.fit(train_features, train_labels)
    get_stats_classification(test_features, test_labels, model=rf)
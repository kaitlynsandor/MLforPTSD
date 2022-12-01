import pandas as pd
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import OrderedDict
from ml_helpers import *

def generate_and_save_rf_importances(train_features, name, rf):

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=list(train_features.columns))

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('./graphs/'+name + "_feature_importances.jpg")

    print('*' * 50)

def generate_and_save_frequency_metric_graph(scores, name):
    score_dict = OrderedDict()
    for score in scores:
        if int(score) in score_dict.keys():
            score_dict[int(score)] += 1
        else:
            score_dict[int(score)] = 1

    lists = sorted(score_dict.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    plt.xticks(range(len(x)), x)
    if len(score_dict) > 10:
        fig, ax = plt.subplots()
        myLocator = mticker.MultipleLocator(4)
        ax.xaxis.set_major_locator(myLocator)
        plt.xlabel("PCL-5 Score")
        plt.ylabel("Instances")
    else:
        plt.xlabel("PHQ-9 Q9 Score")
        plt.ylabel("Instances")
        print('*' * 50)
        print('DISTRIBUTION OF DATA ' + name)
        print('*' * 50)
        total_values = sum(score_dict.values())
        print('total values ' + str(total_values) )
        arr_percents = []
        for key in score_dict.keys():
            percentage = 100 * score_dict[key] / float(total_values)
            arr_percents.append(str(key) + ':' + str(percentage))
        for per in arr_percents:
            print(per)
    plt.bar(np.array(x), y)
    plt.title(name)
    plt.savefig('./graphs/'+name+".jpg")
    plt.clf()

def get_SI_metrics(df):
    #A99/D60
    pre_treatment_scores = df.loc[:, 'A99']
    generate_and_save_frequency_metric_graph(pre_treatment_scores, 'Suicide Ideation Pre Treatment (PHQ-9:Q9)')
    post_treatment_scores = df.loc[:, 'D60']
    generate_and_save_frequency_metric_graph(post_treatment_scores, 'Suicide Ideation Post Treatment (PHQ-9:Q9)')

def get_admissions_discharge_score_metrics(df):
    admission_scores = df.loc[:, 'admissions_score']
    generate_and_save_frequency_metric_graph(admission_scores, 'PCL-5 Admission Score (Pre-Treatment)')
    discharge_scores = df.loc[:, 'discharge_score']
    generate_and_save_frequency_metric_graph(discharge_scores, 'PCL-5 Discharge Score (Post-Treatment)')



# Evaluation models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss


def metrics(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    print("Confusion Matrix:")
    print()
    # print(cm)

    df_cm = pd.DataFrame(cm, index=[i for i in range(len(cm))],
                         columns=[i for i in range(len(cm))])
    plt.figure(figsize=(15, 15))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='PuRd')
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, clf.classes_, rotation=45)
    plt.yticks(tick_marks, clf.classes_, rotation=0)
    plt.show()
    print("-----" * 5)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(classification_report(y_test, y_pred))

    print("-----" * 5)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: {0:.3f}".format(acc))

    kappa = cohen_kappa_score(y_test, y_pred)
    print("Kappa: {0:.3f}".format(kappa))

    logloss = log_loss(y_test, y_proba)
    print("Log loss: {0:.3f}".format(logloss))

    metrics_df = pd.DataFrame({"Accuracy": [acc], "Kappa": [kappa], 'log_loss': [logloss]}).round(3)
    return metrics_df, df_report


def plot_metric_per_class(metric_report, metric='precision', threshold=0.5):
    values = metric_report[metric][:-3]
    x = range(len(values))
    above_threshold = np.maximum(values - threshold, 0)
    below_threshold = np.minimum(values, threshold)

    plt.figure(figsize=(10, 4))
    plt.bar(x, below_threshold, 0.35, color="darkblue")
    plt.bar(x, above_threshold, 0.35, color="green",
            bottom=below_threshold)

    # horizontal line indicating the threshold
    plt.plot([-.5, len(values) + .5], [threshold, threshold], "k--")

    plt.xticks(ticks=np.arange(len(metric_report.index[:-3])), labels=metric_report.index[:-3], rotation=45)
    plt.ylabel(metric)
    plt.xlabel("Classes")
    plt.show()



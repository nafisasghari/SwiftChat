import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from IPython.display import Image , display
# from IPython.display import display
# import json
# import boto3


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix , classification_report, roc_auc_score, f1_score, accuracy_score, precision_score , recall_score
from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss, roc_curve

# from imblearn.under_sampling import EditedNearestNeighbours
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier

import util

df = util.read_df('sample.csv')
df.head()

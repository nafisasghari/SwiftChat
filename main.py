import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, accuracy_score, \
    precision_score, recall_score
from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss, roc_curve

# from imblearn.under_sampling import EditedNearestNeighbours

# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
import util
from util import df_with_selected_features, split_df, split_df_to_x_y, train_model_func
from sklearn.ensemble import RandomForestClassifier

msg = util.read_df('sample.csv')
print(msg.head())

# Add length of messages before cleaning
msg['InBound_length'] = msg['InBound'].apply(len)

# Clean inbound messages and convert them to vector
df = df_with_selected_features(df=msg, col_to_vec='InBound', target=['TemplateID'],
                               cat_feats='ConversationType',
                               cat_feats_name=['cart', 'live_text', 'campaigns', 'platform', 'optin_disc',
                                               ],  # 'optin_conf'
                               num_feats=['InBound_length', 'ConversationLength'], file_path=None,
                               create_vec=True, file_path_vec=None, save=False)

print(df.head(2))
print(df.shape)

# Split Data and remove duplicate rows
train, test = split_df(df, test_size=0.2, random_state=42, drop_duplicate_train=True)

X_train, y_train = split_df_to_x_y(train, target='TemplateID', specify_features=None)
X_test, y_test = split_df_to_x_y(test, target='TemplateID', specify_features=None)

# Train models:

# Random Forests:
rf_clf = RandomForestClassifier(n_jobs=-1, random_state=42)
param_grid_rf = [{'n_estimators': [100], 'max_depth': [15]}]

rf_model = train_model_func(X=X_train, y=np.ravel(y_train),
                            estimator=rf_clf, param_grid=param_grid_rf, cv=5, scoring="neg_log_loss")

rf_metrics, rf_report = util.metrics(rf_model, X_test, y_test)
print(rf_report)
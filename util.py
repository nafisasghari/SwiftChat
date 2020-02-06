# text cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import spacy
from spacy.lang.en import English

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, accuracy_score, \
    precision_score, recall_score
from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss, roc_curve

# import en_core_web_md
# nlp = en_core_web_md.load()
nlp = spacy.load('en_core_web_md')

# Create our list of punctuation marks, stopwords
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

parser = English()


# Tokenizer function
def spacy_tokenizer(sentence):
    """
    Remove punctuations and stopwords, lemmatize tokens and covert them into lowercase
    :param sentence: string
    :return: list of preprocessed tokens
    """
    mytokens = parser(sentence)

    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

    return mytokens


# Covert whole sentence to a vector
def msg_to_vector(series, file_path=None, save=True):
    """

    :param save:
    :param series: pandas series of sentences
    :param file_path: if not None, save result as pandas dataframe in file_path
    :return: Dataframe of vectors (length of input series , 300)
    """
    tokens = series.apply(spacy_tokenizer)
    inbound_clean = tokens.apply(' '.join)
    vec = inbound_clean.apply(lambda x: nlp(x).vector)

    vectors = pd.DataFrame(vec.values.tolist())
    if save:
        # try-error
        # if file_path is not None:
        vectors.to_csv(file_path, index=False)

    return vectors


def create_or_import_vectors(df, col, create=True, file_path=None, save=True):
    if create:
        vectors = msg_to_vector(series=df[col], file_path=file_path, save=save)
    else:
        vectors = pd.read_csv(file_path)
    return vectors


def df_with_selected_features(df, col_to_vec, target, cat_feats=None, cat_feats_name=None,
                              num_feats=None, file_path=None, create_vec=True,
                              file_path_vec=None, save=True):
    """

    :param save:
    :param file_path_vec:
    :param create_vec:
    :param file_path:
    :param df: DataFrame
    :param col_to_vec: DataFrame of vectors, shape: (length of input df , -1)
    :param target: list of strings, target columns names from df
    :param cat_feats: list of strings, categorical features name from df
    :param cat_feats_name: list of strings, name of categories
    :param num_feats: list of strings, numerical features name from df
    :param file_path: if not None, save result as pandas dataframe in file_path
    :return: DataFrame
    """

    df = df.reset_index()
    sentence_vector = create_or_import_vectors(df, col_to_vec, create=create_vec,
                                               file_path=file_path_vec, save=save)

    cat_f = pd.get_dummies(df[cat_feats])
    if cat_feats_name is not None:
        cat_f.columns = cat_feats_name

    df_selected_features = pd.concat([sentence_vector, cat_f, df[target + num_feats]], axis=1)
    if file_path is not None:
        df_selected_features.to_csv(file_path, index=False)
    return df_selected_features


def drop_duplicates_rows(df):
    df_without_dup = df.drop_duplicates(subset=df.columns.tolist(), keep='first')
    return df_without_dup


# Split Data
def split_df(df, test_size=0.2, random_state=None, drop_duplicate_train=False):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    if drop_duplicate_train:
        train = drop_duplicates_rows(train)

    return train, test


def split_df_to_x_y(df, target, specify_features=None):
    """

    :param df:
    :param target: str, target name
    :param specify_features: list of features name
    :return:
    """
    if specify_features is None:
        features = [feat for feat in df.columns if feat != target]
    else:
        features = specify_features
    x = df[features]
    y = df[target]
    return x, y


def train_model_func(X, y, estimator, param_grid, cv=10, scoring='neg_log_loss'):
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring)
    grid_search.fit(X, y)
    print("best parameters: ", grid_search.best_params_)
    print("-------" * 10)
    best_model = grid_search.best_estimator_
    print("best_model: ", best_model)

    return best_model


def metrics(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

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
    print(f"Accuracy: {acc:.3f}")

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"Kappa: {kappa:.3f}".format())

    logloss = log_loss(y_test, y_proba)
    print(f"Log loss: {logloss:.3f}".format())

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


def read_df(file_path, cleaning=True, target='TemplateID'):
    df = pd.read_csv(file_path)
    if cleaning:
        df.dropna(subset=[target], inplace=True)
    return df

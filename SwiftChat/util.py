# some functions to work with dataframe
import pandas as pd

from nlp_preprocessing import create_or_import_vectors


def read_df(file_path: str, drop_na_target: bool = True, target: str = 'TemplateID') -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if drop_na_target:
        df.dropna(subset=[target], inplace=True)
    return df


def df_with_selected_features(df, col_to_vec, nlp, target, cat_feats=None, cat_feats_name=None,
                              num_feats=None, file_path=None, create_vec=True,
                              file_path_vec=None, save=True):
    """

    :param nlp:
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
    sentence_vector = create_or_import_vectors(df, col_to_vec, nlp, create=create_vec,
                                               file_path=file_path_vec, save=save)

    cat_f = pd.get_dummies(df[cat_feats])
    if cat_feats_name is not None:
        cat_f.columns = cat_feats_name

    df_selected_features = pd.concat([df[target + num_feats], sentence_vector, cat_f], axis=1)

    if file_path is not None:
        df_selected_features.to_csv(file_path, index=False)
    return df_selected_features


def drop_duplicates_rows(df: pd.DataFrame) -> pd.DataFrame:
    df_without_dup = df.drop_duplicates(subset=df.columns.tolist(), keep='first')
    return df_without_dup

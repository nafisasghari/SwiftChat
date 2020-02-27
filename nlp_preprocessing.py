# text preprocessing
import pandas as pd
import numpy as np
import string
import spacy
from spacy.lang.en import English
# import en_core_web_md

# Create our list of punctuation marks, stopwords
punctuations = string.punctuation
# stop_words = spacy.lang.en.stop_words.STOP_WORDS
stop_words_to_keep = ['no', "n't", "not", "none", "nothing", "never", "anything", "do", "did", "can", "cannot"]

stop_words = [stop for stop in spacy.lang.en.stop_words.STOP_WORDS if stop not in stop_words_to_keep]

parser = English()


# Tokenizer function
def spacy_tokenizer(sentence):
    """
    Remove punctuations and stopwords, lemmatize tokens and covert them into lowercase
    :param sentence: string
    :return: list of preprocessed tokens
    """
    mytokens = parser(sentence)
    #     if stop_words_to_keep is not None:
    #           stops = [stop for stop in stop_words if stop not in stop_words_to_keep]
    #     else:
    #         stops = stop_words
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

    return mytokens


# Covert whole sentence to a vector
def msg_to_vector(series, nlp, file_path=None, save=True):
    """

    :param nlp:
    :param save:
    :param series: pandas series of sentences
    :param file_path: if not None, save result as pandas dataframe in file_path
    :return: Dataframe of vectors (length of input series , 300)
    """
    tokens = series.apply(spacy_tokenizer)
    inbound_clean = tokens.apply(' '.join)
    vec = inbound_clean.apply(lambda x: nlp(x).vector)

    if save:
        try:
            vec.to_csv(file_path, index=False, header=False)
        except OSError as err:
            print("OS error: {0}".format(err))

    return vec


def create_or_import_vectors(df, col, nlp, create=True, file_path=None, save=True):
    if create:
        vec = msg_to_vector(series=df[col], nlp=nlp, file_path=file_path, save=save)
    else:
        vec = pd.read_csv(file_path, header=None)

    vectors = pd.DataFrame(vec.values.tolist())
    return vectors


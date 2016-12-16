import os
from pickle_load import pickle_load
from gather_and_save_vectors import gather_and_save_vectors


def processdata(path=os.getcwd(), mode='train'):
    """In training mode loads information from a file to enable building a model,
    in testing mode loads the presumably gathered model.
    :param path: the path to the location of the data to be processed.
    :param mode: the mode to use.
        train takes a long time as it needs to train the model.
        test assumes the model already exists.
    """
    # ind_vector: raw counts of ngrams occurring in each industry.
    # example:
    # ('consultant', 'consultant'): 112,
    # ('business', 'analyst'): 106,
    # ('operations', 'manager'): 98,
    # ('network', 'network'): 97,
    # ('director', 'of'): 93,
    # ('account', 'director'): 86,
    # ('co', 'ordinator'): 82,
    # ('product', 'product'): 79,
    # ('it', 'it'): 77,
    # ('programme', 'manager'): 77
    ind_vectors = pickle_load('ind_vectors.data')
    i_features = pickle_load('i_features.data')
    if mode == 'train':
        if not (ind_vectors and i_features):  # False if the files weren't there.
            ind_vectors, i_features = gather_and_save_vectors(path)
        else:
            ind_vectors, i_features = gather_and_save_vectors(path, ind_vectors, i_features)
    elif mode != 'test':
        print('Usage: mode parameter should be either "train" or "test".')
        return None
    return ind_vectors, i_features

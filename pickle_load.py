import pickle
import os
'''used in processdata'''


def pickle_load(path):
    """Loads a pickled data file if the file is there, returns False otherwise.
    :param path: the path and filename to a file"""
    if os.path.isfile(path):
        file = pickle.load(open(path, "rb"))
        return file
    else:
        return False

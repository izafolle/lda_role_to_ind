import csv
import collections
import pickle

from countize import countize
'''used in processdata'''


def gather_and_save_vectors(path, words_vec=collections.defaultdict(list), features=[]):
    """Gathers and pickles vectors from a given csv file.
    :param path: path and filename to pickle into.
    :param words_vec: vector to be pickled.
    :param features:
    """
    with open(path, 'rt', encoding='mac_roman') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        for row in csvreader:
            words_vec, features = countize(row[3], row[2], words_vec, features)  # contains the role and the industry

            try:
                # contains the company name and the industry but not always present
                words_vec, features = countize(row[6], row[2], words_vec, features)
            except:
                pass
    pickle.dump(words_vec, open("ind_vectors.data", "wb"))
    pickle.dump(features, open("i_features.data", "wb"))
    return words_vec, features

import collections


def featurize(vector, features):
    """Gather the features from a vector. Essentially transposing the original vector
    from n-grams as features to industies as features for each n-gram.
    A list of all possible ngrams is made and then the count of these ngrams in each each industry is recorded.
    :param vector:  ngram counts per industry
    :param features: list of all possible ngrams
    """
    # TODO only really need the vector to be input as all possible ngrams should be in those.  But as they are already calculated they are fed in.
    dictionary = collections.defaultdict(lambda: 0)
    for feature in iter(set(features)):
        # populates vectors with zeroes where there's no value in an industry for an n-gram.
        dictionary[feature] = [vector[key][feature] if feature in vector[key] else 0 for key in vector]
    return dictionary

def feature_vector(features, vector):
    """Uses featurize function on a vector.
    :param features: list of all possible ngrams
    :param vector:  ngram counts per industry
    """
    clean_features = set(features)  # dedupe the features
    new_features_vector = featurize(vector, clean_features)
    return new_features_vector

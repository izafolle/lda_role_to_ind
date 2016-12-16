def irrelevant_features(features, threshold=0.9):
    """Remove clutter from features, everything that appears too many times in each industry.
    :param  features: is the list of n-grams with the frequency for each industry
    :param  threshold: ignore n-grams that appear in more than threshold * 100% of all the industries then it is ignored
    """
    if (threshold > 1.) or (threshold < 0.):
        print ("irrelevant_feature threshold outside range 0. to 1., setting to 0.9 instead")
        threshold = 0.9
    irrelevant = []
    for vec in set(features):
        if (1 - (features[vec].count(0)/len(features[vec]))) >= threshold:
            # for each n-gram, look at the number of industries in which it appears.
            # If it occurs in more than threshold (0.9 = 90%) of all the industries then it is ignored
            irrelevant.append(vec)
    return irrelevant

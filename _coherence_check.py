import itertools
'''not used'''


def coherence_check(n, transformed, labels, position=None, unique=True):
    """Prints and returns the (unique if set to True) features inferred from the top n features for each topic
    given the transformed observations, and labels.
    If position is entered then unique is ignored.
    :param n: The n number of top features in a topic.
    :param transformed:
    :param labels:
    :param position: index position topic number
    :param unique: should topic only contain features unique to this topic?
    """
    topics_predicted = list(zip(labels, transformed))
    all_top_features = []
    updated_topfeatures = []
    if position is None:
        for i in range(len(transformed[0])):
            top = [label[0] for label in sorted(topics_predicted, key=lambda x: x[1][i], reverse=True)][:n]
            all_top_features.append(
                ([label[0] for label in sorted(topics_predicted, key=lambda x: x[1][i], reverse=True)][:n]))
            if unique == False:
                return all_top_features
    else:
        topfeatures = [label[0] for label in sorted(topics_predicted, key=lambda x: x[1][position], reverse=True)][:n]
        print(topfeatures)
        return topfeatures
    merged = list(itertools.chain.from_iterable(all_top_features))
    unique = [x for x in merged if merged.count(x) == 1]
    for topic in all_top_features:
        updated_topfeatures.append([gram for gram in topic if gram in unique])
    for idx, top in enumerate(updated_topfeatures):
        print(idx)
        print(top)
        print()
    return updated_topfeatures

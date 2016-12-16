'''used in guess_topic'''

def set_weight(term, irrelevant, indtf_features):  # def set_weight(term, irrelevant):
    """Sets weight of each term depending on ocurrance. 0.2 is decided completely heuristically.
    Features are deemed irrelevant if they appear many times in each industry and
    aren't associated with one in particular.
    :param term:
    :param irrelevant: """
    if term in irrelevant:
        return (0.2 * max([x/sum(indtf_features[term]) for x in indtf_features[term]]))
    elif isinstance(term, tuple):
        return len(term)
    else:
        return 1

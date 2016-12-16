'''used in guess_topic'''

def n_grammize(role):
    """Returns n-grams of a given role. Role can be a string or a tuple of strings.
    (In this latter case they already are assumed to be stripped of unnecessary
    punctuation, certain non-alphanumeric characters and capitalisation).
    :param role: role name to be processed."""
    ngrams = []
    if isinstance(role, str):
        role = role.lower()
        role = role.split()
    if len(role) > 2:
        for i in range(2, len(role)):
            ngrams.append((role[i-2], role[i-1], role[i]))
    if len(role) > 1:
            for i in range(1, len(role)):
                ngrams.append((role[i-1], role[i]))
    for i in range(len(role)):
        ngrams.append(role[i])
    return ngrams

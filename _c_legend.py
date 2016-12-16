from guess_topic import guess_topic
'''not used'''


def c_legend(collapse, lda, features_vec, irrelevant):
    legend = {}
    for key in collapse:
        topicindex = guess_topic(lda, collapse[key], features_vec, irrelevant, verbose=False)
        legend[topicindex] = key
    return legend

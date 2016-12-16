from numpy import array
import numpy
from clean import clean
from n_grammize import n_grammize
from set_weight import set_weight


def guess_topic(lda, query, features_vec, irrelevant, verbose=True):
    """Transforms a short 'document' according to a trained model and weights to infer the topic.
    Each topic is and index.
    Query can be a string, tuple of string or a list of tuple of strings.
    Verbose = False will return only the numeric index.
    Otherwise the topics can be interpreted by a legend in form of a dictionary."""
    query_doc = []
    doc_topic = []
    topic_most_pr = None
    if isinstance(query, str):
        query = clean(query)
        query = n_grammize(query)
        for term in query:
            weight = set_weight(term, irrelevant, features_vec)
            if term in features_vec:
                query_doc.append(weight * array(features_vec[term]))
    elif isinstance(query, tuple):
        if query in features_vec:
            weight = set_weight(query, irrelevant, features_vec)
            query_doc.append(weight * array(features_vec[query]))
    elif isinstance(query, list):
        for term in query:
            weight = set_weight(term, irrelevant, features_vec)
            if term in features_vec:
                query_doc.append(weight * array(features_vec[term]))
    X = array(query_doc)
    if len(X)==1:
        X = X.reshape(1,-1)
    if len(X)==0:
        return topic_most_pr
    doc_topic = lda.transform(X)
    sum_topics = numpy.zeros(len(doc_topic[0]))
    for i in range(len(doc_topic)):
        sum_topics = sum_topics + doc_topic[i]
    topic_most_pr = sum_topics.argmax()
    if verbose == True:
        if topic_most_pr in legend:
            return legend[topic_most_pr]
        else:
            return topic_most_pr
    else:
        return topic_most_pr

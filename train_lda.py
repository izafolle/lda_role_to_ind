import pickle
from sklearn.decomposition import LatentDirichletAllocation


def train_lda(obs, n_topics=42, max_iter=100, doc_topic_prior=0.0001, learning_method='online', learning_offset=50.,
              topic_word_prior=0.001, random_state=0):
    """Train a model using the raw counts.
    LDA expects raw counts, can be used with tfidf scores but the theoretical basis for that isn't well defined.
    :param  obs: observations to fit the lda model to.
    :param  n_topics: LDA parameter for number of topics to create
    :param  max_iter: LDA parameter for number of iterations to perform
    :param  doc_topic_prior: LDA parameter
    :param  learning_method: LDA parameter
    :param  learning_offset: LDA parameter
    :param  topic_word_prior: LDA parameter
    :param  random_state: LDA parameter
    """
    print('Training LDA model...')
    lda = LatentDirichletAllocation(n_topics,
                                    max_iter,
                                    doc_topic_prior,
                                    learning_method,
                                    learning_offset,
                                    topic_word_prior,
                                    random_state)
    lda.fit_transform(obs)
    pickle.dump(lda, open("ilda.data", "wb"))
    return lda

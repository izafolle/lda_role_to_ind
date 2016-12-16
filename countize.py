from clean import clean
'''used in gather_and_save_vectors'''

def countize(word, ind, count_words, features):
    """Counts trigrams, bigrams and unigrams for words.
    This function appends them straight to the data structures they will be used in.
    This populates features and raw counts for each industry.
    :param word:  A string containing eg the name of a role or company, etc.
    :param ind: A the name of an industry
    :param count_words: dictionary of terms already seen by industry
    :param features: list of all possible ngrams (being built up in this function)
    """
    word = clean(word)
    word = word.split()
    if len(word) > 1:
        for i in range(1, len(word)):
            bigram = (word[i-1], word[i])
            count_words[ind].append(bigram)
            features.append(bigram)
    if len(word) > 2:
        for i in range(2, len(word)):
            trigram = (word[i-2], word[i-1], word[i])
            count_words[ind].append(trigram)
            features.append(trigram)
    for i in range(len(word)):
            unigram = word[i]
            count_words[ind].append((unigram))
            features.append((unigram))
    return count_words, features

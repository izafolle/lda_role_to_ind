import re
'''used in countize, get_mostcommon, guess_topic'''

def clean(word):
    """Removes stopwords and some non-alphanumeric characters that are deemed irrelevant for our purposes.
    :param word:  text to be cleaned
    """
    word = word.lower()
    stopwords = ['of', 'and', 'to', 'at', 'in', '@']
    # TODO list of stopwords should be input from config
    word = re.sub(r'[&/ -()|@,\]\[]+', ' ', word)

    for stopword in stopwords:
        pattern = r'\b' + stopword + r'\b'
        pattern = re.compile(pattern)
        word = re.sub(pattern, '', word)
    word = re.sub(r'\s\s+', ' ', word)
    return word.strip()

#
# if __name__ == '__main__':
#     text = "abc\def"
#     cleaned_text = clean(text)
#     print("stop")

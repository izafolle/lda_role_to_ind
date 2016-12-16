from guess_topic import guess_topic
from _get_mostcommon import get_mostcommon
from irrelevant_features import irrelevant_features
'''not used'''

def output(query, lda, features):
    """Outputs results for a query"""
    roles = get_mostcommon(path, 5000)
    all_roles = len(roles)
    irrelevant = irrelevant_features(features)
    # with open("guesses.txt", "w") as text_file:
    #     text_file.write('role:')
    #     text_file.write('\t')
    #     text_file.write("guess: ")
    #     text_file.write('\t')
    #     text_file.write("smatch: ")
    #     text_file.write('\n')
    for query in roles:
        # text_file.write(str(query))
        # text_file.write('\t')
        guess = guess_topic(ilda, query, features, irrelevant)
        # smatch = try_normaliser(query)
        # if guess != smatch:
        #     diff += 1
        print(query)
        #     print(guess, '\t' , smatch )
        print(guess)
        print()
        # text_file.write(str(guess))
        # text_file.write('\t')
        # text_file.write(str(smatch))
        # print('guess: ', str(guess), '\n')
        # print('smatch: ', str(smatch))
        # text_file.write('\t')
        # text_file.write(str(smatch))
        # text_file.write('\n')
        # text_file.write('\n')

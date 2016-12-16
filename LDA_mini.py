import pickle
import os


from processdata import processdata  # uses: pickle_load, gather_and_save_vectors (uses: countize(has test) # uses: clean(has test))
from count_collapse import count_collapse  # contains: declutter (ALL TEST OK)
from feature_vector import feature_vector  # uses: featurize
from guess_topic import guess_topic  # uses: n_grammize, clean(has test), set_weight
from irrelevant_features import irrelevant_features  # (has test)
from train_lda import train_lda
# # from tfidf_vectorizer import tfidf_vectorizer              '''commented out'''
# from c_legend import c_legend                                '''not used'''
# from construct_legend import construct_legend                '''not used'''
# from coherence_check import coherence_check                  '''not used'''
# from output import output                                    '''not used'''
# from get_mostcommon import get_mostcommon                    '''used in output'''  # uses: clean
# from try_normalizer import try_normalizer                    '''not used'''

if __name__ == "__main__":
    # Usage:
    # first command line argument is the directory with the csv files under the cwd,
    # second is keywords 'test'/'train'.
    dirname = 'testdata'  # sys.argv[1]  # where is the data
    legend = {}
    train_test = 'test'  # #sys.argv[2]  # if a model is already trained then use 'test' otherwise use 'train'
    if train_test == 'train':
        for filename in os.listdir(dirname):
            if not filename.startswith('.'):
                print('Currently on file: %s' % filename)
                ind_vectors, i_features = processdata((os.getcwd() + '/' + dirname + '/' + filename), 'train')
        industry_words = count_collapse(ind_vectors, declutter_limit=2)
        indtf_features = feature_vector(i_features, industry_words)
        indtf_samples = [indtf_features[feature] for feature in indtf_features if sum(indtf_features[feature]) > 0]
        indtf_labels = [feature for feature in indtf_features if sum(indtf_features[feature]) > 0]
        ilda = train_lda(indtf_samples)
    elif train_test == 'test':
        ind_vectors, i_features = processdata(mode='test')
        industry_words = count_collapse(ind_vectors, declutter_limit=2)
        indtf_features = feature_vector(i_features, industry_words)
        indtf_samples = [indtf_features[feature] for feature in indtf_features if sum(indtf_features[feature]) > 0]
        indtf_labels = [feature for feature in indtf_features if sum(indtf_features[feature]) > 0]
        ilda = pickle.load(open("ilda.data", "rb"))
        irrelevant = irrelevant_features(indtf_features)
        transformed = ilda.transform(indtf_samples)

        # to get sample output:
        print(guess_topic(lda=ilda, query="project manager", features_vec=indtf_features, irrelevant=irrelevant, verbose=True))
    else:
        print('Usage: location of directory, ')
#features_vector = pickle.load(open("features_vector.data", "rb"))
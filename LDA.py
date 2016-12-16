import csv
import collections
import pickle
import os 
from sklearn.decomposition import LatentDirichletAllocation
import pycurl
from io import BytesIO as BytesIO
#import xmltodict
import itertools
from numpy import array
import numpy
import re
import sys

def pickle_load(path):
    """Loads a pickled data file if the file is there, returns False otherwise.
    :param path: the path and filename to a file"""
    if os.path.isfile(path):
        file = pickle.load(open(path, "rb"))
        return file
    else: 
        return False

def processdata(path = os.getcwd(), mode = 'train'):
    """In training mode loads information from a file to enable building a model,
    in testing mode loads the presumably gathered model.
    :param path: the path to the location of the data to be processed.
    :param mode: the mode to use. train takes a long time as it needs to train the model.  Test assumes the model already exists.
    """
    # ind_vector: raw counts of ngrams occurring in each industry.
    # example: ('consultant', 'consultant'): 112, ('business', 'analyst'): 106, ('operations', 'manager'): 98, ('network', 'network'): 97, ('director', 'of'): 93, ('account', 'director'): 86, ('co', 'ordinator'): 82, ('product', 'product'): 79, ('it', 'it'): 77, ('programme', 'manager'): 77
    ind_vectors = pickle_load('ind_vectors.data')
    i_features = pickle_load('i_features.data')
    if mode == 'train':
        if not (ind_vectors and i_features): # False if the files weren't there.
            ind_vectors, i_features = gather_and_save_vectors(path)
        else: 
            ind_vectors, i_features = gather_and_save_vectors(path,ind_vectors,i_features)
    elif mode != 'test':
        print('Usage: mode parameter should be either "train" or "test".')
        return None
    return ind_vectors, i_features

def featurize(vector,features):
    """Gather the features from a vector. Essentially transposing the original vector
    from n-grams as features to industies as features for each n-gram.
    A list of all possible ngrams is made and then the count of these ngrams in each each industry is recorded.
    :param vector:  ngram counts per industry
    :param features: list of all possible ngrams
    """
    # TODO only really need the vector to be input as all possible ngrams should be in those.  But as they are already calculated they are fed in.
    dictionary = collections.defaultdict(lambda:0)
    for feature in iter(set(features)):
        dictionary[feature] = [vector[key][feature] if feature in vector[key] else 0 for key in vector] #populates vectors with zeroes where there's no value in an industry for an n-gram.
    return dictionary

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
    if len(word)>1:
        for i in range(1,len(word)):
            bigram = (word[i-1],word[i])
            count_words[ind].append(bigram)
            features.append(bigram)
    if len(word)>2:
        for i in range(2,len(word)):
            trigram = (word[i-2],word[i-1], word[i])
            count_words[ind].append(trigram)
            features.append(trigram)
    for i in range(len(word)):
            unigram = word[i]
            count_words[ind].append((unigram))
            features.append((unigram))
    print(count_words)
    print(features)
    return count_words, features

def count_collapse(count_words):
    """Collapse industries from data into stepweb industries.
    :param count_words: dictionary of terms already seen by industry"""
    collapse = {'Accountancy':['Accounting', 'Accountancy'],
                'Administration':['Government Administration','Secretarial & Admin.', 'Secretarial & Administration','Secretarial, PAs, Administration', 'Secretarial & Administrative'],
                'Advertising':[],
                'Animal Care':['Veterinary'],
                'Arts and Entertainment':['Music', 'Computer Games', 'Fine Art', 'Arts and Crafts', 'Photography', 'Motion Pictures and Film', 'Performing Arts', 'Entertainment', 'Events Services'],
                'Banking':['Banking', 'Investment Banking', 'Venture Capital & Private Equity'],
                'Catering':['Food & Beverages', 'Food Production', 'Hospitality', 'Restaurants', 'Hotel & Catering', 'Travel, Catering & Hospitality', 'Wine and Spirits', 'Catering & Hospitality', 'Hospitality & Leisure'],
                'Cleaning':[],
                'Construction':['Construction'],
                'Consulting':['Management Consulting', 'Consultancy'],
                'Customer Service':['Customer Service', 'Customer Services'],
                'Design':['Design', 'Graphic Design', 'Fashion & Design'],
                'Education':['Higher Education', 'Primary/Secondary Education', 'Education', 'Libraries', 'E-Learning', 'Professional Training & Coaching', 'Training'],
                'Engineering':['Aerospace','Engineering', 'Rail Engineers', 'Computer Hardware', 'Electronics', 'Mechanical or Industrial Engineering', 'Civil Engineering', 'Semiconductors', 'Architecture & Planning'],
                'Farming and Agriculture':['Farming', 'Ranching', 'Agriculture, Fishing, Forestry'],
                'Finance':['Banking & Finance', 'Financial Services'],
                'Health':['Medical Practice', 'Mental Health Care', 'Hospital & Health Care', 'Alternative Medicine', 'Health', 'Medical Devices', 'Health, Nursing', 'Health/Healthcare/Social care'],
                'Human Resources':['Human Resources', 'Recruitment', 'Staffing and Recruiting'],
                'Insurance':['Insurance', 'Insurance & Financial Services'],
                'IT':['Program Development','IT & Internet', 'Computer Software', 'Information Tech.', 'Computer & Network Security', 'Information Technology and Services', 'Internet', 'Wireless', 'Information Technology', 'Information Services', 'Computer Networking', 'Finance IT'],
                'Legal':['Law Practice', 'Legal', 'Legal Services', 'Judiciary', 'Alternative Dispute Resolution', 'Legislative Office'],
                'Logistics':['Transport, Logistics','Logistics','Import and Export','Distribution', 'Warehousing', 'Logistics & Transport', 'Package/Freight Delivery', 'Logistics and Supply Chain', 'Transportation/Trucking/Railroad'],'Manufacturing':['Manufacturing', 'Electrical/Electronic Manufacturing', 'Production & Ops.', 'Production & Operations', 'Railroad Manufacture', 'Automotive'],'Management':['Management & Exec.', 'Management & Executive', 'Education Management'], 'Marketing':['Marketing and Advertising', 'Marketing', 'Finance Marketing', 'Client Side', 'Market Research'],'Media':['Media', 'Broadcast Media', 'Media, New Media, Creative', 'Online Media', 'Media Production', 'Writing and Editing', 'Newspapers', 'Animation', 'Publishing'],'Military':['Defence', 'Military', 'Defense & Space', 'Maritime'],'Policing':['Public Policy', 'Think Tanks', 'Political Organization', 'Government Relations', 'International Affairs'],'PR':['Public Relations and Communications'],'Property':['Property','Real Estate', 'Commercial Real Estate'],'Public Sector':['Public Sector'],'Retail':['Wholesale', 'Retail', 'Supermarkets', 'Retail, Wholesale'],'Sales':['Sales'],'Science':['Scientific', 'Nanotechnology', 'Research', 'Biotechnology', 'Chemicals', 'Pharmaceutical & Biotechnology', 'Pharmaceuticals', 'Science'],'Security':['Security', 'Security and Investigations'],'Skilled Trades':['Skilled Trades', 'Dairy', 'Shipbuilding', 'Fishery'],'Social Care':['Individual & Family Services', 'Social Services'],'Sport and Fitness':['Health, Wellness and Fitness', 'Sports', 'Recreational Facilities and Services'],'Third Sector':['Not For Profit, Charities','Philanthropy', 'Charity & Voluntary Work', 'Fund-Raising', 'Charity & Voluntary Work', 'Nonprofit Organization Management', 'Religious Institutions'],'Travel':['Leisure, Travel & Tourism', 'Travel & Hospitality', 'Travel, Leisure, Tourism']}
    new_cw = collections.defaultdict(collections.Counter)
    upd_cw = collections.defaultdict(collections.Counter)
    for key in count_words: #transforms a list of ngrams from the data into a count of those same ngrams per industry.
        ngrams = count_words[key]
        new_cw[key] = collections.Counter(ngrams) 
    new_cw = declutter(new_cw)
    for key in collapse:
        for alternative_name in collapse[key]:
            upd_cw[key].update(new_cw[alternative_name])
    return upd_cw

def gather_and_save_vectors(path, words_vec = collections.defaultdict(list), features = []):
    """Gathers and pickles vectors from a given csv file.
    :param path: path and filename to pickle into.
    :param words_vec: vector to be pickled.
    :param features:
    """
    with open(path, 'rt', encoding='mac_roman') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        for row in csvreader:
            words_vec, features = countize(row[3], row[2], words_vec, features) # contains the role and the industry

            try:
                words_vec, features = countize(row[6], row[2], words_vec, features) # contains the company name and the industry but not always present
            except:
                pass
    pickle.dump(words_vec, open("ind_vectors.data", "wb"))
    pickle.dump(features, open("i_features.data", "wb"))
    return words_vec, features


# def tfidf_vectorizer(vector, maxtf = True):
#    """Transforms a raw count vector into a tfidf vector using maxtf by default, original tfcounts otherwise"""
#     print('recomputing tfidf values for industries...')
#     start_time = time.clock()
#     tfidf_values = collections.defaultdict(dict)
#     alldocs = len(vector)
#     for key in vector:
#         for gram in vector[key]:
#             raw_tf = vector[key][gram]
#             df = len([vector[sector][gram] for sector in vector if gram in vector[sector]]) + 0.001
#             if maxtf:
#                 maxtf = max([vector[key][gram] for gram in vector[key]])
#                 maxtf_norm = 0.4 + ((1 - 0.4) * (raw_tf / maxtf))
#                 max_tfidf = maxtf_norm * math.log(alldocs / df)
#                 tfidf_values[key][gram] = max_tfidf
#             else:
#                 tfidf = raw_tf * math.log(alldocs / df)
#                 tfidf_values[key][gram] = tfidf
#     print('done.')
#     fnames = {1:'tfidf_ivector.data', 2: 'tfidf_cvector.data'}
#     filename = fnames[i]
#     pickle.dump(tfidf_values, open(filename, "wb" ))
#     print( "%.2f" % (time.clock() - start_time), "seconds")
#     return tfidf_values


def declutter(vector, lower_limit = 2):
    """Removes features with raw count values that are less than lower_limit in a vector.
    Infrequent terms in a vector are most likely to be noise.
    :param vector: A vector (industry, ngram, count) to be decluttered.
    :param lower_limit: terms with count below lower_limit are decluttered.
    """
    for key in vector:
        clutter_values = [value for value in vector[key] if vector[key][value]<lower_limit] # gather everything with a value less than two and save it in a list
        for feature in clutter_values: # remove everything in the clutter values from a dictionary
            vector[key].pop(feature,None)
    return vector

def n_grammize(role):
    """Returns n-grams of a given role. Role can be a string or a tuple of strings.
    (In this latter case they already are assumed to be stripped of unnecessary
    punctuation, certain non-alphanumeric characters and capitalisation).
    :param role: role name to be processed."""
    ngrams = []
    if isinstance(role,str):
        role = role.lower()
        role = role.split()
    if len(role)>2:
        for i in range(2, len(role)):
            ngrams.append((role[i-2], role[i-1], role[i]))
    if len(role)>1:
            for i in range(1, len(role)):
                ngrams.append((role[i-1], role[i]))
    for i in range(len(role)):
        ngrams.append(role[i])
    return ngrams


def feature_vector(features, vector):
    """Uses featurize function on a vector.
    :param features: list of all possible ngrams
    :param vector:  ngram counts per industry
    """
    clean_features = set(features) # dedupe the features
    new_features_vector = featurize(vector,clean_features)
    return new_features_vector

def clean(word):
    """Removes stopwords and some non-alphanumeric characters that are deemed irrelevant for our purposes.
    :param word:  text to be cleaned
    """
    word = word.lower()
    stopwords = ['of', 'and','to', 'at', 'in', '@']
    # TODO list of stopwords should be input from config
    word = re.sub(r'[\&/\-\(\)\|\@,\]\[]+', ' ', word)
    for stopword in stopwords:
        pattern = r'\b' + stopword + r'\b'
        pattern = re.compile(pattern)
        word = re.sub(pattern, '', word)
    word = re.sub(r'\s\s+', ' ', word)
    # TODO do we need to remove leading trailing spaces here?
    return word

def construct_legend(indtf_vector, features_vector, lda, labels):
    """Gives the interpretation for each topic.
    That is uses index of the topic (a number) to map to a business name for a topic.
    This mapping can be made from a list of unique topic_name to unique_words mapping.
    :param indtf_vector: industry term frequency vector
    :param features_vector: mapping of topic_name to unique words
    :param lda: The model to be labelled.
    :param labels: list of ngrams.
    """
    # keys = [key for key in indtf_vector]
    # most_imp = [(key,[x[0] for x in sorted(list(indtf_vector[key].items()),key=lambda x:x[1], reverse=True)[:10]]) for key in keys]
    # legend = [(guess_topic(lda, tpl[1], features_vector,verbose=False),tpl[0]) for tpl in most_imp]
    # unique_features = coherence_check(500,transformed,labels)
    # #legend = dict(legend)
    # return legend
    def construct_legend(legend={}):
        c_legend = {
            'Sport': ['fc', 'rugby', 'holistic', 'cricket', ('massage', 'therapist'), ('virgin', 'active'), 'strength',
                      ('head', 'coach'), ('strength', 'conditioning')],
            'Banking': [('barclays', 'capital'), ('barclays', 'corporate'), ('santander', 'uk'), ('banco', 'santander'),
                        ('barclays', 'investment')],
            'Travel': ['tui', ('travel', 'consultant'), ('travel', 'plc'), ('tui', 'travel', 'plc'), ('tui', 'travel'),
                       'tourism', 'tours', ('thomas', 'cook')],
            'Catering': ['inn', 'restaurants', 'sous', ('sous', 'chef'), ('chef', 'de'), 'whitbread', 'drinks',
                         'compass', ('hilton', 'worldwide'), 'cafe', 'partie', ('restaurant', 'manager')],
            'Insurance': ['brokers', ('insurance', 'brokers'), ('insurance', 'services'), 'willis', 'rsa',
                          ('insurance', 'group'), 'marsh', ('insurance', 'company'), ('account', 'handler'),
                          ('direct', 'line'), 'towergate', 'allianz'],
            'HR': ['recruiter', ('hr', 'business'), ('hr', 'business', 'partner'), ('recruitment', 'ltd'),
                   ('hr', 'advisor'), ('senior', 'recruitment'), ('hr', 'consultant'),
                   ('senior', 'recruitment', 'consultant'), 'resourcer', ('recruitment', 'manager'), ('senior', 'hr')],
            'IT': [('senior', 'software'), 'fujitsu', ('technical', 'consultant'), ('senior', 'developer'),
                   ('it', 'consultant'), 'packard', 'hewlett', ('hewlett', 'packard'),
                   ('senior', 'software', 'engineer')],
            'Engineering': [('architectural', 'assistant'), ('civil', 'engineer'), 'amey',
                            ('architectural', 'technician'), 'jacobs', ('graduate', 'engineer'),
                            ('project', 'architect'), ('architects', 'ltd'), 'nuttall', ('consulting', 'engineers'),
                            ('architectural', 'technologist'), 'halcrow', ('bam', 'nuttall'), 'geotechnical'],
            'Social Services': ['funeral', ('family', 'support'), ('social', 'work'), 'nanny', 'childminder',
                                'fostering', 'play', 'ecoclean', ('funeral', 'directors'), 'haringey', 'safeguarding',
                                ('foster', 'carer'), 'domestic', 'baby'],
            'Finance': [('financial', 'adviser'), ('independent', 'financial'), ('independent', 'financial', 'adviser'),
                        'loans', 'place', 'ifa', ('financial', 'planner'), ('place', 'wealth'),
                        ('place', 'wealth', 'management')],
            'Security': [('fire', 'security'), ('security', 'ltd'), ('security', 'consultant'), ('close', 'protection'),
                         ('security', 'systems'), ('security', 'manager'), 'adt', ('security', 'group'),
                         ('g4s', 'security'), 'secure', ('g4s', 'security', 'services')],
            'Logistics': ['dhl', ('royal', 'mail'), ('transport', 'for'), ('transport', 'for', 'london'), 'savills',
                          ('sales', 'negotiator'), 'freight', ('property', 'manager'), 'trains', 'cbre',
                          'transportation'],
            'Utilities': ['swindon', 'lowri', ('lowri', 'beck'), 'beck', ('beck', 'systems'),
                          ('lowri', 'beck', 'systems'), ('a', 'o'), ('swindon', 'borough'),
                          ('swindon', 'borough', 'council'), 'pcubed', 'db2', ('oracle', 'dba'),
                          ('beck', 'systems', 'ltd')],
            'Marketing': [('senior', 'marketing'), 'mccann', ('digital', 'marketing', 'manager'),
                          ('marketing', 'coordinator'), ('digital', 'marketing', 'executive'), ('digital', 'account'),
                          ('media', 'marketing')],
            'Accounting': ['accountants', ('chartered', 'accountants'), ('accounts', 'assistant'), 'grant', 'thornton',
                           ('grant', 'thornton'), ('tax', 'manager'), ('financial', 'accountant'), 'bdo',
                           ('chartered', 'accountant'), ('baker', 'tilly'), 'tilly', ('audit', 'senior')],
            'Legal': [('trainee', 'solicitor'), 'paralegal', 'chambers', 'lawyer', ('solicitors', 'llp'), 'barrister',
                      'litigation', ('legal', 'assistant'), 'attorney', 'dickinson', 'allen', 'eversheds',
                      ('associate', 'solicitor'), 'linklaters', ('legal', 'executive')],
            'Manufacturing': ['rover', 'jaguar', ('land', 'rover'), ('jaguar', 'land', 'rover'), ('jaguar', 'land'),
                              'automotive', 'motors', ('motor', 'company'), 'bmw', ('ford', 'motor'),
                              ('ford', 'motor', 'company'), 'mercedes', 'benz', ('mercedes', 'benz'), 'electric',
                              ('motors', 'ltd'), ('motor', 'group')],
            'Military': ['aircraft', 'selex', 'qinetiq', 'squadron', 'pilot', 'hms', 'commanding', ('selex', 'es'),
                         'es', ('marine', 'engineer'), 'lockheed', ('lockheed', 'martin'), 'joint', 'capability',
                         ('united', 'states'), 'states', 'hq', ('engineer', 'officer'), ('equipment', 'support'),
                         ('defence', 'equipment')],
            'Education': ['headteacher', 'curriculum', ('head', 'teacher'), 'grammar', 'catholic', 'form',
                          ('grammar', 'school'), ('education', 'consultant'), 'early', 'sixth', ('sixth', 'form'),
                          ('community', 'college'), ('school', 'for')],
            'Sales': ['sterling', ('solutions', 'limited'), 'assistance', ('sales', 'support'), ('uk', 'plc'),
                      ('call', 'centre'), ('manager', 'uk'), 'robins', ('director', 'uk'), 'sme', 'call',
                      ('europe', 'limited')],
            'Scientific': ['pharmaceuticals', 'astrazeneca', 'pharma', 'pfizer', ('imperial', 'college', 'london'),
                           'gsk', 'chemist', ('research', 'scientist'), 'pharmaceutical', 'chemical',
                           ('post', 'doctoral'), 'chemicals', 'chemistry'],
            'Design': [('freelance', 'graphic'), ('freelance', 'graphic', 'designer'), ('graphic', 'design'),
                       ('senior', 'designer'), ('freelance', 'designer'), ('digital', 'designer'), ('design', 'intern'),
                       'designs', ('junior', 'designer'), 'illustration', ('design', 'consultant')],
            'Media': ['books', ('producer', 'director'), ('assistant', 'producer'), ('broadcast', 'journalist'),
                      'series', ('editorial', 'assistant')],
            'Health': ['doctor', ('hospitals', 'nhs', 'foundation'), ('healthcare', 'nhs'),
                       ('hospital', 'nhs', 'foundation'), ('university', 'hospitals'), ('healthcare', 'nhs', 'trust'),
                       'gp', ('ambulance', 'service'), ('health', 'nhs'), ('teaching', 'hospitals')],
            'Policy': ['commons', ('house', 'commons'), 'parliamentary', 'political', 'parliament', 'liberal', 'labour',
                       'economist', 'conservative', 'democrats', ('liberal', 'democrats'), 'kirklees',
                       ('kirklees', 'council'), ('senior', 'policy'), ('policy', 'officer'), 'embassy', 'nations',
                       ('labour', 'party'), ('policy', 'adviser'), ('parliamentary', 'assistant')],
            'Construction': [('construction', 'manager'), 'laing', ('senior', 'quantity'),
                             ('senior', 'quantity', 'surveyor'), "o'rourke", ('laing', "o'rourke"), 'isg', 'mace',
                             'joinery', 'dixon', 'sindall', ('morgan', 'sindall'), ('willmott', 'dixon'), 'willmott',
                             'skanska', 'roofing', ('kier', 'group'), 'joiner', 'foreman'],
            'Art and Entertainment': ['musician', 'singer', 'stage', 'composer', ('freelance', 'photographer'), 'opera',
                                      'dj', 'actress', 'songwriter', 'artistic', ('sound', 'engineer'),
                                      ('stage', 'manager'), 'guitar', 'dancer', 'performer', 'musical',
                                      ('theatre', 'company')],
            'Animal Care': [('veterinary', 'nurse'), 'veterinarian', 'pdsa', ('veterinary', 'practice'),
                            ('animal', 'health'), ('laboratories', 'agency'), ('veterinary', 'laboratories'),
                            ('veterinary', 'laboratories', 'agency'), ('veterinary', 'clinic'),
                            ('health', 'veterinary', 'laboratories')],
            'PR': [('press', 'officer'), ('pr', 'manager'), ('internal', 'communications'),
                   ('corporate', 'communications'), ('pr', 'marketing'), ('pr', 'consultant'), ('media', 'relations'),
                   ('pr', 'account'), ('pr', 'intern'), ('junior', 'account'), ('pr', 'assistant'), 'publicity',
                   ('pr', 'communications')],
            'Policy': [('leeds', 'city', 'council'), ('work', 'pensions', 'dwp'), ('pensions', 'dwp'),
                       ('royal', 'borough'), ('kent', 'county'), ('kent', 'county', 'council'), 'oxfordshire',
                       ('hertfordshire', 'county'), ('hertfordshire', 'county', 'council'), ('oxfordshire', 'county'),
                       ('oxfordshire', 'county', 'council'), ('essex', 'county', 'council'), ('essex', 'county'),
                       ('hampshire', 'county')],
            'Agriculture': ['agricultural', 'farmer', 'farming', 'farmers', 'ahdb', 'gardener',
                            ('agriculture', 'horticulture', 'development'), ('agriculture', 'horticulture'),
                            ('horticulture', 'development', 'board'), ('horticulture', 'development'),
                            ('development', 'board')],
            'Retail': [('operative', 'group'), ('co', 'operative', 'group'), 'supermarkets', 'merchandising', 'q',
                       'buying', ('b', 'q'), 'sainsburys', 'wm', ('boots', 'uk'), ('supermarkets', 'plc'),
                       ('wm', 'morrison'), 'dixons', ('morrison', 'supermarkets'), ('morrison', 'supermarkets', 'plc'),
                       ('wm', 'morrison', 'supermarkets'), ('assistant', 'buyer'), 'topshop', 'look', ('new', 'look')],
            'Higher Education': ['hallam', ('sheffield', 'hallam'), ('sheffield', 'hallam', 'university'),
                                 ('hallam', 'university'), ('e', 'learning'), 'salford', ('university', 'aberdeen'),
                                 ('university', 'salford'), 'reader', ('cardiff', 'university'), 'swansea'],
            'Consulting': [('consulting', 'group'), 'growth', ('associate', 'consultant'), ('programme', 'director'),
                           ('consulting', 'limited'), ('pa', 'consulting'), ('consultancy', 'ltd'),
                           ('pa', 'consulting', 'group'), ('change', 'manager'), ('management', 'consulting')]}
        for key in c_legend:
            index = guess_topic(ilda, c_legend[key], indtf_features, irrelevant, verbose=False)
            legend[index] = key
        return legend

def coherence_check(n, transformed, labels, position = None, unique = True):
    """Prints and returns the (unique if set to True) features inferred from the top n features for each topic
    given the transformed observations, and labels.
    If position is entered then unique is ignored.
    :param n: The n number of top features in a topic.
    :param transformed:
    :param labels:
    :param position: index position topic number
    :param unique: should topic only contain features unique to this topic?
    """
    topics_predicted = list(zip(labels,transformed))
    all_top_features = []
    updated_topfeatures = []
    if position == None:
        for i in range(len(transformed[0])):
            top = [label[0] for label in sorted(topics_predicted, key = lambda x: x[1][i], reverse = True)][:n]
            all_top_features.append(([label[0] for label in sorted(topics_predicted, key = lambda x: x[1][i], reverse = True)][:n]))
            if unique == False:
                return all_top_features
    else:
        topfeatures = [label[0] for label in sorted(topics_predicted, key = lambda x: x[1][position], reverse = True)][:n]
        print(topfeatures)
        return topfeatures
    merged = list(itertools.chain.from_iterable(all_top_features))
    unique = [x for x in merged if merged.count(x)==1]
    for topic in all_top_features:
        updated_topfeatures.append([gram for gram in topic if gram in unique])
    for idx,top in enumerate(updated_topfeatures):
        print(idx)
        print(top)
        print()
    return updated_topfeatures

def set_weight(term, irrelevant):
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

def guess_topic(lda, query, features_vec, irrelevant, verbose=True):
    """Transforms a short 'document' according to a trained model and weights to infer the topic. Each topic is and index.
    Query can be a string, tuple of string or a list of tuple of strings. Verbose = False will return only the numeric index.
    Otherwise the topics can be interpreted by a legend in form of a dictionary."""
    query_doc = []
    doc_topic = []
    topic_most_pr = None
    if isinstance(query,str):
        query = clean(query)
        query = n_grammize(query)
        for term in query:
            weight = set_weight(term, irrelevant)
            if term in features_vec:
                query_doc.append(weight * array(features_vec[term]))
    elif isinstance(query,tuple):
        if query in features_vec:
            weight = set_weight(query, irrelevant)
            query_doc.append(weight * array(features_vec[query]))
    elif isinstance(query,list):
        for term in query:
            weight = set_weight(term, irrelevant)
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

def output(query,lda,features):
    """Outputs results for a query"""
    roles = get_mostcommon(path,5000)
    all_roles = len(roles)
    irrelevant = irrelevant_features(features)
    #with open("guesses.txt", "w") as text_file:
    #    text_file.write('role:')
    #    text_file.write('\t')
    #    text_file.write("guess: ")
    #    text_file.write('\t')
    #    text_file.write("smatch: ")
    #    text_file.write('\n')
    for query in roles:
        #text_file.write(str(query))
        #text_file.write('\t')
        guess = guess_topic(ilda,query,features, irrelevant)
        #smatch = try_normaliser(query)
        #if guess != smatch:
        #    diff += 1
        print(query)
        #    print(guess, '\t' , smatch )
        print(guess)
        print()
        #text_file.write(str(guess))
        #text_file.write('\t')
        #text_file.write(str(smatch))
        #print('guess: ', str(guess), '\n')
        #print('smatch: ', str(smatch))
        #text_file.write('\t')
        #text_file.write(str(smatch))
        #text_file.write('\n')
        #text_file.write('\n')

def try_normaliser(query_role):
    """Request the Stepweb normalised industry through the Stepweb API. Somehow the stucture of the xml
    returned by the API seems to change a lot and within a single type of structure there's plenty of variation at times."""
    print(query_role)
    prof_ind = ''
    serviceurl = "http://stepmatch-jbe-frontend.app.tjgprod.ds:4100/tools/jdnorm?query="
    urlends = "&country=uk&language=en&application=totaljobs&environment=live"
    rolename = query_role.replace(' ', '%20')
    queryurl = serviceurl + rolename + urlends
    memoryview = BytesIO()
    c = pycurl.Curl()
    c.setopt(pycurl.URL, queryurl.encode('utf-8'))
    c.setopt(c.WRITEDATA, memoryview)
    c.perform()
    c.close()
    body = memoryview.getvalue()
    str_body = body.decode('utf-8')
    results = xmltodict.parse(str_body)  # TODO needs to be rewritten with different library
    try:
        return results['NormalisationResult']['NormalisationJDs']['JD']['Discipline']['@discipline']
    except:
        try:
            for lg in range(0,len(results['NormalisationResult']['NormalisationJDs']['JD']['Discipline'])):
                if results['NormalisationResult']['NormalisationJDs']['JD']['Discipline'][lg]['@language'] == 'en':
                    prof_ind = results['NormalisationResult']['NormalisationJDs']['JD']['Discipline'][lg]['@discipline']
            z = 1
        except KeyError:
            prof_ind = None
            z = 2
        except TypeError:
            z = 3
            #for lg in range(0,len(results['NormalisationResult']['NormalisationJDs']['JD'][1]['Discipline'])):
            #   if results['NormalisationResult']['NormalisationJDs']['JD'][1]['Discipline'][lg]['@language'] == 'en':
            if 'Discipline' in results['NormalisationResult']['NormalisationJDs']['JD'][0]:
                prof_ind = results['NormalisationResult']['NormalisationJDs']['JD'][0]['Discipline'][3]['@discipline']
            else:
                #prof_ind = results['NormalisationResult']['NormalisationJDs']['JD'][1]['Discipline'][2]['@discipline']
                prof_ind = None
    return prof_ind


def get_mostcommon(path, n, i=3):    
    """Get the n most common rolename from csvfile (I used it) for testing purposes mostly."""
    allroles = []
    with open(path, 'rt', encoding='mac_roman') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        for row in csvreader:
            try:
                role = clean(row[i])
                allroles.append(''.join(role))
            except IndexError:
                pass
    mostc = collections.Counter(allroles)
    roles = mostc.most_common(n)
    mostcroles = [x[0] for x in roles]
    return mostcroles

def irrelevant_features(features):
    """Remove clutter from features, everything that appears too many times in each industry."""
    irrelevant = []
    for vec in set(features):
        if (features[vec].count(0)/len(indtf_features[vec])) < 0.1:
            irrelevant.append(vec)
    return irrelevant

def train_lda(obs):
    """Train a model using the raw counts. LDA expects raw counts, can be used with tfidf scores but the theoretical basis for that isn't well defined."""
    print('Training LDA model...')
    lda = LatentDirichletAllocation(n_topics=42, max_iter=100, 
                                doc_topic_prior=0.0001,
                                learning_method='online',
                                learning_offset=50., 
                                topic_word_prior=0.001,
                                random_state=0)
    lda.fit_transform(obs)
    pickle.dump(lda, open("ilda.data", "wb" ))
    return lda

def c_legend(collapse,lda,features_vec, irrelevant):
    legend = {}
    for key in collapse:
        topicindex = guess_topic(lda, collapse[key], features_vec, irrelevant, verbose = False)
        legend[topicindex] = key
    return legend


if __name__ == "__main__":
    # Usage: first command line argument is the directory with the csv files under the cwd, second is keywords 'test'/'train'.
    dirname = sys.argv[1]  # where is the data
    legend = {}
    train_test = sys.argv[2]  # if a model is already trained already then use 'test' otherwise use 'train'
    if train_test == 'train':
        for filename in os.listdir(dirname):
            if filename.startswith('.') == False:
                print('Currently on file: %s' % filename)
                ind_vectors, i_features = processdata((os.getcwd() + '/' + dirname + '/' + filename), 'train')
        industry_words = count_collapse(ind_vectors)
        indtf_features = feature_vector(i_features, industry_words)
        indtf_samples = [indtf_features[feature] for feature in indtf_features if sum(indtf_features[feature])>0]
        indtf_labels = [feature for feature in indtf_features if sum(indtf_features[feature])>0]
        ilda = train_lda(indtf_samples)
    elif train_test == 'test':
        ind_vectors, i_features = processdata(mode =  'test')
        industry_words = count_collapse(ind_vectors)
        indtf_features = feature_vector(i_features, industry_words)
        indtf_samples = [indtf_features[feature] for feature in indtf_features if sum(indtf_features[feature])>0]
        indtf_labels = [feature for feature in indtf_features if sum(indtf_features[feature])>0]
        ilda = pickle.load(open("ilda.data", "rb"))
        irrelevant = irrelevant_features(indtf_features)
        transformed = ilda.transform(indtf_samples)

        # to get sample output:
        print(guess_topic(lda=ilda, query="project manager", features_vec=indtf_features, irrelevant=irrelevant, verbose=True))
    else:
        print('Usage: location of directory, ')
#features_vector = pickle.load(open("features_vector.data", "rb"))
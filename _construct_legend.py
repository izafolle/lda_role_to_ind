from guess_topic import guess_topic
'''not used'''

# def construct_legend(indtf_vector, features_vector, lda, labels):
#     """Gives the interpretation for each topic.
#     That is uses index of the topic (a number) to map to a business name for a topic.
#     This mapping can be made from a list of unique topic_name to unique_words mapping.
#     :param indtf_vector: industry term frequency vector
#     :param features_vector: mapping of topic_name to unique words
#     :param lda: The model to be labelled.
#     :param labels: list of ngrams.
#     """
#     # keys = [key for key in indtf_vector]
#     # most_imp = [(key,[x[0] for x in sorted(list(indtf_vector[key].items()),key=lambda x:x[1], reverse=True)[:10]]) for key in keys]
#     # legend = [(guess_topic(lda, tpl[1], features_vector,verbose=False),tpl[0]) for tpl in most_imp]
#     # unique_features = coherence_check(500,transformed,labels)
#     # #legend = dict(legend)
#     # return legend
def construct_legend(legend={}):
    """Gives the interpretation for each topic.
    That is uses index of the topic (a number) to map to a business name for a topic.
    This mapping can be made from a list of unique topic_name to unique_words mapping.
    :param indtf_vector: industry term frequency vector
    :param features_vector: mapping of topic_name to unique words
    :param lda: The model to be labelled.
    :param labels: list of ngrams.
    """
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
                   ('labour', 'party'), ('policy', 'adviser'), ('parliamentary', 'assistant'),
                   ('leeds', 'city', 'council'), ('work', 'pensions', 'dwp'), ('pensions', 'dwp'),
                   ('royal', 'borough'), ('kent', 'county'), ('kent', 'county', 'council'), 'oxfordshire',
                   ('hertfordshire', 'county'), ('hertfordshire', 'county', 'council'), ('oxfordshire', 'county'),
                   ('oxfordshire', 'county', 'council'), ('essex', 'county', 'council'), ('essex', 'county'),
                   ('hampshire', 'county')],
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

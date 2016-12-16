import collections


def declutter(vector, lower_limit=2):
    """Removes features with raw count values that are less than lower_limit in a vector.
    Infrequent terms in a vector are most likely to be noise.
    :param vector: A vector (industry, ngram, count) to be decluttered.
    :param lower_limit: terms with count below lower_limit are decluttered.
    """
    for key in vector:
        # gather everything with a value less than two and save it in a list
        clutter_values = [value for value in vector[key] if vector[key][value] < lower_limit]
        for feature in clutter_values:  # remove everything in the clutter values from a dictionary
            vector[key].pop(feature, None)
    return vector

def count_collapse(count_words, declutter_limit = 2):
    """Collapse industries from data into stepweb industries.
    :param count_words: dictionary of terms already seen by industry"""
    collapse = {'Accountancy': ['Accounting', 'Accountancy'],
                'Administration': ['Government Administration', 'Secretarial & Admin.', 'Secretarial & Administration',
                                   'Secretarial, PAs, Administration', 'Secretarial & Administrative'],
                'Advertising': [],
                'Animal Care': ['Veterinary'],
                'Arts and Entertainment': ['Music', 'Computer Games', 'Fine Art', 'Arts and Crafts', 'Photography', 
                                           'Motion Pictures and Film', 'Performing Arts', 'Entertainment', 
                                           'Events Services'],
                'Banking': ['Banking', 'Investment Banking', 'Venture Capital & Private Equity'],
                'Catering': ['Food & Beverages', 'Food Production', 'Hospitality', 'Restaurants', 'Hotel & Catering', 
                             'Travel, Catering & Hospitality', 'Wine and Spirits', 'Catering & Hospitality', 
                             'Hospitality & Leisure'],
                'Cleaning': [],
                'Construction': ['Construction'],
                'Consulting': ['Management Consulting', 'Consultancy'],
                'Customer Service': ['Customer Service', 'Customer Services'],
                'Design': ['Design', 'Graphic Design', 'Fashion & Design'],
                'Education': ['Higher Education', 'Primary/Secondary Education', 'Education', 'Libraries', 'E-Learning', 
                              'Professional Training & Coaching', 'Training'],
                'Engineering': ['Aerospace', 'Engineering', 'Rail Engineers', 'Computer Hardware', 'Electronics', 
                                'Mechanical or Industrial Engineering', 'Civil Engineering', 'Semiconductors', 
                                'Architecture & Planning'],
                'Farming and Agriculture': ['Farming', 'Ranching', 'Agriculture, Fishing, Forestry'],
                'Finance': ['Banking & Finance', 'Financial Services'],
                'Health': ['Medical Practice', 'Mental Health Care', 'Hospital & Health Care', 'Alternative Medicine', 
                           'Health', 'Medical Devices', 'Health, Nursing', 'Health/Healthcare/Social care'],
                'Human Resources': ['Human Resources', 'Recruitment', 'Staffing and Recruiting'],
                'Insurance': ['Insurance', 'Insurance & Financial Services'],
                'IT': ['Program Development', 'IT & Internet', 'Computer Software', 'Information Tech.', 
                       'Computer & Network Security', 'Information Technology and Services', 'Internet', 
                       'Wireless', 'Information Technology', 'Information Services', 'Computer Networking', 
                       'Finance IT'],
                'Legal': ['Law Practice', 'Legal', 'Legal Services', 'Judiciary', 'Alternative Dispute Resolution', 
                          'Legislative Office'],
                'Logistics': ['Transport, Logistics', 'Logistics', 'Import and Export', 'Distribution', 'Warehousing', 
                              'Logistics & Transport', 'Package/Freight Delivery', 'Logistics and Supply Chain', 
                              'Transportation/Trucking/Railroad'],
                'Manufacturing': ['Manufacturing', 'Electrical/Electronic Manufacturing', 'Production & Ops.', 
                                  'Production & Operations', 'Railroad Manufacture', 'Automotive'],
                'Management': ['Management & Exec.', 'Management & Executive', 'Education Management'], 
                'Marketing': ['Marketing and Advertising', 'Marketing', 'Finance Marketing', 'Client Side', 
                              'Market Research'],
                'Media': ['Media', 'Broadcast Media', 'Media, New Media, Creative', 'Online Media', 'Media Production', 
                          'Writing and Editing', 'Newspapers', 'Animation', 'Publishing'],
                'Military': ['Defence', 'Military', 'Defense & Space', 'Maritime'],
                'Policing': ['Public Policy', 'Think Tanks', 'Political Organization', 'Government Relations', 
                             'International Affairs'],
                'PR': ['Public Relations and Communications'],
                'Property': ['Property', 'Real Estate', 'Commercial Real Estate'],
                'Public Sector': ['Public Sector'],
                'Retail': ['Wholesale', 'Retail', 'Supermarkets', 'Retail, Wholesale'],
                'Sales': ['Sales'],
                'Science': ['Scientific', 'Nanotechnology', 'Research', 'Biotechnology', 'Chemicals', 
                            'Pharmaceutical & Biotechnology', 'Pharmaceuticals', 'Science'],
                'Security': ['Security', 'Security and Investigations'],
                'Skilled Trades': ['Skilled Trades', 'Dairy', 'Shipbuilding', 'Fishery'],
                'Social Care': ['Individual & Family Services', 'Social Services'],
                'Sport and Fitness': ['Health, Wellness and Fitness', 'Sports', 'Recreational Facilities and Services'],
                'Third Sector': ['Not For Profit, Charities', 'Philanthropy', 'Charity & Voluntary Work',
                                 'Fund-Raising', 'Charity & Voluntary Work', 'Nonprofit Organization Management',
                                 'Religious Institutions'],
                'Travel': ['Leisure, Travel & Tourism', 'Travel & Hospitality', 'Travel, Leisure, Tourism']}
    new_cw = collections.defaultdict(collections.Counter)
    upd_cw = collections.defaultdict(collections.Counter)
    for key in count_words:  # transforms a list of ngrams from the data into a count of those same ngrams per industry.
        ngrams = count_words[key]
        new_cw[key] = collections.Counter(ngrams)
        # TODO why is ngrams an unexpected argument to collections Counter?
    new_cw = declutter(new_cw, declutter_limit)
    for key in collapse:
        for alternative_name in collapse[key]:
            upd_cw[key].update(new_cw[alternative_name])
    return upd_cw

from nose.tools import assert_equal

import count_collapse
import collections


class TestDeclutter:
    def test_declutter(selfself):
        cw = collections.defaultdict(collections.Counter, {
            'Engineering': collections.Counter({
                ('engineer'): 20,
                ('manager', 'contract'): 2,
                ('reward', 'manager'): 1}),
            'Design': {
                ('manager', 'contract'): 1,
                ('design', 'manager'): 15}})
        expected_result = collections.defaultdict(collections.Counter, {
            'Engineering': collections.Counter({
                'engineer': 20,
                ('manager', 'contract'): 2}),
            'Design':
                {('design', 'manager'): 15}})
        dec = count_collapse.declutter(cw, lower_limit=2)
        assert_equal(dec, expected_result)



class TestCountCollapse:

    def test_correct_count_collapse_no_decluttering(self):
        s = [('Accountancy', ('archers', 'transport')),
             ('Accountancy', ('transport', 'ltd')),
             ('Accountancy', ('archers', 'transport', 'ltd')),
             ('Accountancy', 'archers'),
             ('Accountancy', 'transport'),
             ('Aerospace', ('senior', 'reward')),
             ('Aerospace', ('reward', 'manager')),
             ('Aerospace', ('manager', 'contract')),
             ('Aerospace', ('senior', 'reward', 'manager'))
             ]
        d = collections.defaultdict(list)
        for k, v in s:
            d[k].append(v)
        expected_result = collections.defaultdict(collections.Counter, {'Military': collections.Counter(),
                                                'Health': collections.Counter(),
                                                'Human Resources': collections.Counter(),
                                                'Social Care': collections.Counter(),
                                                'Sales': collections.Counter(),
                                                'IT': collections.Counter(),
                                                'Management': collections.Counter(),
                                                'Legal': collections.Counter(),
                                                'Logistics': collections.Counter(),
                                                'Farming and Agriculture': collections.Counter(),
                                                'Retail': collections.Counter(),
                                                'Accountancy': collections.Counter({('transport', 'ltd'): 1, ('archers', 'transport', 'ltd'): 1, ('archers', 'transport'): 1, 'archers': 1, 'transport': 1}),
                                                'Education': collections.Counter(),
                                                'Animal Care': collections.Counter(),
                                                'PR': collections.Counter(),
                                                'Public Sector': collections.Counter(),
                                                'Science': collections.Counter(),
                                                'Insurance': collections.Counter(),
                                                'Finance': collections.Counter(),
                                                'Arts and Entertainment': collections.Counter(),
                                                'Third Sector': collections.Counter(),
                                                'Policing': collections.Counter(),
                                                'Media': collections.Counter(),
                                                'Administration': collections.Counter(),
                                                'Customer Service': collections.Counter(),
                                                'Sport and Fitness': collections.Counter(),
                                                'Construction': collections.Counter(),
                                                'Engineering': collections.Counter({('manager', 'contract'): 1, ('senior', 'reward'): 1, ('senior', 'reward', 'manager'): 1, ('reward', 'manager'): 1}),
                                                'Design': collections.Counter(),
                                                'Skilled Trades': collections.Counter(),
                                                'Security': collections.Counter(),
                                                'Banking': collections.Counter(),
                                                'Property': collections.Counter(),
                                                'Consulting': collections.Counter(),
                                                'Marketing': collections.Counter(),
                                                'Manufacturing': collections.Counter(),
                                                'Travel': collections.Counter(),
                                                'Catering': collections.Counter()})
        assert_equal(count_collapse.count_collapse(d, declutter_limit=1), expected_result)

    def test_correct_count_collapse_with_decluttering(self):
        s = [('Aerospace', ('senior', 'reward')),
             ('Aerospace', ('reward', 'manager')),
             ('Aerospace', ('manager', 'contract')),
             ('Aerospace', ('senior', 'reward', 'manager')),
             ('Aerospace', ('reward', 'manager')),
             ('Aerospace', ('manager', 'contract'))
        ]
        d = collections.defaultdict(list)
        for k, v in s:
            d[k].append(v)
        expected_result = collections.defaultdict(collections.Counter, {'Military': collections.Counter(),
                                                                        'Health': collections.Counter(),
                                                                        'Human Resources': collections.Counter(),
                                                                        'Social Care': collections.Counter(),
                                                                        'Sales': collections.Counter(),
                                                                        'IT': collections.Counter(),
                                                                        'Management': collections.Counter(),
                                                                        'Legal': collections.Counter(),
                                                                        'Logistics': collections.Counter(),
                                                                        'Farming and Agriculture': collections.Counter(),
                                                                        'Retail': collections.Counter(),
                                                                        'Accountancy': collections.Counter(),
                                                                        'Education': collections.Counter(),
                                                                        'Animal Care': collections.Counter(),
                                                                        'PR': collections.Counter(),
                                                                        'Public Sector': collections.Counter(),
                                                                        'Science': collections.Counter(),
                                                                        'Insurance': collections.Counter(),
                                                                        'Finance': collections.Counter(),
                                                                        'Arts and Entertainment': collections.Counter(),
                                                                        'Third Sector': collections.Counter(),
                                                                        'Policing': collections.Counter(),
                                                                        'Media': collections.Counter(),
                                                                        'Administration': collections.Counter(),
                                                                        'Customer Service': collections.Counter(),
                                                                        'Sport and Fitness': collections.Counter(),
                                                                        'Construction': collections.Counter(),
                                                                        'Engineering': collections.Counter({('manager', 'contract'): 2, ('reward', 'manager'): 2}),
                                                                        'Design': collections.Counter(),
                                                                        'Skilled Trades': collections.Counter(),
                                                                        'Security': collections.Counter(),
                                                                        'Banking': collections.Counter(),
                                                                        'Property': collections.Counter(),
                                                                        'Consulting': collections.Counter(),
                                                                        'Marketing': collections.Counter(),
                                                                        'Manufacturing': collections.Counter(),
                                                                        'Travel': collections.Counter(),
                                                                        'Catering': collections.Counter()})
        assert_equal(count_collapse.count_collapse(d, declutter_limit=2), expected_result)

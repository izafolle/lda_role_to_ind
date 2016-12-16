from nose.tools import assert_equal, assert_false, assert_true, assert_raises
import collections

from irrelevant_features import irrelevant_features


class TestClean:
    s = [('ten_percent', 0),
         ('ten_percent', 0),
         ('ten_percent', 0),
         ('ten_percent', 0),
         ('ten_percent', 0),
         ('ten_percent', 0),
         ('ten_percent', 0),
         ('ten_percent', 0),
         ('ten_percent', 0),
         ('ten_percent', 1),
         ('fifty_percent', 0),
         ('fifty_percent', 0),
         ('fifty_percent', 0),
         ('fifty_percent', 0),
         ('fifty_percent', 0),
         ('fifty_percent', 1),
         ('fifty_percent', 1),
         ('fifty_percent', 1),
         ('fifty_percent', 1),
         ('fifty_percent', 1),
         ('ninety_percent', 0),
         ('ninety_percent', 1),
         ('ninety_percent', 1),
         ('ninety_percent', 1),
         ('ninety_percent', 1),
         ('ninety_percent', 1),
         ('ninety_percent', 1),
         ('ninety_percent', 1),
         ('ninety_percent', 1),
         ('ninety_percent', 1),
         ('onehundred_percent', 1),
         ('onehundred_percent', 1),
         ('onehundred_percent', 1),
         ('onehundred_percent', 1),
         ('onehundred_percent', 1),
         ('onehundred_percent', 1),
         ('onehundred_percent', 1),
         ('onehundred_percent', 1),
         ('onehundred_percent', 1),
         ('onehundred_percent', 1),
         ]
    d = collections.defaultdict(list)
    for k, v in s:
        d[k].append(v)

    def test_correct_irrelevant_features10th(self):
        threshold = 0.11
        found_results = irrelevant_features(self.d, threshold)
        assert_false('ten_percent' in found_results)
        assert_true('fifty_percent' in found_results)
        assert_true('ninety_percent' in found_results)
        assert_true('onehundred_percent' in found_results)

    def test_correct_irrelevant_features50th(self):
        threshold = 0.51
        found_results = irrelevant_features(self.d, threshold)
        assert_false('ten_percent' in found_results)
        assert_false('fifty_percent' in found_results)
        assert_true('ninety_percent' in found_results)
        assert_true('onehundred_percent' in found_results)

    def test_correct_irrelevant_features90th(self):
        threshold = 0.91
        found_results = irrelevant_features(self.d, threshold)
        assert_false('ten_percent' in found_results)
        assert_false('fifty_percent' in found_results)
        assert_false('ninety_percent' in found_results)
        assert_true('onehundred_percent' in found_results)

    def test_correct_irrelevant_features100th(self):
        threshold = 1.
        found_results = irrelevant_features(self.d, threshold)
        assert_false('ten_percent' in found_results)
        assert_false('fifty_percent' in found_results)
        assert_false('ninety_percent' in found_results)
        assert_true('onehundred_percent' in found_results)

    def test_assumedThreshold_correct_irrelevant_features90th(self):
        threshold = 1.5
        found_results = irrelevant_features(self.d, threshold)
        assert_false('ten_percent' in found_results)
        assert_false('fifty_percent' in found_results)
        assert_true('ninety_percent' in found_results)
        assert_true('onehundred_percent' in found_results)
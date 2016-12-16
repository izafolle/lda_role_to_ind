# coding=utf-8
from nose.tools import assert_equal, assert_false, assert_true, assert_raises

from countize import countize
import collections


class TestCountize:

    def test_countize_countwords(self):
        word = 'mechanical engineer'
        ind = 'engineering'
        count_words = collections.defaultdict(list)
        features = []
        expected_result = {'engineering': [('mechanical', 'engineer'), 'mechanical', 'engineer']}

        cw, f = countize(word, ind, count_words, features)
        assert_equal(cw, expected_result)


    def test_countize_features(self):
        word = 'mechanical engineer'
        ind = 'engineering'
        count_words = collections.defaultdict(list)
        features = []
        expected_result = [('mechanical', 'engineer'), 'mechanical', 'engineer']

        cw, f = countize(word, ind, count_words, features)
        assert_equal(cw, expected_result)

# TODO add tests where countwords and features are not initially null.

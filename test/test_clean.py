# coding=utf-8
from nose.tools import assert_equal, assert_false, assert_true, assert_raises

from clean import clean


class TestClean:

    def test_correctCleanAmpersand(self):
        name = "Marks & Spencer"
        expected_result = "marks spencer"
        assert_equal(clean(name), expected_result)

    def test_correctCleanBackslash(self):
        name = "Marks\Spencer"
        expected_result = "marks spencer"
        assert_equal(clean(name), expected_result)

    def test_correctCleanHyphen(self):
        name = "Marks-Spencer"
        expected_result = "marks spencer"
        assert_equal(clean(name), expected_result)

    def test_correctCleanBraces(self):
        name = "Marks (Spencer)"
        expected_result = "marks spencer"
        assert_equal(clean(name), expected_result)

    def test_correctCleanPipe(self):
        name = "Marks | Spencer"
        expected_result = "marks spencer"
        assert_equal(clean(name), expected_result)

    def test_correctCleanAt(self):
        name = "Marks @ Spencer"
        expected_result = "marks spencer"
        assert_equal(clean(name), expected_result)

    def test_correctCleanComma(self):
        name = "Marks, Spencer"
        expected_result = "marks spencer"
        assert_equal(clean(name), expected_result)

    def test_correctCleanSquarebrackets(self):
        name = "Marks [Spencer]"
        expected_result = "marks spencer"
        assert_equal(clean(name), expected_result)

    def test_correctCleanStopwords(self):
        name = "Marks of and to at in Spencer]"
        expected_result = "marks spencer"
        assert_equal(clean(name), expected_result)

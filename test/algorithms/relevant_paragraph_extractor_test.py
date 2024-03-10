import unittest

from src.algorithms.relevant_paragraph_extractor import RelevantParagraphExtractor


class TestRelevantParagraphExtractor(unittest.TestCase):
    maxDiff = None

    def __init__(self, *args, **kwargs):
        super(TestRelevantParagraphExtractor, self).__init__(*args, **kwargs)
        self.url_file = r'C:\Users\alwoj\PycharmProjects\question_answering\src\data\queen_victoria'

    def test_extracting_relevant_paragraphs(self):
        extractor = RelevantParagraphExtractor(['I drink melon soda when I am in Japan.', 'I ate soup',
                                                'One tea for me', 'I got two donuts'], 'I drink tea.')
        actual = extractor.get_relevant_paragraphs()
        self.assertEqual(actual, [['I drink melon soda when I am in Japan.'],
                                  ['One tea for me']])

import os
import unittest

import openai

from src.algorithms.translate_and_answer import TranslateAndAnswer
from src.utils import set_env


class TestTranslateAndAnswer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTranslateAndAnswer, self).__init__(*args, **kwargs)
        self.query = """C'est français"""
        self.gpt = TranslateAndAnswer(self.query)
        set_env()

    def test_translate_to_english(self):
        self.assertEqual(self.gpt.translate_to_english(), """It's French.""")

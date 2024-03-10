import os
import sys
from langdetect import detect
from algorithms.relevant_paragraph_extractor import RelevantParagraphExtractor, split_text_into_paragraphs, \
    concatenate_paragraphs
from src.algorithms.translate_and_answer import TranslateAndAnswer
from src.utils import set_open_ai_key, InformationNotFoundException, set_open_ai_key

path = os.getcwd()
root_directory = r'{}\data'.format(path)


def main(question, similarity_threshold=0.6, min_paragraph_length=30, gpt_temperature=0):
    """

    The analysis is based on text from the "Queen Victoria" book, sourced from Project Gutenberg.

    Processes a given question by identifying relevant paragraphs from a text source,
    then utilizes a GPT model to provide an answer based on the context.

    Parameters:
    - question (str): The question to be processed and answered. Remember to ask straightforward question to gpt!
    - similarity_threshold (float, optional): The cosine similarity threshold used to filter relevant paragraphs.
                                              Defaults to 0.6.
    - min_paragraph_length (int, optional): The minimum length a paragraph must have to be considered in the analysis.
                                             Defaults to 30 characters.
    - gpt_temperature (float, optional): Controls the creativity of the GPT model's responses. Higher values result
    in more creative, but potentially less accurate, answers.
     Ranges from 0 (most deterministic) to 1 (most creative). Defaults to 0.

    Raises:
    - InformationNotFoundException: If no relevant information is found in the text based on the query.

    Outputs:
    - The function writes the original question and the GPT model's answer to standard output.
    """

    # Splits the policy text into paragraphs that are longer than a given minimum length.
    text = split_text_into_paragraphs('{}/queen_victoria'.format(root_directory),
                                      min_paragraph_length=min_paragraph_length)

    # Utilizes a translation and question-answering model to verify the question.
    gpt_question_verifier = TranslateAndAnswer(question, temperature=gpt_temperature)

    # If the question is not in English, translates it to English.
    if detect(question) != 'en':
        question = gpt_question_verifier.translate_to_english()

    # Extracts relevant paragraphs from the text based on the similarity to the question,
    # using a defined similarity threshold.
    relevant_paragraph_extractor = RelevantParagraphExtractor(text, question,
                                                              relevance_threshold=similarity_threshold)
    paragraphs = relevant_paragraph_extractor.get_relevant_paragraphs()

    # Concatenates the relevant paragraphs into a single text.
    paragraphs = concatenate_paragraphs(paragraphs)

    # Raises an exception if no relevant information is found.
    if len(paragraphs) == 0:
        raise InformationNotFoundException('No relevant information found for the query in the text.')

    # Submits the text and question to a GPT model for answering.
    answer = gpt_question_verifier.ask_gpt(paragraphs)

    # Outputs the original question, related paragraphs, and the GPT model's answer.
    sys.stdout.write('Question: \n {} \n\n'.format(question))
    sys.stdout.write('GPT answer: {}.'.format(answer['result']))


if __name__ == "__main__":

    # Ask question in any language for provided text
    set_open_ai_key()
    ask_question = """Who is queen Victoria?"""
    main(question=ask_question)

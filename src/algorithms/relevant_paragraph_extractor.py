from gensim.parsing import remove_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_text_into_paragraphs(url: str, min_paragraph_length=30):
    file = open(url, encoding='utf-8')
    txt = file.read().split('\n\n')
    txt = [line for line in txt if len(line) > min_paragraph_length]
    return txt


def concatenate_paragraphs(paragraphs_: list):
    """Concatenate list of paragraphs into string.
    :param paragraphs_: list
    :return: str
    """
    paragraphs_to_string = ''
    for section in paragraphs_:
        paragraphs_to_string += '\n\n' + section[0]
    return paragraphs_to_string


class RelevantParagraphExtractor:
    """
     Extract relevant paragraphs from a list of paragraphs based on their similarity to a query using TF-IDF
     and cosine similarity.

     Attributes:
         paragraphs_ (list of str): The list of input paragraphs.
         query_ (str): The query for which relevant paragraphs are extracted.
         relevance_threshold (float): The threshold for paragraph relevance. Defaults to 0.25.
         vectorizer (TfidfVectorizer): TF-IDF vectorizer for text processing.

     Methods:
         remove_stopwords(text):
             Remove common stopwords from a given text.

         get_relevant_paragraphs():
             Calculate the relevance of each paragraph to the query and return relevant paragraphs.

     Example:
         paragraphs = ["This is the first paragraph.", "The second paragraph contains relevant information.",
         "A third paragraph is here."]
         query = "I had a dream about relevant information."

         extractor = RelevantParagraphExtractor(paragraphs, query)
         relevant_paragraphs = extractor.get_relevant_paragraphs()

         for paragraph in relevant_paragraphs:
             print(paragraph)

        "The second paragraph contains relevant information."
     """

    def __init__(self, paragraphs_, query_, relevance_threshold=0.25):
        self.paragraphs_ = paragraphs_
        self.query_ = remove_stopwords(query_)
        self.relevance_threshold = relevance_threshold
        self.vectorizer = TfidfVectorizer()

    def get_relevant_paragraphs(self):
        # Create a TF-IDF vectorizer and calculate TF-IDF scores for query and document paragraphs
        query_tfidf = self.vectorizer.fit_transform([self.query_])
        document_tfidf = self.vectorizer.transform(self.paragraphs_)

        # Calculate cosine similarity between query and document paragraphs
        similarity_scores = cosine_similarity(query_tfidf, document_tfidf)
        paragraph_importance = list(enumerate(similarity_scores[0]))

        paragraphs_ = []
        # Define a threshold for paragraph relevance (adjust as needed)
        for i, score in paragraph_importance:
            if score >= self.relevance_threshold:
                paragraphs_.append([self.paragraphs_[i]])
        return paragraphs_

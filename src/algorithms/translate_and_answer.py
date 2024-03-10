# langchain: This is a Python library developed by OpenAI. It provides tools for working with language models,
# including GPT. It is used to create the QA chain and for splitting the text into chunks that the
# GPT model can process.
import openai
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA, StuffDocumentsChain, create_qa_with_sources_chain


class TranslateAndAnswer:
    def __init__(self, query, temperature):
        self.query = query
        self.temperature = temperature

    def ask_gpt(self, paragraphs):
        """
          Ask a question to GPT-4 using a structured retrieval-based approach.

          Args:
              paragraphs (str): A string containing paragraphs of text from which information is retrieved.

          Returns:
              dict: A dictionary containing the response from the GPT-4 model with relevant source documents.
          """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8192,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]
        )

        # Splits the input paragraphs into smaller texts for processing.
        texts = text_splitter.create_documents([paragraphs])

        # Embeds the text fragments using the SentenceTransformer model.
        # Sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space
        # and can be used for tasks like clustering or semantic search.
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Constructs a searchable database of text embeddings using FAISS.
        db = FAISS.from_documents(texts, embedding_function)

        # Initializes a ChatOpenAI model for answering questions. A temperature of 0 means that the model will always
        # select the highest probability word. A higher temperature means that the model might select a word with
        # slightly lower probability, leading to more variation, randomness and creativity.
        llm_src = ChatOpenAI(temperature=self.temperature, model="gpt-4")

        # Creates a question answering chain that returns answers with sources.
        qa_chain = create_qa_with_sources_chain(llm_src)

        # Uses the StuffDocumentsChain to combine the QA chain with the source documents.
        final_qa_chain = StuffDocumentsChain(
            llm_chain=qa_chain,
            document_variable_name='context',
        )
        # Performs a retrieval-based question answering using the constructed database.
        retrieval_qa = RetrievalQA(
            retriever=db.as_retriever(),
            combine_documents_chain=final_qa_chain,
            return_source_documents=True
        )

        # Returns the answer along with relevant source documents.
        answer = retrieval_qa({"query": """"{}.""".format(self.query)})
        return answer

    def translate_to_english(self):
        """
        Translate a query to English using the OpenAI GPT-4 model.

        Returns:
            str: The translated text in English.

        This method uses the OpenAI GPT-4 model to translate a query to English.

        Example:
            translator = Translator(query="Je suis un étudiant.")
            english_translation = translator.translate_to_english()
            print(english_translation)
        """
        return openai.ChatCompletion.create(
            temperature=0,
            model="gpt-4",
            messages=[  # Change the prompt parameter to the messages parameter
                {'role': 'user', 'content': "Translate query to english: {}".format(self.query)}
            ],
        )["choices"][0]['message']['content'].strip()

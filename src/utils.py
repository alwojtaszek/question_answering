import os

import openai

class InformationNotFoundException(Exception):
    def __init__(self, message):
        super(InformationNotFoundException, self).__init__(message)


def set_open_ai_key():
    """
     Set the OpenAI API key in the environment for authentication.

     This function sets the OpenAI API key as an environment variable for authentication purposes.
     It ensures that the API key is available for making authenticated API requests to OpenAI's services.
     """
    os.getenv('OPENAI_API_KEY')

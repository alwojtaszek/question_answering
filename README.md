**Text Analysis Toolkit Overview**

**Introduction**


This tool uses advanced technology like TF-IDF, Sentence Transformer Embeddings
and GPT to help you deeply understand and explore texts.
Initially designed with the "Queen Victoria" book from Project Gutenberg. 
It can be adapted to analyze any text document by modifying split_text_into_paragraphs method 
in relevant_paragraph_extractor.py.

**Core Features**

>1) Employs TF-IDF to identify key sections in the text.
>2) Enhances query comprehension with Sentence Transformer Embeddings.
>3) Generates appropriate responses using GPT based on the context.

**Getting Started**
>1) Initial Setup: Clone the repository to your local machine. Ensure Python 3.x is installed on your system.
>2) Dependency Installation: Execute pip install -r requirements.txt in your terminal to install required libraries.
>3) Add your OPENAI_API_KEY to environment variables.
>4) Running the tool: Use the command python main.py --query "your question" to begin the analysis process.
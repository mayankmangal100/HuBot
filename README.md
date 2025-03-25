<<<<<<< HEAD
# HuBot

#### Steps
Create a virtual env with python 3.10 and install dependencies from requirement.txt file

#### Key Components

Document Processing: Uses unstructured to extract text from PDFs

Text Splitting: Implements RecursiveCharacterTextSplitter for chunking documents

Embedding Model: Uses BGE-small from HuggingFace for high accuracy and low latency

Hybrid Retrieval: Combines BM25 (lexical search) and embedding-based search

Vector Storage: Implements Qdrant for vector storage and retrieval, we need to explore mongo db atlas as well as suggested by Abhinav(will take this up later)

LLM Interface: Uses Mistral 7B Instruct GGUF with quantization for fast inference

Conversational Handling: Processes follow-up questions by generating standalone questions
=======

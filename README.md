#### Key Components

Document Processing: Uses unstructured to extract text from PDFs
Text Splitting: Implements RecursiveCharacterTextSplitter for chunking documents
Embedding Model: Uses BGE-small from HuggingFace for high accuracy and low latency
Hybrid Retrieval: Combines BM25 (lexical search) and embedding-based search
Vector Storage: Implements Qdrant for vector storage and retrieval
LLM Interface: Uses Mistral 7B Instruct GGUF with quantization for fast inference
Conversational Handling: Processes follow-up questions by generating standalone questions

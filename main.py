"""Main entry point for the RAG system."""

import os
from src.utils import Config, logger
from src.rag_system import RAGSystem

def main():
    # Load configuration
    config = Config.load_config("config/config.json")
    
    # Initialize RAG system
    rag = RAGSystem(config=config)
    
    # Check if documents are already processed
    document_path = "Documents/EveriseHandbook.pdf"
    # if not os.path.exists(os.path.join(os.getcwd(), "vector_store")):
    if not os.path.exists(document_path):
        logger.error(f"Document not found: {document_path}")
        return
        
    # Process document
    logger.info(f"Processing document: {document_path}")
    rag.ingest(document_path)
    # else:
    #     logger.info("Using existing vector store")
    
    # Interactive loop
    try:
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            # Get answer with streaming
            stream = True
            if stream:
                for token in rag.answer_question(query, stream=True):
                    print(token, end="", flush=True)
            else:
                answer = rag.answer_question(query, stream=False)
                print(f"\nAnswer: {answer}")
            print("\nSources:")
            for doc, score in rag.last_context_docs:
                print(f"Source: {doc['metadata']['source']} (Chunk {doc['metadata']['chunk_id']}, Score: {score:.3f})")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    
if __name__ == "__main__":
    main()

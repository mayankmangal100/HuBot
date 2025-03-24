"""Main RAG system that orchestrates the entire process."""

from typing import Dict, Any, List, Optional
from src.utils import Config, LatencyTracker
from src.document_processor import DocumentProcessor
from src.text_splitter import TextSplitter
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.llm_interface import LLMInterface
from src.utils import logger

class RAGSystem:
    """Main RAG system that orchestrates the entire process"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Load configuration
        self.config = config
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.text_splitter = TextSplitter(
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"]
        )
        self.embedding_model = EmbeddingModel(
            model_name=config["model"]["embedding_model"],
            device=config["model"]["device"]
        )
        self.vector_store = VectorStore(config)
        self.llm = LLMInterface(
            model_name=config["model"]["llm_model_path"],
            config=config
        )
        
        # Store conversation history
        self.conversation_history = ""
        self.last_context_docs = []
        
    def ingest(self, file_path: str):
        """Process a document and add it to the vector store"""
        # Extract text
        text = self.document_processor.process_pdf(file_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text, metadata={"source": file_path})
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_texts(
            [chunk["text"] for chunk in chunks]
        )
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
        
    def get_relevant_context(self, query: str, query_embedding: Optional[List[float]] = None) -> str:
        """Retrieve relevant context for a query"""
        if query_embedding is None:
            query_embedding = self.embedding_model.embed_query(query)
            
        results = self.vector_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=self.config["retrieval"]["top_k"]
        )
        
        # Store for later use
        self.last_context_docs = results
        
        # Format context with clear structure
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Excerpt {i}]\n{doc['text'].strip()}")
            
        context = "\n\n".join(context_parts)
        
        # Log retrieved context
        logger.debug(f"Retrieved context ({len(results)} chunks):\n{context}")
        
        return context
        
    def answer_question(self, question: str, stream: bool = False) -> str:
        """Answer a question using the RAG pipeline"""
        tracker = LatencyTracker().start()
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.embed_query(question)
            
            # Get relevant context
            context = self.get_relevant_context(
                query=question,
                query_embedding=query_embedding
            )
            
            if not context.strip():
                return "I couldn't find any relevant information to answer your question."
            
            # Generate answer
            answer = self.llm.generate_answer(
                question=question,
                context=context,
                max_tokens=self.config["llm"]["max_tokens"],
                isStream=stream
            )
            
            tracker.end("Complete RAG pipeline")
            return answer
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return "I encountered an error while processing your question. Please try again."

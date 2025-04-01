"""Main RAG system that orchestrates the entire process."""

from typing import Dict, Any, List, Optional
from src.utils import Config, LatencyTracker
from src.document_processor import DocumentProcessor
from src.text_splitter import TextSplitter
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.llm_interface import LLMInterface
from src.rewrite_query import QueryRewriter
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
        self.query_rewriter = QueryRewriter(
            query_rewrite_model=config.get("query_rewrite_model", "MBZUAI/LaMini-Flan-T5-248M"),
            device=config["model"]["device"]
        )
        self.llm = LLMInterface(config=config)
        
        # Store context and query information
        self.last_context_docs = []
        self.last_query_embedding = None
        self.last_retrieval_query = None
        
        logger.info("RAG system initialized with all components")
    
    def load_existing_data(self) -> bool:
        """Load existing vector store data if available. Returns True if data was loaded successfully."""
        return self.vector_store.load_collection()
        
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
        
    def answer_question(self, question: str, stream: bool = False) -> str:
        """Answer a question using the RAG pipeline"""
        
        try:
            tracker = LatencyTracker().start()
            # Get query embedding
            query_embedding = self.embedding_model.embed_query(question)
            self.last_query_embedding = query_embedding  # Store for streaming case
            
            # Process query through rewriter
            query_info = self.query_rewriter.process_query(
                query=question,
                query_embedding=query_embedding
            )
            
            # Use rewritten query for retrieval if available
            retrieval_query = query_info["rewritten_query"] or question
            self.last_retrieval_query = retrieval_query  # Store for streaming case
            
            # Retrieve relevant documents
            results = self.vector_store.hybrid_search(
                query=retrieval_query,
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
            
            if not context.strip():
                return "I couldn't find any relevant information to answer your question."
            
            # Generate answer
            answer = self.llm.generate_answer(
                question=retrieval_query,
                context=context,
                isStream=stream
            )
            
            # Handle conversation history update
            if stream:
                # For streaming responses, we need to wait until the stream is complete
                # The conversation update will happen in the app layer after streaming completes
                pass
            else:
                # For non-streaming responses, update immediately
                self.query_rewriter.add_exchange(
                    query=question,
                    rewritten_query=retrieval_query,
                    answer=answer,
                    query_embedding=query_embedding,
                    context_docs=self.last_context_docs
                )
            
            tracker.end("Complete RAG pipeline")
            return answer
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return "I encountered an error while processing your question. Please try again."

"""Vector store with hybrid search capabilities."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.utils import LatencyTracker
from src.utils import logger
import os

class VectorStore:
    """Vector store with hybrid search capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_qdrant = config["vector_store"].get("engine", "qdrant") == "qdrant"
        self.collection_name = config["vector_store"].get("collection_name", "documents")
        self.use_faiss_fallback = config["vector_store"].get("use_faiss_fallback", True)
        
        # Initialize components
        self.documents = []
        self.embeddings = None
        self.client = None
        self.faiss_index = None
        self.collection_exists = False
        
        self.bm25 = None
        self.tokenizer = lambda x: x.lower().split()

        # Initialize Qdrant client
        if self.use_qdrant:
            try:
                from qdrant_client import QdrantClient
                
                # Use persistent storage
                self.storage_path = os.path.join(os.getcwd(), "vector_store")
                os.makedirs(self.storage_path, exist_ok=True)
                # Clean up any existing lock files
                lock_file = os.path.join(self.storage_path, "lock.file")
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                    logger.info("Removed existing lock file")
                
                self.client = QdrantClient(
                    path=self.storage_path,
                    force_disable_multiple_clients_check=True  # Allow multiple clients
                )
                
                logger.info("Initialized Qdrant client with persistent storage at: %s", self.storage_path)
                
            except ImportError:
                logger.error("qdrant-client not installed. Install with: pip install qdrant-client")
                raise
                
        # Initialize FAISS if needed
        if not self.use_qdrant or self.use_faiss_fallback:
            try:
                import faiss
            except ImportError:
                logger.error("FAISS not installed. Install with: pip install faiss-cpu")
                raise

    def load_collection(self) -> bool:
        """Load existing collection if it exists. Returns True if collection was loaded successfully."""
        if not self.use_qdrant or not self.client:
            return False

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            self.collection_exists = self.collection_name in collection_names
            
            if not self.collection_exists:
                logger.info("No existing collection found with name: %s", self.collection_name)
                return False
            
            # Load collection data
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Adjust based on your needs
            )[0]
            
            if not points:
                logger.info("Collection exists but is empty: %s", self.collection_name)
                return False
            
            # Reconstruct documents and embeddings
            self.documents = []
            self.embeddings = []
            for point in points:
                self.documents.append({
                    "text": point.payload["text"],
                    "metadata": point.payload["metadata"]
                })
                self.embeddings.append(point.vector)
            
            # Initialize BM25 with loaded documents
            from rank_bm25 import BM25Okapi
            texts = [doc["text"] for doc in self.documents]
            tokenized_texts = [self.tokenizer(text) for text in texts]
            self.bm25 = BM25Okapi(tokenized_texts)
            
            logger.info("Successfully loaded collection with %d documents", len(self.documents))
            return True
            
        except Exception as e:
            logger.error("Error loading collection: %s", str(e))
            return False

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add documents and their embeddings to the store"""
        tracker = LatencyTracker().start()
        
        # Store documents and embeddings
        self.documents = documents
        self.embeddings = embeddings
        
        # Initialize BM25
        from rank_bm25 import BM25Okapi
        texts = [doc["text"] for doc in documents]
        tokenized_texts = [self.tokenizer(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Add to Qdrant
        if self.use_qdrant:
            try:
                from qdrant_client.models import Distance, VectorParams
                from qdrant_client.models import PointStruct
                
                # Create collection
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=len(embeddings[0]),
                        distance=Distance.COSINE
                    )
                )
                
                # Add points
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(
                        id=idx,
                        vector=embedding,
                        payload={
                            "text": doc["text"],
                            "metadata": doc["metadata"]
                        }
                    ) for idx, (doc, embedding) in enumerate(zip(documents, embeddings))]
                )
            except Exception as e:
                logger.error(f"Error adding documents to Qdrant: {str(e)}")
                if self.use_faiss_fallback:
                    logger.info("Falling back to FAISS")
                else:
                    raise
                    
        # Add to FAISS if needed
        if not self.use_qdrant or self.use_faiss_fallback:
            import faiss
            dimension = len(embeddings[0])
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(np.array(embeddings).astype(np.float32))
            
        tracker.end(f"Vector store initialization ({len(documents)} documents)")
        
    def hybrid_search(self, query: str, query_embedding: List[float], top_k: int = 3, alpha: float = 0.5) -> List[Tuple[Dict[str, Any], float]]:
        """Hybrid search combining dense and sparse retrievers"""
        tracker = LatencyTracker().start()
        
        # Dense search
        if self.use_qdrant:
            try:
                # Search in Qdrant
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k
                )
                
                # Extract scores and documents
                dense_scores = [hit.score for hit in results]
                doc_indices = [hit.id for hit in results]
                
            except Exception as e:
                logger.error(f"Error searching in Qdrant: {str(e)}")
                if self.use_faiss_fallback:
                    logger.info("Falling back to FAISS")
                    self.use_qdrant = False
                else:
                    raise
        
        if not self.use_qdrant:
            # Fallback to FAISS
            D, I = self.faiss_index.search(np.array([query_embedding]).astype(np.float32), top_k)
            dense_scores = 1 / (1 + D[0])  # Convert distances to similarities
            doc_indices = I[0]
        
        # Sparse search with BM25
        query_tokens = self.tokenizer(query)
        sparse_scores = self.bm25.get_scores(query_tokens)
        
        # Combine scores
        final_results = []
        for i, dense_score in zip(doc_indices, dense_scores):
            doc = self.documents[i]
            sparse_score = sparse_scores[i]
            
            # Hybrid score
            hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
            
            final_results.append((doc, hybrid_score))
            
        # Sort by score
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        tracker.end(f"Hybrid search (top-{top_k})")
        return final_results

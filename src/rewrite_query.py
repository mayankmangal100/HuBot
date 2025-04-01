"""Query rewriting and understanding module."""

import time
from typing import List, Dict, Tuple, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import spacy
from spacy.tokens import Doc
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.conversation_store import ConversationStore
from src.utils import logger, LatencyTracker

class QueryRewriter:
    """
    Handles follow-up questions and standalone queries in a RAG-based chatbot system.
    Focuses on low latency (target: <500ms) and high accuracy for query understanding.
    """
    
    def __init__(
        self,
        query_rewrite_model: str = "MBZUAI/LaMini-Flan-T5-248M",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_history: int = 5,
        debug: bool = False
    ):
        """
        Initialize QueryRewriter with specified models.
        
        Args:
            query_rewrite_model: Model for query rewriting (needs to be small and fast)
            device: Device to run models on ('cuda' or 'cpu')
            max_history: Maximum number of conversation turns to retain
            debug: Whether to print debug information
        """
        self.device = device
        self.debug = debug
        
        # Initialize conversation store
        self.conversation = ConversationStore(max_history=max_history)
        
        # Query type detection thresholds
        self.similarity_threshold = 0.65  # Threshold for detecting follow-up questions
        self.new_info_threshold = 0.40   # Threshold for detecting new information
        
        # Load query rewriting model
        self._load_models(query_rewrite_model)
        
        if debug:
            logger.info(f"QueryRewriter initialized with model: {query_rewrite_model} on {device}")
    
    def _load_models(self, query_rewrite_model: str):
        """Load query rewriting model with optimizations for speed."""
        tracker = LatencyTracker().start()
        
        # Load small and fast query rewriting model
        self.tokenizer = AutoTokenizer.from_pretrained(query_rewrite_model)
        self.rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(
            query_rewrite_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # Load NLP pipeline for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback if model not installed
            logger.info("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Create optimized pipeline for entity detection
        self.entity_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER", 
            tokenizer="dslim/bert-base-NER",
            device=0 if self.device == "cuda" else -1,
            grouped_entities=True
        )
        
        tracker.end("Loading query rewriting models")
    
    def _detect_query_type(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> Tuple[str, float]:
        """Detect if query is standalone or a follow-up."""
            
        # Get last exchange
        last_exchange = self.conversation.get_last_exchange()
        if not last_exchange:
            logger.info("No last exchange found, query type: standalone")
            return "standalone", 1.0
            
        # Check for explicit references to previous context
        reference_indicators = [
            "it", "this", "that", "these", "those", "they", "them", 
            "he", "she", "his", "her", "their", "its",
            "the", "mentioned", "above", "previous", "before",
            "you said", "you mentioned", "you noted", "you stated",
            "what about", "how about", "tell me more", "elaborate", "explain"
        ]
        
        has_reference = any(indicator in query.lower() for indicator in reference_indicators)
        
        if has_reference:
            logger.info(f"Reference indicators found in query: {query}")
            return "followup", 0.9
            
        # Use embeddings for semantic similarity if available
        if query_embedding is not None and hasattr(last_exchange, 'embedding'):
            similarity = self._calculate_similarity(query_embedding, last_exchange.embedding)
            logger.info(f"Query similarity with last exchange: {similarity:.4f}")
            
            if similarity > self.similarity_threshold:
                return "followup", similarity
                
        # Default to standalone
        logger.info("No strong indicators for followup, query type: standalone")
        return "standalone", 0.8
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract named entities from query."""
        # Use spaCy for basic entity extraction
        doc = self.nlp(query)
        entities = {ent.text: {"type": ent.label_, "text": ent.text} for ent in doc.ents}
        
        # Use transformer model for additional entity detection
        # ner_results = self.entity_pipeline(query)
        # for ent in ner_results:
        #     text = ent["word"]
        #     if text not in entities:
        #         entities[text] = {"type": ent["entity"], "text": text}
        
        return entities
    
    def _rewrite_query(
        self,
        query: str,
        query_type: str,
        last_exchange: Optional[Dict[str, str]]
    ) -> Optional[str]:
        """Rewrite query if needed based on context."""
        if query_type != "followup":
            return None
            
        try:
            # Get last assistant response
            last_response = last_exchange.answer
            
            # Create prompt for rewriting
            prompt = f"""Given the conversation context and a follow-up question, rewrite the follow-up to be a standalone question.
            
Previous response: {last_response}

Follow-up question: {query}

Standalone question:"""
            
            # Generate rewritten query
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.rewrite_model.generate(
                **inputs,
                max_length=256,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )
            
            # Extract and clean rewritten query
            rewritten = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Log rewrite details
            logger.info(f"Query rewriting: '{query}' -> '{rewritten}'")
            
            return rewritten
            
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return None
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vec1, vec2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def process_query(
        self,
        query: str,
        query_embedding: List[float]
    ) -> Dict[str, Any]:
        """
        Process a query to detect type and rewrite if needed.
        Uses pre-computed query embedding from RAG system.
        """
        tracker = LatencyTracker().start()
        
        # 1. Detect query type
        query_type, confidence = self._detect_query_type(query, query_embedding)
        logger.info(f"Query type detected: {query_type}")
        
        # 2. Extract entities
        entities = self._extract_entities(query)
        self.conversation.update_entity_memory(entities)
        
        # 3. Rewrite query if needed
        last_exchange = self.conversation.get_last_exchange()
        rewritten_query = self._rewrite_query(query, query_type, last_exchange)
        if rewritten_query:
            logger.info(f"Rewritten query: {rewritten_query}")
        
        # 4. Return results
        processing_time = tracker.end("Query processing")
        
        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "query_type": query_type,
            "confidence": confidence,
            "entities": entities,
            "processing_time_ms": processing_time
        }
        
    def add_exchange(
        self,
        query: str,
        rewritten_query: str,
        answer: str,
        query_embedding: List[float],
        context_docs: List[Dict[str, Any]]
    ):
        """Add a completed exchange to conversation history."""
        try:
            # Add exchange to conversation store
            self.conversation.add_exchange(
                query=query,
                rewritten_query=rewritten_query or query,  # Use original if no rewrite
                answer=answer,
                query_embedding=query_embedding,
                context_docs=context_docs,
                metadata={}
            )
            logger.info(f"Added exchange to conversation history: '{query}' -> '{answer[:30]}...'")
        except Exception as e:
            logger.error(f"Error adding exchange to conversation history: {str(e)}")
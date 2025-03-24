"""Optimized embedding model with caching and batching."""

import torch
from typing import List, Union
import numpy as np
from src.utils import LatencyTracker
from functools import lru_cache
from src.utils import logger
import os


class EmbeddingModel:
    """Optimized embedding model with caching and quantization"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = None):
        self.model_name = model_name
        self.device = device or "cpu"  # Default to CPU if not specified
        
        logger.info(f"Using device: {self.device} for embeddings")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            
            # Apply dynamic quantization to individual modules instead of whole model
            try:
                logger.info("Applying dynamic quantization to embedding model")
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        module = torch.quantization.quantize_dynamic(
                            module,
                            {torch.nn.Linear},
                            dtype=torch.qint8
                        )
            except Exception as e:
                logger.warning(f"Quantization failed: {str(e)}")
                
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
            
    # Add LRU cache to avoid re-computing embeddings for the same text
    @lru_cache(maxsize=1024)
    def _embed_single_text(self, text: str) -> List[float]:
        """Generate embedding for a single text with caching"""
        with torch.no_grad():
            embedding = self.model.encode(text)
            
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy().tolist()
            
        return embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        tracker = LatencyTracker().start()
        
        # Process in optimal batch size
        batch_size = 8  # Smaller batch size for CPU
        all_embeddings = []
        
        # Use thread pool for parallel processing on CPU
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit tasks in batches
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                futures.append(executor.submit(self._batch_embed, batch))
                
            # Collect results
            for future in futures:
                all_embeddings.extend(future.result())
        
        tracker.end(f"Embedding {len(texts)} texts")
        return all_embeddings
    
    def _batch_embed(self, batch: List[str]) -> List[List[float]]:
        """Process a batch of embeddings"""
        with torch.no_grad():
            embeddings = self.model.encode(batch, convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy().tolist()
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        tracker = LatencyTracker().start()
        result = self._embed_single_text(query)
        tracker.end("Query embedding")
        return result

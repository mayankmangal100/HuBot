"""Text splitting module for document chunking."""

from typing import List, Dict, Any
from src.utils import LatencyTracker
from src.utils import logger


class TextSplitter:
    """Handles document chunking using various strategies"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except ImportError:
            logger.error("langchain_text_splitters not installed. Install with: pip install langchain-text-splitters")
            raise
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        tracker = LatencyTracker().start()
        
        if metadata is None:
            metadata = {}
            
        chunks = self.splitter.create_documents([text], [metadata])
        chunks_with_metadata = []
        
        for i, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                "id": f"{metadata.get('source', 'doc')}_{i}",
                "text": chunk.page_content,
                "metadata": {**chunk.metadata, "chunk_id": i}
            })
        
        tracker.end(f"Text splitting ({len(chunks_with_metadata)} chunks)")
        return chunks_with_metadata

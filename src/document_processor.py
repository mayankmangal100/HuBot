"""Document processor for extracting text from various file formats."""

import os
from typing import Dict, Any
from src.utils import LatencyTracker, logger

class DocumentProcessor:
    """Processes documents to extract text"""
    
    def __init__(self):
        try:
            from unstructured.partition.pdf import partition_pdf
            self.partition_pdf = partition_pdf
        except ImportError:
            logger.error("unstructured package not installed. Install with: pip install 'unstructured[all-docs]' python-magic-bin tesseract poppler-utils")
            raise
            
    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        tracker = LatencyTracker().start()
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Extract text using unstructured
            elements = self.partition_pdf(
                filename=file_path,
                strategy="fast",
                include_metadata=True
            )
            
            # Combine text from all elements
            text = "\n\n".join([str(element) for element in elements])
            
            tracker.end(f"PDF processing ({os.path.basename(file_path)})")
            return text
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            if "tesseract" in str(e).lower():
                logger.error("Tesseract not found. Please install tesseract and add it to PATH")
            elif "poppler" in str(e).lower():
                logger.error("Poppler not found. Please install poppler-utils")
            raise

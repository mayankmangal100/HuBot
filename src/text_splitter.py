"""Text splitting module for document chunking."""

import json
import re
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
    
    def split_text(self, grouped: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        tracker = LatencyTracker().start()
        
        if metadata is None:
            metadata = {}

        chunks_with_metadata = []
        chunk_id = 0
        

        def flush_chunk(texts: List[str], section_title: str) -> List[Dict[str, Any]]:
            nonlocal chunk_id
            final_chunks = []
            combined_text = "\n".join(texts).strip()

            if len(combined_text) > self.chunk_size:
                sub_chunks = self.splitter.create_documents([combined_text], [metadata])
                for sub in sub_chunks:
                    final_chunks.append({
                        "id": f"{metadata.get('source', 'doc')}_{chunk_id}",
                        "text": sub.page_content,
                        "metadata": {
                            **sub.metadata,
                            "chunk_id": chunk_id,
                            "section": section_title
                        }
                    })
                    chunk_id += 1
            else:
                final_chunks.append({
                    "id": f"{metadata.get('source', 'doc')}_{chunk_id}",
                    "text": combined_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": chunk_id,
                        "section": section_title
                    }
                })
                chunk_id += 1

            return final_chunks

        for section in grouped:
            chunks_with_metadata.extend(
                flush_chunk(section["content"], section["title"])
            )
            for subsection in section.get("subsections", []):
                full_title = f"{section['title']} > {subsection['title']}"
                chunks_with_metadata.extend(
                    flush_chunk(subsection["content"], full_title)
                )
        
        tracker.end(f"Logical Text splitting ({len(chunks_with_metadata)} chunks)")

        # with open("debug_chunks.json", "w", encoding="utf-8") as f:
        #     json.dump(chunks_with_metadata, f, indent=2, ensure_ascii=False)

        return chunks_with_metadata

            
        # # chunks = self.splitter.create_documents([text], [metadata])
        # all_split_chunks = []

        # for block in blocks:
        #     split_chunks = self.splitter.split_text(block["text"], block["metadata"])
        #     all_split_chunks.extend(split_chunks)
        # chunks_with_metadata = []
        
        # # for i, chunk in enumerate(chunks):
        # #     chunks_with_metadata.append({
        # #         "id": f"{metadata.get('source', 'doc')}_{i}",
        # #         "text": chunk.page_content,
        # #         "metadata": {**chunk.metadata, "chunk_id": i}
        # #     })

        # for i, chunk in enumerate(all_split_chunks):
        #     combined_metadata = {
        #         **chunk.metadata,
        #         "chunk_id": i,
        #         "page_number": chunk.metadata.get("page_number")
        #     }
            
        #     chunks_with_metadata.append({
        #         "id": f"{metadata.get('source', 'doc')}_{i}",
        #         "text": chunk.page_content,
        #         "metadata": combined_metadata
        #     })
        
        # tracker.end(f"Text splitting ({len(chunks_with_metadata)} chunks)")
        # return chunks_with_metadata

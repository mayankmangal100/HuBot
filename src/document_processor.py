"""Document processor for extracting text from various file formats."""

import os
from typing import Dict, Any, List
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
            
    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
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

            blocks = self.convert_elements_to_blocks(elements)
            
            grouped = self.group_by_title_with_subheaders(blocks)
            
            # Combine text from all elements
            # text = "\n\n".join([str(element) for element in elements])
            
            tracker.end(f"PDF processing ({os.path.basename(file_path)})")
            return grouped
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            if "tesseract" in str(e).lower():
                logger.error("Tesseract not found. Please install tesseract and add it to PATH")
            elif "poppler" in str(e).lower():
                logger.error("Poppler not found. Please install poppler-utils")
            raise

    
    def convert_elements_to_blocks(self, elements: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert unstructured elements to standardized blocks.
        """
        blocks = []
        for el in elements:
            el_type = el.__class__.__name__
            if el_type in ["Title", "NarrativeText", "ListItem", "Text"]:
                text = el.text.strip()
                page_number = getattr(el.metadata, "page_number", None)
                if text:
                    blocks.append({"type": el_type, "text": text, "page_number": page_number,})
        return blocks

    def group_by_title_with_subheaders(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group blocks using Title and nested Title elements (subheaders), with page number tracking.
        """
        grouped = []
        current_section = None
        current_subsection = None
        untitled_count = 0

        for block in blocks:
            page_number = block.get("page_number")

            if block["type"] == "Title":
                # Heuristic: If previous block was also a Title, it's likely a subsection
                if current_section and current_subsection is None:
                    current_subsection = {
                        "title": block["text"],
                        "content": [],
                        "page_number": page_number
                    }
                elif current_section and current_subsection:
                    # Save previous subsection
                    current_section.setdefault("subsections", []).append(current_subsection)
                    current_subsection = {
                        "title": block["text"],
                        "content": [],
                        "page_number": page_number
                    }
                else:
                    # New top-level section
                    if current_section:
                        # Save last subsection if any
                        if current_subsection:
                            current_section.setdefault("subsections", []).append(current_subsection)
                            current_subsection = None
                        grouped.append(current_section)
                    current_section = {
                        "title": block["text"],
                        "content": [],
                        "subsections": [],
                        "page_number": page_number
                    }
            else:
                # Narrative, ListItem, etc.
                if current_subsection:
                    current_subsection["content"].append(block["text"])
                elif current_section:
                    current_section["content"].append(block["text"])
                else:
                    # Fallback: No title seen yet
                    if not grouped or grouped[-1]["title"].startswith("Untitled"):
                        grouped.append({
                            "title": f"Untitled Section {untitled_count}",
                            "content": [block["text"]],
                            "subsections": [],
                            "page_number": page_number
                        })
                        untitled_count += 1
                    else:
                        grouped[-1]["content"].append(block["text"])

        # Save any remaining section/subsection
        if current_subsection and current_section:
            current_section.setdefault("subsections", []).append(current_subsection)
        if current_section:
            grouped.append(current_section)

        return grouped
"""Response generation module for RAG system."""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np

from src.utils import logger, LatencyTracker

class ResponseGenerator:
    """Generates coherent responses using retrieved documents and conversation history."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        temperature: float = 0.7,
        debug: bool = False
    ):
        """
        Initialize response generator.
        
        Args:
            model_name: Name of the language model to use
            device: Device to run model on
            max_length: Maximum length of generated responses
            temperature: Sampling temperature for generation
            debug: Whether to print debug information
        """
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.debug = debug
        
        # Load model and tokenizer
        self._load_model(model_name)
        
        if debug:
            logger.info(f"ResponseGenerator initialized with model: {model_name}")
    
    def _load_model(self, model_name: str):
        """Load language model with optimizations."""
        tracker = LatencyTracker().start()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # Create optimized pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        tracker.end("Loading response generation model")
    
    def _format_context(
        self,
        docs: List[Dict[str, Any]],
        max_tokens: int = 1024
    ) -> str:
        """Format retrieved documents into context string."""
        context = []
        current_tokens = 0
        
        for doc in docs:
            # Estimate tokens (rough approximation)
            doc_tokens = len(doc["content"].split())
            if current_tokens + doc_tokens > max_tokens:
                break
                
            context.append(f"Document: {doc['content']}")
            current_tokens += doc_tokens
        
        return "\n\n".join(context)
    
    def _create_prompt(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Create a prompt for response generation."""
        # Start with system instruction
        prompt = [
            "You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely.",
            "If you're not sure about something, say so rather than making things up.",
            "Always base your answers on the given context.\n"
        ]
        
        # Add chat history if available
        if chat_history:
            for exchange in chat_history[-2:]:  # Last 2 exchanges
                prompt.append(f"Human: {exchange['query']}")
                prompt.append(f"Assistant: {exchange['answer']}\n")
        
        # Add context and current query
        prompt.extend([
            "Context:",
            context,
            "\nHuman: " + query,
            "\nAssistant:"
        ])
        
        return "\n".join(prompt)
    
    def generate_response(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Generate a response using retrieved documents and chat history.
        
        Args:
            query: User's query
            docs: Retrieved relevant documents
            chat_history: Optional chat history
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing:
                - response: Generated response
                - confidence: Model's confidence score
                - processing_time_ms: Processing time
        """
        tracker = LatencyTracker().start()
        
        # Format context from documents
        context = self._format_context(docs)
        
        # Create generation prompt
        prompt = self._create_prompt(query, context, chat_history)
        
        # Generate response
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # Extract response and clean it
            response = outputs[0]["generated_text"]
            response = response.split("Assistant:")[-1].strip()
            
            # Calculate confidence (using logits from last token)
            confidence = float(torch.softmax(
                torch.tensor(outputs[0]["scores"][-1]), dim=0
            ).max())
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            response = "I apologize, but I encountered an error while generating a response."
            confidence = 0.0
        
        processing_time = tracker.end("Full response generation")
        
        return {
            "response": response,
            "confidence": confidence,
            "processing_time_ms": processing_time,
            "processing_details": tracker.get_marks()
        }

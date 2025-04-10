"""Optimized interface for CPU inference using llama-cpp."""

import os
import psutil
from typing import Dict, Any, Optional, Generator
from llama_cpp import Llama
from src.utils import logger, LatencyTracker

class LLMInterface:
    """Optimized interface for fast CPU inference using llama-cpp."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM interface with CPU optimizations."""
        self.config = config
        
        # Optimize thread usage - use physical cores
        self.n_threads = min(8, psutil.cpu_count(logical=False))
        
        # Optimize context window based on available memory
        # available_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        # self.n_ctx = max(2048, min(8192, int(available_ram * 256))) # Ensure context window is at least 2048
        self.n_ctx = 2048
        # Load model with optimizations
        self._load_model()
        
        # For storing streamed responses
        self._last_streamed_response = None
    
    def _load_model(self):
        """Load model with memory and performance optimizations."""
        tracker = LatencyTracker().start()
        
        try:
            model_dir = os.path.join(os.getcwd(), "models")
            model_path = os.path.join(model_dir, self.config["model"]["llm_model_path"])
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Initialize with optimized parameters
            self.llm = Llama(
                model_path=model_path,
                n_threads=self.n_threads,
                n_ctx=self.n_ctx,
                n_batch=512,      # Increased batch size for better throughput
                verbose=False,
                use_mlock=True,   # Lock memory to prevent swapping
                use_mmap=True,    # Use memory mapping for faster loading
                n_gpu_layers=0,
                seed=-1    # CPU-only mode
            )
            
            logger.info(
                f"Model loaded with optimizations:\n"
                f"- Model: {self.config['model']['llm_model_path']}\n"
                f"- Threads: {self.n_threads}\n"
                f"- Context: {self.n_ctx}\n"
                f"- Batch size: 256"
            )
            tracker.end("Model loading")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create an optimized prompt format."""
        system_prompt = """You are a helpful HR assistant. Follow these rules strictly:
- Use only the information from the Context.
- If an answer can be inferred from the Context (e.g., based on numbers, limits, dates), do so and explain briefly.
- DO NOT invent any facts not supported by the Context.
- If nothing at all is relevant to the question, respond with: "No relevant information found in the handbook."
- Avoid saying 'context' — say 'the handbook' instead.
- Be concise, professional, and accurate."""
        
        prompt = f"""<s>[INST] {system_prompt}

Question: {question}

Context:
{context}

Answer:[/INST] """
        return prompt
    
    def generate_answer(
        self,
        question: str,
        context: str = "",
        isStream: bool = False
    ) -> str | Generator[str, None, None]:
        """Generate answer with optimized parameters."""
        tracker = LatencyTracker().start()
        
        try:
            # Create prompt
            prompt = self._create_prompt(question, context)
            print(context)
            max_tokens = self.config["llm"]["max_tokens"]
            
            # Check prompt length
            if len(prompt) > (self.n_ctx - max_tokens - 20):  # Add safety margin
                # Truncate context to fit within context window
                max_context_len = self.n_ctx - max_tokens - len(question) - 100  # Extra safety margin
                context = context[:max_context_len] + "..." if len(context) > max_context_len else context
                prompt = self._create_prompt(question, context)
                logger.info(f"Context truncated to {len(context)} characters")
            
            # Set generation parameters
            params = {
                "max_tokens": max_tokens,
                "temperature": self.config["llm"]["temperature"],
                "stop": ["</s>", "[INST]", "Context:", "Question:", "[/INST]"],
                "top_p": self.config["llm"]["top_p"],
                "top_k": self.config["llm"]["top_k"],
            }
            
            # Generate response
            if isStream:
                # For streaming responses
                response = self.llm(prompt, stream=True, **params)
                return self._handle_streaming(response, tracker)
            else:
                # For non-streaming responses
                response = self.llm(prompt, **params)
                return self._handle_single_response(response, tracker)
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while generating the answer. Please try again."
            
    def get_last_streamed_response(self) -> str:
        """Get the last complete streamed response for conversation history."""
        if self._last_streamed_response is None:
            return "No streamed response available"
        return self._last_streamed_response
    
    def _handle_streaming(
        self,
        response: Generator,
        tracker: LatencyTracker
    ) -> Generator[str, None, None]:
        """Handle streaming response with proper cleanup."""
        try:
            full_response = ""
            for chunk in response:
                if "choices" in chunk and chunk["choices"]:
                    token = chunk["choices"][0]["text"]
                    full_response += token
                    yield token
                else:
                    break
                    
            tracker.end("Streaming generation")
            
            # Store the complete response for conversation history
            self._last_streamed_response = full_response
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield " [Error during streaming]"
    
    def _handle_single_response(
        self,
        response: Dict[str, Any],
        tracker: LatencyTracker
    ) -> str:
        """Handle single response with validation."""
        if response["choices"] and response["choices"][0]["text"]:
            answer = response["choices"][0]["text"].strip()
        else:
            logger.error(f"Empty response from LLM: {response}")
            answer = "I apologize, but I was unable to generate a response."
        
        processing_time = tracker.end("Complete generation")
        logger.debug(f"Answer generated in {processing_time:.2f}ms")
        
        return answer

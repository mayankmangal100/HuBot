"""Optimized interface for interacting with LLMs."""

import os
import psutil
from typing import Dict, Any
from src.utils import LatencyTracker
from src.utils import logger

class LLMInterface:
    """Optimized interface for interacting with LLMs"""
    
    def __init__(self, model_name: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf", config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.n_threads = min(8, psutil.cpu_count(logical=False))
        self.n_ctx = 2048  # Increased context window
        
        try:
            from llama_cpp import Llama
            
            model_dir = os.path.join(os.getcwd(), "models")
            model_path = os.path.join(model_dir, model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            logger.info(f"Loading model from {model_path}")
            self.llm = Llama(
                model_path=model_path,
                n_threads=self.n_threads,
                n_ctx=self.n_ctx,
                n_batch=512,
                verbose=False,
                use_mlock=True,
                use_mmap=True,
                n_gpu_layers=0  # Use CPU only for now
            )
            
            logger.info(f"LLM initialized with {self.n_threads} threads, context size: {self.n_ctx}")
            
        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise
            
    def generate_answer(self, question: str, context: str, max_tokens: int = 512, isStream: bool = False) -> str:
        """Generate an answer given a question and context"""
        tracker = LatencyTracker().start()
        
        # Format prompt with clear instruction
        prompt = f"""[INST] You are a helpful AI assistant. Use the following context to answer the question. Be direct and concise.

Context:
{context}

Question: {question}

Answer: [/INST]"""

        logger.debug(f"Prompt length: {len(prompt.split())}")
        
        try:
            # Generate response with optimized parameters
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,  # Increased for more exploratory responses
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stream=isStream
            )

            if isStream:
                full_response = []
                for chunk in response:
                    if "choices" in chunk and chunk["choices"]:
                        token = chunk["choices"][0]["text"]
                        # print(token, end="", flush=True)
                        full_response.append(token)
                        yield token
                    else:
                        break
                print()
                answer = "".join(full_response)
            else:
                if response["choices"] and response["choices"][0]["text"]:
                    answer = response["choices"][0]["text"].strip()
                else:
                    logger.error(f"Empty response from LLM: {response}")
                    answer = "I apologize, but I was unable to generate a response. Please try rephrasing your question."

            tracker.end("Answer generation")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while generating the response. Please try again."
            
    def rewrite_question(self, current_question: str, conversation_history: str = "") -> str:
        """Rewrite a follow-up question as a standalone question"""
        tracker = LatencyTracker().start()
        
        prompt = f"""[INST] Given the conversation history below, rewrite the follow-up question as a standalone question that captures the full context. If the question is already standalone, return it as is.

Conversation history: {conversation_history}

Question to rewrite: {current_question}

Rewritten question: [/INST]"""
        
        response = self.llm(
            prompt,
            max_tokens=128,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.1
        )
        
        rewritten = response["choices"][0]["text"].strip()
        tracker.end("Question rewriting")
        return rewritten

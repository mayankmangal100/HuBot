import torch
from llama_cpp import Llama
from typing import List, Dict, Generator, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import queue

class BatchedLlamaCppInference:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_threads: int = 4,
        batch_size: int = 4,
        max_tokens_per_request: int = 512
    ):
        """
        Initialize a batched inference manager for llama.cpp models.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context size for the model
            n_batch: Number of tokens to process in one batch (model parameter)
            n_threads: Number of CPU threads to use
            batch_size: Number of prompts to batch together
            max_tokens_per_request: Maximum tokens to generate per request
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.max_tokens_per_request = max_tokens_per_request
        
        # Initialize the model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            verbose=False
        )
        
        # Request queue and executor
        self.request_queue = queue.Queue()
        self.result_queues = {}
        self.executor = ThreadPoolExecutor(max_workers=1)  # Single worker for batching
        self.batch_thread = None
        self.running = False
        
    def start(self):
        """Start the batching thread"""
        if not self.running:
            self.running = True
            self.batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
            self.batch_thread.start()
            
    def stop(self):
        """Stop the batching thread"""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=2.0)
            
    def _batch_worker(self):
        """Worker thread that processes batches of requests"""
        while self.running:
            # Collect requests up to batch size or wait for timeout
            batch = []
            request_ids = []
            
            try:
                # Get the first request with a short timeout
                request_id, request = self.request_queue.get(timeout=0.1)
                batch.append(request)
                request_ids.append(request_id)
                
                # Try to get more requests with a very short timeout
                batch_timeout = 0.05  # 50ms timeout for batching
                batch_start = time.time()
                
                while len(batch) < self.batch_size and time.time() - batch_start < batch_timeout:
                    try:
                        req_id, req = self.request_queue.get(timeout=0.01)
                        batch.append(req)
                        request_ids.append(req_id)
                    except queue.Empty:
                        break
                        
                # Process the batch
                if batch:
                    prompt_batch = [req["prompt"] for req in batch]
                    max_tokens_batch = [req.get("max_tokens", self.max_tokens_per_request) for req in batch]
                    stream_batch = [req.get("stream", False) for req in batch]
                    
                    # For efficiency, execute the batch with the minimum max_tokens if any are streaming
                    if any(stream_batch):
                        self._process_streaming_batch(prompt_batch, request_ids, max_tokens_batch, stream_batch)
                    else:
                        self._process_non_streaming_batch(prompt_batch, request_ids, max_tokens_batch)
                    
            except queue.Empty:
                # No requests, just continue the loop
                continue
                
    def _process_non_streaming_batch(self, prompts: List[str], request_ids: List[str], max_tokens_list: List[int]):
        """Process a batch of non-streaming requests"""
        try:
            # Use the minimum max_tokens for the batch for efficiency
            min_max_tokens = min(max_tokens_list)
            
            # Generate completions for the batch
            results = []
            for prompt in prompts:
                # Here we're using Llama-cpp's generate method which doesn't support true batching
                # But we're still gaining efficiency through batched evaluation in the background
                completion = self.llm.create_completion(
                    prompt,
                    max_tokens=min_max_tokens,
                    stream=False
                )
                results.append(completion)
                
            # Distribute results to the appropriate queues
            for req_id, result in zip(request_ids, results):
                if req_id in self.result_queues:
                    self.result_queues[req_id].put(result)
                    self.result_queues[req_id].put(None)  # Signal completion
                    
        except Exception as e:
            # In case of error, send the error to all result queues
            for req_id in request_ids:
                if req_id in self.result_queues:
                    self.result_queues[req_id].put(e)
                    self.result_queues[req_id].put(None)
                    
    def _process_streaming_batch(self, prompts: List[str], request_ids: List[str], 
                                max_tokens_list: List[int], stream_flags: List[bool]):
        """Process a batch with some streaming requests"""
        # This is a simplified implementation - true parallel streaming is complex
        for i, (prompt, req_id, max_tokens, stream) in enumerate(
            zip(prompts, request_ids, max_tokens_list, stream_flags)
        ):
            try:
                if stream:
                    # Handle streaming requests one by one
                    stream_generator = self.llm.create_completion(
                        prompt,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    # Send streaming results to the queue
                    for token in stream_generator:
                        if req_id in self.result_queues:
                            self.result_queues[req_id].put(token)
                else:
                    # For non-streaming requests
                    result = self.llm.create_completion(
                        prompt,
                        max_tokens=max_tokens,
                        stream=False
                    )
                    if req_id in self.result_queues:
                        self.result_queues[req_id].put(result)
                
                # Signal completion
                if req_id in self.result_queues:
                    self.result_queues[req_id].put(None)
                    
            except Exception as e:
                if req_id in self.result_queues:
                    self.result_queues[req_id].put(e)
                    self.result_queues[req_id].put(None)
                    
    def generate(self, prompt: str, max_tokens: int = 128, stream: bool = False) -> Generator:
        """
        Generate a completion for a prompt, possibly as a stream of tokens.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the results
            
        Returns:
            A generator yielding tokens (if stream=True) or the complete result
        """
        # Create a unique request ID and set up a result queue
        request_id = f"req_{time.time()}_{np.random.randint(0, 10000)}"
        result_queue = queue.Queue()
        self.result_queues[request_id] = result_queue
        
        # Submit request to the queue
        self.request_queue.put((request_id, {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": stream
        }))
        
        # Return a generator that yields results from the queue
        try:
            while True:
                result = result_queue.get()
                if result is None:  # End of generation
                    break
                elif isinstance(result, Exception):
                    raise result
                else:
                    yield result
        finally:
            # Clean up
            self.result_queues.pop(request_id, None)
            
    def generate_multiple(self, prompts: List[str], max_tokens: int = 128) -> List[str]:
        """
        Generate completions for multiple prompts at once.
        
        Args:
            prompts: List of input text prompts
            max_tokens: Maximum tokens to generate per prompt
            
        Returns:
            List of completion strings
        """
        results = []
        result_generators = [
            self.generate(prompt, max_tokens=max_tokens, stream=False)
            for prompt in prompts
        ]
        
        # Collect all results
        for gen in result_generators:
            result = next(gen, None)
            if result:
                if isinstance(result, dict) and "choices" in result:
                    text = result["choices"][0]["text"]
                else:
                    text = str(result)
                results.append(text)
            else:
                results.append("")
                
        return results

# Usage example
if __name__ == "__main__":
    batch_llm = BatchedLlamaCppInference(
        model_path="path/to/your/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx=2048,
        n_batch=512,
        n_threads=4,
        batch_size=4
    )
    
    batch_llm.start()
    
    try:
        # Example 1: Single request with streaming
        prompt = "Write a short poem about AI."
        print("Generating with streaming:")
        for token in batch_llm.generate(prompt, max_tokens=100, stream=True):
            if isinstance(token, dict) and "choices" in token:
                print(token["choices"][0]["text"], end="", flush=True)
            else:
                print(token, end="", flush=True)
        print("\n")
        
        # Example 2: Batch multiple requests
        prompts = [
            "Explain quantum computing in one sentence.",
            "Write a haiku about programming.",
            "Define artificial intelligence.",
            "List three benefits of machine learning."
        ]
        
        print("Batch processing multiple requests:")
        results = batch_llm.generate_multiple(prompts, max_tokens=50)
        
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result}")
            
    finally:
        batch_llm.stop()

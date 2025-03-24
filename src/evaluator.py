"""Evaluator module for assessing RAG system performance."""

import numpy as np
from typing import List, Dict, Any
from src.utils import LatencyTracker, logger
from rouge_score import rouge_scorer
from bert_score import score
import torch

class RAGEvaluator:
    """Evaluates RAG system performance using various metrics"""
    
    def __init__(self, use_gpu: bool = False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict[str, Any]], relevant_doc_ids: List[str]) -> Dict[str, float]:
        """Evaluate retrieval performance using precision, recall, and MRR"""
        tracker = LatencyTracker().start()
        
        # Get retrieved document IDs
        retrieved_ids = [doc["id"] for doc, _ in retrieved_docs]
        
        # Calculate metrics
        true_positives = set(retrieved_ids).intersection(relevant_doc_ids)
        
        # Precision and recall
        precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(true_positives) / len(relevant_doc_ids) if relevant_doc_ids else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_doc_ids:
                mrr = 1 / (i + 1)
                break
                
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr
        }
        
        tracker.end("Retrieval evaluation")
        return metrics
        
    def evaluate_answer(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """Evaluate answer quality using ROUGE and BERTScore"""
        tracker = LatencyTracker().start()
        
        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
        
        # Calculate BERTScore
        P, R, F1 = score([generated_answer], [reference_answer], lang="en", device=self.device)
        bert_f1 = F1.mean().item()
        
        metrics = {
            "rouge1_f1": rouge_scores["rouge1"].fmeasure,
            "rouge2_f1": rouge_scores["rouge2"].fmeasure,
            "rougeL_f1": rouge_scores["rougeL"].fmeasure,
            "bert_score": bert_f1
        }
        
        tracker.end("Answer evaluation")
        return metrics
        
    def evaluate_latency(self, latencies: Dict[str, float]) -> Dict[str, float]:
        """Evaluate system latency metrics"""
        metrics = {
            "mean": np.mean(list(latencies.values())),
            "p50": np.percentile(list(latencies.values()), 50),
            "p90": np.percentile(list(latencies.values()), 90),
            "p95": np.percentile(list(latencies.values()), 95),
            "p99": np.percentile(list(latencies.values()), 99)
        }
        return metrics
        
    def log_metrics(self, metrics: Dict[str, Dict[str, float]], experiment_name: str = "default"):
        """Log evaluation metrics"""
        logger.info(f"\nEvaluation Results ({experiment_name}):")
        
        for category, values in metrics.items():
            logger.info(f"\n{category.upper()}:")
            for metric, value in values.items():
                logger.info(f"{metric}: {value:.4f}")
                
    def evaluate_all(self, 
                    query: str,
                    retrieved_docs: List[Dict[str, Any]],
                    relevant_doc_ids: List[str],
                    generated_answer: str,
                    reference_answer: str,
                    latencies: Dict[str, float],
                    experiment_name: str = "default") -> Dict[str, Dict[str, float]]:
        """Run all evaluations and return combined metrics"""
        
        # Get all metrics
        retrieval_metrics = self.evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)
        answer_metrics = self.evaluate_answer(generated_answer, reference_answer)
        latency_metrics = self.evaluate_latency(latencies)
        
        # Combine metrics
        all_metrics = {
            "retrieval": retrieval_metrics,
            "answer": answer_metrics,
            "latency": latency_metrics
        }
        
        # Log results
        self.log_metrics(all_metrics, experiment_name)
        
        return all_metrics

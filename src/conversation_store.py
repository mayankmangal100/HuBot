"""Centralized conversation history management."""

from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConversationExchange:
    """Single exchange in a conversation"""
    query: str
    rewritten_query: str
    answer: str
    query_embedding: List[float]
    timestamp: datetime
    context_docs: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ConversationStore:
    """Manages conversation history and state"""
    
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.reset()
    
    def reset(self):
        """Reset conversation state"""
        self.history: List[ConversationExchange] = []
        self.entity_memory = {}  # Track entities mentioned in conversation
    
    def add_exchange(
        self,
        query: str,
        rewritten_query: str,
        answer: str,
        query_embedding: List[float],
        context_docs: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ):
        """Add a new exchange to history"""
        exchange = ConversationExchange(
            query=query,
            rewritten_query=rewritten_query,
            answer=answer,
            query_embedding=query_embedding,
            timestamp=datetime.now(),
            context_docs=context_docs,
            metadata=metadata or {}
        )
        
        self.history.append(exchange)
        
        # Maintain max history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_last_exchange(self) -> ConversationExchange:
        """Get the last exchange in history"""
        return self.history[-1] if self.history else None
    
    def get_recent_exchanges(self, n: int = None) -> List[ConversationExchange]:
        """Get n most recent exchanges"""
        n = n or self.max_history
        return self.history[-n:]
    
    def update_entity_memory(self, entities: Dict[str, Any]):
        """Update entity memory with new entities"""
        self.entity_memory.update(entities)
    
    def get_entity_memory(self) -> Dict[str, Any]:
        """Get current entity memory"""
        return self.entity_memory.copy()

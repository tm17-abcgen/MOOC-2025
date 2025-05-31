"""
Simplified Embedding Database for DSPy Lean 4 workflow

This module provides RAG capabilities for the DSPy agents.
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

class EmbeddingDB:
    """Simple embedding database for RAG support."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        
    def load_from_pickle(self, chunks_path: str, embeddings_path: str = None):
        """Load chunks and optionally embeddings from pickle files."""
        # Load chunks
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Load embeddings if path provided
        if embeddings_path and os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        elif embeddings_path:
            # Try with .npy extension
            npy_path = embeddings_path.replace('.pkl', '.npy')
            if os.path.exists(npy_path):
                self.embeddings = np.load(npy_path)
        
        print(f"[EmbeddingDB] Loaded {len(self.chunks)} chunks")
        if self.embeddings is not None:
            print(f"[EmbeddingDB] Loaded embeddings with shape {self.embeddings.shape}")
    
    def generate_embeddings(self):
        """Generate embeddings for all chunks if not already loaded."""
        if self.embeddings is None and self.chunks:
            print(f"[EmbeddingDB] Generating embeddings for {len(self.chunks)} chunks...")
            self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)
            print(f"[EmbeddingDB] Generated embeddings with shape {self.embeddings.shape}")
    
    def get_similar_examples(self, query: str, k: int = 5) -> List[str]:
        """
        Get k most similar examples to the query.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar text chunks
        """
        if not self.chunks:
            return []
        
        # Ensure embeddings exist
        if self.embeddings is None:
            self.generate_embeddings()
        
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return corresponding chunks
        return [self.chunks[i] for i in top_k_indices]
    
    def search_proofs(self, query: str, k: int = 3) -> List[str]:
        """Search specifically for proof examples."""
        # Add "proof" context to query
        proof_query = f"Lean 4 proof {query}"
        return self.get_similar_examples(proof_query, k)
    
    def search_implementations(self, query: str, k: int = 3) -> List[str]:
        """Search specifically for implementation examples."""
        # Add implementation context to query  
        impl_query = f"Lean 4 function implementation {query}"
        return self.get_similar_examples(impl_query, k)

# Compatibility class for existing code
class VectorDB:
    """Compatibility wrapper for VectorDB interface."""
    
    @staticmethod
    def get_top_k(npy_file: str, embedding_model, query: str, k: int = 5, verbose: bool = False) -> Tuple[List[str], List[float]]:
        """Get top k results - compatibility method."""
        # Create EmbeddingDB instance
        db = EmbeddingDB()
        
        # Load data
        chunks_file = npy_file.replace('.npy', '_chunks.pkl')
        if os.path.exists(chunks_file):
            db.load_from_pickle(chunks_file, npy_file)
        else:
            print(f"[VectorDB] Warning: Chunks file not found: {chunks_file}")
            return [], []
        
        # Get similar examples
        results = db.get_similar_examples(query, k)
        
        # Return with dummy scores for compatibility
        scores = [1.0 - (i * 0.1) for i in range(len(results))]
        
        if verbose:
            for i, (chunk, score) in enumerate(zip(results, scores)):
                print(f"Result #{i+1} (Score: {score:.4f})")
                print(f"{chunk[:200]}...")
                print("-" * 50)
        
        return results, scores
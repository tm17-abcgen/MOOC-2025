#!/usr/bin/env python3
"""
Generate embeddings from Lean 4 documentation for RAG support.

This script processes all documentation files and creates embeddings
for the DSPy workflow to use.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import argparse
from tqdm import tqdm

def chunk_text(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better embeddings.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    sentences = text.split('\n\n')  # Split by paragraphs
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def read_documents(docs_dir: str) -> List[str]:
    """
    Read all text files from the documents directory.
    
    Args:
        docs_dir: Path to documents directory
        
    Returns:
        List of text chunks from all documents
    """
    all_chunks = []
    
    print(f"[EMBEDDINGS] Reading documents from {docs_dir}")
    
    for filename in os.listdir(docs_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(docs_dir, filename)
            print(f"  Processing {filename}...")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add document context to chunks
            doc_name = filename.replace('.txt', '').replace('_', ' ').title()
            
            # Chunk the document
            chunks = chunk_text(content, max_chunk_size=500)
            
            # Add document context to each chunk
            for i, chunk in enumerate(chunks):
                contextualized_chunk = f"Document: {doc_name}\n\n{chunk}"
                all_chunks.append(contextualized_chunk)
    
    print(f"[EMBEDDINGS] Generated {len(all_chunks)} chunks from {len(os.listdir(docs_dir))} documents")
    return all_chunks

def generate_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embeddings for text chunks.
    
    Args:
        chunks: List of text chunks
        model_name: Sentence transformer model name
        
    Returns:
        NumPy array of embeddings
    """
    print(f"[EMBEDDINGS] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"[EMBEDDINGS] Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)
    
    print(f"[EMBEDDINGS] Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def save_embeddings(chunks: List[str], embeddings: np.ndarray, output_dir: str):
    """
    Save chunks and embeddings to files.
    
    Args:
        chunks: Text chunks
        embeddings: Corresponding embeddings
        output_dir: Directory to save files
    """
    # Save chunks
    chunks_path = os.path.join(output_dir, "lean4_embeddings_chunks.pkl")
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"[EMBEDDINGS] Saved chunks to: {chunks_path}")
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "lean4_embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"[EMBEDDINGS] Saved embeddings to: {embeddings_path}")
    
    # Create index for quick reference
    index_data = {
        'num_chunks': len(chunks),
        'embedding_dim': embeddings.shape[1],
        'model_used': 'all-MiniLM-L6-v2',
        'chunk_preview': [chunk[:100] + "..." if len(chunk) > 100 else chunk for chunk in chunks[:5]]
    }
    
    index_path = os.path.join(output_dir, "embedding_index.json")
    import json
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    print(f"[EMBEDDINGS] Saved index to: {index_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from Lean 4 documentation")
    parser.add_argument("--docs-dir", type=str, default="documents",
                       help="Directory containing documentation files")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Directory to save embeddings")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence transformer model to use")
    
    args = parser.parse_args()
    
    # Resolve paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, args.docs_dir)
    output_dir = os.path.join(current_dir, args.output_dir)
    
    if not os.path.exists(docs_dir):
        print(f"[ERROR] Documents directory not found: {docs_dir}")
        return
    
    # Read documents
    chunks = read_documents(docs_dir)
    
    if not chunks:
        print("[ERROR] No text chunks found!")
        return
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks, args.model)
    
    # Save everything
    save_embeddings(chunks, embeddings, output_dir)
    
    print(f"\n[SUCCESS] Embeddings generation complete!")
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Files saved to: {output_dir}")

if __name__ == "__main__":
    main()
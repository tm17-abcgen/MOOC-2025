#!/usr/bin/env python3
"""
Test script to verify DSPy setup is working correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test environment setup."""
    print("🔧 Testing Environment Setup...")
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print("  ✓ OPENROUTER_API_KEY is set")
    else:
        print("  ✗ OPENROUTER_API_KEY not found")
        return False
    
    # Check embeddings
    if os.path.exists("lean4_embeddings_chunks.pkl"):
        print("  ✓ Embeddings chunks file found")
    else:
        print("  ✗ Embeddings chunks file not found")
        return False
    
    if os.path.exists("lean4_embeddings.npy"):
        print("  ✓ Embeddings vectors file found")
    else:
        print("  ✗ Embeddings vectors file not found")
        return False
    
    return True

def test_imports():
    """Test that all required imports work."""
    print("\n📦 Testing Imports...")
    
    try:
        import dspy
        print("  ✓ DSPy imported successfully")
    except ImportError as e:
        print(f"  ✗ DSPy import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("  ✓ SentenceTransformers imported successfully")
    except ImportError as e:
        print(f"  ✗ SentenceTransformers import failed: {e}")
        return False
    
    try:
        from src.dspy_agents import DSPyPlanningAgent, configure_openrouter_lm
        print("  ✓ DSPy agents imported successfully")
    except ImportError as e:
        print(f"  ✗ DSPy agents import failed: {e}")
        return False
    
    return True

def test_openrouter_connection():
    """Test OpenRouter connection."""
    print("\n🌐 Testing OpenRouter Connection...")
    
    try:
        from src.dspy_agents import configure_openrouter_lm
        
        # Try to configure with a simple model
        lm = configure_openrouter_lm("gpt-4o")
        print("  ✓ OpenRouter LM configuration successful")
        
        # Try a simple DSPy call
        import dspy
        
        # Simple signature for testing
        class SimpleTest(dspy.Signature):
            question = dspy.InputField()
            answer = dspy.OutputField()
        
        predictor = dspy.ChainOfThought(SimpleTest)
        result = predictor(question="What is 2+2?")
        
        if result.answer:
            print("  ✓ DSPy prediction successful")
            print(f"    Answer: {result.answer[:50]}...")
            return True
        else:
            print("  ✗ DSPy prediction returned empty result")
            return False
            
    except Exception as e:
        print(f"  ✗ OpenRouter test failed: {e}")
        return False

def test_embeddings_rag():
    """Test embeddings and RAG functionality."""
    print("\n🔍 Testing RAG System...")
    
    try:
        from src.embedding_db import EmbeddingDB
        
        # Initialize embedding database
        db = EmbeddingDB()
        db.load_from_pickle("lean4_embeddings_chunks.pkl", "lean4_embeddings.npy")
        
        print(f"  ✓ Loaded {len(db.chunks)} chunks")
        
        # Test similarity search
        results = db.get_similar_examples("natural numbers addition", k=3)
        
        if results:
            print(f"  ✓ RAG search returned {len(results)} results")
            print(f"    Sample: {results[0][:100]}...")
            return True
        else:
            print("  ✗ RAG search returned no results")
            return False
            
    except Exception as e:
        print(f"  ✗ RAG test failed: {e}")
        return False

def main():
    print("🚀 DSPy Lean 4 Code Generator - Setup Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_environment()
    all_tests_passed &= test_imports()
    all_tests_passed &= test_embeddings_rag()
    
    # Only test OpenRouter if other tests pass
    if all_tests_passed:
        all_tests_passed &= test_openrouter_connection()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  - Test a single task: python run_dspy_tests.py --task task_id_0")
        print("  - Test all tasks: python run_dspy_tests.py --all")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
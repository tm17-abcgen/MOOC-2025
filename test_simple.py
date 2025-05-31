#!/usr/bin/env python3
"""
Test simple DSPy agent with very explicit instructions.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables and fix warnings
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import dspy
from dspy_agents import configure_openrouter_lm

# Simple test signature
class SimpleCodeGen(dspy.Signature):
    """Generate ONLY a single Lean 4 expression - nothing else!
    
    Task: You must generate EXACTLY one expression that would replace {{code}}.
    - For identity function: return exactly "x"
    - For addition: return exactly "a + b"  
    - NO imports, NO definitions, NO comments, NO extra text
    """
    problem = dspy.InputField(desc="What the function should do")
    template_context = dspy.InputField(desc="Context about the Lean template")
    
    expression = dspy.OutputField(desc="EXACTLY ONE Lean expression - nothing more!")

def test_simple_generation():
    """Test with very simple, explicit instructions."""
    print("🧪 Testing simple code generation...")
    
    try:
        # Configure OpenRouter
        configure_openrouter_lm("gpt-4o")
        
        # Create predictor
        predictor = dspy.ChainOfThought(SimpleCodeGen)
        
        # Test with identity function
        result = predictor(
            problem="Create an identity function that returns its input unchanged",
            template_context="Template: def ident (x : Nat) : Nat := {{code}}"
        )
        
        print(f"  Generated: '{result.expression}'")
        print(f"  Expected: 'x'")
        
        if result.expression.strip() == "x":
            print("  ✅ Perfect!")
        else:
            print(f"  ⚠️  Needs cleaning: '{result.expression.strip()}'")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    test_simple_generation()
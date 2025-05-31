#!/usr/bin/env python3
"""
Manual signature optimization by adding few-shot examples.
"""

# Example of manually optimized signature with few-shot examples

OPTIMIZED_CODE_SIGNATURE = '''
class OptimizedCodeGeneration(dspy.Signature):
    """Generate ONLY the function body expression for Lean 4 template replacement.
    
    Examples:
    Input: "identity function that returns its input" 
    Template: "def ident (x : Nat) : Nat := {{code}}"
    Output: "x"
    
    Input: "check if number is divisible by 11"
    Template: "def isDivisibleBy11 (n : Int) : Bool := {{code}}"  
    Output: "n % 11 == 0"
    
    Input: "add two numbers"
    Template: "def add (a b : Nat) : Nat := {{code}}"
    Output: "a + b"
    
    Input: "return zero"
    Template: "def zero (_ : Nat) : Nat := {{code}}"
    Output: "0"
    """
    problem_description = dspy.InputField(desc="What the function should do")
    lean_template = dspy.InputField(desc="The Lean template showing function signature")
    
    expression = dspy.OutputField(desc="Single Lean expression to replace {{code}}")
'''

OPTIMIZED_PROOF_SIGNATURE = '''
class OptimizedProofGeneration(dspy.Signature):
    """Generate ONLY the proof tactics for Lean 4 template replacement.
    
    Examples:
    Input: "prove identity function correctness after unfolding"
    Template: "theorem ident_correct : ident_spec x (ident x) := by {{proof}}"
    Output: "rfl"
    
    Input: "prove divisibility check correctness"  
    Template: "theorem div_correct : n % 11 = 0 ↔ isDivisibleBy11 n := by {{proof}}"
    Output: "simp [isDivisibleBy11]"
    
    Input: "prove addition correctness"
    Template: "theorem add_correct : add a b = a + b := by {{proof}}"
    Output: "rfl"
    """
    problem_description = dspy.InputField(desc="What needs to be proven")
    lean_template = dspy.InputField(desc="The Lean template showing theorem")
    
    tactics = dspy.OutputField(desc="Tactic sequence to replace {{proof}}")
'''

print("Manual optimization templates created!")
print("These can be used to replace the existing signatures for better performance.")
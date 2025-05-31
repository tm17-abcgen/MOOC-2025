#!/usr/bin/env python3
"""
Update signatures with more explicit instructions and examples.
"""

# Updated signature that should work better
BETTER_CODE_SIGNATURE = '''
class CodeGeneration(dspy.Signature):
    """Generate ONLY the function body expression for Lean 4 template replacement.
    
    You are given a Lean template like:
    def ident (x : Nat) : Nat := {{code}}
    
    Your job is to replace {{code}} with EXACTLY the right expression.
    
    Examples:
    - Identity function: return "x" 
    - Add two numbers: return "a + b"
    - Return zero: return "0"
    - Return first element: return "arr[0]"
    
    CRITICAL: Return ONLY the expression, no extra text!
    """
    problem_description = dspy.InputField(desc="What the function should do")
    lean_template = dspy.InputField(desc="The Lean template showing function signature")
    function_signature = dspy.InputField(desc="Function parameters and return type")
    
    expression = dspy.OutputField(desc="Single Lean expression to replace {{code}} - NOTHING ELSE!")
'''

BETTER_PROOF_SIGNATURE = '''
class ProofGeneration(dspy.Signature):
    """Generate ONLY the proof tactics for Lean 4 template replacement.
    
    You are given a Lean template like:
    theorem name : statement := by {{proof}}
    
    Your job is to replace {{proof}} with EXACTLY the right tactics.
    
    Examples:
    - Simple equality: return "rfl"
    - After unfolding: return "rfl" 
    - Case analysis: return "cases h <;> simp"
    - Induction: return "induction n with | zero => simp | succ n ih => simp [ih]"
    
    CRITICAL: Return ONLY the tactic sequence, no extra text!
    """
    problem_description = dspy.InputField(desc="What needs to be proven")
    lean_template = dspy.InputField(desc="The Lean template showing theorem")
    goal_after_unfold = dspy.InputField(desc="What the goal looks like after unfolding")
    
    tactics = dspy.OutputField(desc="Tactic sequence to replace {{proof}} - NOTHING ELSE!")
'''

print("Updated signature templates:")
print(BETTER_CODE_SIGNATURE)
print("\n" + "="*50 + "\n")
print(BETTER_PROOF_SIGNATURE)
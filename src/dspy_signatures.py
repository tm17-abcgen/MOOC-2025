"""
DSPy Signatures for Lean 4 Code Generation Agents

This module defines all DSPy signatures used by the multi-agent system
for generating Lean 4 code and proofs.
"""

import dspy
from typing import Optional, List, Dict

# Planning Agent Signatures
class ProblemAnalysis(dspy.Signature):
    """Analyze a Lean 4 problem to understand its complexity and requirements."""
    problem_description = dspy.InputField(desc="Natural language description of the problem")
    lean_template = dspy.InputField(desc="Lean 4 template with {{code}} and {{proof}} placeholders")
    
    complexity_level = dspy.OutputField(desc="Complexity: simple|medium|complex")
    problem_type = dspy.OutputField(desc="Type: arithmetic|boolean|conditional|array|other")
    key_concepts = dspy.OutputField(desc="List of key mathematical concepts involved")
    
class ImplementationPlan(dspy.Signature):
    """Create a detailed implementation plan for a Lean 4 function."""
    problem_description = dspy.InputField(desc="Natural language description of the problem")
    lean_template = dspy.InputField(desc="Lean 4 template with {{code}} and {{proof}} placeholders")
    complexity_analysis = dspy.InputField(desc="Analysis of problem complexity and type")
    
    implementation_strategy = dspy.OutputField(desc="Step-by-step strategy for implementing the function")
    expected_implementation = dspy.OutputField(desc="Expected Lean 4 code structure")
    potential_challenges = dspy.OutputField(desc="Potential implementation challenges")

class ProofStrategy(dspy.Signature):
    """Create a proof strategy for a Lean 4 theorem."""
    problem_description = dspy.InputField(desc="Natural language description of the problem")
    lean_template = dspy.InputField(desc="Lean 4 template with {{code}} and {{proof}} placeholders")
    implementation = dspy.InputField(desc="The function implementation in Lean 4")
    
    proof_approach = dspy.OutputField(desc="Overall approach: direct|induction|cases|contradiction")
    proof_tactics = dspy.OutputField(desc="List of Lean 4 tactics to use in order")
    proof_outline = dspy.OutputField(desc="High-level outline of the proof steps")

# Generation Agent Signatures
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
    implementation_plan = dspy.InputField(desc="Strategy for implementation")
    similar_examples = dspy.InputField(desc="Similar Lean 4 code examples", default="")
    
    expression = dspy.OutputField(desc="Single Lean expression to replace {{code}} - NOTHING ELSE!")
    code_explanation = dspy.OutputField(desc="Brief explanation of the implementation")

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
    implementation = dspy.InputField(desc="The function implementation")
    proof_strategy = dspy.InputField(desc="Strategy for the proof")
    similar_proofs = dspy.InputField(desc="Similar proof examples", default="")
    failed_attempts = dspy.InputField(desc="Previous failed attempts", default="")
    
    tactics = dspy.OutputField(desc="Tactic sequence to replace {{proof}} - NOTHING ELSE!")
    proof_explanation = dspy.OutputField(desc="Step-by-step explanation of the proof")

# Verification Agent Signatures
class ErrorAnalysis(dspy.Signature):
    """Analyze Lean 4 compilation errors to suggest fixes."""
    lean_code = dspy.InputField(desc="The Lean 4 code that failed")
    error_message = dspy.InputField(desc="Lean 4 compiler error message")
    problem_context = dspy.InputField(desc="Problem description and requirements")
    
    error_type = dspy.OutputField(desc="Type of error: syntax|type|tactic|goal")
    root_cause = dspy.OutputField(desc="Root cause analysis of the error")
    suggested_fix = dspy.OutputField(desc="Specific fix to apply to the code")

class ImplementationVerification(dspy.Signature):
    """Verify if a Lean 4 template replacement is correct.
    
    CONTEXT: You are verifying a template replacement, not a complete function!
    
    The implementation is just the expression that replaces {{code}} in a template like:
    def ident (x : Nat) : Nat := {{code}}
    
    When {{code}} is replaced with "x", the complete function becomes:
    def ident (x : Nat) : Nat := x
    
    Your job is to verify if the replacement expression is logically correct for the problem.
    
    Examples of CORRECT replacements:
    - Identity function: "x" → def ident (x : Nat) : Nat := x
    - Addition: "a + b" → def add (a b : Nat) : Nat := a + b
    - Zero function: "0" → def zero (_ : Nat) : Nat := 0
    """
    problem_description = dspy.InputField(desc="What the function should accomplish")
    implementation = dspy.InputField(desc="The expression that replaces {{code}} in the template")
    execution_result = dspy.InputField(desc="Result of executing the complete Lean code with template")
    lean_template = dspy.InputField(desc="The Lean template showing the complete function context")
    
    is_correct = dspy.OutputField(desc="Whether the template replacement is correct: true|false")
    reasoning = dspy.OutputField(desc="Reasoning focused on whether the expression solves the problem")
    improvements = dspy.OutputField(desc="Suggested improvements if any")

# Proof Enhancement Signatures
class ProofGoalAnalysis(dspy.Signature):
    """Analyze the mathematical structure of a proof goal."""
    goal_statement = dspy.InputField(desc="The Lean 4 proof goal after unfolding")
    implementation = dspy.InputField(desc="The function implementation")
    failed_attempts = dspy.InputField(desc="Previous failed proof attempts", default="")
    
    goal_structure = dspy.OutputField(desc="Mathematical structure: equation|inequality|logical|inductive")
    key_properties = dspy.OutputField(desc="Key mathematical properties to leverage")
    recommended_tactics = dspy.OutputField(desc="Ordered list of Lean 4 tactics to try")

class ComprehensiveProofPlan(dspy.Signature):
    """Create a comprehensive proof plan with fallback strategies."""
    problem_description = dspy.InputField(desc="Natural language description of the problem")
    implementation = dspy.InputField(desc="The function implementation")
    goal_analysis = dspy.InputField(desc="Analysis of the proof goal structure")
    failed_attempts = dspy.InputField(desc="Previous failed proof attempts with errors", default="")
    
    primary_strategy = dspy.OutputField(desc="Primary proof strategy with detailed steps")
    fallback_strategies = dspy.OutputField(desc="Alternative strategies if primary fails")
    tactic_sequence = dspy.OutputField(desc="Specific sequence of Lean 4 tactics")

# Pattern Detection Signatures
class PatternDetection(dspy.Signature):
    """Detect specific patterns in Lean 4 problems."""
    problem_description = dspy.InputField(desc="Natural language description of the problem")
    lean_template = dspy.InputField(desc="Lean 4 template code")
    
    has_boolean_logic = dspy.OutputField(desc="Contains Bool/Prop conversion: true|false")
    has_conditionals = dspy.OutputField(desc="Contains if-then-else logic: true|false")
    has_arrays = dspy.OutputField(desc="Contains array operations: true|false")
    special_tactics_needed = dspy.OutputField(desc="Special tactics required for this pattern")

# Error Recovery Signatures
class ErrorRecovery(dspy.Signature):
    """Recover from Lean 4 errors with targeted fixes."""
    original_code = dspy.InputField(desc="The code that produced an error")
    error_message = dspy.InputField(desc="The Lean 4 error message")
    error_analysis = dspy.InputField(desc="Analysis of what went wrong")
    previous_fixes = dspy.InputField(desc="Previous fix attempts that failed", default="")
    
    fixed_code = dspy.OutputField(desc="Corrected Lean 4 code")
    fix_explanation = dspy.OutputField(desc="Explanation of what was fixed and why")

# Meta-learning Signatures
class FailureAnalysis(dspy.Signature):
    """Analyze patterns in failed attempts to improve future generations."""
    failed_attempts = dspy.InputField(desc="List of failed code/proof attempts with errors")
    problem_type = dspy.InputField(desc="Type of problem being solved")
    
    common_mistakes = dspy.OutputField(desc="Common patterns in the failures")
    learned_constraints = dspy.OutputField(desc="Constraints to apply in future attempts")
    improved_approach = dspy.OutputField(desc="Improved approach based on failures")
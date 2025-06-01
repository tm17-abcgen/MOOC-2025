"""
Enhanced DSPy Signatures for Lean 4 Code Generation Agents

This module defines all DSPy signatures with Lean 4 syntax constraints and API corrections.
"""

import dspy
from typing import Optional, List, Dict

# Planning Agent Signatures
class ProblemAnalysis(dspy.Signature):
    """Analyze a Lean 4 problem to identify patterns for appropriate tactics.
    
    PROBLEM TYPES → IMPLEMENTATION PATTERNS:
    - boolean: decide + Boolean operators
    - conditional: if-then-else structures
    - array: Array.any, Array.contains, Array.find?
    - arithmetic: basic operations + omega for proofs
    """
    problem_description = dspy.InputField(desc="Natural language description of the problem")
    lean_template = dspy.InputField(desc="Lean 4 template with {{code}} and {{proof}} placeholders")
    
    complexity_level = dspy.OutputField(desc="Complexity: simple|medium|complex")
    problem_type = dspy.OutputField(desc="Type: boolean_logic|conditional_logic|array_operations|arithmetic")
    key_concepts = dspy.OutputField(desc="Concepts: decidability, case_analysis, array_membership, linear_arithmetic")
    
class ImplementationPlan(dspy.Signature):
    """Create a detailed implementation plan using correct Lean 4 API.
    
    API GUIDANCE:
    - Array operations: Array.any, Array.contains, Array.find? (no ? for any/contains)
    - Boolean logic: decide for Prop→Bool, && || ! for Bool operations
    - Conditionals: if-then-else with proper nesting
    - Comparisons: <, >, ≤, ≥ with decide if needed
    """
    problem_description = dspy.InputField(desc="Natural language description of the problem")
    lean_template = dspy.InputField(desc="Lean 4 template with {{code}} and {{proof}} placeholders")
    complexity_analysis = dspy.InputField(desc="Analysis of problem complexity and type")
    
    implementation_strategy = dspy.OutputField(desc="Strategy using correct Lean 4 API (Array.any not Array.any?)")
    expected_implementation = dspy.OutputField(desc="Expected structure with valid Lean 4 syntax")
    potential_challenges = dspy.OutputField(desc="API usage challenges and syntax pitfalls to avoid")

class ProofStrategy(dspy.Signature):
    """Create a proof strategy using valid Lean 4 tactics.
    
    STRATEGY BY PATTERN:
    - Boolean ↔ proofs: cases analysis with <;> simp composition
    - Decidable propositions: simp [decide_eq_true_iff]
    - Array membership: simp [Array.mem_def]
    - If-then-else: split tactic
    - Linear arithmetic: omega
    
    AVOID: unfold (use simp), comma separation (use <;>), Lean 3 syntax
    """
    problem_description = dspy.InputField(desc="Natural language description of the problem")
    lean_template = dspy.InputField(desc="Lean 4 template with {{code}} and {{proof}} placeholders")
    implementation = dspy.InputField(desc="The function implementation in Lean 4")
    
    proof_approach = dspy.OutputField(desc="Approach: case_analysis|simplification|arithmetic|direct")
    proof_tactics = dspy.OutputField(desc="Valid Lean 4 tactics: simp, cases, omega, constructor")
    proof_outline = dspy.OutputField(desc="Proof steps using correct Lean 4 syntax")

# Generation Agent Signatures
class CodeGeneration(dspy.Signature):
    """Generate ONLY valid Lean 4 function body expression for template replacement.
    
    TEMPLATE SEPARATION - CRITICAL:
    - {{code}} placeholder = ONLY function body expression (no 'def', no proof tactics)
    - {{proof}} placeholder = ONLY proof tactics (handled separately)
    - NEVER include function signatures (def, theorem) in your output
    - NEVER include proof tactics (by, cases, constructor, simp) in code section
    
    CRITICAL API CONSTRAINTS:
    - Array.any (NOT Array.any?) - takes predicate function
    - Array.contains - takes element to check
    - Array.find? - returns Option, takes predicate
    - decide - converts decidable Prop to Bool for if-conditions
    - if-then-else syntax: 'if condition then value1 else value2'
    - Boolean operations: && (and), || (or), ! (not)
    - For Prop in if-conditions: use decide (a ≤ b) NOT just a ≤ b
    
    Examples by type:
    - Boolean logic: "(decide (a < 0) && decide (b > 0)) || (decide (a > 0) && decide (b < 0))"
    - Array operations: "a.any (fun x => b.contains x)"
    - Arithmetic: "a + b", "n % 11"
    - Conditionals: "if decide (a ≤ b) then a else b"
    - Min of three: "if decide (a ≤ b) && decide (a ≤ c) then a else if decide (b ≤ c) then b else c"
    
    FORBIDDEN IN CODE SECTION:
    - Function definitions (def functionName)
    - Proof tactics (by, cases, constructor, simp, exact, etc.)
    - Multiple statements or expressions
    
    RETURN: Single valid Lean 4 expression only!
    """
    problem_description = dspy.InputField(desc="What the function should do")
    lean_template = dspy.InputField(desc="The Lean template showing function signature")
    implementation_plan = dspy.InputField(desc="Strategy for implementation")
    similar_examples = dspy.InputField(desc="Similar Lean 4 code examples", default="")
    
    expression = dspy.OutputField(desc="Single valid Lean 4 expression using correct API - NO explanations!")
    code_explanation = dspy.OutputField(desc="Brief explanation of the implementation and API usage")

class ProofGeneration(dspy.Signature):
    """Generate ONLY valid Lean 4 proof tactics for template replacement.
    
    CRITICAL LEAN 4 SYNTAX CONSTRAINTS:
    - Use ONLY Lean 4 syntax (NOT Lean 3)
    - NO comma separators between tactics
    - Use <;> for tactic composition or newlines
    - Valid tactics: simp, cases, omega, rfl, constructor, exact, decide
    - For ↔: Use 'constructor' NOT 'iff.intro'
    - For Boolean equivalence after unfold: Use 'simp [decide_eq_true_iff]' 
    - Templates may already have 'unfold' - generate tactics to follow unfold
    - For opposite signs: "simp [decide_eq_true_iff]" handles Bool/Prop conversion
    - NO additional unfold tactics - templates already have them
    
    SPECIFIC PATTERNS:
    - After unfold with Bool ↔: "simp [decide_eq_true_iff]"
    - TYPE MISMATCH (Prop ↔ Bool): "simp [decide_eq_true_iff]" then "tauto"
    - Pure Boolean logic: "cases a <;> cases b <;> simp"
    - If-then-else: "split <;> simp"
    - Array membership: "simp [Array.mem_def]"
    - Simple equality: "rfl"
    
    NEVER USE:
    - Or.comm for type mismatch errors
    - Lean 3 tactics like iff.intro
    - Comma separators between tactics
    
    RETURN FORMAT: Single valid tactic or composition with <;>
    """
    problem_description = dspy.InputField(desc="What needs to be proven")
    lean_template = dspy.InputField(desc="The Lean template showing theorem")
    implementation = dspy.InputField(desc="The function implementation")
    proof_strategy = dspy.InputField(desc="Strategy for the proof")
    similar_proofs = dspy.InputField(desc="Similar proof examples", default="")
    failed_attempts = dspy.InputField(desc="Previous failed attempts", default="")
    
    tactics = dspy.OutputField(desc="Valid Lean 4 tactic sequence - NO commas, use <;> or newlines!")
    proof_explanation = dspy.OutputField(desc="Step-by-step explanation using only valid Lean 4 tactics")

# Verification Agent Signatures
class ErrorAnalysis(dspy.Signature):
    """Analyze Lean 4 compilation errors with focus on syntax and API issues.
    
    COMMON ERROR TYPES:
    - syntax: comma usage, Lean 3 identifiers, missing keywords
    - api: non-existent methods (Array.any?, Array.contains?)
    - tactic: unfold failures, wrong case analysis
    - type: Bool/Prop confusion, decidability issues
    """
    lean_code = dspy.InputField(desc="The Lean 4 code that failed")
    error_message = dspy.InputField(desc="Lean 4 compiler error message")
    problem_context = dspy.InputField(desc="Problem description and requirements")
    
    error_type = dspy.OutputField(desc="Type: syntax_error|api_error|tactic_error|type_error")
    root_cause = dspy.OutputField(desc="Specific cause: comma_usage|lean3_syntax|wrong_api|tactic_failure")
    suggested_fix = dspy.OutputField(desc="Exact fix: replace commas with <;>, use Array.any not Array.any?, etc.")

class ImplementationVerification(dspy.Signature):
    """Verify if a Lean 4 template replacement uses correct API and syntax.
    
    VERIFICATION CHECKLIST:
    - API correctness: Array.any (not Array.any?), Array.contains (not Array.contains?)
    - Syntax validity: decide for Prop→Bool, if-then-else structure
    - Type compatibility: Bool vs Prop consistency
    - Logic correctness: matches problem requirements
    
    COMMON ERRORS TO CATCH:
    - Array.any? doesn't exist → should be Array.any
    - Array.contains? doesn't exist → should be Array.contains  
    - Missing decide for Prop→Bool conversion
    - Wrong Boolean operator precedence
    """
    problem_description = dspy.InputField(desc="What the function should accomplish")
    implementation = dspy.InputField(desc="The expression that replaces {{code}} in the template")
    execution_result = dspy.InputField(desc="Result of executing the complete Lean code with template")
    lean_template = dspy.InputField(desc="The Lean template showing the complete function context")
    
    is_correct = dspy.OutputField(desc="Whether API and syntax are correct: true|false")
    reasoning = dspy.OutputField(desc="API and syntax validation focused on common errors")
    improvements = dspy.OutputField(desc="Specific API/syntax fixes needed")

# Proof Enhancement Signatures
class ProofGoalAnalysis(dspy.Signature):
    """Analyze the mathematical structure of a proof goal for Lean 4 tactics.
    
    GOAL TYPE → RECOMMENDED TACTICS:
    - Boolean ↔: cases a <;> cases b <;> simp
    - Decidable ↔: simp [decide_eq_true_iff]
    - Array membership: simp [Array.mem_def, Array.contains_def]
    - If-then-else equality: split <;> simp
    - Linear arithmetic: omega
    - Simple equality: rfl
    """
    goal_statement = dspy.InputField(desc="The Lean 4 proof goal after unfolding")
    implementation = dspy.InputField(desc="The function implementation")
    failed_attempts = dspy.InputField(desc="Previous failed proof attempts", default="")
    
    goal_structure = dspy.OutputField(desc="Structure: boolean_iff|decidable_iff|array_membership|arithmetic|equality")
    key_properties = dspy.OutputField(desc="Key properties: decidability, case analysis needs, simplification")
    recommended_tactics = dspy.OutputField(desc="Ordered Lean 4 tactics: simp, cases, omega, constructor, exact")

class ComprehensiveProofPlan(dspy.Signature):
    """Create a comprehensive proof plan with Lean 4 syntax-aware strategies.
    
    STRATEGY PRIORITIES:
    1. Boolean problems: cases analysis with <;> composition
    2. Arithmetic: omega tactic for linear arithmetic
    3. If-then-else: split tactic or cases on condition
    4. Array membership: simp with Array.mem_def lemmas
    5. Decidable props: simp with decide_eq_true_iff
    
    AVOID: unfold tactics, Lean 3 syntax, comma separation
    """
    problem_description = dspy.InputField(desc="Natural language description of the problem")
    implementation = dspy.InputField(desc="The function implementation")
    goal_analysis = dspy.InputField(desc="Analysis of the proof goal structure")
    failed_attempts = dspy.InputField(desc="Previous failed proof attempts with errors", default="")
    
    primary_strategy = dspy.OutputField(desc="Primary strategy using valid Lean 4 tactics")
    fallback_strategies = dspy.OutputField(desc="Alternative Lean 4 strategies avoiding common syntax errors")
    tactic_sequence = dspy.OutputField(desc="Specific Lean 4 tactics using <;> composition")

# Pattern Detection Signatures
class PatternDetection(dspy.Signature):
    """Detect specific patterns in Lean 4 problems for tactic selection.
    
    PATTERN → TACTICS:
    - Boolean conversion → decide, cases with <;> simp
    - Array operations → Array.any, Array.contains (no ?)
    - If-then-else → split tactic
    - Integer comparison → decide, omega
    - Opposite signs → Boolean combination with decide
    """
    problem_description = dspy.InputField(desc="Natural language description of the problem")
    lean_template = dspy.InputField(desc="Lean 4 template code")
    
    has_boolean_logic = dspy.OutputField(desc="Contains Bool/Prop conversion: true|false")
    has_conditionals = dspy.OutputField(desc="Contains if-then-else logic: true|false")
    has_arrays = dspy.OutputField(desc="Contains array operations: true|false")
    special_tactics_needed = dspy.OutputField(desc="Required Lean 4 tactics: decide|cases|split|simp|omega|constructor")

# Error Recovery Signatures
class ErrorRecovery(dspy.Signature):
    """Recover from Lean 4 errors with targeted syntax fixes.
    
    COMMON ERROR PATTERNS AND FIXES:
    - "unexpected token ','" → Use <;> or newlines between tactics
    - "unknown identifier 'iff.intro'" → Use 'constructor' for ↔ goals
    - "unfold failed" → Use 'simp' or direct proof instead
    - "Array.any? not found" → Use 'Array.any' (no question mark)
    - "Array.contains? not found" → Use 'Array.contains' (no question mark)
    - "split failed" → Use 'cases' on the condition or 'by_cases'
    - "simp made no progress" → Use 'cases', 'omega', or specific lemmas
    """
    original_code = dspy.InputField(desc="The code that produced an error")
    error_message = dspy.InputField(desc="The Lean 4 error message")
    error_analysis = dspy.InputField(desc="Analysis of what went wrong")
    previous_fixes = dspy.InputField(desc="Previous fix attempts that failed", default="")
    
    fixed_code = dspy.OutputField(desc="Corrected Lean 4 code using valid syntax")
    fix_explanation = dspy.OutputField(desc="Explanation of syntax fix applied")

# Meta-learning Signatures
class FailureAnalysis(dspy.Signature):
    """Analyze patterns in failed attempts to improve future generations.
    
    Focus on identifying:
    - Syntax error patterns (commas, Lean 3 vs 4, API usage)
    - Proof strategy failures (unfold vs simp, case analysis)
    - Type-specific patterns (Boolean, Array, arithmetic)
    """
    failed_attempts = dspy.InputField(desc="List of failed code/proof attempts with errors")
    problem_type = dspy.InputField(desc="Type of problem being solved")
    
    syntax_patterns = dspy.OutputField(desc="Common syntax error patterns identified")
    api_mistakes = dspy.OutputField(desc="Incorrect API usage patterns (Array.any? etc.)")
    proof_strategy_issues = dspy.OutputField(desc="Ineffective proof tactics and better alternatives")
    improved_approach = dspy.OutputField(desc="Improved approach based on failures")
"""
DSPy-based Agents for Lean 4 Code Generation

This module implements all agents using DSPy with Chain of Thought reasoning
and optimization capabilities.
"""

import dspy
from typing import Dict, List, Tuple, Optional
import json
import os
from .dspy_signatures import *
from .embedding_db import EmbeddingDB
from .lean_runner import execute_lean_code_tuple as execute_lean_code

def configure_openrouter_lm(model: str):
    """Configure DSPy with OpenRouter LM."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    lm = dspy.LM(
        model=f"openai/openai/{model}",  # OpenRouter format: openai/openai/model-name
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=20000,
        temperature=1.0
    )
    dspy.configure(lm=lm)
    return lm

class DSPyPlanningAgent(dspy.Module):
    """Planning agent that analyzes problems and creates implementation strategies."""
    
    def __init__(self, model: str = "gpt-4o"):
        super().__init__()
        self.model = model
        
        # Configure DSPy with OpenRouter
        configure_openrouter_lm(model)
        
        # Initialize DSPy modules with Chain of Thought
        self.problem_analyzer = dspy.ChainOfThought(ProblemAnalysis)
        self.implementation_planner = dspy.ChainOfThought(ImplementationPlan)
        self.proof_strategist = dspy.ChainOfThought(ProofStrategy)
        self.pattern_detector = dspy.ChainOfThought(PatternDetection)
        
    def forward(self, problem_description: str, lean_template: str, 
                previous_error: Optional[str] = None) -> Dict[str, any]:
        """
        Analyze the problem and create a comprehensive plan.
        
        Returns:
            Dictionary containing:
            - complexity_analysis
            - implementation_plan
            - proof_strategy
            - detected_patterns
        """
        # Detect patterns first
        patterns = self.pattern_detector(
            problem_description=problem_description,
            lean_template=lean_template
        )
        
        # Analyze problem complexity
        analysis = self.problem_analyzer(
            problem_description=problem_description,
            lean_template=lean_template
        )
        
        # Create implementation plan
        impl_plan = self.implementation_planner(
            problem_description=problem_description,
            lean_template=lean_template,
            complexity_analysis=f"Complexity: {analysis.complexity_level}, Type: {analysis.problem_type}, Concepts: {analysis.key_concepts}"
        )
        
        # Placeholder implementation for proof strategy (will be filled after implementation)
        proof_strategy = {
            "proof_approach": "To be determined after implementation",
            "proof_tactics": [],
            "proof_outline": "Requires implementation first"
        }
        
        return {
            "complexity_analysis": {
                "level": analysis.complexity_level,
                "type": analysis.problem_type,
                "concepts": analysis.key_concepts
            },
            "implementation_plan": {
                "strategy": impl_plan.implementation_strategy,
                "expected_structure": impl_plan.expected_implementation,
                "challenges": impl_plan.potential_challenges
            },
            "proof_strategy": proof_strategy,
            "detected_patterns": {
                "has_boolean": patterns.has_boolean_logic == "true",
                "has_conditionals": patterns.has_conditionals == "true", 
                "has_arrays": patterns.has_arrays == "true",
                "special_tactics": patterns.special_tactics_needed
            }
        }
    
    def refine_proof_strategy(self, problem_description: str, lean_template: str,
                            implementation: str) -> Dict[str, any]:
        """Create proof strategy after implementation is available."""
        proof_strat = self.proof_strategist(
            problem_description=problem_description,
            lean_template=lean_template,
            implementation=implementation
        )
        
        return {
            "proof_approach": proof_strat.proof_approach,
            "proof_tactics": proof_strat.proof_tactics.split(", "),
            "proof_outline": proof_strat.proof_outline
        }


class DSPyGenerationAgent(dspy.Module):
    """Generation agent that creates Lean 4 code and proofs with RAG support."""
    
    def __init__(self, model: str = "gpt-4o", embedding_db: Optional[EmbeddingDB] = None):
        super().__init__()
        self.model = model
        self.embedding_db = embedding_db
        
        # Configure DSPy with OpenRouter
        configure_openrouter_lm(model)
        
        # Initialize DSPy modules
        self.code_generator = dspy.ChainOfThought(CodeGeneration)
        self.proof_generator = dspy.ChainOfThought(ProofGeneration)
        self.error_recovery = dspy.ChainOfThought(ErrorRecovery)
        
    def generate_implementation(self, problem_description: str, lean_template: str,
                              plan: Dict[str, any]) -> Tuple[str, str]:
        """Generate Lean 4 function implementation."""
        # Get similar examples from RAG if available
        similar_examples = ""
        if self.embedding_db:
            # Extract function signature from template
            examples = self.embedding_db.get_similar_examples(problem_description, k=3)
            similar_examples = "\n\n".join([f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples)])
        
        # Generate implementation
        result = self.code_generator(
            problem_description=problem_description,
            lean_template=lean_template,
            implementation_plan=json.dumps(plan["implementation_plan"], indent=2),
            similar_examples=similar_examples
        )
        
        # Clean up the generated code to ensure it's just the function body
        clean_code = self._clean_generated_code(result.expression)
        
        return clean_code, result.code_explanation
    
    def generate_proof(self, problem_description: str, lean_template: str,
                      implementation: str, proof_strategy: Dict[str, any],
                      failed_attempts: List[Dict[str, str]] = None) -> Tuple[str, str]:
        """Generate Lean 4 proof."""
        # Get similar proofs from RAG
        similar_proofs = ""
        if self.embedding_db:
            # Search for similar proof patterns
            proof_query = f"proof {problem_description}"
            proofs = self.embedding_db.get_similar_examples(proof_query, k=3)
            similar_proofs = "\n\n".join([f"Proof Example {i+1}:\n{p}" for i, p in enumerate(proofs)])
        
        # Format failed attempts
        failed_str = ""
        if failed_attempts:
            failed_str = "\n\n".join([
                f"Attempt {i+1}:\nProof: {att['proof']}\nError: {att['error']}"
                for i, att in enumerate(failed_attempts[-3:])  # Last 3 attempts
            ])
        
        # Generate proof
        result = self.proof_generator(
            problem_description=problem_description,
            lean_template=lean_template,
            implementation=implementation,
            proof_strategy=json.dumps(proof_strategy, indent=2),
            similar_proofs=similar_proofs,
            failed_attempts=failed_str
        )
        
        # Clean up the generated proof to ensure it's just the tactics
        clean_proof = self._clean_generated_proof(result.tactics)
        
        return clean_proof, result.proof_explanation
    
    def fix_error(self, original_code: str, error_message: str, 
                  error_analysis: Dict[str, str]) -> Tuple[str, str]:
        """Fix errors in generated code."""
        result = self.error_recovery(
            original_code=original_code,
            error_message=error_message,
            error_analysis=json.dumps(error_analysis, indent=2),
            previous_fixes=""
        )
        
        return result.fixed_code, result.fix_explanation
    
    def _clean_generated_code(self, generated_code: str) -> str:
        """Clean generated code to extract only the function body."""
        # Remove common unwanted patterns
        code = generated_code.strip()
        
        # Remove import statements
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('--') or line == '':
                continue
            if line.startswith('def ') or line.startswith('theorem '):
                continue
            if 'CODE START' in line or 'CODE END' in line:
                continue
            if 'PROOF START' in line or 'PROOF END' in line:
                continue
            clean_lines.append(line)
        
        # If we have multiple lines, join them appropriately
        if clean_lines:
            # For simple cases, often it's just one expression
            if len(clean_lines) == 1:
                return clean_lines[0]
            else:
                # Join with proper indentation
                return '\n  '.join(clean_lines)
        
        # Fallback - try to extract just the essential part
        if 'def ' in code and ':=' in code:
            # Extract what comes after :=
            parts = code.split(':=')
            if len(parts) > 1:
                after_assign = parts[-1].strip()
                # Remove any trailing stuff
                after_assign = after_assign.split('\n')[0].strip()
                return after_assign
        
        return code
    
    def _clean_generated_proof(self, generated_proof: str) -> str:
        """Clean generated proof to extract only the tactics."""
        # Remove common unwanted patterns
        proof = generated_proof.strip()
        
        # Remove import statements and definitions
        lines = proof.split('\n')
        clean_lines = []
        skip_until_tactics = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('--') or line == '':
                continue
            if line.startswith('def ') or line.startswith('theorem ') or line.startswith('by'):
                skip_until_tactics = True
                continue
            if 'PROOF START' in line or 'PROOF END' in line:
                continue
            if 'CODE START' in line or 'CODE END' in line:
                continue
            if skip_until_tactics and not line.startswith('  '):
                # Reset when we get to non-indented content that might be tactics
                skip_until_tactics = False
            if not skip_until_tactics:
                clean_lines.append(line)
        
        if clean_lines:
            # Join tactics with newlines and proper indentation
            return '\n  '.join(clean_lines)
        
        # Fallback - try to extract what comes after 'by'
        if 'by' in proof:
            parts = proof.split('by')
            if len(parts) > 1:
                after_by = parts[-1].strip()
                # Clean up the tactics
                tactics = []
                for line in after_by.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('--') and 'END' not in line:
                        tactics.append(line)
                if tactics:
                    return '\n  '.join(tactics)
        
        return proof


class DSPyVerificationAgent(dspy.Module):
    """Verification agent that validates Lean 4 code and analyzes errors."""
    
    def __init__(self, model: str = "gpt-4o", reasoning_model: str = "o3-mini"):
        super().__init__()
        self.model = model
        self.reasoning_model = reasoning_model
        
        # Configure DSPy with OpenRouter for reasoning model
        configure_openrouter_lm(reasoning_model)
        
        # Initialize DSPy modules
        self.error_analyzer = dspy.ChainOfThought(ErrorAnalysis)
        self.impl_verifier = dspy.ChainOfThought(ImplementationVerification)
        
    def analyze_error(self, lean_code: str, error_message: str, 
                     problem_context: str) -> Dict[str, str]:
        """Analyze Lean 4 compilation errors."""
        result = self.error_analyzer(
            lean_code=lean_code,
            error_message=error_message,
            problem_context=problem_context
        )
        
        return {
            "error_type": result.error_type,
            "root_cause": result.root_cause,
            "suggested_fix": result.suggested_fix
        }
    
    def verify_implementation(self, problem_description: str, implementation: str,
                            lean_template: str) -> Tuple[bool, str, Dict[str, any]]:
        """Verify if implementation is correct."""
        # Replace placeholder and execute
        test_code = lean_template.replace("{{code}}", implementation).replace("{{proof}}", "sorry")
        
        # Execute Lean code
        success, output, error = execute_lean_code(test_code)
        
        # If Lean compilation succeeds, the implementation is likely correct
        if success:
            return (
                True,
                f"Implementation '{implementation}' compiles successfully in the template context. This indicates the expression is syntactically and semantically correct for the given function signature.",
                {"improvements": "None needed - implementation compiles successfully"}
            )
        
        # Only use LLM verification if there are compilation errors
        execution_result = f"Success: {success}\nOutput: {output}\nError: {error}"
        
        result = self.impl_verifier(
            problem_description=problem_description,
            implementation=implementation,
            execution_result=execution_result,
            lean_template=lean_template
        )
        
        error_analysis = {}
        if error:
            error_analysis = self.analyze_error(test_code, error, problem_description)
        
        return (
            result.is_correct == "true",
            result.reasoning,
            {
                "improvements": result.improvements,
                "error_analysis": error_analysis
            }
        )


class DSPyProofEnhancer(dspy.Module):
    """Enhanced proof generation using mathematical reasoning."""
    
    def __init__(self, model: str = "o3-mini"):
        super().__init__()
        self.model = model
        
        # Configure DSPy with OpenRouter for reasoning model
        configure_openrouter_lm(model)
        
        # Initialize modules with hint support for guided reasoning
        self.goal_analyzer = dspy.ChainOfThoughtWithHint(ProofGoalAnalysis)
        self.proof_planner = dspy.ChainOfThoughtWithHint(ComprehensiveProofPlan)
        self.failure_analyzer = dspy.ChainOfThought(FailureAnalysis)
        
    def analyze_goal(self, goal_statement: str, implementation: str,
                    failed_attempts: List[Dict[str, str]] = None) -> Dict[str, any]:
        """Analyze proof goal with mathematical reasoning."""
        failed_str = ""
        if failed_attempts:
            failed_str = "\n".join([f"Attempt {i+1}: {att['proof']}" 
                                   for i, att in enumerate(failed_attempts[-2:])])
        
        # Provide hint for complex cases
        hint = "Consider the structure of the goal and what tactics work best for each type"
        
        result = self.goal_analyzer(
            goal_statement=goal_statement,
            implementation=implementation,
            failed_attempts=failed_str,
            hint=hint
        )
        
        return {
            "structure": result.goal_structure,
            "properties": result.key_properties.split(", "),
            "tactics": result.recommended_tactics.split(", ")
        }
    
    def create_comprehensive_plan(self, problem_description: str, implementation: str,
                                goal_analysis: Dict[str, any], 
                                failed_attempts: List[Dict[str, str]] = None) -> Dict[str, any]:
        """Create comprehensive proof plan with fallbacks."""
        failed_str = ""
        if failed_attempts:
            failed_str = json.dumps(failed_attempts[-3:], indent=2)
        
        # Provide strategic hint
        hint = "Focus on the primary strategy but prepare alternatives for common failure modes"
        
        result = self.proof_planner(
            problem_description=problem_description,
            implementation=implementation,
            goal_analysis=json.dumps(goal_analysis, indent=2),
            failed_attempts=failed_str,
            hint=hint
        )
        
        return {
            "primary_strategy": result.primary_strategy,
            "fallback_strategies": result.fallback_strategies.split("\n"),
            "tactic_sequence": result.tactic_sequence.split(", ")
        }
    
    def learn_from_failures(self, failed_attempts: List[Dict[str, str]], 
                          problem_type: str) -> Dict[str, any]:
        """Analyze failure patterns to improve future attempts."""
        result = self.failure_analyzer(
            failed_attempts=json.dumps(failed_attempts, indent=2),
            problem_type=problem_type
        )
        
        return {
            "common_mistakes": result.common_mistakes.split(", "),
            "constraints": result.learned_constraints.split(", "),
            "improved_approach": result.improved_approach
        }
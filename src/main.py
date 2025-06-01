"""
Main entry point for DSPy-based Lean 4 Code Generation

This module provides the main workflow function that will be called by tests.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Fix tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dspy_workflow_enhanced import DSPyLean4WorkflowEnhanced, DSPyOptimizerEnhanced
from src.dspy_workflow import DSPyLean4Workflow, DSPyOptimizer  # Keep for compatibility
from src.lean_runner import execute_lean_code

# Load environment variables
load_dotenv()

# Type alias for Lean code
LeanCode = Dict[str, str]

# Global workflow instance (will be initialized once)
_workflow_instance = None

def get_workflow_instance() -> DSPyLean4WorkflowEnhanced:
    """Get or create the global enhanced workflow instance with complete configuration."""
    global _workflow_instance
    
    if _workflow_instance is None:
        print("[INIT] Initializing Enhanced DSPy Lean 4 Workflow...")
        print("[INIT] Features: Pattern Detection, Model Routing, Syntax Validation, Progressive Enhancement")
        
        # Check for embedding database with enhanced documentation
        embedding_db_path = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Look for embeddings in various locations (enhanced docs should be available)
        possible_paths = [
            os.path.join(parent_dir, "lean4_embeddings_chunks.pkl"),
            os.path.join(parent_dir, "embeddings", "lean4_embeddings_chunks.pkl"),
            os.path.join(current_dir, "lean4_embeddings_chunks.pkl")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                embedding_db_path = path
                print(f"[INIT] Found enhanced embeddings at: {path}")
                break
        
        if not embedding_db_path:
            print("[INIT] Warning: No embedding database found - RAG features disabled")
        
        # Get configuration from environment
        planning_model = os.getenv("PLANNING_AGENT_MODEL", "gpt-4o")
        generation_model = os.getenv("GENERATION_AGENT_MODEL", "gpt-4o")
        verification_model = os.getenv("VERIFICATION_AGENT_MODEL", "gpt-4o")
        reasoning_model = os.getenv("PROOF_AGENT_MODEL", "o3-mini")
        
        enable_routing = os.getenv("USE_MODEL_ROUTING", "true").lower() == "true"
        max_impl_attempts = int(os.getenv("MAX_IMPLEMENTATION_ATTEMPTS", "5"))
        max_proof_attempts = int(os.getenv("MAX_PROOF_ATTEMPTS", "10"))
        use_progressive = os.getenv("USE_PROGRESSIVE_ENHANCEMENT", "true").lower() == "true"
        
        print(f"[INIT] Model Configuration:")
        print(f"  - Planning: {planning_model}")
        print(f"  - Generation: {generation_model}")
        print(f"  - Verification: {verification_model}")
        print(f"  - Reasoning/Proof: {reasoning_model}")
        print(f"[INIT] Enhanced Features:")
        print(f"  - Model Routing: {enable_routing}")
        print(f"  - Max Implementation Attempts: {max_impl_attempts}")
        print(f"  - Max Proof Attempts: {max_proof_attempts}")
        print(f"  - Progressive Enhancement: {use_progressive}")
        
        # Initialize enhanced workflow with complete configuration
        _workflow_instance = DSPyLean4WorkflowEnhanced(
            planning_model=planning_model,
            generation_model=generation_model, 
            verification_model=verification_model,
            reasoning_model=reasoning_model,
            embedding_db_path=embedding_db_path,
            enable_model_routing=enable_routing
        )
        
        # Apply configuration overrides
        _workflow_instance.max_implementation_attempts = max_impl_attempts
        _workflow_instance.max_proof_attempts = max_proof_attempts
        _workflow_instance.use_progressive_enhancement = use_progressive
        
        print("[INIT] Enhanced DSPy Lean 4 Workflow initialized successfully!")
        
    return _workflow_instance

def main_workflow(problem_description: str, task_lean_code: str = "") -> LeanCode:
    """
    Main workflow for the coding agent using the complete enhanced DSPy pipeline.
    
    This workflow implements a 5-phase enhanced process:
    1. Enhanced Planning with pattern detection and complexity analysis
    2. Model-routed Implementation Generation with API validation 
    3. Enhanced Proof Strategy with pattern-aware tactics
    4. Progressive Proof Enhancement with failure learning
    5. Final Validation with metrics collection
    
    Args:
        problem_description: Problem description in natural language. This file is read from "description.txt"
        task_lean_code: Lean code template. This file is read from "task.lean"
    
    Returns:
        LeanCode: Final generated solution, which is a dictionary with two keys: "code" and "proof".
    """
    print(f"[MAIN] Starting enhanced workflow for task...")
    print(f"[MAIN] Using DSPyLean4WorkflowEnhanced with complete pipeline")
    
    # Get or create enhanced workflow instance
    workflow = get_workflow_instance()
    
    # Validate inputs
    if not problem_description.strip():
        print("[ERROR] Empty problem description provided")
        return {"code": "sorry", "proof": "sorry"}
    
    if not task_lean_code.strip():
        print("[ERROR] Empty task template provided")
        return {"code": "sorry", "proof": "sorry"}
    
    # Generate solution using enhanced DSPy workflow with complete pipeline
    try:
        print(f"[MAIN] Invoking enhanced workflow.generate_solution()...")
        
        # The enhanced workflow implements:
        # - Pattern detection (Boolean, Array, Arithmetic, Three-way minimum)
        # - Model routing (GPT-4o for generation, o3-mini for proofs)
        # - Syntax validation (API fixes, Lean 4 compliance)
        # - Progressive enhancement (iterative proof improvement)
        # - Error learning (failure pattern analysis)
        result = workflow.generate_solution(
            problem_description=problem_description,
            task_lean_code=task_lean_code
        )
        
        # Enhanced result validation
        code = result.get("code", "sorry").strip()
        proof = result.get("proof", "sorry").strip()
        
        # Check for empty or invalid responses
        if not code or code.lower() == "sorry":
            print("[WARNING] Generated code is empty or 'sorry'")
            code = "sorry"
        
        if not proof or proof.lower() == "sorry":
            if code != "sorry":
                print("[INFO] Generated proof uses 'sorry' (acceptable for complex proofs)")
            else:
                print("[WARNING] Generated proof is empty or 'sorry'")
            proof = "sorry"
        
        # Enhanced success reporting - accept implementation + sorry as success
        if code != "sorry" and proof != "sorry":
            success_level = "COMPLETE"
        elif code != "sorry":
            success_level = "SUCCESS"  # Implementation works, proof complex
        else:
            success_level = "FAILED"
        
        print(f"[MAIN] Enhanced workflow completed with status: {success_level}")
        print(f"[MAIN] Code: {'✓' if code != 'sorry' else '✗'}")
        print(f"[MAIN] Proof: {'✓' if proof != 'sorry' else '○' if code != 'sorry' else '✗'}")
        
        return {
            "code": code,
            "proof": proof
        }
        
    except Exception as e:
        print(f"[ERROR] Enhanced workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Enhanced error handling with context
        print(f"[ERROR] Problem description length: {len(problem_description)}")
        print(f"[ERROR] Template length: {len(task_lean_code)}")
        print(f"[ERROR] Template contains placeholders: {('{{code}}' in task_lean_code and '{{proof}}' in task_lean_code)}")
        
        # Return sorry as fallback with error context
        return {
            "code": "sorry",
            "proof": "sorry"
        }

def get_problem_and_code_from_taskpath(task_path: str) -> Tuple[str, str]:
    """
    Reads a directory in the format of task_id_*. It will read the file "task.lean" and also read the file 
    that contains the task description, which is "description.txt".
    
    After reading the files, it will return a tuple of the problem description and the Lean code template.
    
    Args:
        task_path: Path to the task file
    """
    problem_description = ""
    lean_code_template = ""
    
    with open(os.path.join(task_path, "description.txt"), "r") as f:
        problem_description = f.read()

    with open(os.path.join(task_path, "task.lean"), "r") as f:
        lean_code_template = f.read()

    return problem_description, lean_code_template

def get_unit_tests_from_taskpath(task_path: str) -> List[str]:
    """
    Reads a directory in the format of task_id_*. It will read the file "tests.lean" and return the unit tests.
    """
    with open(os.path.join(task_path, "tests.lean"), "r") as f:
        unit_tests = f.read()
    
    return unit_tests

def get_task_lean_template_from_taskpath(task_path: str) -> str:
    """
    Reads a directory in the format of task_id_*. It will read the file "task.lean" and return the Lean code template.
    """
    with open(os.path.join(task_path, "task.lean"), "r") as f:
        task_lean_template = f.read()
    return task_lean_template

def optimize_workflow(task_directory: str = None):
    """
    Optimize the DSPy workflow using available task data.
    
    Args:
        task_directory: Directory containing task_id_* folders
    """
    if task_directory is None:
        # Default to tasks directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        task_directory = os.path.join(os.path.dirname(current_dir), "tasks")
    
    print(f"\n[OPTIMIZE] Starting workflow optimization...")
    print(f"[OPTIMIZE] Task directory: {task_directory}")
    
    # Get workflow instance
    workflow = get_workflow_instance()
    
    # Create optimizer
    optimizer = DSPyOptimizer(workflow)
    
    # Create training dataset
    training_examples = optimizer.create_training_dataset(task_directory)
    
    if len(training_examples) == 0:
        print("[OPTIMIZE] No training examples found!")
        return
    
    # Define metric function for optimization
    def lean_metric(example, prediction):
        """Evaluate the quality of a Lean 4 solution."""
        try:
            # Check if implementation compiles
            impl_test = example.lean_template.replace("{{code}}", prediction.implementation).replace("{{proof}}", "sorry")
            impl_success, _, _ = execute_lean_code(impl_test)
            
            # Check if proof is valid (not sorry)
            has_valid_proof = prediction.proof and prediction.proof.strip().lower() != "sorry"
            
            # Check if complete solution compiles
            if has_valid_proof:
                complete_test = example.lean_template.replace("{{code}}", prediction.implementation).replace("{{proof}}", prediction.proof)
                proof_success, _, _ = execute_lean_code(complete_test)
            else:
                proof_success = False
            
            # Score: 0.5 for valid implementation, 0.5 for valid proof
            score = 0.0
            if impl_success:
                score += 0.5
            if proof_success:
                score += 0.5
                
            return score
            
        except Exception as e:
            print(f"[METRIC] Error evaluating: {e}")
            return 0.0
    
    # Optimize with MIPROv2
    optimized_workflow = optimizer.optimize_with_mipro(
        training_examples=training_examples[:10],  # Start with subset
        metric_function=lean_metric,
        num_iterations=50  # Reduced for testing
    )
    
    # Update global instance
    global _workflow_instance
    _workflow_instance = optimized_workflow
    
    print("[OPTIMIZE] Workflow optimization completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSPy-based Lean 4 Code Generator")
    parser.add_argument("--task", type=str, help="Path to specific task directory")
    parser.add_argument("--optimize", action="store_true", help="Run workflow optimization")
    parser.add_argument("--test-single", type=str, help="Test a single task by ID (e.g., task_id_0)")
    
    args = parser.parse_args()
    
    if args.optimize:
        optimize_workflow()
    elif args.test_single:
        # Test a single task
        current_dir = os.path.dirname(os.path.abspath(__file__))
        task_path = os.path.join(os.path.dirname(current_dir), "tasks", args.test_single)
        
        if os.path.exists(task_path):
            print(f"\n[TEST] Testing {args.test_single}...")
            description, template = get_problem_and_code_from_taskpath(task_path)
            result = main_workflow(description, template)
            
            print(f"\n[RESULT] Implementation:")
            print(result["code"])
            print(f"\n[RESULT] Proof:")
            print(result["proof"])
            
            # Test the complete solution
            complete_code = template.replace("{{code}}", result["code"]).replace("{{proof}}", result["proof"])
            from lean_runner import execute_lean_code_tuple
            success, output, error = execute_lean_code_tuple(complete_code)
            
            print(f"\n[VERIFICATION] Success: {success}")
            if error:
                print(f"[VERIFICATION] Error: {error}")
        else:
            print(f"[ERROR] Task directory not found: {task_path}")
    else:
        print("DSPy-based Lean 4 Code Generator")
        print("Use --help for usage information")
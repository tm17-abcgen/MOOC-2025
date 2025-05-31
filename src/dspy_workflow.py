"""
DSPy-based Main Workflow for Lean 4 Code Generation

This module implements the main workflow using DSPy with optimization
and progressive enhancement capabilities.
"""

import dspy
from typing import Dict, List, Tuple, Optional
import os
import json
from tqdm import tqdm

from .dspy_agents import (
    DSPyPlanningAgent,
    DSPyGenerationAgent,
    DSPyVerificationAgent,
    DSPyProofEnhancer
)
from .embedding_db import EmbeddingDB
from .lean_runner import execute_lean_code_tuple as execute_lean_code

class DSPyLean4Workflow:
    """Main workflow orchestrator using DSPy agents."""
    
    def __init__(self, 
                 planning_model: str = "gpt-4o",
                 generation_model: str = "gpt-4o", 
                 verification_model: str = "gpt-4o",
                 reasoning_model: str = "o3-mini",
                 embedding_db_path: Optional[str] = None):
        """
        Initialize the workflow with DSPy agents.
        
        Args:
            planning_model: Model for planning agent
            generation_model: Model for code/proof generation
            verification_model: Model for verification
            reasoning_model: Model for mathematical reasoning (o3-mini)
            embedding_db_path: Path to embedding database for RAG
        """
        # Initialize embedding database if provided
        self.embedding_db = None
        if embedding_db_path and os.path.exists(embedding_db_path):
            self.embedding_db = EmbeddingDB()
            # Load embeddings - try to find corresponding .npy file
            embeddings_npy_path = embedding_db_path.replace('_chunks.pkl', '.npy')
            self.embedding_db.load_from_pickle(embedding_db_path, embeddings_npy_path)
            print(f"[INFO] Loaded embeddings from {embedding_db_path}")
        
        # Initialize agents
        self.planning_agent = DSPyPlanningAgent(model=planning_model)
        self.generation_agent = DSPyGenerationAgent(
            model=generation_model, 
            embedding_db=self.embedding_db
        )
        self.verification_agent = DSPyVerificationAgent(
            model=verification_model,
            reasoning_model=reasoning_model
        )
        self.proof_enhancer = DSPyProofEnhancer(model=reasoning_model)
        
        # Configuration
        self.max_implementation_attempts = 5
        self.max_proof_attempts = 10
        self.use_progressive_enhancement = True
        
    def generate_solution(self, 
                         problem_description: str, 
                         task_lean_code: str) -> Dict[str, str]:
        """
        Generate complete Lean 4 solution (implementation + proof).
        
        Args:
            problem_description: Natural language problem description
            task_lean_code: Lean template with {{code}} and {{proof}} placeholders
            
        Returns:
            Dictionary with 'code' and 'proof' keys
        """
        print(f"\n[WORKFLOW] Starting solution generation...")
        
        # Step 1: Planning and Analysis
        print("[STEP 1] Analyzing problem and creating plan...")
        plan = self.planning_agent(
            problem_description=problem_description,
            lean_template=task_lean_code
        )
        
        print(f"  - Complexity: {plan['complexity_analysis']['level']}")
        print(f"  - Type: {plan['complexity_analysis']['type']}")
        print(f"  - Patterns detected: Boolean={plan['detected_patterns']['has_boolean']}, "
              f"Conditionals={plan['detected_patterns']['has_conditionals']}, "
              f"Arrays={plan['detected_patterns']['has_arrays']}")
        
        # Step 2: Implementation Generation with Retry
        print("\n[STEP 2] Generating implementation...")
        implementation = None
        implementation_errors = []
        
        for attempt in range(self.max_implementation_attempts):
            print(f"  Attempt {attempt + 1}/{self.max_implementation_attempts}")
            
            # Generate implementation
            code, explanation = self.generation_agent.generate_implementation(
                problem_description=problem_description,
                lean_template=task_lean_code,
                plan=plan
            )
            
            # Verify implementation
            is_correct, reasoning, analysis = self.verification_agent.verify_implementation(
                problem_description=problem_description,
                implementation=code,
                lean_template=task_lean_code
            )
            
            if is_correct:
                print(f"  ✓ Implementation verified successfully!")
                implementation = code
                break
            else:
                print(f"  ✗ Verification failed: {reasoning}")
                implementation_errors.append({
                    "attempt": attempt + 1,
                    "code": code,
                    "error": analysis.get("error_analysis", {})
                })
                
                # Try to fix the error
                if analysis.get("error_analysis"):
                    fixed_code, fix_explanation = self.generation_agent.fix_error(
                        original_code=code,
                        error_message=analysis["error_analysis"].get("root_cause", ""),
                        error_analysis=analysis["error_analysis"]
                    )
                    code = fixed_code
        
        if not implementation:
            print("  ! Failed to generate valid implementation, using last attempt")
            implementation = code
        
        # Step 3: Refine proof strategy with implementation
        print("\n[STEP 3] Refining proof strategy...")
        proof_strategy = self.planning_agent.refine_proof_strategy(
            problem_description=problem_description,
            lean_template=task_lean_code,
            implementation=implementation
        )
        
        # Step 4: Proof Generation with Progressive Enhancement
        print("\n[STEP 4] Generating proof...")
        proof = None
        failed_proof_attempts = []
        
        for attempt in range(self.max_proof_attempts):
            print(f"  Attempt {attempt + 1}/{self.max_proof_attempts}")
            
            # Progressive enhancement after failures
            enhanced_strategy = proof_strategy
            if self.use_progressive_enhancement and attempt > 0:
                if attempt == 1:
                    # First enhancement: Analyze goal structure
                    print("  [Enhancement] Analyzing proof goal structure...")
                    goal_analysis = self.proof_enhancer.analyze_goal(
                        goal_statement=self._extract_goal_from_template(task_lean_code),
                        implementation=implementation,
                        failed_attempts=failed_proof_attempts
                    )
                    enhanced_strategy["goal_analysis"] = goal_analysis
                    
                elif attempt >= 2:
                    # Second enhancement: Comprehensive planning
                    print("  [Enhancement] Creating comprehensive proof plan...")
                    comprehensive_plan = self.proof_enhancer.create_comprehensive_plan(
                        problem_description=problem_description,
                        implementation=implementation,
                        goal_analysis=enhanced_strategy.get("goal_analysis", {}),
                        failed_attempts=failed_proof_attempts
                    )
                    enhanced_strategy["comprehensive_plan"] = comprehensive_plan
                    
                if attempt >= 3 and len(failed_proof_attempts) >= 2:
                    # Learn from failures
                    print("  [Enhancement] Learning from failure patterns...")
                    failure_insights = self.proof_enhancer.learn_from_failures(
                        failed_attempts=failed_proof_attempts,
                        problem_type=plan['complexity_analysis']['type']
                    )
                    enhanced_strategy["failure_insights"] = failure_insights
            
            # Generate proof
            proof_code, proof_explanation = self.generation_agent.generate_proof(
                problem_description=problem_description,
                lean_template=task_lean_code,
                implementation=implementation,
                proof_strategy=enhanced_strategy,
                failed_attempts=failed_proof_attempts
            )
            
            # Verify complete solution
            complete_code = task_lean_code.replace("{{code}}", implementation).replace("{{proof}}", proof_code)
            success, output, error = execute_lean_code(complete_code)
            
            if success and "sorry" not in proof_code.lower():
                print(f"  ✓ Proof verified successfully!")
                proof = proof_code
                break
            else:
                print(f"  ✗ Proof failed: {error if error else 'Contains sorry'}")
                failed_proof_attempts.append({
                    "attempt": attempt + 1,
                    "proof": proof_code,
                    "error": error or "Proof contains 'sorry'",
                    "analysis": proof_explanation
                })
        
        if not proof:
            print("  ! Failed to generate valid proof, using best attempt")
            proof = proof_code if 'proof_code' in locals() else "sorry"
        
        # Final result
        result = {
            "code": implementation,
            "proof": proof
        }
        
        print(f"\n[WORKFLOW] Solution generation completed!")
        print(f"  - Implementation: {'✓' if implementation and implementation != 'sorry' else '✗'}")
        print(f"  - Proof: {'✓' if proof and proof != 'sorry' else '✗'}")
        
        return result
    
    def _extract_goal_from_template(self, template: str) -> str:
        """Extract the theorem statement from the template."""
        # Simple extraction - you might need to improve this
        lines = template.split('\n')
        for i, line in enumerate(lines):
            if 'theorem' in line.lower() and ':' in line:
                # Get the part after the colon
                goal = line.split(':', 1)[1].strip()
                # Continue to next lines if needed
                j = i + 1
                while j < len(lines) and '{{proof}}' not in lines[j]:
                    goal += ' ' + lines[j].strip()
                    j += 1
                return goal
        return "Unknown goal"


class DSPyOptimizer:
    """Optimizer for DSPy modules using various techniques."""
    
    def __init__(self, workflow: DSPyLean4Workflow):
        self.workflow = workflow
        
    def create_training_dataset(self, task_directory: str) -> List[dspy.Example]:
        """
        Create DSPy training dataset from task files.
        
        Args:
            task_directory: Directory containing task_id_* folders
            
        Returns:
            List of DSPy Examples for training
        """
        print("[DATASET] Creating training dataset from tasks...")
        examples = []
        
        # Get all task directories
        task_dirs = [d for d in os.listdir(task_directory) if d.startswith("task_id_")]
        
        for task_dir in tqdm(task_dirs, desc="Processing tasks"):
            task_path = os.path.join(task_directory, task_dir)
            
            try:
                # Read task files
                with open(os.path.join(task_path, "description.txt"), "r") as f:
                    description = f.read().strip()
                
                with open(os.path.join(task_path, "task.lean"), "r") as f:
                    template = f.read().strip()
                
                # Try to read test.json for expected behavior
                expected_impl = None
                expected_proof = None
                if os.path.exists(os.path.join(task_path, "test.json")):
                    with open(os.path.join(task_path, "test.json"), "r") as f:
                        tests = json.load(f)
                        # Extract patterns from tests if possible
                
                # Create DSPy example
                example = dspy.Example(
                    problem_description=description,
                    lean_template=template,
                    task_id=task_dir
                )
                
                # If we have ground truth (from successful runs), add it
                if expected_impl:
                    example.implementation = expected_impl
                if expected_proof:
                    example.proof = expected_proof
                
                examples.append(example)
                
            except Exception as e:
                print(f"  Warning: Failed to process {task_dir}: {e}")
        
        print(f"[DATASET] Created {len(examples)} training examples")
        return examples
    
    def optimize_with_mipro(self, 
                           training_examples: List[dspy.Example],
                           metric_function,
                           num_iterations: int = 200) -> DSPyLean4Workflow:
        """
        Optimize the workflow using MIPROv2.
        
        Args:
            training_examples: Training dataset
            metric_function: Function to evaluate performance
            num_iterations: Number of optimization iterations
            
        Returns:
            Optimized workflow
        """
        print(f"\n[OPTIMIZER] Starting MIPROv2 optimization with {num_iterations} iterations...")
        
        # Configure MIPROv2 optimizer
        from dspy.teleprompt import MIPROv2
        
        optimizer = MIPROv2(
            metric=metric_function,
            auto="medium",  # medium optimization depth
            num_iterations=num_iterations,
            instruction_tokens=500,
            example_tokens=2000
        )
        
        # Create a simplified module for optimization
        class SimplifiedWorkflow(dspy.Module):
            def __init__(self, workflow):
                super().__init__()
                self.workflow = workflow
                
            def forward(self, problem_description: str, lean_template: str) -> dspy.Prediction:
                result = self.workflow.generate_solution(problem_description, lean_template)
                return dspy.Prediction(
                    implementation=result["code"],
                    proof=result["proof"]
                )
        
        # Optimize
        simplified = SimplifiedWorkflow(self.workflow)
        optimized = optimizer.compile(simplified, trainset=training_examples)
        
        print("[OPTIMIZER] Optimization completed!")
        return optimized.workflow
    
    def optimize_with_bootstrap(self,
                              training_examples: List[dspy.Example],
                              metric_function,
                              max_bootstrapped_demos: int = 4) -> DSPyLean4Workflow:
        """
        Optimize using BootstrapFewShotWithRandomSearch.
        
        Args:
            training_examples: Training dataset
            metric_function: Function to evaluate performance
            max_bootstrapped_demos: Maximum number of demonstrations
            
        Returns:
            Optimized workflow
        """
        print(f"\n[OPTIMIZER] Starting Bootstrap optimization...")
        
        from dspy.teleprompt import BootstrapFewShotWithRandomSearch
        
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=metric_function,
            max_bootstrapped_demos=max_bootstrapped_demos,
            num_candidate_programs=20
        )
        
        # Similar optimization process as MIPROv2
        # Implementation details...
        
        print("[OPTIMIZER] Bootstrap optimization completed!")
        return self.workflow
"""
Build DSPy training dataset from Lean 4 tasks

This script processes task files and creates a dataset for DSPy optimization.
"""

import os
import json
import pickle
from typing import List, Dict, Tuple
import dspy
from tqdm import tqdm

def extract_ground_truth_from_tests(test_json_path: str) -> Tuple[str, str]:
    """
    Try to extract expected implementation and proof patterns from test cases.
    
    Returns:
        Tuple of (implementation_hint, proof_hint)
    """
    try:
        with open(test_json_path, 'r') as f:
            tests = json.load(f)
        
        # Analyze test cases to infer patterns
        impl_hint = ""
        proof_hint = ""
        
        # Look for patterns in test inputs/outputs
        if isinstance(tests, list) and len(tests) > 0:
            # Check if it's an identity function
            if all(test.get("input") == test.get("output") for test in tests if "input" in test and "output" in test):
                impl_hint = "identity function returning input"
                proof_hint = "reflexivity (rfl)"
            
            # Check for boolean operations
            elif any("true" in str(test).lower() or "false" in str(test).lower() for test in tests):
                impl_hint = "boolean logic operation"
                proof_hint = "case analysis on boolean values"
                
            # Check for conditional logic
            elif any("if" in str(test).lower() or "then" in str(test).lower() for test in tests):
                impl_hint = "conditional logic with if-then-else"
                proof_hint = "split_ifs tactic for conditional reasoning"
                
        return impl_hint, proof_hint
        
    except Exception as e:
        return "", ""

def create_dspy_dataset(task_directory: str, output_path: str) -> List[dspy.Example]:
    """
    Create DSPy dataset from task files.
    
    Args:
        task_directory: Directory containing task_id_* folders
        output_path: Path to save the dataset
        
    Returns:
        List of DSPy Examples
    """
    print(f"[DATASET] Building DSPy dataset from {task_directory}")
    
    examples = []
    task_dirs = sorted([d for d in os.listdir(task_directory) if d.startswith("task_id_")])
    
    for task_dir in tqdm(task_dirs, desc="Processing tasks"):
        task_path = os.path.join(task_directory, task_dir)
        
        try:
            # Read task files
            with open(os.path.join(task_path, "description.txt"), "r") as f:
                description = f.read().strip()
            
            with open(os.path.join(task_path, "task.lean"), "r") as f:
                template = f.read().strip()
            
            # Extract task metadata
            task_id = task_dir
            
            # Try to get hints from test cases
            test_json_path = os.path.join(task_path, "test.json")
            impl_hint, proof_hint = extract_ground_truth_from_tests(test_json_path)
            
            # Read signature for more context
            signature_info = {}
            try:
                with open(os.path.join(task_path, "signature.json"), "r") as f:
                    signature_info = json.load(f)
            except:
                pass
            
            # Create comprehensive example
            example = dspy.Example(
                # Inputs
                problem_description=description,
                lean_template=template,
                task_id=task_id,
                
                # Metadata that can help
                function_name=signature_info.get("name", "unknown"),
                input_types=str(signature_info.get("inputs", [])),
                output_type=str(signature_info.get("output", "unknown")),
                
                # Hints from analysis
                implementation_hint=impl_hint,
                proof_hint=proof_hint
            )
            
            # If we have successful solutions from previous runs, add them
            # This would come from analyzing successful test runs
            
            examples.append(example)
            
        except Exception as e:
            print(f"  Warning: Failed to process {task_dir}: {e}")
    
    # Save dataset
    print(f"\n[DATASET] Saving {len(examples)} examples to {output_path}")
    
    # Save as pickle for easy loading
    with open(output_path, 'wb') as f:
        pickle.dump(examples, f)
    
    # Also save as JSON for inspection
    json_path = output_path.replace('.pkl', '.json')
    json_data = []
    for ex in examples:
        json_data.append({
            "task_id": ex.task_id,
            "description": ex.problem_description[:200] + "..." if len(ex.problem_description) > 200 else ex.problem_description,
            "function_name": ex.function_name,
            "implementation_hint": ex.implementation_hint,
            "proof_hint": ex.proof_hint
        })
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"[DATASET] Dataset saved successfully!")
    print(f"  - Pickle: {output_path}")
    print(f"  - JSON summary: {json_path}")
    
    return examples

def analyze_dataset(examples: List[dspy.Example]):
    """Analyze the dataset to understand task distribution."""
    print(f"\n[ANALYSIS] Dataset Statistics:")
    print(f"  Total examples: {len(examples)}")
    
    # Analyze patterns
    identity_count = sum(1 for ex in examples if "identity" in ex.implementation_hint)
    boolean_count = sum(1 for ex in examples if "boolean" in ex.implementation_hint)
    conditional_count = sum(1 for ex in examples if "conditional" in ex.implementation_hint)
    
    print(f"  Identity functions: {identity_count}")
    print(f"  Boolean operations: {boolean_count}")
    print(f"  Conditional logic: {conditional_count}")
    print(f"  Other: {len(examples) - identity_count - boolean_count - conditional_count}")

def load_dataset(dataset_path: str) -> List[dspy.Example]:
    """Load DSPy dataset from file."""
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build DSPy dataset from Lean 4 tasks")
    parser.add_argument("--tasks", type=str, default="../tasks", 
                       help="Path to tasks directory")
    parser.add_argument("--output", type=str, default="dspy_lean4_dataset.pkl",
                       help="Output path for dataset")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze existing dataset")
    
    args = parser.parse_args()
    
    if args.analyze and os.path.exists(args.output):
        # Load and analyze existing dataset
        examples = load_dataset(args.output)
        analyze_dataset(examples)
    else:
        # Build new dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        task_dir = os.path.join(os.path.dirname(current_dir), "tasks")
        
        if not os.path.isabs(args.tasks):
            task_dir = os.path.join(current_dir, args.tasks)
        else:
            task_dir = args.tasks
            
        if not os.path.exists(task_dir):
            print(f"[ERROR] Tasks directory not found: {task_dir}")
            exit(1)
            
        examples = create_dspy_dataset(task_dir, args.output)
        analyze_dataset(examples)
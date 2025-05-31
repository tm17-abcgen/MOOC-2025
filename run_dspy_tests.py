#!/usr/bin/env python3
"""
Run tests using the DSPy-based implementation

This script tests the DSPy workflow on individual tasks or all tasks.
"""

import os
import sys
import argparse
from datetime import datetime
import json

# Fix tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.main import main_workflow, get_problem_and_code_from_taskpath
from src.lean_runner import execute_lean_code

def test_single_task(task_id: str, tasks_dir: str) -> dict:
    """Test a single task and return results."""
    print(f"\n{'='*60}")
    print(f"Testing {task_id}")
    print('='*60)
    
    task_path = os.path.join(tasks_dir, task_id)
    
    if not os.path.exists(task_path):
        return {
            "task_id": task_id,
            "status": "error",
            "error": "Task directory not found"
        }
    
    try:
        # Load task
        description, template = get_problem_and_code_from_taskpath(task_path)
        
        print(f"\nDescription preview: {description[:100]}...")
        
        # Generate solution
        print("\nGenerating solution...")
        result = main_workflow(description, template)
        
        print(f"\nGenerated code: {result['code']}")
        print(f"Generated proof: {result['proof']}")
        
        # Test implementation only
        impl_code = template.replace("{{code}}", result['code']).replace("{{proof}}", "sorry")
        impl_success, impl_output, impl_error = execute_lean_code(impl_code)
        
        # Test complete solution
        complete_code = template.replace("{{code}}", result['code']).replace("{{proof}}", result['proof'])
        complete_success, complete_output, complete_error = execute_lean_code(complete_code)
        
        # Check for sorry in proof
        has_sorry = "sorry" in result['proof'].lower()
        
        # Determine overall status
        if complete_success and not has_sorry:
            status = "success"
        elif impl_success and has_sorry:
            status = "partial"
        else:
            status = "failed"
        
        test_result = {
            "task_id": task_id,
            "status": status,
            "implementation": result['code'],
            "proof": result['proof'],
            "impl_success": impl_success,
            "complete_success": complete_success,
            "has_sorry": has_sorry,
            "impl_error": impl_error,
            "complete_error": complete_error
        }
        
        print(f"\nResult: {status.upper()}")
        if impl_error:
            print(f"Implementation error: {impl_error}")
        if complete_error and not has_sorry:
            print(f"Proof error: {complete_error}")
            
        return test_result
        
    except Exception as e:
        import traceback
        print(f"\nERROR: {str(e)}")
        traceback.print_exc()
        
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e)
        }

def test_all_tasks(tasks_dir: str) -> list:
    """Test all tasks in the directory."""
    # Standard task IDs to test
    task_ids = ["task_id_0", "task_id_58", "task_id_77", "task_id_127", 
                "task_id_227", "task_id_404", "task_id_431", "task_id_433",
                "task_id_435", "task_id_441", "task_id_447"]
    
    results = []
    
    print(f"\nTesting {len(task_ids)} tasks...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for task_id in task_ids:
        result = test_single_task(task_id, tasks_dir)
        results.append(result)
    
    return results

def print_summary(results: list):
    """Print summary of test results."""
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    partial = sum(1 for r in results if r['status'] == 'partial')
    failed = sum(1 for r in results if r['status'] == 'failed')
    error = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\nTotal tasks: {total}")
    print(f"✓ Success (complete): {success} ({success/total*100:.1f}%)")
    print(f"⚠ Partial (impl only): {partial} ({partial/total*100:.1f}%)")
    print(f"✗ Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"! Error: {error} ({error/total*100:.1f}%)")
    
    print("\nDetailed results:")
    for r in results:
        status_symbol = {
            'success': '✓',
            'partial': '⚠',
            'failed': '✗',
            'error': '!'
        }.get(r['status'], '?')
        
        print(f"  {status_symbol} {r['task_id']}: {r['status']}")
        if r['status'] == 'error' and 'error' in r:
            print(f"    Error: {r['error']}")

def save_results(results: list, output_path: str):
    """Save test results to file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test DSPy-based Lean 4 code generation")
    parser.add_argument("--task", type=str, help="Test specific task ID (e.g., task_id_0)")
    parser.add_argument("--all", action="store_true", help="Test all standard tasks")
    parser.add_argument("--tasks-dir", type=str, default="tasks", 
                       help="Directory containing task folders")
    parser.add_argument("--output", type=str, default="dspy_test_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Resolve tasks directory
    if not os.path.isabs(args.tasks_dir):
        args.tasks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.tasks_dir)
    
    if not os.path.exists(args.tasks_dir):
        print(f"ERROR: Tasks directory not found: {args.tasks_dir}")
        return
    
    if args.task:
        # Test single task
        result = test_single_task(args.task, args.tasks_dir)
        print_summary([result])
        
    elif args.all:
        # Test all tasks
        results = test_all_tasks(args.tasks_dir)
        print_summary(results)
        save_results(results, args.output)
        
    else:
        print("Please specify --task <task_id> or --all")
        parser.print_help()

if __name__ == "__main__":
    main()
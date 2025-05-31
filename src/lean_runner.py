import subprocess
import os

def execute_lean_code(code: str) -> str:
    """
    Writes Lean code to TempProject.lean in temp_project directory, 
    executes it, and returns the output or errors.
    
    Args:
        code: The Lean code to execute
        
    Returns:
        str: Execution result message
    """
    temp_file = "TempTest.lean"  # Fixed filename
    
    try:
        # Write the Lean code to the temp file
        temp_path = os.path.join("lean_playground", temp_file)
        os.makedirs("lean_playground", exist_ok=True)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Execute Lean within the temp_project directory
        result = subprocess.run(
            ["lake", "lean", temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise exception on non-zero return code
        )

        # If execution was successful, return success message
        if result.returncode == 0:
            return "Lean code executed successfully"

        # If there was an error, return error message
        error_message = result.stderr.strip()
        if not error_message and result.stdout.strip():
            # Some Lean errors might be in stdout instead of stderr
            error_message = result.stdout.strip()
            
        error_msg = error_message if error_message else f"Lean execution failed with return code {result.returncode}"
        return f"Lean Error: {error_msg}"

    except FileNotFoundError:
        return "Error: Lean executable not found or temp_project directory doesn't exist."
    except PermissionError:
        return f"Error: Permission denied when writing to or executing {temp_file}"
    except Exception as e:
        return f"Unexpected error while running Lean: {str(e)}"


def execute_lean_code_tuple(code: str) -> tuple[bool, str, str]:
    """
    DSPy-compatible version that returns tuple format.
    
    Args:
        code: The Lean code to execute
        
    Returns:
        tuple[bool, str, str]: (success, output, error)
    """
    temp_file = "TempTest.lean"  # Fixed filename
    
    try:
        # Write the Lean code to the temp file
        temp_path = os.path.join("lean_playground", temp_file)
        os.makedirs("lean_playground", exist_ok=True)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Execute Lean within the temp_project directory
        result = subprocess.run(
            ["lake", "lean", temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise exception on non-zero return code
        )

        # If execution was successful, return success with output
        if result.returncode == 0:
            output = result.stdout.strip()
            return True, output, ""

        # If there was an error, return failure with error message
        error_message = result.stderr.strip()
        if not error_message and result.stdout.strip():
            # Some Lean errors might be in stdout instead of stderr
            error_message = result.stdout.strip()
            
        error_msg = error_message if error_message else f"Lean execution failed with return code {result.returncode}"
        return False, "", error_msg

    except FileNotFoundError:
        return False, "", "Error: Lean executable not found or temp_project directory doesn't exist."
    except PermissionError:
        return False, "", f"Error: Permission denied when writing to or executing {temp_file}"
    except Exception as e:
        return False, "", f"Unexpected error while running Lean: {str(e)}"


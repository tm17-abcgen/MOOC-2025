#!/usr/bin/env python3
"""
Test the lean_runner function to make sure it returns the correct format.
"""

import sys
import os
sys.path.append('src')

from lean_runner import execute_lean_code

def test_lean_runner():
    """Test that lean_runner returns the correct tuple format."""
    print("🧪 Testing lean_runner function...")
    
    # Test with simple Lean code
    test_code = """
-- Simple test
#eval 2 + 3
"""
    
    try:
        result = execute_lean_code(test_code)
        print(f"  Result type: {type(result)}")
        print(f"  Result length: {len(result)}")
        print(f"  Result: {result}")
        
        # Check if it's a 3-tuple
        if isinstance(result, tuple) and len(result) == 3:
            success, output, error = result
            print(f"  ✅ Success: {success}")
            print(f"  📄 Output: {output}")
            print(f"  ❌ Error: {error}")
            print("  ✅ Function returns correct 3-tuple format!")
            return True
        else:
            print(f"  ❌ Function should return 3-tuple, got: {type(result)}")
            return False
            
    except Exception as e:
        print(f"  ❌ Function failed with error: {e}")
        return False

if __name__ == "__main__":
    test_lean_runner()
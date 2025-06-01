"""
Lean 4 Syntax Checker and API Validator

This module provides utilities to validate Lean 4 syntax and API usage
before sending code to the Lean compiler.
"""

import re
from typing import List, Tuple, Dict
from enum import Enum

class ErrorSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class SyntaxIssue:
    def __init__(self, severity: ErrorSeverity, message: str, line: int = -1, suggestion: str = ""):
        self.severity = severity
        self.message = message
        self.line = line
        self.suggestion = suggestion
    
    def __str__(self):
        line_info = f" (line {self.line})" if self.line >= 0 else ""
        return f"[{self.severity.value.upper()}]{line_info}: {self.message}"

class Lean4SyntaxChecker:
    """Validates Lean 4 syntax and API usage."""
    
    def __init__(self):
        # Common API errors
        self.api_errors = {
            'Array.any?': 'Array.any',
            'Array.contains?': 'Array.contains',
            '.any?': '.any',
            '.contains?': '.contains',
            'List.any?': 'List.any',
            'List.contains?': 'List.contains'
        }
        
        # Lean 3 to Lean 4 mappings
        self.lean3_to_lean4 = {
            'iff.intro': 'constructor',
            'iff.left': '(.mp)',
            'iff.right': '(.mpr)',
            'and.intro': 'constructor',
            'and.left': '(.left)',
            'and.right': '(.right)',
            'or.inl': 'Or.inl',
            'or.inr': 'Or.inr'
        }
        
        # Problematic tactic patterns
        self.tactic_patterns = [
            (r'(\w+)\s*,\s*(\w+)', r'\1 <;> \2', "Use <;> instead of comma for tactic composition"),
            (r'unfold\s+\w+\s*,?', '', "Avoid unfold tactics, use simp instead"),
            (r'split\s*$', 'split\n  · sorry\n  · sorry', "Split needs case handling")
        ]
    
    def check_code(self, code: str) -> Tuple[str, List[SyntaxIssue]]:
        """Check and fix code implementation."""
        issues = []
        fixed_code = code
        
        # Check API usage
        fixed_code, api_issues = self._check_api_usage(fixed_code)
        issues.extend(api_issues)
        
        # Check Boolean/Prop consistency
        bool_issues = self._check_boolean_consistency(fixed_code)
        issues.extend(bool_issues)
        
        # Check if-then-else syntax
        ite_issues = self._check_if_then_else(fixed_code)
        issues.extend(ite_issues)
        
        return fixed_code, issues
    
    def check_proof(self, proof: str) -> Tuple[str, List[SyntaxIssue]]:
        """Check and fix proof tactics."""
        issues = []
        fixed_proof = proof
        
        # Check tactic composition
        fixed_proof, tactic_issues = self._check_tactic_composition(fixed_proof)
        issues.extend(tactic_issues)
        
        # Check Lean 3 vs Lean 4 syntax
        fixed_proof, syntax_issues = self._check_lean4_syntax(fixed_proof)
        issues.extend(syntax_issues)
        
        # Check problematic tactics
        problematic_issues = self._check_problematic_tactics(fixed_proof)
        issues.extend(problematic_issues)
        
        return fixed_proof, issues
    
    def _check_api_usage(self, code: str) -> Tuple[str, List[SyntaxIssue]]:
        """Check for incorrect API usage."""
        issues = []
        fixed_code = code
        
        for wrong_api, correct_api in self.api_errors.items():
            if wrong_api in fixed_code:
                fixed_code = fixed_code.replace(wrong_api, correct_api)
                issues.append(SyntaxIssue(
                    ErrorSeverity.ERROR,
                    f"Incorrect API usage: {wrong_api}",
                    suggestion=f"Use {correct_api} instead"
                ))
        
        return fixed_code, issues
    
    def _check_boolean_consistency(self, code: str) -> List[SyntaxIssue]:
        """Check Bool/Prop consistency."""
        issues = []
        
        # Check for missing decide wrappers
        prop_pattern = r'([a-zA-Z_]\w*)\s*([<>=≤≥]+)\s*([a-zA-Z_0-9]+)'
        matches = re.findall(prop_pattern, code)
        
        for match in matches:
            var1, op, var2 = match
            comparison = f"{var1} {op} {var2}"
            if comparison in code and '&&' in code and 'decide' not in code:
                issues.append(SyntaxIssue(
                    ErrorSeverity.WARNING,
                    f"May need decide wrapper for: {comparison}",
                    suggestion=f"Use decide ({comparison}) for Bool context"
                ))
        
        return issues
    
    def _check_if_then_else(self, code: str) -> List[SyntaxIssue]:
        """Check if-then-else syntax."""
        issues = []
        
        # Check for proper if-then-else structure
        if_pattern = r'if\s+.*\s+then\s+.*\s+else\s+.*'
        if 'if' in code and not re.search(if_pattern, code, re.DOTALL):
            issues.append(SyntaxIssue(
                ErrorSeverity.WARNING,
                "Incomplete if-then-else structure",
                suggestion="Ensure proper if condition then value1 else value2 syntax"
            ))
        
        return issues
    
    def _check_tactic_composition(self, proof: str) -> Tuple[str, List[SyntaxIssue]]:
        """Check and fix tactic composition."""
        issues = []
        fixed_proof = proof
        
        for pattern, replacement, message in self.tactic_patterns:
            matches = re.findall(pattern, fixed_proof)
            if matches:
                fixed_proof = re.sub(pattern, replacement, fixed_proof)
                issues.append(SyntaxIssue(
                    ErrorSeverity.ERROR,
                    message,
                    suggestion=f"Pattern fixed: {pattern} → {replacement}"
                ))
        
        return fixed_proof, issues
    
    def _check_lean4_syntax(self, proof: str) -> Tuple[str, List[SyntaxIssue]]:
        """Check and fix Lean 3 vs Lean 4 syntax."""
        issues = []
        fixed_proof = proof
        
        for lean3, lean4 in self.lean3_to_lean4.items():
            if lean3 in fixed_proof:
                fixed_proof = fixed_proof.replace(lean3, lean4)
                issues.append(SyntaxIssue(
                    ErrorSeverity.ERROR,
                    f"Lean 3 syntax detected: {lean3}",
                    suggestion=f"Use Lean 4 syntax: {lean4}"
                ))
        
        return fixed_proof, issues
    
    def _check_problematic_tactics(self, proof: str) -> List[SyntaxIssue]:
        """Check for problematic tactic usage."""
        issues = []
        
        # Check for standalone unfold
        if re.search(r'\bunfold\b', proof):
            issues.append(SyntaxIssue(
                ErrorSeverity.WARNING,
                "unfold tactic may fail",
                suggestion="Consider using simp [function_name] instead"
            ))
        
        # Check for standalone simp that might fail
        if proof.strip() == 'simp':
            issues.append(SyntaxIssue(
                ErrorSeverity.WARNING,
                "Standalone simp may not make progress",
                suggestion="Consider adding case analysis: cases a <;> cases b <;> simp"
            ))
        
        # Check for split without context
        if proof.strip() == 'split':
            issues.append(SyntaxIssue(
                ErrorSeverity.WARNING,
                "Split needs proper case handling",
                suggestion="Add case branches or use split <;> simp"
            ))
        
        return issues

class BooleanPatternAnalyzer:
    """Analyzes Boolean logic patterns for appropriate tactics."""
    
    @staticmethod
    def suggest_tactics(goal: str, implementation: str) -> List[str]:
        """Suggest tactics based on goal and implementation patterns."""
        tactics = []
        
        # Boolean equivalence patterns
        if '↔' in goal or 'iff' in goal.lower():
            if 'Bool' in goal or 'decide' in goal:
                tactics.append("simp [decide_eq_true_iff]")
            elif any(var in goal for var in ['a', 'b', 'c']) and 'Bool' in implementation:
                tactics.append("cases a <;> cases b <;> simp")
            else:
                tactics.append("constructor")
        
        # Array membership patterns
        if 'Array' in goal or '∈' in goal:
            tactics.extend(["simp [Array.mem_def]", "simp [Array.contains_def]"])
        
        # If-then-else patterns
        if 'if' in goal and 'then' in goal:
            tactics.extend(["split", "cases h <;> simp"])
        
        # Arithmetic patterns
        if any(op in goal for op in ['<', '>', '≤', '≥', '=']):
            tactics.extend(["omega", "simp <;> omega"])
        
        # Default fallback
        if not tactics:
            tactics.extend(["simp", "rfl", "exact"])
        
        return tactics

class ProofPatternMatcher:
    """Matches proof patterns and suggests appropriate strategies."""
    
    def __init__(self):
        self.patterns = {
            'boolean_iff': {
                'keywords': ['Bool', 'decide', '↔', 'true', 'false'],
                'tactics': ['cases a <;> cases b <;> simp', 'simp [decide_eq_true_iff]'],
                'description': 'Boolean equivalence proof'
            },
            'array_membership': {
                'keywords': ['Array', '∈', 'contains', 'any'],
                'tactics': ['simp [Array.mem_def]', 'simp [Array.contains_def]', 'simp [Array.any_eq_true]'],
                'description': 'Array membership proof'
            },
            'conditional': {
                'keywords': ['if', 'then', 'else'],
                'tactics': ['split', 'split <;> simp', 'cases h <;> simp'],
                'description': 'Conditional statement proof'
            },
            'arithmetic': {
                'keywords': ['Nat', 'Int', '+', '-', '*', '/', '%', '<', '>', '≤', '≥'],
                'tactics': ['omega', 'simp <;> omega', 'norm_num'],
                'description': 'Arithmetic proof'
            }
        }
    
    def match_pattern(self, goal: str, implementation: str) -> Dict[str, any]:
        """Match goal and implementation to known patterns."""
        combined_text = f"{goal} {implementation}".lower()
        
        scores = {}
        for pattern_name, pattern_info in self.patterns.items():
            score = sum(1 for keyword in pattern_info['keywords'] 
                       if keyword.lower() in combined_text)
            if score > 0:
                scores[pattern_name] = score
        
        if not scores:
            return {
                'pattern': 'unknown',
                'tactics': ['simp', 'exact', 'rfl'],
                'description': 'General proof pattern'
            }
        
        best_pattern = max(scores.keys(), key=lambda k: scores[k])
        return {
            'pattern': best_pattern,
            'tactics': self.patterns[best_pattern]['tactics'],
            'description': self.patterns[best_pattern]['description']
        }

def validate_lean_syntax(code: str, is_proof: bool = False) -> Tuple[str, List[SyntaxIssue]]:
    """Main function to validate and fix Lean 4 syntax."""
    checker = Lean4SyntaxChecker()
    
    if is_proof:
        return checker.check_proof(code)
    else:
        return checker.check_code(code)

def suggest_proof_tactics(goal: str, implementation: str) -> List[str]:
    """Suggest appropriate proof tactics based on patterns."""
    analyzer = BooleanPatternAnalyzer()
    return analyzer.suggest_tactics(goal, implementation)

def match_proof_pattern(goal: str, implementation: str) -> Dict[str, any]:
    """Match proof to known patterns and suggest strategies."""
    matcher = ProofPatternMatcher()
    return matcher.match_pattern(goal, implementation)
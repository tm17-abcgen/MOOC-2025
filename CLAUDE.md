# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Lab 2 (Deliverable 2: Coding Agent)** from the **Advanced LLM Agents MOOC, Spring 2025**. It's a DSPy-based Lean 4 automated theorem prover that generates code implementations and formal proofs from natural language descriptions. Students must implement an agentic workflow that solves programming tasks and proves implementations satisfy given specifications.

The project uses a starter code template with multi-agent workflows using GPT-4o and o3-mini models, incorporating Chain of Thought reasoning, RAG (Retrieval-Augmented Generation), and automatic prompt optimization.

## Architecture

### Core Components
- **DSPy Workflow**: Multi-agent system with declarative signatures and automatic prompt optimization
- **Agent Types**: Planning (GPT-4o), Generation (GPT-4o + RAG), Verification (GPT-4o + o3-mini), Proof Enhancement (o3-mini)
- **RAG System**: Embedding database built from Lean 4 documentation in `documents/`
- **Task Structure**: Each task in `tasks/task_id_*/` contains description.txt, task.lean template, and tests.lean

### Key Workflow
1. Planning agent analyzes problem complexity and patterns
2. Generation agent creates Lean 4 implementations using RAG examples
3. Verification agent tests compilation and correctness
4. Proof enhancer iteratively improves proofs with failure learning
5. Progressive enhancement through multiple refinement attempts

## Development Commands

### Setup and Installation
```bash
# Setup virtual environment and dependencies
./setup.sh

# Activate environment (after setup)
source venv/bin/activate
```

### Testing
```bash
# Run all tests
make test

# Test single task
python run_dspy_tests.py --task task_id_0

# Test all tasks
python run_dspy_tests.py --all

# Test via main workflow
python src/main.py --test-single task_id_0
```

### Optimization
```bash
# Optimize DSPy prompts using MIPROv2
python src/main.py --optimize

# Build/rebuild dataset
python src/build_dataset.py
```

### Lean Development
```bash
# Verify Lean installation
lake --version
lake lean Lean4CodeGenerator.lean

# Update Lean dependencies (takes 10+ minutes)
lake update
```

### Submission
```bash
# Package for submission
make zip
```

## Key Implementation Details

### Assignment Requirements
- **Core Task**: Implement `main_workflow()` function in `src/main.py` 
- **Input**: Problem description (string) and Lean template (string)
- **Output**: Dictionary with "code" and "proof" keys containing Lean 4 implementations
- **Constraint**: No hard-coding solutions (submissions audited for violations)
- **Models**: Use provided GPT-4o and o3-mini agents in `src/agents.py`

### Main Entry Point  
- `src/main.py::main_workflow()` - Primary function called by test harness
- Takes problem description and Lean template, returns dict with "code" and "proof" keys
- Uses global workflow instance for efficiency across multiple calls

### Starter Code Components
- `src/dspy_signatures.py` - Declarative task definitions for each agent
- `src/dspy_agents.py` - Agent implementations with specific model assignments  
- `src/dspy_workflow.py` - Main orchestration logic and progressive enhancement
- `src/agents.py` - Provided GPT-4o and o3-mini agent classes (DO NOT MODIFY)

### RAG System
- `src/embedding_db.py` - Processes documents/ into embeddings for similarity search
- `src/embedding_models.py` - Provides OpenAI and lightweight embedding models
- `generate_embeddings.py` - Builds embedding database from Lean 4 documentation
- Documents split on `<EOC>` tags, uses cosine similarity for retrieval

### Testing Framework
- `tests/tests.py` - Test harness that scans tasks/ directory (DO NOT MODIFY)
- Tests both implementation-only (proof = "sorry") and complete solutions
- Uses `src/lean_runner.py` to execute Lean code via filesystem

## Environment Variables
Required in `.env`:
- `OPENROUTER_API_KEY` - For GPT-4o and o3-mini model access

## Task Structure
Each `tasks/task_id_*/` contains:
- `description.txt` - Natural language problem description
- `task.lean` - Template with {{code}} and {{proof}} placeholders
- `tests.lean` - Unit tests for verification
- `signature.json` - Function signature metadata

## Assignment Constraints & Submission
- **No "sorry"**: Must generate non-trivial proofs (no "sorry" allowed in final output)
- **No hard-coding**: Hard-coded responses in prompts or database will result in disqualification
- **Model restrictions**: Use only provided GPT-4o and o3-mini agents (no o1-pro, etc.)
- **Available libraries**: Mathlib, Aesop
- **Submission**: Run `make zip` to package entire project directory

### Recommended Multi-Agent Architecture
1. **Planning Agent**: Task decomposition, strategy, feedback from Lean execution
2. **Generation Agent**: Code and proof generation with RAG integration  
3. **Verification/Feedback Agent**: Execute Lean code via `execute_lean_code` function

### Optional PDF Documentation
Students may submit a 2-page PDF describing agent architecture for partial credit, including:
- Number and role of each agent
- Planning, generation, verification, and feedback handling
- RAG usage and error-handling logic
- Design choices and trade-offs
# DSPy-based Lean 4 Code Generator

This is a complete reimplementation of the Lean 4 code generation system using DSPy (Declarative Self-improving Python) with Chain of Thought reasoning and automatic prompt optimization.

## 🚀 Features

- **DSPy Signatures**: Declarative task definitions for each agent
- **Chain of Thought (CoT)**: All agents use CoT reasoning for better results
- **Progressive Enhancement**: Gradual improvement of proofs through multiple strategies
- **RAG Support**: Retrieval-Augmented Generation using embedding database
- **Prompt Optimization**: MIPROv2 and BootstrapFewShot optimization
- **Multi-Agent Architecture**: Specialized agents for planning, generation, verification, and proof enhancement
- **Failed Proof Learning**: System learns from previous failures to improve future attempts

## 📋 Prerequisites

- Python 3.8+
- `uv` package manager (install from https://github.com/astral-sh/uv)
- OpenAI API key with access to GPT-4o and o3-mini models
- Lean 4 installed (for verification)

## 🛠️ Installation

1. Run the setup script:
```bash
./setup.sh
```

2. Configure your API keys in `.env`:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. Test the setup:
```bash
python test_dspy_setup.py
```

## 🏗️ Architecture

### DSPy Agents

1. **DSPyPlanningAgent** (GPT-4o)
   - Analyzes problem complexity
   - Detects patterns (Boolean, Conditional, Array)
   - Creates implementation and proof strategies

2. **DSPyGenerationAgent** (GPT-4o with RAG)
   - Generates Lean 4 implementations
   - Generates formal proofs
   - Uses RAG for similar examples
   - Includes error recovery

3. **DSPyVerificationAgent** (GPT-4o + o3-mini)
   - Verifies implementations
   - Analyzes compilation errors
   - Uses o3-mini for mathematical reasoning

4. **DSPyProofEnhancer** (o3-mini)
   - Analyzes proof goal structure
   - Creates comprehensive proof plans
   - Learns from failed attempts

### DSPy Signatures

All agents use declarative signatures that define:
- Input fields with descriptions
- Output fields with expected formats
- Automatic prompt generation
- Type constraints and validation

Example:
```python
class CodeGeneration(dspy.Signature):
    """Generate Lean 4 function implementation based on plan."""
    problem_description = dspy.InputField(desc="Natural language description")
    lean_template = dspy.InputField(desc="Lean 4 template with {{code}} placeholder")
    implementation_plan = dspy.InputField(desc="Detailed implementation strategy")
    similar_examples = dspy.InputField(desc="Similar Lean 4 code examples from RAG")
    
    lean_code = dspy.OutputField(desc="Complete Lean 4 function implementation")
    code_explanation = dspy.OutputField(desc="Brief explanation of the implementation")
```

## 🎯 Usage

### Test a Single Task
```bash
python run_dspy_tests.py --task task_id_0
```

### Test All Tasks
```bash
python run_dspy_tests.py --all
```

### Run with Main Workflow
```bash
python src/main.py --test-single task_id_0
```

### Optimize Prompts
```bash
python src/main.py --optimize
```

### Build Dataset
```bash
python src/build_dataset.py
```

## 🔧 Configuration

Edit `.env` to configure:
- Model choices for each agent
- Maximum retry attempts
- Progressive enhancement settings
- Embedding database path

## 📊 Workflow

1. **Problem Analysis**: Pattern detection and complexity assessment
2. **Planning**: Create structured implementation and proof strategies
3. **Implementation Generation**: Generate code with RAG examples
4. **Verification**: Test implementation correctness
5. **Proof Strategy Refinement**: Update strategy based on implementation
6. **Proof Generation**: Generate proof with progressive enhancement
   - Attempt 1: Basic generation
   - Attempt 2: Goal structure analysis (o3-mini)
   - Attempt 3+: Comprehensive planning with failure learning
7. **Final Verification**: Ensure complete solution works

## 🧪 Testing

The system is tested on 11 standard tasks:
- task_id_0: Identity function
- task_id_58: Boolean operations
- task_id_77: Conditional logic
- task_id_127-447: Various complexity levels

## 🚀 Optimization

DSPy provides automatic prompt optimization:

1. **MIPROv2**: Optimizes instructions and few-shot examples
2. **BootstrapFewShot**: Learns from successful examples
3. **Custom Metrics**: Lean-specific evaluation for optimization

Run optimization:
```bash
python src/main.py --optimize
```

## 📁 Project Structure

```
lab2-starter-code-dspy/
├── src/
│   ├── dspy_signatures.py    # DSPy signature definitions
│   ├── dspy_agents.py        # Agent implementations
│   ├── dspy_workflow.py      # Main workflow orchestration
│   ├── main.py               # Entry point
│   ├── build_dataset.py      # Dataset creation
│   ├── embedding_db.py       # RAG support
│   └── lean_runner.py        # Lean execution
├── tasks/                    # Test tasks
├── run_dspy_tests.py        # Test runner
├── setup.sh                 # Setup script
├── requirements.txt         # Dependencies
└── .env.example            # Configuration template
```

## 🔍 Key Innovations

1. **Declarative Agents**: No manual prompt engineering needed
2. **Automatic Optimization**: DSPy learns optimal prompts from data
3. **Progressive Enhancement**: Gradual improvement strategy
4. **Failed Proof Learning**: System improves from mistakes
5. **Hybrid Models**: GPT-4o for generation, o3-mini for reasoning

## 🐛 Troubleshooting

1. **Import Errors**: Ensure virtual environment is activated
2. **API Errors**: Check .env configuration and API keys
3. **Lean Errors**: Ensure Lean 4 is installed and in PATH
4. **Memory Issues**: Reduce batch size in optimization

## 📈 Performance

The DSPy implementation provides:
- Better prompt consistency through signatures
- Automatic prompt optimization
- Learning from failures
- Progressive enhancement for complex proofs
- RAG-based example retrieval

## 🤝 Contributing

To extend the system:
1. Add new signatures in `dspy_signatures.py`
2. Create new agents in `dspy_agents.py`
3. Extend workflow in `dspy_workflow.py`
4. Add optimization metrics in `main.py`

## 📝 License

This project is part of the Advanced LLM Agents MOOC.
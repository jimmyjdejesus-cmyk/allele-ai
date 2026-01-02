# Chain of Thought (COT) Prompt Support

## ADDED Requirements

### REQ-COT-001: COT Prompt Building
The system MUST provide a function to wrap prompts with Chain of Thought instructions.

#### Scenario: Build COT prompt for reasoning task
Given a base prompt "Solve: If 5 apples cost $2, how much do 15 apples cost?"
When build_cot_prompt() is called
Then it returns: "Solve: If 5 apples cost $2, how much do 15 apples cost?\n\nLet's think step by step:"

#### Scenario: Build COT prompt for multiple choice
Given a base prompt "Question: What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Madrid\nAnswer:"
When build_cot_prompt() is called
Then it returns the prompt with COT instruction appended
And preserves the original prompt structure

### REQ-COT-002: COT Mode in GenomeModel
The system MUST support COT mode as a special personality configuration that applies COT prompts without genome traits.

#### Scenario: Generate with COT mode
Given a GenomeModel instance
And cot_mode=True
And a user prompt "What is 2+2?"
When generate() is called
Then it applies COT prompt wrapper
And does NOT apply genome trait system prompts
And sends the COT-wrapped prompt to the LLM

#### Scenario: Generate with COT mode and baseline comparison
Given a GenomeModel instance with cot_mode=True
And a baseline GenomeModel instance (no genome, no COT)
When both generate responses to the same prompt
Then COT mode response includes step-by-step reasoning
And baseline response may not include explicit reasoning steps

### REQ-COT-003: COT Integration with Benchmarks
The system MUST apply COT prompts to reasoning benchmarks (GSM8K, ARC) when COT mode is selected.

#### Scenario: Apply COT to GSM8K benchmark
Given benchmark "gsm8k"
And personality "cot"
When the benchmark is executed
Then COT prompt wrapper is applied to each GSM8K question
And the model is instructed to think step by step

#### Scenario: Apply COT to MMLU benchmark
Given benchmark "mmlu"
And personality "cot"
When the benchmark is executed
Then COT prompt wrapper is applied to each MMLU question
And the model is instructed to think step by step before answering

## MODIFIED Requirements

### REQ-COT-004: Extend build_system_prompt
The existing build_system_prompt() function MUST be extended or a new function MUST be added to support COT prompts.

#### Scenario: COT prompt does not conflict with personality prompts
Given a personality configuration (e.g., technical_expert)
And COT mode enabled
When both are applied
Then COT wrapper is applied AFTER personality system prompt
And both instructions are present in the final prompt


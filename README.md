# Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

This repository contains the implementation for the paper *"Learn from Your Mistakes:
Tree-like Self-Play for Secure Code LLMs"*.

## Abstract

While Large Language Models (LLMs) have demonstrated remarkable performance in code generation, their effectiveness is undermined by a propensity to replicate subtle yet critical security vulnerabilities endemic to their training data. Current security alignment techniques often treat code as an indivisible unit, making it difficult to precisely correct localized, single-token errors that cause vulnerabilities. In this work, we introduce Tree-like Self-Play (TSP), a novel training framework that reframes secure code generation as a targeted self-play process. TSP models secure code generation as a tree-structured sequential decision problem, where each node represents a state in the generation process. At these "CWE Risk Nodes," the model engages in self-play by generating multiple code variants as branches. The main player's core task is to learn from its own mistakes by distinguishing the secure "golden path" from the potentially vulnerable branches. Our extensive experiments demonstrate that TSP substantially improves the security pass rate of models like CodeLlama-7B and shows strong generalization to unseen vulnerability types and other programming languages.

## Methodology

The core problem with standard fine-tuning (SFT) and reinforcement learning (RL) is that feedback is often provided at the program level, failing to pinpoint the specific, often single-token, error that introduces a vulnerability.

Our solution, **Tree-like Self-Play (TSP)**, addresses this by:
1. **Modeling Code Generation as a Tree**: We treat the process of generating code from a prompt as a path traversal through a decision tree. Each node is a state in the generation, and each edge is a token choice.
2. **Identifying CWE Risk Nodes**: We identify specific, critical junctures in the generation process where an insecure choice could be made. These are forks in the path where a vulnerability might be introduced.
3. **Targeted Self-Play**: At these risk nodes, the model (the "opponent player") generates multiple exploratory branches, creating plausible but insecure code variants. The model being trained (the "main player") is then tasked with learning to distinguish the secure "golden path" from these flawed alternatives. This provides a highly targeted, on-policy learning signal for self-correction.

This iterative process of self-improvement allows the model to build a more generalizable understanding of security principles by confronting and correcting its own local errors.

## Key Results

Our experiments show that TSP significantly enhances the security of code generation models while preserving their general coding abilities.

- **Enhanced Security**: TSP improved the Security Pass Rate (SPR@1) of CodeLlama-7B to **75.8%**, outperforming standard SFT (57.0%) and a standard self-play baseline (74.5%).
- **Generalization to Unseen Vulnerabilities**: The model robustly defends against previously unseen vulnerability types, achieving a **32% reduction in vulnerabilities** over SFT on novel CWEs.
- **Cross-Lingual Transfer**: Security principles learned from C/C++ data were successfully transferred to other languages, including **Python, JavaScript, Go, and Ruby**.
- **Minimal Performance Impact**: These security gains were achieved with a negligible impact on the models' general-purpose coding abilities, as measured by the HumanEval benchmark.

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU with at least 24GB VRAM (for 7B model training)
- [vLLM](https://github.com/vllm-project/vllm) for inference
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for DPO training

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Easonnoway/TSP.git
cd TSP

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Set up LLaMA-Factory (required for training)
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

# 5. Create environment configuration
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY (only needed for CWE annotation)
```

### Dataset

Two annotated CWE vulnerability datasets are included in `data/annotated_datasets/`:

- `diversevul_new_annotated.json`: DiverseVul-derived annotated dataset (~7.2MB, 1,353 items)
- `sec-new-desc_annotated.json`: Security description annotated dataset (~2.8MB, 421 items)

These serve as the input for the TSP pipeline.

### Quick Start

Run the full TSP pipeline with CodeLlama-7B:

```bash
# Set environment variables
export TSP_MODEL_PATH="CodeLlama-7b-Instruct-hf"

# Run the pipeline
bash src/training/TSP_example.sh \
    -i data/annotated_datasets/sec-new-desc_annotated_with_nodes.json \
    -o ./output
```

The pipeline consists of three steps:
1. **Inference** — Generate code completions at CWE risk nodes using vLLM
2. **Preference Pair Creation** — Build DPO training pairs from secure/insecure generations
3. **Training** — Fine-tune the model using DPO via LLaMA-Factory

### Reproducing Evaluation

See `src/evaluation/` for evaluation scripts used in our experiments:

```bash
# Run inference for evaluation
bash src/evaluation/inference/script.sh

# Run LLM-based vulnerability evaluation
python src/evaluation/llm_evaluate/llm_evaluate_vuln.py

# Run CodeQL-based evaluation (RQ1)
bash src/evaluation/codeql/codeql_analysis_script.sh
```

### Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key for GPT-4o annotation |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | API base URL for annotation |
| `TSP_MODEL_PATH` | `CodeLlama-7b-Instruct-hf` | Model name or path for inference |
| `LLAMA_FACTORY_DIR` | `<repo>/LLaMA-Factory` | Path to LLaMA-Factory installation |
| `TSP_CACHE_DIR` | `./cache` | Cache directory for API responses |
| `TSP_DATA_DIR` | `../../data` | Default data directory (shell scripts) |
| `TSP_OUTPUT_DIR` | `../../output_data` | Default output directory (shell scripts) |
| `TSP_MODEL_DIR` | `../../models` | Default model directory (shell scripts) |

## Repository Structure

```
TSP/
├── src/
│   ├── api_annotation/                      # GPT-4o CWE vulnerability annotation
│   │   ├── api_annotation.py                # Ray-based parallel API client
│   │   ├── api_prompt.py                    # System prompt template
│   │   └── script.sh                        # Annotation pipeline wrapper
│   ├── dataset_processing/                  # Dataset processing & DPO conversion
│   │   ├── utils.py                         # Post-processing & DPO format conversion
│   │   └── inference/
│   │       ├── inference_with_template.py   # vLLM code generation at risk nodes
│   │       └── script.sh                    # Inference pipeline wrapper
│   ├── training/                            # DPO training
│   │   ├── TSP_example.sh                   # Main 3-step pipeline entry point
│   │   └── config/
│   │       ├── codellama_7b.yaml            # CodeLlama-7B DPO training config
│   │       └── qwencoder_7b.yaml            # Qwen2.5-Coder-7B DPO training config
│   └── evaluation/                          # Evaluation scripts (RQ1/RQ2/RQ3)
│       ├── inference/
│       │   ├── inference_with_template.py   # Unified evaluation inference
│       │   ├── convert_to_database.py       # Convert outputs to CodeQL databases
│       │   └── script.sh
│       ├── llm_evaluate/
│       │   ├── llm_evaluate_vuln.py         # LLM-based vulnerability evaluation
│       │   └── calculate_results.py         # Result aggregation
│       └── codeql/
│           ├── analyze_cwe_vulnerabilities.py  # CodeQL CWE analysis
│           └── codeql_analysis_script.sh
├── data/
│   ├── annotated_datasets/                  # Training datasets
│   │   ├── diversevul_new_annotated.json
│   │   └── sec-new-desc_annotated.json
│   ├── evaluation_datasets/                 # Evaluation datasets (per RQ)
│   │   ├── rq1_dataset.json
│   │   ├── rq2_dataset.json
│   │   ├── rq1_cwe_evaluate.json
│   │   └── rq3_cwe_evaluate_ablation.json
│   └── testcases.zip                        # Generated test cases (all RQs)
├── results/                                 # Experimental results
│   ├── rq1/
│   │   ├── inference_outputs/               # Model inference outputs
│   │   ├── evaluation_results/              # LLM evaluation results
│   │   └── codeql_results/                  # CodeQL analysis results
│   ├── rq2/
│   │   ├── inference_outputs/
│   │   └── evaluation_results/
│   └── rq3/
│       ├── inference_outputs/
│       └── evaluation_results/
├── requirements.txt                         # Python dependencies
├── .env.example                             # Environment variable template
├── LICENSE                                  # MIT License
├── CITATION.cff                             # Citation metadata
└── README.md                                # This file
```

## Citation

If you use this work, please cite our paper:

```bibtex
@article{tsp2025,
  title={Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

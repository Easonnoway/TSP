"""Step 1: Opponent Player Branch Generation.

Generates self-play code branches at CWE Risk Nodes using the opponent
player model (vLLM). The opponent produces alternative code continuations
from the prefix before each risk node.

Corresponds to Algorithm 1:
    y_{i,v} ~ p_{θ_{t-1}}(·|x_i, y_{i,<k_v})

Default hyperparameters match Table 7 in Appendix E:
    temperature=1.0, top_p=0.95, samples_per_node=4, max_tokens=1024
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports for GPU dependencies (mockable in tests)
_LLM = None
_SamplingParams = None


def _get_llm():
    """Lazy-load vLLM LLM class (mockable in tests)."""
    global _LLM
    if _LLM is None:
        from vllm import LLM
        _LLM = LLM
    return _LLM


def _get_sampling_params():
    """Lazy-load vLLM SamplingParams class (mockable in tests)."""
    global _SamplingParams
    if _SamplingParams is None:
        from vllm import SamplingParams
        _SamplingParams = SamplingParams
    return _SamplingParams


def _create_codellama_prompt(description: str, code_prefix: str) -> str:
    """Create prompt for CodeLlama-style models."""
    question = f"Please directly complete the following code generation task. Only provide the code without explanation:\n\n{description}"
    return f"<s>[INST] {question} [/INST] {code_prefix}"


def _split_code_at_node(func_src: str, code_line: str) -> str:
    """Split source code at the vulnerability line, returning the prefix.

    Uses exact line match first, falls back to stripped exact match,
    then to substring match with minimum length guard.
    """
    target_stripped = code_line.strip()

    # Guard: skip very short patterns that would match too broadly
    if len(target_stripped) < 4:
        logger.warning(f"Code line too short for reliable matching: '{target_stripped}'")
        return ""

    lines = func_src.split("\n")
    prefix_lines = []

    # Pass 1: exact stripped match
    for i, line in enumerate(lines):
        if line.strip() == target_stripped:
            return "\n".join(lines[:i])

    # Pass 2: substring match (target must be at least 50% of line to avoid false positives)
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if target_stripped in line_stripped and len(target_stripped) >= len(line_stripped) * 0.5:
            return "\n".join(lines[:i])

    logger.warning(f"Code line not found in source: '{target_stripped[:60]}...'")
    return ""


def generate_branches(
    data_file: str,
    output_file: str,
    model_path: str = "CodeLlama-7b-Instruct-hf",
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    samples_per_node: int = 4,
    tensor_parallel: int = 4,
    gpu_ids: str = "0,1,2,3",
    max_model_len: int = 16024,
) -> str:
    """Generate self-play code branches at CWE Risk Nodes.

    For each data item with annotated Nodes, splits the source code at
    each vulnerability line and generates alternative continuations from
    the prefix using the opponent player model.

    Args:
        data_file: Path to input JSON with Nodes[].Code_Line fields.
        output_file: Path to write output JSON with Generated_Code added.
        model_path: Model name or path for vLLM inference.
        temperature: Sampling temperature (paper default: 1.0).
        top_p: Top-p sampling (paper default: 0.95).
        max_tokens: Max new tokens per generation (paper default: 1024).
        samples_per_node: Number of branches per risk node (paper default: 4).
        tensor_parallel: Number of GPUs for tensor parallelism.
        gpu_ids: Comma-separated GPU IDs.
        max_model_len: Maximum model context length.

    Returns:
        Path to the output file.
    """
    LLM = _get_llm()
    SamplingParams = _get_sampling_params()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} items from {data_file}")

    # Collect all prompts (one per node, per sample)
    generation_tasks = []
    for item_idx, item in enumerate(data):
        if "Nodes" not in item:
            continue
        description = item["description"]
        func_src = item["func_src_after"]

        for node_idx, node in enumerate(item["Nodes"]):
            code_line = node.get("Code_Line", "")
            code_prefix = _split_code_at_node(func_src, code_line)
            prompt = _create_codellama_prompt(description, code_prefix)

            for sample_idx in range(samples_per_node):
                generation_tasks.append({
                    "item_idx": item_idx,
                    "node_idx": node_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "code_prefix": code_prefix,
                })

    logger.info(f"Total generation tasks: {len(generation_tasks)}")

    # Batch inference
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        max_model_len=max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    prompts = [task["prompt"] for task in generation_tasks]
    outputs = llm.generate(prompts, sampling_params)

    # Organize results: group generated code by (item_idx, node_idx)
    generated_map = {}
    for task, output in zip(generation_tasks, outputs):
        key = (task["item_idx"], task["node_idx"])
        generated_text = output.outputs[0].text
        full_code = task["code_prefix"] + "\n" + generated_text

        if key not in generated_map:
            generated_map[key] = []
        generated_map[key].append(full_code)

    # Write output: expand items so each sample creates its own entry
    # This allows downstream pairs.py to build one pair per sample
    expanded_data = []
    for item_idx, item in enumerate(data):
        if "Nodes" not in item:
            expanded_data.append(item)
            continue

        # Create one expanded item per sample_idx
        max_samples = 0
        for node in item["Nodes"]:
            key = (item_idx, item.get("_node_offset", 0) + item["Nodes"].index(node))
            count = len(generated_map.get(key, []))
            max_samples = max(max_samples, count)

        if max_samples == 0:
            # Fallback: use original item with first available generation
            for node in item["Nodes"]:
                node_key = (item_idx, item["Nodes"].index(node))
                if node_key in generated_map and generated_map[node_key]:
                    node["Generated_Code"] = generated_map[node_key][0]
            expanded_data.append(item)
        else:
            for s in range(max_samples):
                new_item = {**item}
                new_nodes = []
                for n_idx, node in enumerate(item["Nodes"]):
                    new_node = {**node}
                    node_key = (item_idx, n_idx)
                    branches = generated_map.get(node_key, [])
                    new_node["Generated_Code"] = branches[s] if s < len(branches) else (branches[0] if branches else "")
                    new_nodes.append(new_node)
                new_item["Nodes"] = new_nodes
                expanded_data.append(new_item)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(expanded_data, f, indent=2, ensure_ascii=False)

    total_nodes = len(generated_map)
    logger.info(f"Generated branches for {total_nodes} risk nodes "
                f"({len(expanded_data)} expanded items) -> {output_file}")
    return output_file

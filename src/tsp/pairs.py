"""Step 2: Preference Pair Construction.

Builds (golden_path, self_play_path) pairs from the generated branches,
then converts them into the training format for preference optimization.

Corresponds to Algorithm 1:
    P_t ← P_t ∪ {(y_i, y_{i,v})}
"""

import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)


def build_preference_pairs(generated_file: str, output_file: str) -> str:
    """Build preference pairs from generated self-play branches.

    For each data item's CWE Risk Node, creates a pair:
        - prompt: the task description
        - original_code: the golden path (secure code from func_src_after)
        - generated_code: the self-play branch (opponent's generation)

    Corresponds to Algorithm 1: P_t ← P_t ∪ {(y_i, y_{i,v})}

    Args:
        generated_file: Path to JSON with Nodes[].Generated_Code fields.
        output_file: Path to write preference pairs JSON.

    Returns:
        Path to the output file.
    """
    with open(generated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        original_code = item["func_src_after"]
        prompt = item["description"]

        for node in item.get("Nodes", []):
            generated_code = node.get("Generated_Code", "")
            if not generated_code:
                continue

            pairs.append({
                "prompt": prompt,
                "original_code": original_code,
                "generated_code": generated_code,
            })

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    logger.info(f"Built {len(pairs)} preference pairs -> {output_file}")
    return output_file


def convert_to_training_format(pairs_file: str, output_file: str) -> str:
    """Convert preference pairs to TSP training format.

    Transforms pairs into the instruction/chosen/rejected format used
    by preference optimization training:
        - instruction: the task description (prompt)
        - input: empty (not used)
        - chosen: the golden path (secure code, preferred)
        - rejected: the self-play branch (potentially insecure, dispreferred)

    Args:
        pairs_file: Path to preference pairs JSON.
        output_file: Path to write training format JSON.

    Returns:
        Path to the output file.
    """
    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    training_data = []
    for pair in pairs:
        prompt = pair.get("prompt", "")
        original_code = pair.get("original_code", "")
        generated_code = pair.get("generated_code", "")

        training_data.append({
            "instruction": prompt,
            "input": "",
            "chosen": original_code,
            "rejected": generated_code,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Converted {len(training_data)} pairs to training format -> {output_file}")
    return output_file

"""Step 3: TSP Training.

Trains the main player using the collected preference pairs via
LLaMA-Factory's preference optimization pipeline.

Corresponds to Algorithm 1:
    θ_t ← argmin L_TSP(θ, θ_t) over P_t

Uses DeepSpeed ZeRO-2 for distributed training with BF16 mixed precision.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def _detect_config(model_path: str) -> str:
    """Auto-detect the training config file based on model name."""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    model_lower = model_path.lower()

    if "qwen" in model_lower and "coder" in model_lower:
        return os.path.join(config_dir, "qwencoder_7b.yaml")
    return os.path.join(config_dir, "codellama_7b.yaml")


def train_tsp(
    model_path: str,
    data_file: str,
    output_dir: str,
    config_file: Optional[str] = None,
    llama_factory_dir: Optional[str] = None,
) -> str:
    """Execute one round of TSP training using LLaMA-Factory.

    Copies the training data and patched config into LLaMA-Factory's
    directory structure, then launches training via llamafactory-cli.

    Args:
        model_path: Path or name of the model to fine-tune.
        data_file: Path to TSP training data in instruction/chosen/rejected format.
        output_dir: Directory to save model checkpoints and logs.
        config_file: Path to YAML training config (auto-detected if None).
        llama_factory_dir: Path to LLaMA-Factory installation
            (default: $LLAMA_FACTORY_DIR or ./LLaMA-Factory).

    Returns:
        Path to the output directory.
    """
    if config_file is None:
        config_file = _detect_config(model_path)

    if llama_factory_dir is None:
        llama_factory_dir = os.environ.get(
            "LLAMA_FACTORY_DIR",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "LLaMA-Factory"),
        )

    if not os.path.isdir(llama_factory_dir):
        raise FileNotFoundError(
            f"LLaMA-Factory not found at {llama_factory_dir}. "
            "Install from https://github.com/hiyouga/LLaMA-Factory and set LLAMA_FACTORY_DIR."
        )

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Copy training data into LLaMA-Factory's data directory
    lf_data_dir = os.path.join(llama_factory_dir, "data")
    os.makedirs(lf_data_dir, exist_ok=True)

    data_filename = "tsp_data.json"
    lf_data_path = os.path.join(lf_data_dir, data_filename)
    shutil.copy2(data_file, lf_data_path)
    logger.info(f"Copied training data to {lf_data_path}")

    # Ensure dataset_info.json registers tsp_data
    info_path = os.path.join(lf_data_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f) if os.path.getsize(info_path) > 0 else {}
    else:
        info = {}

    if "tsp_data" not in info:
        info["tsp_data"] = {
            "file_name": data_filename,
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "chosen": "chosen", "rejected": "rejected"},
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    # Patch config with correct model path and output dir
    with open(config_file, "r", encoding="utf-8") as f:
        config_text = f.read()

    config_text = config_text.replace(
        "model_name_or_path: CodeLlama-7b-Instruct-hf",
        f"model_name_or_path: {model_path}",
    )
    config_text = config_text.replace(
        "model_name_or_path: Qwen/Qwen2.5-Coder-7B-Instruct",
        f"model_name_or_path: {model_path}",
    )

    patched_config = os.path.join(output_dir, "training_config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(patched_config, "w", encoding="utf-8") as f:
        # Update output_dir in config
        for line in config_text.split("\n"):
            if line.startswith("output_dir:"):
                f.write(f"output_dir: {output_dir}\n")
            else:
                f.write(line + "\n")

    logger.info(f"Patched config written to {patched_config}")

    # Launch training
    cmd = [
        sys.executable, "-m", "llamafactory.cli",
        "train", patched_config,
    ]
    logger.info(f"Launching training: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=llama_factory_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Training failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        raise RuntimeError(f"TSP training failed with return code {result.returncode}")

    logger.info(f"Training completed. Output at {output_dir}")
    return output_dir

#!/usr/bin/env python3
"""TSP (Tree-like Self-Play) Training Pipeline.

Complete implementation of Algorithm 1 from the paper:
"Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs"

Usage:
    python -m run_tsp --input data/annotated_datasets/sec-new-desc_annotated.json \\
                       --output ./output --model CodeLlama-7b-Instruct-hf --iterations 2

Pipeline steps (per iteration):
    Step 0: Parse CWE Risk Node annotations (offline, once)
    Step 1: Generate self-play branches at risk nodes (opponent player)
    Step 2: Build preference pairs (golden path vs. self-play path)
    Step 3: Train main player via preference optimization
"""

import argparse
import logging
import os
import sys

# Ensure tsp package is importable when running from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tsp import (
    annotate_dataset,
    build_preference_pairs,
    convert_to_training_format,
    generate_branches,
    parse_risk_nodes,
    train_tsp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_tsp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSP Training Pipeline (Algorithm 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_tsp \\
    --input data/annotated_datasets/sec-new-desc_annotated.json \\
    --output ./output \\
    --model CodeLlama-7b-Instruct-hf \\
    --iterations 2
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True,
        help="Input annotated dataset JSON (with GPT-4o 'output' fields)",
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="Output directory for all intermediate and final results",
    )
    parser.add_argument(
        "--model", "-m", default="CodeLlama-7b-Instruct-hf",
        help="Model name or path (default: CodeLlama-7b-Instruct-hf)",
    )
    parser.add_argument(
        "--iterations", "-T", type=int, default=2,
        help="Number of TSP iterations (default: 2)",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Training config YAML (auto-detected from model name if omitted)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for branch generation (paper: 1.0)",
    )
    parser.add_argument(
        "--samples-per-node", type=int, default=4,
        help="Number of self-play branches per risk node (paper: 4)",
    )
    parser.add_argument(
        "--skip-annotation", action="store_true",
        help="Skip Step 0 if input already has 'Nodes' fields",
    )
    parser.add_argument(
        "--gpu-ids", default="0,1,2,3",
        help="Comma-separated GPU IDs (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=4,
        help="Number of GPUs for tensor parallelism (default: 4)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    model_path = args.model
    annotated_file = args.input

    # =========================================================================
    # Step 0: Parse CWE Risk Node annotations (if not already done)
    # =========================================================================
    if not args.skip_annotation:
        needs_parsing = True
        try:
            import json
            with open(annotated_file, "r") as f:
                sample = json.load(f)
            if sample and "Nodes" in sample[0]:
                needs_parsing = False
                logger.info("Input already has 'Nodes' fields. Skipping annotation parsing.")
        except Exception:
            pass

        if needs_parsing:
            parsed_file = os.path.join(args.output, "parsed_annotations.json")
            logger.info("=" * 60)
            logger.info("Step 0: Parsing CWE Risk Node annotations")
            logger.info("=" * 60)
            parse_risk_nodes(annotated_file, parsed_file)
            annotated_file = parsed_file

    # =========================================================================
    # TSP Iterative Loop (Algorithm 1)
    # =========================================================================
    for t in range(1, args.iterations + 1):
        iter_dir = os.path.join(args.output, f"iteration_{t}")
        os.makedirs(iter_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"TSP Iteration {t}/{args.iterations}")
        logger.info(f"Opponent model: {model_path}")
        logger.info("=" * 60)

        # -----------------------------------------------------------------
        # Step 1: Opponent player generates self-play branches
        #         y_{i,v} ~ p_{θ_{t-1}}(·|x_i, y_{i,<k_v})
        # -----------------------------------------------------------------
        generated_file = os.path.join(iter_dir, "generated_branches.json")
        logger.info(f"Step 1: Generating self-play branches (temp={args.temperature}, "
                     f"samples/node={args.samples_per_node})")

        generate_branches(
            data_file=annotated_file,
            output_file=generated_file,
            model_path=model_path,
            temperature=args.temperature,
            samples_per_node=args.samples_per_node,
            tensor_parallel=args.tensor_parallel,
            gpu_ids=args.gpu_ids,
        )

        # -----------------------------------------------------------------
        # Step 2: Build preference pairs
        #         P_t ← P_t ∪ {(y_i, y_{i,v})}
        # -----------------------------------------------------------------
        pairs_file = os.path.join(iter_dir, "preference_pairs.json")
        training_file = os.path.join(iter_dir, "tsp_training_data.json")
        logger.info("Step 2: Building preference pairs")

        build_preference_pairs(generated_file, pairs_file)
        convert_to_training_format(pairs_file, training_file)

        # -----------------------------------------------------------------
        # Step 3: Train main player
        #         θ_t ← argmin L_TSP(θ, θ_t) over P_t
        # -----------------------------------------------------------------
        checkpoint_dir = os.path.join(iter_dir, "checkpoint")
        logger.info("Step 3: Training main player")

        train_tsp(
            model_path=model_path,
            data_file=training_file,
            output_dir=checkpoint_dir,
            config_file=args.config,
        )

        # Update model path for next iteration
        model_path = checkpoint_dir
        logger.info(f"Iteration {t} complete. Updated model -> {model_path}")

    logger.info("=" * 60)
    logger.info(f"TSP training complete after {args.iterations} iterations")
    logger.info(f"Final model: {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

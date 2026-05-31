#!/bin/bash

# TSP Evaluation Inference Script
# Usage: bash script.sh --rq RQ_NUM --model MODEL_NAME [--dataset FILE] [--output-dir DIR]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Defaults
RQ=""
MODEL=""
DATASET=""
OUTPUT_DIR=""
DATA_DIR="$REPO_ROOT/data"
RESULTS_DIR="$REPO_ROOT/results"

show_help() {
    echo "Usage: $0 --rq RQ_NUM --model MODEL_NAME [options]"
    echo
    echo "Required:"
    echo "  --rq RQ_NUM        RQ number (1, 2, or 3)"
    echo "  --model MODEL      Model name (e.g. codellama7b_tsp)"
    echo
    echo "Optional:"
    echo "  --dataset FILE     Input dataset JSON (auto-detected if omitted)"
    echo "  --output-dir DIR   Output directory (default: results/rq<N>/inference_outputs)"
    echo "  --gpu-ids IDS      GPU IDs (default: 0,1,2,3)"
    echo "  --tensor-parallel  Number of GPUs (default: 4)"
    echo "  -h, --help         Show this help"
}

GPU_IDS="0,1,2,3"
TENSOR_PARALLEL=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rq) RQ="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --tensor-parallel) TENSOR_PARALLEL="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

if [[ -z "$RQ" || -z "$MODEL" ]]; then
    echo "Error: --rq and --model are required"
    show_help
    exit 1
fi

# Auto-detect dataset if not specified
if [[ -z "$DATASET" ]]; then
    case "$RQ" in
        1) DATASET="$DATA_DIR/evaluation_datasets/rq1_cwe_evaluate.json" ;;
        2) DATASET="$DATA_DIR/evaluation_datasets/rq2_dataset.json" ;;
        3) DATASET="$DATA_DIR/evaluation_datasets/rq3_cwe_evaluate_ablation.json" ;;
        *) echo "Error: Invalid RQ number: $RQ"; exit 1 ;;
    esac
fi

# Auto-set output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$RESULTS_DIR/rq${RQ}/inference_outputs"
fi

if [[ ! -f "$DATASET" ]]; then
    echo "Error: Dataset not found: $DATASET"
    exit 1
fi

OUTPUT_FILE="$OUTPUT_DIR/${MODEL}_inference_output.json"
mkdir -p "$OUTPUT_DIR"

echo "=== TSP Evaluation Inference ==="
echo "RQ: $RQ"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_FILE"
echo "==============================="

python "$SCRIPT_DIR/inference_with_template.py" \
    --data_file "$DATASET" \
    --output_file "$OUTPUT_FILE" \
    --model "$MODEL" \
    --gpu_ids "$GPU_IDS" \
    --tensor_parallel_size "$TENSOR_PARALLEL"

if [ $? -eq 0 ]; then
    echo "Inference completed: $OUTPUT_FILE"
else
    echo "Error: Inference failed"
    exit 1
fi

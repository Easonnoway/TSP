#!/bin/bash
# TSP Inference Pipeline
# Paths can be overridden via environment variables

DATA_DIR="${TSP_DATA_DIR:-../../data}"
OUTPUT_DIR="${TSP_OUTPUT_DIR:-../../output_data}"
MODEL_DIR="${TSP_MODEL_DIR:-../../models}"
INPUT_FILE="${INPUT_FILE:-$DATA_DIR/sec-new-desc_annotated_with_nodes.json}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/sec-new-desc_annotated_with_nodes_generation.json}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/CodeLlama-7b-Instruct-hf}"

python ./inference_with_template.py \
    --data_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model "$MODEL_PATH"

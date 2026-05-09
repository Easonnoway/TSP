#!/bin/bash
# API Annotation Pipeline
# Input/output paths can be overridden via environment variables

DATA_DIR="${TSP_DATA_DIR:-../../data}"
OUTPUT_DIR="${TSP_OUTPUT_DIR:-../../output_data}"
INPUT_FILE="${INPUT_FILE:-$DATA_DIR/sec-new-desc.json}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/sec-new-desc_annotated.json}"

python ./api_annotation.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"

echo "$OUTPUT_FILE generated"

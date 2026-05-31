#!/bin/bash

# TSP CodeQL Analysis Script
# Usage: bash codeql_analysis_script.sh --codeql-path PATH [--models MODEL1,MODEL2] [--testcases-dir DIR]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Defaults
CODEQL_PATH="${CODEQL_PATH:-}"
MODELS=("codellama7b_tsp")
TESTCASES_DIR="$REPO_ROOT/data/testcases"

show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --codeql-path PATH    Path to CodeQL binary (required, or set CODEQL_PATH env)"
    echo "  --models MODELS       Comma-separated model names (default: codellama7b_tsp)"
    echo "  --testcases-dir DIR   Test cases directory (default: data/testcases)"
    echo "  -h, --help            Show this help"
    echo
    echo "Examples:"
    echo "  CODEQL_PATH=/usr/local/bin/codeql $0 --models codellama7b_tsp,codellama7b_sft"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --codeql-path) CODEQL_PATH="$2"; shift 2 ;;
        --models) IFS=',' read -ra MODELS <<< "$2"; shift 2 ;;
        --testcases-dir) TESTCASES_DIR="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

if [[ -z "$CODEQL_PATH" ]]; then
    echo "Error: CodeQL path not set. Use --codeql-path or set CODEQL_PATH environment variable."
    exit 1
fi

for MODEL_NAME in "${MODELS[@]}"; do
    echo "==============================================="
    echo "Processing model: ${MODEL_NAME}"
    echo "==============================================="

    TEST_DIR="$TESTCASES_DIR/rq1/rq1_1/Testcases_${MODEL_NAME}"
    if [[ ! -d "$TEST_DIR" ]]; then
        echo "Warning: Directory does not exist - $TEST_DIR"
        echo "Trying top-level search..."
        TEST_DIR=$(find "$TESTCASES_DIR" -type d -name "Testcases_${MODEL_NAME}" | head -1)
        if [[ -z "$TEST_DIR" ]]; then
            echo "Skipping ${MODEL_NAME}..."
            continue
        fi
    fi

    DB_DIR="$REPO_ROOT/results/rq1/codeql_results/databases"
    mkdir -p "$DB_DIR"

    echo "Creating CodeQL database..."
    "$CODEQL_PATH" database create --overwrite --language=python "$DB_DIR/Testcases_${MODEL_NAME}" --source-root "$TEST_DIR"

    if [[ $? -ne 0 ]]; then
        echo "Error: Database creation failed for ${MODEL_NAME}"
        continue
    fi

    echo "Running CodeQL analysis..."
    RESULT_DIR="$REPO_ROOT/results/rq1/codeql_results/testcases_${MODEL_NAME}"
    mkdir -p "$RESULT_DIR"
    "$CODEQL_PATH" database analyze "$DB_DIR/Testcases_${MODEL_NAME}" \
        codeql/python-queries:Security --format=csv --output="$RESULT_DIR"

    echo "Model ${MODEL_NAME} processing complete"
done

echo "All model analyses complete!"

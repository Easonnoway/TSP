import os
import json
import argparse
from tqdm import tqdm


def convert_to_database_for_eval(model, input_dir, output_dir):
    """Convert inference output JSON to individual code files for evaluation.

    Args:
        model: Model name (e.g. codellama7b_tsp)
        input_dir: Directory containing inference output JSON files
        output_dir: Directory to write extracted code files
    """
    input_path = os.path.join(input_dir, f"{model}_inference_output.json")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found - {input_path}")
        return

    with open(input_path, 'r') as f:
        json_data = json.load(f)

    base_path = os.path.join(output_dir, f"Testcases_{model}")
    os.makedirs(base_path, exist_ok=True)

    for item in tqdm(json_data):
        file_id = item.get("ID", item.get("id", "unknown"))
        generation_code = item.get("Generation", "")

        parts = file_id.split('_')
        file_name = parts[1] if len(parts) >= 2 else file_id

        file_path = os.path.join(base_path, f"{file_name}.c")
        with open(file_path, 'w') as f:
            f.write(generation_code)

    print(f"Created {len(json_data)} files in {base_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert inference output to database format")
    parser.add_argument("model", help="Model name (e.g. codellama7b_tsp)")
    parser.add_argument("--input-dir", default=None,
                        help="Directory containing inference output JSONs (default: results/rq<N>/inference_outputs)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for extracted code files (default: ./testcases_output)")
    args = parser.parse_args()

    input_dir = args.input_dir or "./inference_outputs"
    output_dir = args.output_dir or "./testcases_output"

    convert_to_database_for_eval(args.model, input_dir, output_dir)

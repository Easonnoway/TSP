#!/usr/bin/env python3
import os
import csv
import json
import argparse
from collections import defaultdict


def extract_cwe_info(result_dir, model_name):
    """
    Extract CWE vulnerability information from CSV files in specified directory.

    Args:
        result_dir: Directory path containing CSV files (organized as testcases_<model>/)
        model_name: Model name used to locate the subdirectory
    """
    csv_dir = os.path.join(result_dir, f"testcases_{model_name}")

    if not os.path.exists(csv_dir):
        print(f"Error: Directory does not exist - {csv_dir}")
        return {}

    vulnerabilities = defaultdict(list)
    cwe_type_counts = defaultdict(int)

    print(f"Analyzing directory: {csv_dir}")
    for file_name in sorted(os.listdir(csv_dir)):
        if not file_name.endswith('.csv'):
            continue

        cwe_type = file_name.split('_')[2].split('.')[0]
        file_path = os.path.join(csv_dir, file_name)

        try:
            if os.path.getsize(file_path) == 0:
                print(f"Warning: Empty file - {file_name}")
                continue

            print(f"Processing file: {file_name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue

                    vuln_type = row[0]
                    description = row[1]
                    severity = row[2]
                    detail = row[3]
                    code_file_path = row[4]
                    line_start = row[5] if len(row) > 5 else ""
                    col_start = row[6] if len(row) > 6 else ""
                    line_end = row[7] if len(row) > 7 else ""
                    col_end = row[8] if len(row) > 8 else ""

                    vuln_info = {
                        "cwe_type": cwe_type,
                        "vulnerability_type": vuln_type,
                        "description": description,
                        "severity": severity,
                        "detail": detail,
                        "line_start": line_start,
                        "col_start": col_start,
                        "line_end": line_end,
                        "col_end": col_end
                    }

                    vulnerabilities[code_file_path].append(vuln_info)
                    cwe_type_counts[cwe_type] += 1

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    print("\nVulnerability type statistics:")
    for cwe_type, count in sorted(cwe_type_counts.items()):
        print(f"CWE-{cwe_type}: {count} vulnerabilities")

    return vulnerabilities


def save_results(vulnerabilities, output_file):
    """Save results to JSON file and generate summary CSV."""
    result = {k: v for k, v in vulnerabilities.items()}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Found {len(vulnerabilities)} code files with vulnerabilities")

    total_vulns = sum(len(vulns) for vulns in vulnerabilities.values())
    print(f"Total vulnerabilities: {total_vulns}")

    summary_file = output_file.replace('.json', '_summary.csv')
    with open(summary_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Code File', 'CWE Types', 'Vulnerability Count'])
        for code_file, vulns in vulnerabilities.items():
            cwe_types = set(vuln['cwe_type'] for vuln in vulns)
            writer.writerow([code_file, ', '.join(sorted(cwe_types)), len(vulns)])

    print(f"Summary report saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CWE vulnerabilities from CodeQL CSV results")
    parser.add_argument("--result-dir", required=True,
                        help="Directory containing CodeQL result subdirectories (testcases_<model>/)")
    parser.add_argument("--model", required=True,
                        help="Model name to analyze")
    parser.add_argument("--output",
                        help="Output JSON file path (default: <result_dir>/analyze_res/<model>_vulnerabilities.json)")
    args = parser.parse_args()

    if args.output:
        output_file = args.output
    else:
        output_dir = os.path.join(args.result_dir, "analyze_res")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{args.model}_vulnerabilities.json")

    vulnerabilities = extract_cwe_info(args.result_dir, args.model)
    save_results(vulnerabilities, output_file)

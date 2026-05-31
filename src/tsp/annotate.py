"""Step 0: CWE Risk Node Annotation.

Uses GPT-4o to identify CWE Risk Nodes in secure code, then parses
the raw LLM output into structured node annotations.

Corresponds to Section 2.1 (CWE Risk Node Identification) and Appendix C
of the paper.
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .prompts import system_prompt, user_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPT-4o API calling (Ray-based parallel execution)
# ---------------------------------------------------------------------------

def _make_api_request(
    system_content: str,
    user_content: str,
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    temperature: float = 0.05,
    max_retries: int = 5,
) -> Optional[str]:
    """Make a single OpenAI API request with retry logic."""
    import urllib3
    from urllib3.util.retry import Retry

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    http = urllib3.PoolManager(retries=retry_strategy)

    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(1000):
        try:
            resp = http.request("POST", url, body=payload, headers=headers)
            result = json.loads(resp.data.decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt % 10 == 0:
                logger.warning(f"API request failed (attempt {attempt}): {e}")
            time.sleep(3)

    return None


def _generate_hash_uid(to_hash: Any) -> str:
    """Generate SHA256 hash UID for caching."""
    content = json.dumps(to_hash, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def annotate_dataset(
    input_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    cache_dir: Optional[str] = None,
    num_workers: int = 50,
) -> str:
    """Annotate a dataset with CWE Risk Nodes using GPT-4o.

    Reads a JSON array of {description, func_src_after} items, calls GPT-4o
    with the annotation prompt for each item, and writes results with an
    added 'output' field containing the raw LLM annotations.

    Args:
        input_file: Path to input JSON dataset.
        output_file: Path to write annotated output JSON.
        api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
        api_base: OpenAI API base URL.
        model: LLM model name for annotation.
        cache_dir: Directory for caching API responses.
        num_workers: Number of parallel Ray workers.

    Returns:
        Path to the output file.
    """
    try:
        import ray
    except ImportError:
        raise ImportError("ray is required for parallel annotation. Install with: pip install ray")

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required: pass api_key or set OPENAI_API_KEY env var")

    if cache_dir is None:
        cache_dir = os.environ.get("TSP_CACHE_DIR", "./cache")
    os.makedirs(cache_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} items from {input_file}")

    ray.init(num_cpus=num_workers, ignore_reinit_error=True)

    @ray.remote
    def remote_annotate(item):
        uid = _generate_hash_uid(item)
        cache_path = os.path.join(cache_dir, f"{uid}.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

        sys_content = system_prompt.strip()
        usr_content = user_prompt.format(
            description=item["description"],
            code=item["func_src_after"],
        )

        result_text = _make_api_request(sys_content, usr_content, api_key, api_base, model)

        item_with_output = {**item, "output": result_text or ""}
        with open(cache_path, "w") as f:
            json.dump(item_with_output, f, ensure_ascii=False)

        return item_with_output

    refs = [remote_annotate.remote(item) for item in dataset]
    results = ray.get(refs)
    ray.shutdown()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Annotated {len(results)} items -> {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# Parse raw annotations into structured Risk Nodes
# ---------------------------------------------------------------------------

def _parse_item_nodes(item: Dict) -> Dict:
    """Parse the 'output' field of an annotated item into a 'Nodes' list."""
    if "output" not in item:
        return item

    output_text = item["output"]
    nodes = []

    matches = re.findall(
        r"\[\[(\d+)\]\]\n(.*?)(?=\[\[\d+\]\]|\Z)", output_text, re.DOTALL
    )

    for match in matches:
        node_num = int(match[0])
        content = match[1].strip()

        code_line = None
        cwe_id = None
        description_lines = []

        for line in content.split("\n"):
            if line.startswith("[Code_Line]"):
                code_line = line.replace("[Code_Line]", "").strip()
            elif line.startswith("[CWE_ID]"):
                cwe_id = line.replace("[CWE_ID]", "").strip()
            elif line.startswith("[Description]"):
                # First line of description
                description_lines.append(line.replace("[Description]", "").strip())
            elif description_lines:
                # Continuation lines for multi-line descriptions
                if not line.startswith("["):
                    description_lines.append(line.strip())
                else:
                    break

        description = " ".join(description_lines).strip()

        # Remove trailing markdown code fence artifacts
        if description.endswith("```"):
            description = description[:-3].strip()

        if code_line is not None and cwe_id is not None and description:
            nodes.append({
                "Node_Number": node_num,
                "Code_Line": code_line,
                "CWE_ID": cwe_id,
                "Description": description,
            })
        else:
            logger.warning(f"Incomplete data for node [[{node_num}]]. Skipping.")

    if nodes:
        return {**item, "Nodes": nodes}
    else:
        logger.warning("No valid nodes found in output. 'Nodes' key not added.")
        return item


def parse_risk_nodes(input_file: str, output_file: str) -> str:
    """Parse GPT-4o raw annotations into structured CWE Risk Nodes.

    Reads a JSON file with raw 'output' fields and writes a new file where
    each item has a structured 'Nodes' list extracted from the annotation text.

    Args:
        input_file: Path to annotated JSON (with 'output' fields).
        output_file: Path to write parsed JSON (with 'Nodes' fields).

    Returns:
        Path to the output file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed_data = [_parse_item_nodes(item) for item in data]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=4, ensure_ascii=False)

    total_nodes = sum(len(item.get("Nodes", [])) for item in parsed_data)
    logger.info(f"Parsed {total_nodes} risk nodes from {len(parsed_data)} items -> {output_file}")
    return output_file

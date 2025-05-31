import json
from pathlib import Path

def find_crash_files(directory):
    crash_files = []
    directory = Path(directory)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.name.endswith('crash'):
            crash_files.append(str(file_path))
    return crash_files

def find_dicts(crash_file):
    with open(crash_file, 'r', encoding='utf-8') as f:
        content = f.read()
    json_blocks = content.strip().split('}\n{')
    json_blocks = [block if block.startswith('{') else '{' + block for block in json_blocks]
    json_blocks = [block if block.endswith('}') else block + '}' for block in json_blocks]
    return json_blocks

def merge_dicts(json_blocks):
    merged_result = {}
    for block in json_blocks:
        try:
            data = json.loads(block)
            merged_result.update(data)
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping invalid JSON block. Error: {e}")
    return merged_result

if __name__ == "__main__":
    ratte_dirs = [
    "//workspace/mlir-inconsistent/cov_collection/ratte_seed_v1/fullreset_2025-05-02_16-19-37",
    "//workspace/mlir-inconsistent/cov_collection/ratte_seed_v2/fullreset_2025-05-02_16-20-19",
    "//workspace/mlir-inconsistent/cov_collection/ratte_seed_v3/fullreset_2025-05-02_18-43-59",
    "//workspace/mlir-inconsistent/cov_collection/ratte_seed_v4/fullreset_2025-05-02_18-44-11",
    "//workspace/mlir-inconsistent/cov_collection/ratte_seed_v5/fullreset_2025-05-02_23-48-36"
    ]

    tosasmith_dirs = [
    "//workspace/mlir-inconsistent/cov_collection/tosa_seed_v1/fullreset_2025-05-01_17-48-17",
    "//workspace/mlir-inconsistent/cov_collection/tosa_seed_v2/fullreset_2025-05-01_17-49-30",
    "//workspace/mlir-inconsistent/cov_collection/tosa_seed_v3/fullreset_2025-05-01_20-06-52",
    "//workspace/mlir-inconsistent/cov_collection/tosa_seed_v4/fullreset_2025-05-01_20-06-50",
    "//workspace/mlir-inconsistent/cov_collection/tosa_seed_v5/fullreset_2025-05-01_21-37-40"
    ]
    output_path = '//workspace/mlir-inconsistent/cov_collection/mergebug/tosa.json'
    crash_files = []
    for d in tosasmith_dirs:
        crash_files.extend(find_crash_files(d))
    all_blocks = []
    for crash_file in crash_files:
        all_blocks.extend(find_dicts(crash_file))
    merged_dict = merge_dicts(all_blocks)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_dict, f, ensure_ascii=False, indent=4)

    print(f"[+] Merged result written to: {output_path}")

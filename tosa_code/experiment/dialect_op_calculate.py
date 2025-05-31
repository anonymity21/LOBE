import os
import json
def parse_dialects_and_ops_from_dir(directory):
    dialects_set = set()
    ops_set = set()

    for filename in os.listdir(directory):
        if filename.endswith(".cov"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("dialects:"):
                        dialects = line[len("dialects:"):].strip().split()
                        dialects_set.update(dialects)
                    elif line.startswith("ops:"):
                        ops = line[len("ops:"):].strip().split()
                        ops_set.update(ops)

    return dialects_set, ops_set


output_dir = "//workspace/mlir-inconsistent/cov_collection/mregedialectop"
os.makedirs(output_dir, exist_ok=True)

def write_to_file(file_name, source_dirs):
    all_dialects = set()
    all_ops = set()
    for single_dir in source_dirs:
        dialects, ops = parse_dialects_and_ops_from_dir(single_dir)
        all_dialects.update(dialects)
        all_ops.update(ops)

    dialects_file = os.path.join(output_dir, f"{file_name}_dialects.json")
    ops_file = os.path.join(output_dir, f"{file_name}_ops.json")

    with open(dialects_file, "w", encoding="utf-8") as f:
        json.dump(sorted(list(all_dialects)), f, indent=4, ensure_ascii=False)

    with open(ops_file, "w", encoding="utf-8") as f:
        json.dump(sorted(list(all_ops)), f, indent=4, ensure_ascii=False)

    print(f"Saved: {dialects_file}, {ops_file}")



ratte_dirs = [
"//workspace/mlir-inconsistent/cov_collection/ratte_seed_v1/fullreset_2025-05-11_22-42-00",
"//workspace/mlir-inconsistent/cov_collection/ratte_seed_v2/fullreset_2025-05-12_00-36-59",
"//workspace/mlir-inconsistent/cov_collection/ratte_seed_v3/fullreset_2025-05-12_02-30-42",
"//workspace/mlir-inconsistent/cov_collection/ratte_seed_v4/fullreset_2025-05-12_03-51-31",
"//workspace/mlir-inconsistent/cov_collection/ratte_seed_v5/fullreset_2025-05-12_05-25-13"
]

ratte_semantics_dir = [
"//workspace/mlir-inconsistent/cov_collection/ratte_seed_v1_semantics/sem_2025-05-12_15-30-33",
"//workspace/mlir-inconsistent/cov_collection/ratte_seed_v2_semantics/sem_2025-05-12_15-31-26",
"//workspace/mlir-inconsistent/cov_collection/ratte_seed_v3_semantics/sem_2025-05-12_16-36-00",
"//workspace/mlir-inconsistent/cov_collection/ratte_seed_v4_semantics/sem_2025-05-12_16-36-15",
"//workspace/mlir-inconsistent/cov_collection/ratte_seed_v5_semantics/sem_2025-05-12_17-41-26"
]

tosasmith_dirs = [
"//workspace/mlir-inconsistent/cov_collection/tosa_seed_v1/fullreset_2025-05-11_19-32-43",
"//workspace/mlir-inconsistent/cov_collection/tosa_seed_v2/fullreset_2025-05-11_20-41-18",
"//workspace/mlir-inconsistent/cov_collection/tosa_seed_v3/fullreset_2025-05-11_21-50-43",
"//workspace/mlir-inconsistent/cov_collection/tosa_seed_v4/fullreset_2025-05-11_22-57-18",
"//workspace/mlir-inconsistent/cov_collection/tosa_seed_v5/fullreset_2025-05-12_00-16-27"
]



write_to_file("ratte", ratte_dirs)
write_to_file("ratte_semantics", ratte_semantics_dir)   
write_to_file("tosasmith", tosasmith_dirs)
def extract_mlir_paths_from_log(log_path, output_path):
    mlir_paths = set()

    with open(log_path, 'r') as f:
        for line in f:
            if line.startswith("---- Process") and line.strip().endswith("----"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    mlir_path = parts[2]
                    if mlir_path.endswith(".mlir"):
                        mlir_paths.add(mlir_path)
                else:
                    print(f"❗ Unexpected line format: {line.strip()}")

    with open(output_path, 'a+') as out:
        for path in mlir_paths:
            out.write(path + '\n')

    print(f"✅ Extracted {len(mlir_paths)} unique .mlir file paths to {output_path}")

if __name__ == "__main__":
    from pathlib import Path
    log_dir = Path("//workspace/mlir-inconsistent/multiple_log/ratte_seed_v1/2025-04-02_19-46-59")  # 替换为你的目录路径
    log_files = list(log_dir.glob("*.log_filtered.log"))

    for log_file in log_files:
        extract_mlir_paths_from_log(log_file, "extracted_mlir_files.txt")


    log_files = list(log_dir.glob("*.log_otherresult.log_notsame.log"))
    for log_file in log_files:
        extract_mlir_paths_from_log(log_file, "extracted_mlir_files.txt")


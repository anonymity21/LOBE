import os
import subprocess
from pathlib import Path

input_dir = "//workspace/mlir-inconsistent/tosa_seed_v1"
log_file = "mlir_errors.log"
unique_error_file = "unique_errors.txt"

def collect_mlir_files(directory):
    return list(Path(directory).rglob("*.mlir"))

def run_mlir_opt(file_path):
    try:
        result = subprocess.run(
            ["//workspace/mlir-inconsistent/third_party_tools/mlir-opt-449e2f5d66", file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stderr.strip()
    except Exception as e:
        return f"[Runner Error] {file_path}: {str(e)}"

def main():
    # all_files = ['//workspace/mlir-inconsistent/tosa_seed4constraints/tosa.0aa66dd8e3fdb570.mlir']
    all_files = collect_mlir_files(input_dir)
    print(f"Found {len(all_files)} .mlir files.")

    seen_errors = set()
    seen_errors_withfile = set()
    with open(log_file, "w") as log:
        for file in all_files:
            err = run_mlir_opt(str(file))
            if err:
                log.write(f"--- Error in file: {file} ---\n")
                log.write(err + "\n\n")
                for line in err.splitlines():
                    if "error:" in line:
                        clean_error = line.split("error:", 1)[1].strip()
                        seen_errors.add(clean_error)
                        seen_errors_withfile.add(f"{os.path.basename(file)}: {clean_error}")
    with open(unique_error_file, "w") as out:
        for e in seen_errors_withfile:
            # out.write("==== Unique Error ====\n")
            out.write(e + "\n\n")

    print(f"Finished. All errors saved to {log_file}")
    print(f"Unique errors saved to {unique_error_file}")

if __name__ == "__main__":
    main()

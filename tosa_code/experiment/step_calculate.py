#!/usr/bin/env python3
from pathlib import Path
import os

program_dict = {}
def parse_cmd(cmd_line):
    pre_timeout = cmd_line.split('| timeout')[0].strip()
    opt_invocations = pre_timeout.split('| //workspace/mlir-inconsistent/third_party_tools/mlir-opt-449e2f5d66')
    # print(opt_invocations)
    return len(opt_invocations)-1


def parse_log(log_path):
    """
    Parse the log file to extract, for each successfully lowered MLIR file,
    the file path and its corresponding Total cmd line.
    """
    
    file_dict = {}
    results = []
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Detect start of a new process block
        if line.startswith("---- Process") and line.endswith("----"):
            # Extract the .mlir file path from between the markers
             parts = line.split("/")
             if parts:
                file_name = parts[-1].replace("----", "").strip()
        # Detect a successful lowering result
        elif line.startswith("LoweringResult: LoweringResult.NORMAL"):
            # Look ahead to the next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("Total cmd:"):
                    cmd = next_line[len("Total cmd:"):].strip()
                    results.append(parse_cmd(cmd))
            # skip next line so we don't re-process it
            i += 1
        i += 1
    return file_name.strip(), sum(results), len(results)

def main():
    dir_path = "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v1/conversion_2025-05-16_23-31-00_deduplicate_priority"  # adjust path as needed
    for root, dirs, files in os.walk(dir_path):
        total_sum = 0
        total_len = 0
        for file in files:
            full_path = os.path.join(root, file)
            process_file, sum, number = parse_log(full_path)
            total_sum += sum
            total_len += number
    print(f"tosa: average steps: {total_sum/total_len}")

    dir_path = "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v1/conversion_2025-05-16_23-33-43_deduplicate_priority"
    for root, dirs, files in os.walk(dir_path):
        arith_total_sum = 0
        arith_total_len = 0
        linalg_total_sum = 0
        linalg_total_len = 0
        tensor_total_sum = 0
        tensor_total_len = 0
        for file in files:
            full_path = os.path.join(root, file)
            process_file, sum, number = parse_log(full_path)
            # print(process_file + ' ' + str(sum) + ' ' + str(number))
            if 'arithsem' in process_file:
                arith_total_sum += sum
                arith_total_len += number
            elif 'tensor' in process_file:
                tensor_total_sum += sum
                tensor_total_len += number
            elif 'linalggeneric'in process_file:
                linalg_total_sum += sum
                linalg_total_len += number
    print(f"arith: average steps: {arith_total_sum/arith_total_len}")
    print(f"tensor: average steps: {tensor_total_sum/tensor_total_len}")
    print(f"linalg: average steps: {linalg_total_sum/linalg_total_len}")
            
if __name__ == "__main__":
    main()

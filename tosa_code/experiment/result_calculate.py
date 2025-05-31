import re
from pathlib import Path
from collections import defaultdict

def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by "---- Process"
    sections = content.split("---- Process")
    
    results = {}

    for section in sections[1:]:  # skip the first part before first "---- Process"
        lines = section.strip().splitlines()
        if not lines:
            continue

        # First line should contain the program name
        header = lines[0].strip()
        program_name = header if header else "Unknown"

        # Count result types
        normal_count = 0
        error_count = 0
        execute_error_count = 0
        noresult_error_count = 0
        timeout_error_count = 0

        max_opt_count = 0

        for line in lines:
            if "LoweringResult: LoweringResult.NORMAL" in line:
                normal_count += 1
            elif "LoweringResult: LoweringResult.CONVERT_ERROR" in line:
                error_count += 1
            elif 'LoweringResult: LoweringResult.EXECUTE_ERROR' in line:
                execute_error_count += 1
            elif "LoweringResult: LoweringResult.TIMEOUT" in line:
                noresult_error_count += 1
            elif "LoweringResult: LoweringResult.NORESULT" in line:
                noresult_error_count += 1
            elif "[FullResetLowering] Already reach the max applied opts." in line:
                max_opt_count += 1


        results[program_name] = {
            "NORMAL": normal_count,
            "CONVERT_ERROR": error_count,
            "EXECUTE_ERROR": execute_error_count,
            "NORESULT": noresult_error_count,
            "TIMEOUT": timeout_error_count,
            "max_opt_count": max_opt_count
        }

    return results


def obtain_results(file_path):
    dirpath = Path(file_path)
    total_files = 0
    total_normal = 0
    total_error = 0
    total_execute = 0
    total_noresult = 0
    total_timeout = 0
    total_max_opt_count = 0

    for file_path in dirpath.iterdir():
        if file_path.is_file():
            stats = parse_log_file(file_path)
            total_files += len(stats)
            for prog, counts in stats.items():
                total_normal += counts.get("NORMAL", 0)
                total_error += counts.get("CONVERT_ERROR", 0)
                total_execute += counts.get("EXECUTE_ERROR", 0)
                total_noresult += counts.get("NORESULT", 0)
                total_timeout += counts.get("TIMEOUT", 0)
                total_max_opt_count += counts.get("max_opt_count", 0)

    # 输出最终统计
    print("==== Summary ====")
    print(f"Total files: {total_files}")
    print(f"Total NORMAL results: {total_normal}")
    print(f"Total CONVERT_ERROR results: {total_error}")
    print(f"Total Lowering Success Rate: {100 * total_normal / (total_error + total_normal):.2f}%")
    # print(f"Total EXECUTE_ERROR results: {total_execute}")
    # print(f"Total NORESULT results: {total_noresult}")  
    # print(f"Total TIMEOUT results: {total_timeout}")
    # print(f"Total max_opt_count: {total_max_opt_count}")

    # if total_files > 0:
    #     print(f"Average NORMAL per file: {total_normal / total_files:.2f}")
    #     print(f"Average CONVERT_ERROR per file: {total_error / total_files:.2f}")
    #     print(f"Average EXECUTE_ERROR per file: {total_execute / total_files:.2f}")
    #     print(f"Average NORESULT per file: {total_noresult / total_files:.2f}")
    #     print(f"Average TIMEOUT per file: {total_timeout / total_files:.2f}")
    #     print(f"Average max_opt_count per file: {total_max_opt_count / total_files:.2f}")
    return total_normal, total_error

def compute_total(log_dirs):
    total_normal = 0
    total_error = 0
    for file in log_dirs:
        normal,error = obtain_results(file)
        total_normal += normal
        total_error += error
    rate = f"{100 * total_normal / (total_error + total_normal):.2f}%"
    return total_normal/5,total_error/5,rate
        

if __name__ == "__main__":
    ratte_dirs_w = [
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v1/fullreset_2025-05-11_22-42-00_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v2/fullreset_2025-05-12_00-36-59_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v3/fullreset_2025-05-12_02-30-42_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v4/fullreset_2025-05-12_03-51-31_deduplicate_priority",
         "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v5/fullreset_2025-05-12_05-25-13_deduplicate_priority"
    ]
    ratte_dirs_wo = [
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v1/fullreset_2025-05-16_19-48-09_deduplicate",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v2/fullreset_2025-05-16_15-31-50_deduplicate",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v3/fullreset_2025-05-16_16-35-25_deduplicate",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v3/fullreset_2025-05-16_16-35-25_deduplicate",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v5/fullreset_2025-05-16_18-36-00_deduplicate"
    ]

    tosa_dirs_w = [
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v1/fullreset_2025-05-11_19-32-43_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v2/fullreset_2025-05-11_20-41-18_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v3/fullreset_2025-05-11_21-50-43_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v4/fullreset_2025-05-11_22-57-18_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v5/fullreset_2025-05-12_00-16-27_deduplicate_priority"

    ]
    tosa_dirs_wo = [
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v1/fullreset_2025-05-11_20-06-26_deduplicate",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v2/fullreset_2025-05-11_21-14-11_deduplicate",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v3/fullreset_2025-05-11_22-23-42_deduplicate",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v4/fullreset_2025-05-11_23-34-52_deduplicate",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v5/fullreset_2025-05-12_01-01-41_deduplicate"
    ]

    ratte_w_normal, ratte_w_error, ratte_w_rate =  compute_total(ratte_dirs_w)
    ratte_wo_normal, ratte_wo_error, ratte_wo_rate =  compute_total(ratte_dirs_wo)

    tosa_w_normal, tosa_w_error, tosa_w_rate =  compute_total(tosa_dirs_w)
    tosa_wo_normal, tosa_wo_error, tosa_wo_rate =  compute_total(tosa_dirs_wo)

    total_w_normal = ratte_w_normal + tosa_w_normal
    total_w_error = ratte_w_error + tosa_w_error
    total_w_rate = f"{100 * total_w_normal / (total_w_error + total_w_normal):.2f}%"


    total_wo_normal = ratte_wo_normal + tosa_wo_normal
    total_wo_error = ratte_wo_error + tosa_wo_error
    total_wo_rate = f"{100 * total_wo_normal / (total_wo_error + total_wo_normal):.2f}%"

    print("==== Lowering Success Rate Summary ====")
    print("With Priority Update:")
    print(f"  TOSASmith: Success = {tosa_w_normal}, Fail = {tosa_w_error}, Rate = {tosa_w_rate}")
    print(f"  Ratte    : Success = {ratte_w_normal}, Fail = {ratte_w_error}, Rate = {ratte_w_rate}")
    print(f"  Total    : Success = {total_w_normal}, Fail = {total_w_error}, Rate = {total_w_rate}")

    print("\nWithout Priority Update:")
    print(f"  TOSASmith: Success = {tosa_wo_normal}, Fail = {tosa_wo_error}, Rate = {tosa_wo_rate}")
    print(f"  Ratte    : Success = {ratte_wo_normal}, Fail = {ratte_wo_error}, Rate = {ratte_wo_rate}")
    print(f"  Total    : Success = {total_wo_normal}, Fail = {total_wo_error}, Rate = {total_wo_rate}")





# ==== Summary ====
# Total files: 24
# Total NORMAL results: 8581
# Total CONVERT_ERROR results: 256
# Total Lowering Success Rate: 97.10%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 8475
# Total CONVERT_ERROR results: 202
# Total Lowering Success Rate: 97.67%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 10083
# Total CONVERT_ERROR results: 457
# Total Lowering Success Rate: 95.66%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 13805
# Total CONVERT_ERROR results: 219
# Total Lowering Success Rate: 98.44%
# ==== Summary ====
# Total files: 23
# Total NORMAL results: 9082
# Total CONVERT_ERROR results: 262
# Total Lowering Success Rate: 97.20%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 6798
# Total CONVERT_ERROR results: 237
# Total Lowering Success Rate: 96.63%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 5093
# Total CONVERT_ERROR results: 194
# Total Lowering Success Rate: 96.33%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 7112
# Total CONVERT_ERROR results: 274
# Total Lowering Success Rate: 96.29%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 7112
# Total CONVERT_ERROR results: 274
# Total Lowering Success Rate: 96.29%
# ==== Summary ====
# Total files: 23
# Total NORMAL results: 5775
# Total CONVERT_ERROR results: 210
# Total Lowering Success Rate: 96.49%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 12714
# Total CONVERT_ERROR results: 329
# Total Lowering Success Rate: 97.48%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 13871
# Total CONVERT_ERROR results: 414
# Total Lowering Success Rate: 97.10%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 11680
# Total CONVERT_ERROR results: 2700
# Total Lowering Success Rate: 81.22%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 11073
# Total CONVERT_ERROR results: 1628
# Total Lowering Success Rate: 87.18%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 11503
# Total CONVERT_ERROR results: 2580
# Total Lowering Success Rate: 81.68%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 6560
# Total CONVERT_ERROR results: 6980
# Total Lowering Success Rate: 48.45%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 9145
# Total CONVERT_ERROR results: 5178
# Total Lowering Success Rate: 63.85%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 5908
# Total CONVERT_ERROR results: 6742
# Total Lowering Success Rate: 46.70%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 5136
# Total CONVERT_ERROR results: 6167
# Total Lowering Success Rate: 45.44%
# ==== Summary ====
# Total files: 24
# Total NORMAL results: 5318
# Total CONVERT_ERROR results: 5375
# Total Lowering Success Rate: 49.73%
# ==== Lowering Success Rate Summary ====
# With Priority Update:
#   TOSASmith: Success = 12168.2, Fail = 1530.2, Rate = 88.83%
#   Ratte    : Success = 10005.2, Fail = 279.2, Rate = 97.29%
#   Total    : Success = 22173.4, Fail = 1809.4, Rate = 92.46%

# Without Priority Update:
#   TOSASmith: Success = 6413.4, Fail = 6088.4, Rate = 51.30%
#   Ratte    : Success = 6378.0, Fail = 237.8, Rate = 96.41%
#   Total    : Success = 12791.4, Fail = 6326.2, Rate = 66.91%
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
            # elif 'LoweringResult: LoweringResult.EXECUTE_ERROR' in line:
            #     execute_error_count += 1
            # elif "LoweringResult: LoweringResult.TIMEOUT" in line:
            #     noresult_error_count += 1
            # elif "LoweringResult: LoweringResult.NORESULT" in line:
            #     noresult_error_count += 1
            elif "[FullResetLowering] Already reach the max applied opts." in line:
                max_opt_count += 1

        if error_count > 100:
            print(f"Warning: {program_name} has too many errors: {error_count}.")
            
        results[program_name] = {
            "NORMAL": normal_count,
            "CONVERT_ERROR": error_count,
            # "EXECUTE_ERROR": execute_error_count,
            # "NORESULT": noresult_error_count,
            # "TIMEOUT": timeout_error_count,
            "max_opt_count": max_opt_count
        }

    return results


def obtain_results(file_path):
    dirpath = Path(file_path)
    total_files = 0
    total_normal = 0
    total_error = 0
    # total_execute = 0
    # total_noresult = 0
    # total_timeout = 0
    total_max_opt_count = 0

    for file_path in dirpath.iterdir():
        if file_path.is_file():
            stats = parse_log_file(file_path)
            total_files += len(stats)
            for prog, counts in stats.items():
                total_normal += counts.get("NORMAL", 0)
                total_error += counts.get("CONVERT_ERROR", 0)
                # total_execute += counts.get("EXECUTE_ERROR", 0)
                # total_noresult += counts.get("NORESULT", 0)
                # total_timeout += counts.get("TIMEOUT", 0)
                total_max_opt_count += counts.get("max_opt_count", 0)

    # 输出最终统计
    # print("==== Summary ====")
    # print(f"Total files: {total_files}")
    # print(f"Total NORMAL results: {total_normal}")
    # print(f"Total CONVERT_ERROR results: {total_error}")
    # print(f"Total Lowering Success Rate: {100 * total_normal / (total_error + total_normal):.2f}%")
    return total_normal, total_error

def compute_total(log_dirs):
    total_normal = 0
    total_error = 0
    for file in log_dirs:
        normal,error = obtain_results(file)
        total_normal += normal
        total_error += error
    rate = f"{100 * total_normal / (total_error + total_normal):.2f}%"
    print(f"Total NORMAL results: {total_normal/5}")
    print(f"Total CONVERT_ERROR results: {total_error/5}")
    print(f"Lowering Success Rate: {rate}")
    return total_normal/5,total_error/5,rate
        

if __name__ == "__main__":
    # ratte_dirs_w = [
    #     "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v1/fullreset_2025-05-11_22-42-00_deduplicate_priority",
    #     "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v2/fullreset_2025-05-12_00-36-59_deduplicate_priority",
    #     "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v3/fullreset_2025-05-12_02-30-42_deduplicate_priority",
    #     "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v4/fullreset_2025-05-12_03-51-31_deduplicate_priority",
    #      "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v5/fullreset_2025-05-12_05-25-13_deduplicate_priority"
    # ]
    # ratte_dirs_wo = [
    #     "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v1/fullreset_2025-05-16_19-48-09_deduplicate",
    #     "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v2/fullreset_2025-05-16_15-31-50_deduplicate",
    #     "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v3/fullreset_2025-05-16_16-35-25_deduplicate",
    #     "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v3/fullreset_2025-05-16_16-35-25_deduplicate",
    #     "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v5/fullreset_2025-05-16_18-36-00_deduplicate"
    # ]
    ratte_dirs_20  = [
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v1/fullreset_2025-05-27_21-33-46_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v2/fullreset_2025-05-28_00-06-06_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v3/fullreset_2025-05-28_02-37-13_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v4/fullreset_2025-05-28_05-07-53_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v5/fullreset_2025-05-28_07-51-51_deduplicate_priority"
    ]
    ratte_dirs_30 = [
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v1/fullreset_2025-05-27_22-09-13_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v2/fullreset_2025-05-28_00-42-32_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v3/fullreset_2025-05-28_03-13-13_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v4/fullreset_2025-05-28_05-51-07_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v5/fullreset_2025-05-28_08-27-31_deduplicate_priority"
    ]
    ratte_dirs_40 = [
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v1/fullreset_2025-05-27_22-47-20_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v2/fullreset_2025-05-28_01-21-30_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v3/fullreset_2025-05-28_03-50-02_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v4/fullreset_2025-05-28_06-28-18_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v5/fullreset_2025-05-28_09-02-12_deduplicate_priority"
    ]
    ratte_dirs_50 = [
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v1/fullreset_2025-05-27_23-25-30_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v2/fullreset_2025-05-28_02-00-15_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v3/fullreset_2025-05-28_04-26-29_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v4/fullreset_2025-05-28_07-10-08_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/ratte_seed_v5/fullreset_2025-05-28_09-40-28_deduplicate_priority"
    ]
    tosa_dirs_20 = [
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v1/fullreset_2025-05-27_21-29-18_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v2/fullreset_2025-05-27_23-13-05_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v3/fullreset_2025-05-28_01-27-03_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v4/fullreset_2025-05-28_03-41-54_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v5/fullreset_2025-05-28_05-54-54_deduplicate_priority"
    ]
    tosa_dirs_30 = [
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v1/fullreset_2025-05-27_20-56-54_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v2/fullreset_2025-05-27_23-47-01_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v3/fullreset_2025-05-28_01-58-40_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v4/fullreset_2025-05-28_04-16-00_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v5/fullreset_2025-05-28_06-26-48_deduplicate_priority"



    ]
    tosa_dirs_40 = [
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v1/fullreset_2025-05-27_22-05-07_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v2/fullreset_2025-05-28_00-20-53_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v3/fullreset_2025-05-28_02-34-13_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v4/fullreset_2025-05-28_04-49-21_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v5/fullreset_2025-05-28_07-00-27_deduplicate_priority"


    ]
    tosa_dirs_50 = [
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v1/fullreset_2025-05-27_22-38-49_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v2/fullreset_2025-05-28_00-54-29_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v3/fullreset_2025-05-28_03-08-23_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v4/fullreset_2025-05-28_05-23-14_deduplicate_priority",
        "//workspace/mlir-inconsistent/experiment_log/tosa_seed_v5/fullreset_2025-05-28_07-33-25_deduplicate_priority"
    ]
    print("==== Ratte 20 ====")
    normal, error, rate = compute_total(ratte_dirs_20)
    print("==== Ratte 30 ====")
    normal, error, rate = compute_total(ratte_dirs_30)
    print("==== Ratte 40 ====")
    normal, error, rate = compute_total(ratte_dirs_40)
    print("==== Ratte 50 ====")
    normal, error, rate = compute_total(ratte_dirs_50)

    print("==== Tosa 20 ====")
    normal, error, rate = compute_total(tosa_dirs_20)
    print("==== Tosa 30 ====")
    normal, error, rate = compute_total(tosa_dirs_30)
    print("==== Tosa 40 ====")
    normal, error, rate = compute_total(tosa_dirs_40)
    print("==== Tosa 50 ====")
    normal, error, rate = compute_total(tosa_dirs_50)

    # tosa_dirs_50 = [
    # ]




# ==== Ratte 20 ====
# Warning: //workspace/mlir-inconsistent/ratte_seed_v3/arithsem.e70e0400e981759f.mlir ---- has too many errors: 206.
# Total NORMAL results: 8999.6
# Total CONVERT_ERROR results: 470.8
# Lowering Success Rate: 95.03%
# ==== Ratte 30 ====
# Warning: //workspace/mlir-inconsistent/ratte_seed_v3/arithsem.e70e0400e981759f.mlir ---- has too many errors: 158.
# Total NORMAL results: 9081.0
# Total CONVERT_ERROR results: 307.6
# Lowering Success Rate: 96.72%
# ==== Ratte 40 ====
# Warning: //workspace/mlir-inconsistent/ratte_seed_v3/arithsem.e70e0400e981759f.mlir ---- has too many errors: 128.
# Total NORMAL results: 9057.6
# Total CONVERT_ERROR results: 227.8
# Lowering Success Rate: 97.55%
# ==== Ratte 50 ====
# Warning: //workspace/mlir-inconsistent/ratte_seed_v3/arithsem.e70e0400e981759f.mlir ---- has too many errors: 122.
# Total NORMAL results: 9366.6
# Total CONVERT_ERROR results: 214.8
# Lowering Success Rate: 97.76%
# ==== Tosa 20 ====
# Warning: //workspace/mlir-inconsistent/tosa_seed_v1/tosa.1a557c5726c02b4e.mlir ---- has too many errors: 107.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v1/tosa.4be3fee74f9f597d.mlir ---- has too many errors: 171.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v1/tosa.499c58727d48db3e.mlir ---- has too many errors: 275.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v1/tosa.e2eb58cdfa90dcac.mlir ---- has too many errors: 165.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v1/tosa.aec1cf549956ee15.mlir ---- has too many errors: 144.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v1/tosa.ce11734f63754b76.mlir ---- has too many errors: 186.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v1/tosa.0e4cd9fbfe6fcea9.mlir ---- has too many errors: 224.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v1/tosa.a4bf431c664146ab.mlir ---- has too many errors: 182.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v1/tosa.74730dbeeed68761.mlir ---- has too many errors: 149.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v1/tosa.95cebca668e44303.mlir ---- has too many errors: 328.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v2/tosa.9e35cfc79a6a1718.mlir ---- has too many errors: 164.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v2/tosa.1eb0cbcf849dc569.mlir ---- has too many errors: 278.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v2/tosa.fa1f33dce9a4fcc5.mlir ---- has too many errors: 144.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v2/tosa.3101717532a423fe.mlir ---- has too many errors: 210.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v2/tosa.cff16e8da3509ca5.mlir ---- has too many errors: 192.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v3/tosa.5af2823296af14d4.mlir ---- has too many errors: 151.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v3/tosa.0aad848a4c9c50b7.mlir ---- has too many errors: 201.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v3/tosa.43e83446d40af484.mlir ---- has too many errors: 108.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v3/tosa.44b9ef1c603ab6ab.mlir ---- has too many errors: 104.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v3/tosa.d67c5a5634644045.mlir ---- has too many errors: 211.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v3/tosa.a571f1fa079616a4.mlir ---- has too many errors: 128.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v3/tosa.6f2ee74e49162a0c.mlir ---- has too many errors: 135.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v3/tosa.04875f49a20fd15f.mlir ---- has too many errors: 156.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v4/tosa.4fff777b97ec1f3e.mlir ---- has too many errors: 122.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v4/tosa.a40d7cb3d68950b2.mlir ---- has too many errors: 185.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v4/tosa.29e5380ac5cc313c.mlir ---- has too many errors: 117.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v4/tosa.95c85c190ff6af12.mlir ---- has too many errors: 107.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v4/tosa.e5c038e40f59d944.mlir ---- has too many errors: 122.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v5/tosa.096e220ac00af868.mlir ---- has too many errors: 134.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v5/tosa.08f4ac4175ccd9a3.mlir ---- has too many errors: 243.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v5/tosa.28899998cf28d1ff.mlir ---- has too many errors: 148.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v5/tosa.715ce5c6c5ef3a4b.mlir ---- has too many errors: 101.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v5/tosa.681cba8729bdeab9.mlir ---- has too many errors: 151.
# Warning: //workspace/mlir-inconsistent/tosa_seed_v5/tosa.961719c1df193af5.mlir ---- has too many errors: 197.
# Total NORMAL results: 11329.4
# Total CONVERT_ERROR results: 1759.8
# Lowering Success Rate: 86.56%
# ==== Tosa 30 ====
# Total NORMAL results: 12822.2
# Total CONVERT_ERROR results: 330.0
# Lowering Success Rate: 97.49%
# ==== Tosa 40 ====
# Total NORMAL results: 12727.2
# Total CONVERT_ERROR results: 254.8
# Lowering Success Rate: 98.04%
# ==== Tosa 50 ====
# Total NORMAL results: 12471.2
# Total CONVERT_ERROR results: 236.2
# Lowering Success Rate: 98.14%
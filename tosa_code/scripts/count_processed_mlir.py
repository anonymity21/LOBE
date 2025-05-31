import glob
import os

def count_occurrences(log_dir, search_string):
    total_count = 0
    path_pattern = os.path.join(log_dir, '*.log')
    log_files = glob.glob(path_pattern)
    for logfile in log_files:
        count = 0
        with open(logfile, 'r') as file:
            for line in file:
                if search_string in line:
                    count += 1
        print(f"{logfile}: {count} occurrences")
        total_count += count
    return total_count


log_dir = '//workspace/mlir-inconsistent/multiple_log/tosa_seed_v13/2024-12-24_15-37-24'
search_term = '---- Process'

total_count = count_occurrences(log_dir, search_term)
print(f"{total_count} mlir files have been processed.")

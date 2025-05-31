import os
import shutil
from collections import defaultdict

SOURCE_DIR = '//workspace/mlir-inconsistent/ratte_seed_semantics' 
TARGET_BASE_DIR = '//workspace/mlir-inconsistent'  
PREFIXES = ["arithsem", "linalggeneric", "tensor"]
FILES_PER_TYPE = 2000 * 12
NUM_BUCKETS = 5
TOTAL = FILES_PER_TYPE * NUM_BUCKETS

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def collect_files_by_prefix(directory, prefixes):
    collected = defaultdict(list)
    for fname in os.listdir(directory):
        for prefix in prefixes:
            if fname.startswith(prefix):
                collected[prefix].append(fname)
                break
    return collected

def distribute_to_rattes(collected_files):
    for prefix, files in collected_files.items():
        files = files[:TOTAL]
        files.sort() 
        total = len(files)
        chunk_size = FILES_PER_TYPE 

        for i in range(NUM_BUCKETS):
            ratte_dir = os.path.join(TARGET_BASE_DIR, f'ratte_seed_v{i+1}_semantics')
            ensure_dir(ratte_dir)
            start = i * chunk_size
            end = start + chunk_size if i < NUM_BUCKETS - 1 else total
            for fname in files[start:end]:
                src = os.path.join(SOURCE_DIR, fname)
                dst = os.path.join(ratte_dir, fname)
                shutil.copy(src, dst)

def main():
    collected = collect_files_by_prefix(SOURCE_DIR, PREFIXES)
    distribute_to_rattes(collected)

if __name__ == "__main__":
    main()

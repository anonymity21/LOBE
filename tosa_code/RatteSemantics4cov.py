import util
import multiprocessing
import argparse
import time 
import json
from datetime import datetime
import os
import random
from tqdm import tqdm
import re
from multiprocessing import Process, current_process



# seed_name = 'ratte_seed_v1_semantics'
class Configuration:
    def __init__(self):
        self.project_dir = '//workspace/mlir-inconsistent'
        self.run_option = {}
        self.run_option['arithsem'] = '-arith-expand -test-lower-to-llvm'
        self.run_option['linalggeneric'] = '-one-shot-bufferize="bufferize-function-boundaries" -cse -canonicalize -convert-vector-to-scf -test-lower-to-llvm'
        self.run_option['tensor'] = '-one-shot-bufferize="bufferize-function-boundaries" -cse -canonicalize -convert-vector-to-scf -test-lower-to-llvm'
        self.mlir_opt = self.project_dir + '/third_party_tools/mlir-opt-449e2f5d66'
        self.scan_func = self.project_dir + '/third_party_tools/mlir-scan'
        self.jit_runner = '//MLIR/llvm-release/llvm-project/build/bin/mlir-cpu-runner'
        self.jit_arg = ('-e main -entry-point-result=void --shared-libs=//MLIR/llvm-release/llvm-project/build/lib/libmlir_c_runner_utils.so'
        ' --shared-libs=//MLIR/llvm-release/llvm-project/build/lib/libmlir_runner_utils.so' 
        ' --shared-libs=//MLIR/llvm-release/llvm-project/build/lib/libmlir_async_runtime.so')

    def init_dirs(self, seed_name):
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.project_dir, 'multiple_log', seed_name, f'sem_{current_datetime}')
        self.cov_dir = os.path.join(self.project_dir, 'cov_collection', seed_name, f'sem_{current_datetime}')
        os.makedirs(self.cov_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tmp_dir = self.project_dir + '/sample_tmp/'

    def process_stacktrace(self, stacktrace):
        new_stacktrace = []
        for line in stacktrace:
            line = line.strip()
            if not line.startswith('#'):
                continue
            line = line.split(' ', 2)[2].split(f'({self.mlir_opt}')[0]
            new_stacktrace.append(line)
        return '\n'.join(new_stacktrace)

config = Configuration()
def parse_args():
    parser = argparse.ArgumentParser(description="Running Configuration")
    parser.add_argument(
        '--seedname', type=str, default='tosa_seed_v15', 
        help="Directory name of tosa seed to process"
    )
    return parser.parse_args()

def is_empty(file_path):
    """Check if a file is empty."""
    content = util.get_file_content(file_path).strip()
    if len(content) == 0  or content == '\n':
        return True
    return False
def get_crash_key(error_message):
    lines = error_message.split('\n')
    for line in lines:
        if 'mlir-opt:' in line:
            return line.replace('Testing: ', '')
    return config.process_stacktrace(lines)

def normalize_memref_base(actual_result):
    pattern = r'(Memref base@ = )0x[0-9a-fA-F]+( rank)'
    replacement = r'\g<1>0x000000000000\g<2>'
    result = re.sub(pattern, replacement, actual_result)
    return result

def log_crash(result_error_file, process_file, crash_log_path, log_file):
    error_message = util.get_file_content(result_error_file)
    if error_message.strip() != '':
        util.append_content_to_file(log_file, f'process_file: {process_file}\n error_message: {error_message}\n' )
    if 'PLEASE submit a bug report' not in error_message:
        return False
    crash_key = get_crash_key(error_message)
    if crash_key:
        print(f'[crash key] find a crash in {process_file}')
        util.append_content_to_file(crash_log_path, f'{crash_key} | {process_file}\n')
        return True
    else:
        print(f'[crash key] None\n{error_message}')
        return False

def log_inconsistent(result_file, expected_file, cmd, inconsistent_log_path, log_file):
    expected_result = util.get_file_content(expected_file)
    actual_result = normalize_memref_base(util.get_file_content(result_file))
    if expected_result.strip() != actual_result.strip():
        # util.append_content_to_file(log_file, error_message)
        print(f'[log inconsistent] find an inconsistent result in {cmd}')
        util.append_content_to_file(inconsistent_log_path, f'[inconsistent] {cmd}\nExpected: {expected_result}\nActual:   {actual_result}\n\n')
        return True
    return False

def collection_cov(tmp_file, cov_file):
    cmd = (' ').join([config.scan_func, tmp_file])
    # print(f'cmd: {cmd}')
    res = util.execmd(cmd)
    cov_ops = set()
    cov_dialects = set()
    for line in res.split('\n'):
            line = line.strip()
            if line.startswith('[main]') or not line:
                continue
            try:
                dialect,op = line.split(' ')
                cov_ops.add(op)
                cov_dialects.add(dialect)
            except ValueError as e:
                print(f"Error: {e}")
                print(f'[scan_mlir]: scan file {tmp_file},line: {line} has error')
    # self.log_info(f'[scan_mlir] current ops: {ops}')
    util.append_content_to_file(cov_file, f'dialects: {(" ").join(cov_dialects)}\n')
    util.append_content_to_file(cov_file, f'ops: {(" ").join(cov_ops)}\n')


def run_and_compare(mlir_file):
    file_name = os.path.basename(mlir_file)
    pid = os.getpid()
    dialect, _ = file_name.split('.', 1)

    tmp_dir = os.path.join(config.tmp_dir, file_name)
    os.makedirs(tmp_dir, exist_ok=True)

    log_file = os.path.join(config.log_dir, f'{pid}.log')
    cov_file = os.path.join(config.cov_dir, f'{pid}.cov')
    crash_log = os.path.join(config.cov_dir, f'{pid}.crash')
    inconsistent_log = os.path.join(config.cov_dir, f'{pid}.inconsistent')

    llvm_file = os.path.join(tmp_dir, util.random_file_prefix(file_name))
    err_file = f'{llvm_file}.err'
    result_file = f'{llvm_file}.res'
    expected_file = f'{mlir_file}.res'



    # Step 1: run mlir-opt
    # opt_cmd = f'timeout 100 {config.mlir_opt} {mlir_file} {config.run_option[dialect]} 1> {llvm_file} 2> {err_file}'
    # util.execmd(opt_cmd)
    current_input = mlir_file
    options = (config.run_option[dialect]).split()
    for  option in options:
        # print(f'option: {option}')
        tmp_file = os.path.join(tmp_dir, util.random_file_prefix(file_name))
        opt_cmd = f'timeout 100 {config.mlir_opt} {current_input} {option} 1> {tmp_file} 2> {tmp_file}.err'
        # print(f'opt_cmd: {opt_cmd}')
        util.execmd(opt_cmd)
        collection_cov(tmp_file, cov_file)
        current_input = tmp_file
        # if log_crash(err_file, mlir_file, crash_log, log_file):
        #     return 

    # Step 2: run JIT
    jit_cmd = f'timeout 100 {config.jit_runner} {current_input} {config.jit_arg}  1> {result_file} 2> {err_file}'
    util.execmd(jit_cmd)

    # Step 3: compare result
    cmd = f'timeout 100 {config.mlir_opt} {mlir_file} {config.run_option[dialect]} | timeout 100 {config.jit_runner}  {config.jit_arg}'
    log_inconsistent(result_file, expected_file, cmd, inconsistent_log, log_file)
    util.remove_dir(tmp_dir)


def run_worker(mlir_files_subset, process_id):
    start_time = time.time()
    time_limit = 60 * 60  # 1 hour
    count = 0
    progress_bar = tqdm(mlir_files_subset, desc=f"Process-{process_id}", position=process_id, ncols=100)
    for i, mlir_file in enumerate(progress_bar):
        if i % 10 == 0:
            if time.time() - start_time >= time_limit:
                print(f"[Process-{process_id}] Reached 1-hour limit. Processed {count} files.")
                break
        run_and_compare(mlir_file)
        count += 1

    elapsed = time.time() - start_time
    print(f"[Process-{process_id}] Done. Executed {count} files in {elapsed:.2f} seconds.")

# python3 tosa_code/RatteSemantics4cov.py --seedname=ratte_seed_v1_semantics
def main():
    args = parse_args()
    all_mlirfiles = util.get_all_files_in_directory(f'{config.project_dir}/{args.seedname}')
    random.shuffle(all_mlirfiles)
    config.init_dirs(args.seedname)
    num_procs = 12
    chunk_size = len(all_mlirfiles) // num_procs
    file_chunks = [all_mlirfiles[i * chunk_size: (i + 1) * chunk_size] for i in range(num_procs)]

    processes = []
    for i in range(num_procs):
        p = Process(target=run_worker, args=(file_chunks[i], i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time / 3600
    print(f"All tasks spent: {hours:.2f} hours")

from RandomPassLowering import RandomPassLowering
from FullResetLowering import FullResetLowering
from StatefulLowering import StatefulLowering
from BaseLoweringStrategy import LoweringState
from ConversionPassLowering import ConversionPassLowering
from util import Configuration
import util
from util import LoweringResult
import os
import random
from tqdm import tqdm 
import shutil
import multiprocessing
import argparse
from functools import partial
import time 
import sys
import json




config = Configuration()

class ExecutionResult:
    def __init__(self, results, cmd=""):
        self.output = self.process_multi_results(results)
        self.cmd = cmd

    def process_multi_results(self, raw_text):
        results = []
        current_block = []
        for line in raw_text.splitlines():
            if line.startswith('%') and '=' in line:
                if current_block:
                    results.append('\n'.join(current_block))
                    current_block = []
            current_block.append(line.strip())
        if current_block:
            results.append('\n'.join(current_block))

        cleaned_blocks = []
        for block in results:
            lines = block.splitlines()
            first_line = lines[0]
            if 'Memref base@' in first_line and 'data =' in first_line:
                prefix = first_line.split('=')[0].strip() + '='
                data_start = first_line.split('data =', 1)[1].strip()
                cleaned = [prefix]
                if data_start:
                    cleaned.append(data_start)
                cleaned.extend(lines[1:])
                cleaned_blocks.append('\n'.join(cleaned))
            else:
                cleaned_blocks.append(block)
        return cleaned_blocks

# run multiple_run_num times, and then compare the results
def multiple_run_and_compare(mlir_file, lowering_strategy):
    pid = os.getpid()
    log_file = f'{config.log_dir}/{pid}.log'
    cov_file = f'{config.cov_dir}/{pid}.cov'
    crash_file = f'{config.cov_dir}/{pid}.crash'
    inconsistent_file = f'{config.cov_dir}/{pid}.inconsistent'
    file_name = os.path.basename(mlir_file)
    tmp_dir = os.path.join(config.tmp_dir, file_name)
    util.mkdir_dir(tmp_dir)
    util.append_content_to_file(log_file, f'---- Process {mlir_file} ----\n')
    results = []
    lowering_state = LoweringState()
    all_stacktrace = {}

    i = 0
    start_time = time.time()  # Start time of the program
    check_interval = 50
    while True:
        util.append_content_to_file(log_file, f'Run {i} for mlir file: {mlir_file}\n')
        strategy = lowering_strategy(config, mlir_file, tmp_dir, log_file, cov_file)
        lowering_result, process_file = strategy.lowering(lowering_state)
        if lowering_result == LoweringResult.NORMAL:
            result = strategy.execute_mlir(process_file)
            if isinstance(result, tuple) and result[0] == LoweringResult.NORMAL:
                res = ExecutionResult(result[1], result[2])
                results.append(res)
            elif result not in [LoweringResult.CONVERT_ERROR, LoweringResult.TIMEOUT, LoweringResult.NORESULT, LoweringResult.EXECUTE_ERROR]:
                assert False, f'[multiple_run_and_compare] Other case in result: {result}'
        print(f'Run {i} for mlir file: {mlir_file}', flush=True)
        i += 1
        # dump the crach
        if len(strategy.stacktrace) > 0:
            all_stacktrace.update(strategy.stacktrace)
        if i % check_interval == 0:
            print(f'Pid {pid} has run {i} times for mlir file: {mlir_file}', flush=True)
            elapsed_time = time.time() - start_time  # Calculate the elapsed time
            if elapsed_time >= config.execution_time:
                print(f'Time limit is reached. Pid {pid} has run {elapsed_time:.4f} for mlir file: {mlir_file}', flush=True)
                break  # Break the loop if the time limit is reached
    
    elapsed_time = (time.time() - start_time) / 3600  # Calculate the elapsed time in hours
    util.append_content_to_file(log_file, f'Pid {pid} has run {elapsed_time:.2f} hours for mlir file: {mlir_file}, total numbers of run {i}.\n')
    print(f'Pid {pid} has run {elapsed_time:.2f} hours for mlir file: {mlir_file}, total numbers of run {i}.\n')

    if len(results) == 0:
        util.append_content_to_file(log_file, f'[multiple_run_and_compare] No results for mlir file: {mlir_file}\n')
        return

    # compare the results and output the different results
    baseline = results[0].output
    baseline_file = f"{mlir_file}.baseline.txt"
    with open(baseline_file, "w") as f:
        f.write("\n".join(baseline))
    print(f"[Compare] Baseline written to: {baseline_file}")
    util.append_content_to_file(log_file, f'[Compare] baseline with others\n') 
    inconsistent_dict = {}
    for idx, res in enumerate(results[1:], 1):
        current = res.output
        baseline_len = len(baseline)
        current_len = len(current)
        max_len = max(baseline_len, current_len)

        for i in range(max_len):
            base_line = baseline[i] if i < baseline_len else ""
            test_line = current[i] if i < current_len else ""
            if base_line.strip() != test_line.strip() and test_line.strip() != "1111111111": # for placeholder
                util.append_content_to_file(log_file,f"[Diff Found] Run [Command] {res.cmd}\n")
                util.append_content_to_file(log_file, f"{base_line} is different from result_line {i}: {test_line}\n")
                inconsistent_dict[res.cmd] = {
                    "base_line": base_line,
                    "test_line": test_line
                }
                break

    with open(inconsistent_file, "a") as f:
        f.write(json.dumps(inconsistent_dict, indent=4))
        f.write('\n')
    with open(crash_file, 'a') as f:
        f.write(json.dumps(all_stacktrace, indent=4))
        f.write('\n')
    util.delete_mlir_files()
    util.remove_dir(tmp_dir)

# python3 tosa_code/main_ratte.py --strategy=fullreset --multiprocess=1 --executiontime=1 --seedname=ratte_seed_v1

def parse_args():
    parser = argparse.ArgumentParser(description="Running Configuration")
    parser.add_argument(
        '--strategy', choices=['fullreset', 'stateful', 'conversion'], default='fullreset',
        help="Choose the lowering strategy ('fullreset' or 'stateful')"
    )
    parser.add_argument(
        '--multiprocess', type=int, default=1, 
        help="Number of processes for multiprocessing (must be an integer)"
    )
    parser.add_argument(
        '--runtimes', type=int, default=500, 
        help="Running times for each mlir program"
    )
    parser.add_argument(
        '--executiontime', type=float, default=1, 
        help="The time of execution for each mlir program, default is 1 hour"
    )
    parser.add_argument(
        '--seedname', type=str, default='tosa_seed_v15', 
        help="Directory name of tosa seed to process"
    )
    parser.add_argument(
        '--cov',
        action='store_true',
        help="Enable coverage information recording."
    )
    parser.add_argument(
        '--priority',
        action='store_true',
        help="Enable priority of dialect conversion."
    )
    parser.add_argument(
        '--deduplicateprint',
        action='store_false',
        help="Disable the unique print."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config.init(args.seedname, args.runtimes, args.cov, args.executiontime, args.priority, args.deduplicateprint, args.strategy) 
    # TODO: temporary
    # config.all_mlirfiles = ['//workspace/mlir-inconsistent/ratte_seed_v3/ratte.103b813bb2f17bf1.mlir']
    # config.all_mlirfiles = util.get_file_content('//workspace/mlir-inconsistent/extracted_mlir_files.txt').split('\n')
    if args.strategy == 'fullreset':
        strategy = FullResetLowering
    elif args.strategy == 'conversion':
        strategy = ConversionPassLowering
    else:
        strategy = StatefulLowering

    # for mlir_file in config.all_mlirfiles:
    #     multiple_run_and_compare(mlir_file, strategy)

    partial_function = partial(multiple_run_and_compare, lowering_strategy=strategy)
    # Use a Pool of workers to process the files in parallel
    with multiprocessing.Pool(processes=args.multiprocess) as pool:
        list(tqdm(pool.imap(partial_function, config.all_mlirfiles), 
                  desc="Processing All MLIR Files", 
                  total=len(config.all_mlirfiles), 
                  unit="files"))


def check_time_limit(start_time, limit=24 * 60 * 60):
    """ Check the time limit of the program and exit if the time limit is reached."""
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= limit:
            print("Time limit reached: 24 hours. Stopping the program.")
            os._exit(0)  # Exit the program
        time.sleep(1800)  # Sleep for 10 seconds

def timed_main():
    max_runtime = 24 * 60 * 60  # 24 hours
    start_time = time.time()
    # Start a thread to monitor the time limit
    monitor_thread = threading.Thread(target=check_time_limit, args=(start_time, max_runtime), daemon=True)
    monitor_thread.start()
    # Run the main function
    main()


if __name__ == "__main__":
    # timed_main()
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time / 3600
    print(f"All tasks spent: {hours:.2f} hours")

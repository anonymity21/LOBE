import util
from util import Configuration
import os
import random
from tqdm import tqdm 
import shutil
from collections import deque
import multiprocessing
from datetime import datetime 


config = Configuration()
original_mlir_file = '//workspace/mlir-inconsistent/tosa_seed_enumeration/tosa.0b0efabec7f81c16.mlir'


def scan_mlir(tmp_mlir_file):
    ops = set()
    cmd = (' ').join([config.scan_func, tmp_mlir_file])
    res = util.execmd(cmd)
    for line in res.split('\n'):
        line = line.strip()
        if line.startswith('[main]') or not line:
            continue
        dialect,op = line.split(' ')
        if op == 'builtin.module' or dialect == 'llvm' or op == 'builtin.unrealized_conversion_cast' or dialect == 'func':
            continue
        ops.add(op)
    print(f'[scan_mlir] current ops: {ops}')
    return ops

def is_equal(result1, result2):
    content1 = util.get_file_content(result1)
    content2 = util.get_file_content(result2)
    return content1 == content2
        
def is_empty(mlir_file):
    content = util.get_file_content(mlir_file).strip()
    if content == '' or content =='module {\n}' or content == '\n':
        return True
    return False

def is_empty(file_path):
    """Check if a file is empty."""
    content = util.get_file_content(file_path).strip()
    if len(content) == 0 or content =='module {\n}' or content == '\n':
        return True
    return False

def construct_func_opt(pass_name):
    if pass_name.startswith('-'):
        return pass_name
    else:
        return f'-pass-pipeline="builtin.module(func.func({pass_name}))"'

def pick_conversion_pass(next_op, op_pass_dict):
    dialect,op_name = next_op.split('.',1)
    all_op = f'{dialect}.op'
    if next_op in op_pass_dict:
        return [construct_func_opt(one_pass) for one_pass in op_pass_dict[next_op]]
    elif all_op in op_pass_dict:
        return [construct_func_opt(one_pass) for one_pass in op_pass_dict[all_op]]
    else:
        assert False, f'[Warning] need add pass for op {next_op}'


def convert_pass_for_ops(ops):
    priority_list = ['-pass-pipeline="builtin.module(func.func(tosa-to-linalg-named))"', 
    '-tosa-to-scf', ['-tosa-to-arith', '-tosa-to-tensor',  '-pass-pipeline="builtin.module(func.func(tosa-to-linalg))"'], '-convert-scf-to-cf']
    last_list = ['-convert-func-to-llvm', '-convert-cf-to-llvm']

    all_conversion_pass = set()
    for op in ops:
        all_conversion_pass.update(pick_conversion_pass(op, config.op_convert_dict))
    results = []
    for conversion_pass in priority_list:
        if isinstance(conversion_pass, list):
            for one_pass in conversion_pass:
                if one_pass in all_conversion_pass:
                    results.append(one_pass)
            return results
        else:
            if conversion_pass in all_conversion_pass:
                results.append(conversion_pass)
                return results
    tem_all_conversion_pass = all_conversion_pass.copy()
    tem_all_conversion_pass.discard('-convert-func-to-llvm')
    tem_all_conversion_pass.discard('-convert-cf-to-llvm')
    if len(tem_all_conversion_pass) <= 0:
        return all_conversion_pass
    else:
        return tem_all_conversion_pass

# lowering_index = 0
def dfs_lowering(process_file, current_path, all_paths, tmp_dir, final_results, log_file):
    if is_empty(process_file):
        current_path.append('error')
        all_paths.append(list(current_path))
        current_path.pop()
        return
    ops = scan_mlir(process_file)
    all_conversion_pass = convert_pass_for_ops(ops)
    # all_optimization_pass = pick_optimization_pass(ops)
    if len(all_conversion_pass) == 0:
        all_paths.append(list(current_path))
        res_file = os.path.join(tmp_dir, process_file +'.re')
        cmd = (' ').join([config.mlir_opt, process_file, '-convert-func-to-llvm',  '-reconcile-unrealized-casts','|', config.jit_runner, config.jit_arg, '1>', res_file  ])
        util.execmd(cmd)
        final_result = util.get_file_content(res_file)
        print(f'current_path: {current_path}')
        print(final_result)
        all_cmd = (' ').join([config.mlir_opt, original_mlir_file, (f' | {config.mlir_opt} ').join(current_path), '-convert-func-to-llvm',  '-reconcile-unrealized-casts', '|', 'timeout 10', config.jit_runner, config.jit_arg ])
        util.append_content_to_file(log_file, f'Totoal cmd: {all_cmd}\n')
        print(final_result)
        util.append_content_to_file(log_file, f'final_result: {final_result}\n')
        final_results.append(final_result)
        return
    elif len(current_path) > 25:
        print(f'[Warning] There are ops that cannot be converted. {ops}')
        all_cmd = (' ').join([config.mlir_opt, original_mlir_file, (f' | {config.mlir_opt} ').join(current_path), '-convert-func-to-llvm',  '-reconcile-unrealized-casts', '|', 'timeout 10', config.jit_runner, config.jit_arg ])
        # util.append_content_to_file(log_file, f'Totoal cmd: {all_cmd}\n')
        return 
    else:
        for conversion_pass in all_conversion_pass:
            print(f'Pick conversion_pass {conversion_pass} from all_conversion_pass: {all_conversion_pass}')
            result_mlir_file = os.path.join(tmp_dir, util.random_file_prefix('file_name.'))
            cmd = (' ').join([config.mlir_opt, process_file, conversion_pass, '1>', result_mlir_file , '2>', f'{result_mlir_file}.err'])
            util.execmd(cmd)
            current_path.append(conversion_pass)
            print(f'current_path: {current_path}')
            dfs_lowering(result_mlir_file, current_path, all_paths, tmp_dir, final_results, log_file)
            current_path.pop()
            

# enumerate all conversion paths (without optimization)
def lowering_process(mlir_file, tmp_dir, log_file):
    file_name = os.path.basename(mlir_file)
    tmp_dir = os.path.join(config.tmp_dir, file_name)
    util.mkdir_dir(tmp_dir)
    util.append_content_to_file(log_file, f'---- Process {mlir_file} ----\n')
    all_paths = []
    final_results = []
    results = []
    dfs_lowering(mlir_file, [], all_paths,tmp_dir, final_results, log_file)
    for result in final_results:
        if 'data =' in result:
                result = result.split('data =')[1]
                results.append(result)
    if all(result == results[0] for result in results):
        util.append_content_to_file(log_file, 'Same Results!\n')
    else:
        util.append_content_to_file(log_file, f'Not Same in {mlir_file}\n')
        util.append_content_to_file(log_file, f'results[0]: {results[0]}\n')
        different_set = set()
        for result in results:
            if result != results[0]:
                different_set.add(result)
        different_results = '\n'.join(list(different_set))
        util.append_content_to_file(log_file, f'Different result:\n{different_results} \n')
    util.remove_dir(tmp_dir)


def multiple_run_and_compare(mlir_file):
    tmp_dir = '//workspace/mlir-inconsistent/tmp'
    pid = os.getpid()
    log_file = f'//workspace/mlir-inconsistent/enumerate_log/{pid}.log'
    lowering_process(mlir_file, tmp_dir, log_file)


# def main():
#     config.init() 
#     mlir_dir = '//workspace/mlir-inconsistent/tosa_seed_enumeration'
#     all_mlirfiles = util.get_all_files_in_directory(mlir_dir) 
#     # all_mlirfiles = ['//workspace/mlir-inconsistent/tobereported/tosa1.mlir']
#     num_processes = 1
#     # Use a Pool of workers to process the files in parallel
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         # Map the process_mlir_file function to all files
#         list(tqdm(pool.imap(multiple_run_and_compare, all_mlirfiles), 
#                   desc="Processing All MLIR Files", 
#                   total=len(all_mlirfiles), 
#                   unit="files", 
#                   ncols=100))

def run_single_file():
    config.init('', 0, False, 1)
    mlir_file = '//workspace/mlir-inconsistent/tosa_seed_enumeration/tosa.0b0efabec7f81c16.mlir'
    tmp_dir = '//workspace/mlir-inconsistent/tmp'
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f'//workspace/mlir-inconsistent/enumerate_log/{current_datetime}.log'
    lowering_process(mlir_file, tmp_dir, log_file)

if __name__ == '__main__':
    run_single_file()
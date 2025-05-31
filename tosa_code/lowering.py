import util
from util import Configuration
import os
import random
from tqdm import tqdm 
import shutil
import time

config = Configuration()

def scan_mlir(tmp_mlir_file):
    ops = set()
    dialects = set()
    cmd = (' ').join([config.scan_func, tmp_mlir_file])
    res = util.execmd(cmd)
    for line in res.split('\n'):
        line = line.strip()
        if line.startswith('[main]') or not line:
            continue
        dialect,op = line.split(' ')
        dialects.add(dialect)
        if op == 'builtin.module' or dialect == 'llvm' or op == 'builtin.unrealized_conversion_cast':
            continue
        ops.add(op)
    print(f'[scan_mlir] current ops: {ops}')
    print(f'[scan_mlir] current dialects: {dialects}')
    return ops

def is_empty(mlir_file):
    content = util.get_file_content(mlir_file).strip()
    if content == '' or content =='module {\n}' or content == '\n':
        return True
    return False


# need to check
priority_list = [
    ["tosa.conv2d", "tosa.matmul", "tosa.transpose", "tosa.conv3d", "tosa.fully_connected", 
        "tosa.depthwise_conv", "tosa.maxpool_2d", "tosa.avg_pool2d"],
    ["tosa.cond_if", "tosa.while_loop", "tosa.yield"],
    ['tosa'],
    ['scf']   
]
# lowering must follow the priority
def pick_one_op(ops):
    assert len(ops) > 0,'[pick_one_op] ops is empty!'
    if isinstance(ops, set):
        ops = list(ops)
    # Prioritize converting dialects in the priority_list.
    for priority_ops in priority_list:
        available_ops = [op for op in ops if any(op.startswith(p) for p in priority_ops)]
        if available_ops:
            return random.choice(available_ops)
    
    # Convert other dialects
    available_dialect_ops = [op for op in ops if not op.startswith('func')]
    if available_dialect_ops:
        return random.choice(available_dialect_ops)
    
    # Last step, convert func.op 
    available_func_ops = [op for op in ops if op.startswith('func')]
    if available_func_ops:
        return random.choice(available_func_ops)
    return random.choice(ops)

def construct_func_opt(pass_name):
    if pass_name.startswith('-'):
        return pass_name
    else:
        return f'-pass-pipeline="builtin.module(func.func({pass_name}))"'

def pick_one_pass(next_op, op_pass_dict):
    dialect,op_name = next_op.split('.',1)
    all_op = f'{dialect}.op'
    if next_op in op_pass_dict:
        return construct_func_opt(random.choice(op_pass_dict[next_op]))
    elif all_op in op_pass_dict:
        return construct_func_opt(random.choice(op_pass_dict[all_op]))
    elif next_op in config.unconverted_ops:
        return ''
    else:
        assert False, f'[Warning] need add pass for op {next_op}'


def pick_current_opt(ops):
    assert len(ops) > 0,'[pick_current_opt] ops is empty!'
    current_opt = {}
    for op in ops: 
        dialect = op.split('.',1)[0]
        new_op = f'{dialect}.op'
        if new_op in config.specific_opts:
            current_opt.update(config.specific_opts[new_op])
    current_opt.update(config.general_opts)
    results = []
    # len(current_opt)
    # opt_num = util.random_int(config.opt_minnum, len(current_opt))
    opt_num = 1
    for i in range(opt_num):
        opt = random.choice(list(current_opt.keys()))
        if len(current_opt[opt]) > 0:
            # randomly select a options for opt
            option_id = util.random_int(-1, len(current_opt[opt])-1)
            if option_id == -1:
                results.append(f'{opt}')
            else:
                value = current_opt[opt][option_id]
                results.append(f'{opt}="{value}"')
        else:
            results.append(opt)
    return (' ').join(results)

def lowering_process_with_generalopts(mlir_file, tmp_dir):
    process_file = mlir_file
    ops = scan_mlir(process_file)
    applied_opts = []
    all_general = 0
    file_name = os.path.basename(mlir_file)
    while len(ops) != 0 and len(applied_opts) < config.max_applied_optnnum:
        print(f'[Already applied opts] {len(applied_opts)}')
        result_mlir_file = os.path.join(tmp_dir, util.random_file_prefix(file_name+'.'))
        # perform the optimization
        next_generalopt = pick_current_opt(ops)
        cmd = (' ').join([config.mlir_opt, process_file, next_generalopt, '1>', f'{result_mlir_file}.opt', '2>', f'{result_mlir_file}.err' ])
        util.execmd(cmd)
        if is_empty(f'{result_mlir_file}.err'):
            if is_empty(f'{result_mlir_file}.opt'):
                print(f'[lowering_process_with_generalopts] {result_mlir_file}.opt is empty after {next_generalopt}')
            else:
                all_general = all_general + len(next_generalopt.split(' '))
                print(f'[Already applied optimization opt] {all_general}')
                applied_opts.append(next_generalopt)
                process_file = f'{result_mlir_file}.opt'
        else:
            optimization_errors = util.get_file_content(f'{result_mlir_file}.err')
            if 'Stack dump:' in optimization_errors:
                print('[lowering_process_with_generalopts: optimization errors] ' + optimization_errors)
        print(f'[lowering_process_with_generalopts] pick optimization options: {next_generalopt}')
        ops = scan_mlir(process_file)
        if len(ops) == 0:  # optimization has converted all ops
            break
        # pick one op and one op conversion pass
        next_op = pick_one_op(ops)
        next_pass = pick_one_pass(next_op,config.op_convert_dict)
        if next_pass == '':
            continue
            # print(f'[Warning] There are ops that cannot be converted, current ops: {next_op}')
            # return 'data = convert_error'
        cmd = (' ').join([config.mlir_opt, process_file, next_pass, '1>', result_mlir_file, '2>', f'{result_mlir_file}.err' ])            
        util.execmd(cmd)
        print(f'[lowering_process_with_generalopts] pick conversion op: {next_op}, pick pass: {next_pass}')
        if is_empty(f'{result_mlir_file}.err'):
            process_file = result_mlir_file
            applied_opts.append(next_pass)
        ops = scan_mlir(process_file)
        # assert(next_op not in ops, f'[lowering_process_with_generalopts] {next_pass} cannot lower {next_op}')
        # process_file = result_mlir_file

    for op in ops:
        if op in config.unsupported_ops:
            print(f'[Warning] There exits op that cannot be converted, unsupported op: {op}')
            return 'data = convert_error'

    # if there are ops that cannot be converted, report them.
    # assert len(ops) == 0, f'There are ops that cannot be converted, current ops: {ops}'
    applied_opts.append('-reconcile-unrealized-casts')
    res_file = os.path.join(tmp_dir, file_name+'.re')
    err_file = os.path.join(tmp_dir, file_name+'.err')
    all_cmd = (' ').join([config.mlir_opt, mlir_file, (f' | {config.mlir_opt} ').join(applied_opts), '|', 'timeout 10', config.jit_runner, config.jit_arg ])
    print(f'Totoal cmd: {all_cmd}')
    cmd = (' ').join([ config.mlir_opt, process_file, '-reconcile-unrealized-casts','|', 'timeout 10', config.jit_runner, config.jit_arg, '1>', res_file, '2>', err_file  ])
    util.execmd(cmd)
    final_result = util.get_file_content(res_file)
    final_err = util.get_file_content(err_file)
    # print(f'final_result: {final_result}\n final_error: {final_err}')
    if is_empty(err_file):
        if is_empty(res_file):
            final_result = 'data = timeout'
    print(f'final_result: {final_result}\n final_error: {final_err}')
    if len(final_result.strip()) == 0:
        final_result = 'data = noresult'
    return final_result

def lowering_process(mlir_file, tmp_dir):
    process_file = mlir_file
    ops = scan_mlir(process_file)
    applied_opts = []
    file_name = os.path.basename(mlir_file)
    while len(ops) != 0 and len(applied_opts) < config.max_applied_optnnum:
        result_mlir_file = os.path.join(tmp_dir, util.random_file_prefix(file_name+'.'))
        # pick one op and one op conversion pass
        next_op = pick_one_op(ops)
        next_pass = pick_one_pass(next_op,config.op_convert_dict)
        if next_pass == '':
            print(f'[Warning] There are ops that cannot be converted, current ops: {next_op}')
            return 'data = convert_error'
        print(f'[lowering_process_with_generalopts] pick op: {next_op}, pick pass: {next_pass}')
        cmd = (' ').join([config.mlir_opt, process_file, next_pass, '1>', result_mlir_file ])
        print(cmd)
        util.execmd(cmd)
        ops = scan_mlir(result_mlir_file)
        assert(next_op not in ops, f'[lowering_process_with_generalopts] {next_pass} cannot lower {next_op}')
        applied_opts.append(next_pass)
        process_file = result_mlir_file
    for op in ops:
        if op in config.unsupported_ops:
            print(f'[Warning] There exits op that cannot be converted, unsupported op: {op}')
            return 'data = convert_error'
    # if there are ops that cannot be converted, report them.
    assert len(ops) == 0, f'There are ops that cannot be converted, current ops: {ops}'
    applied_opts.append('-reconcile-unrealized-casts')
    res_file = os.path.join(tmp_dir, file_name+'.re')
    err_file = os.path.join(tmp_dir, file_name+'.err')
    all_cmd = (' ').join([config.mlir_opt, mlir_file, (f' | {config.mlir_opt} ').join(applied_opts), '|', 'timeout 10', config.jit_runner, config.jit_arg ])
    print(f'Totoal cmd: {all_cmd}')
    cmd = (' ').join([ config.mlir_opt, process_file, '-convert-func-to-llvm -reconcile-unrealized-casts','|', 'timeout 10', config.jit_runner, config.jit_arg, '1>', res_file, '2>', err_file  ])
    util.execmd(cmd)
    final_result = util.get_file_content(res_file)
    final_err = util.get_file_content(err_file)
    # print(f'final_result: {final_result}\n final_error: {final_err}')
    if is_empty(err_file):
        if is_empty(res_file):
            final_result = 'data = timeout'
    print(f'final_result: {final_result}\n final_error: {final_err}')
    if len(final_result.strip()) == 0:
        final_result = 'data = noresult'
    return final_result

def multiple_run_and_compare(mlir_file):
    print(f'---- Process {mlir_file} ----\n')
    file_name = os.path.basename(mlir_file)
    tmp_dir = os.path.join(config.project_dir, 'sample_tmp', file_name)
    if not os.path.exists(tmp_dir):
        try:
            os.makedirs(tmp_dir)
        except Exception as e:
            print(f"Error creating folder {folder_path}: {e}")
    results = []
    for i in range(10):
        print(f'----- Run {i}')
        result = lowering_process_with_generalopts(mlir_file, tmp_dir)
        if 'data =' in result:
            result = result.split('data =')[1]
            results.append(result)
        else:
            assert False, f'[multiple_run_and_compare] Other case in result {result}' 

    if all(result == results[0] for result in results):
        print('Same!')
    else:
        print(f'Not Same in {mlir_file}')
        print(f'results[0]: {results[0]}')
        for result in results:
            if result != results[0]:
                print(f'Different result: {result}')
    # if os.path.exists(tmp_dir):
    #     try:
    #         shutil.rmtree(tmp_dir)
    #     except Exception as e:
    #         print(f"Error deleting folder {tmp_dir}: {e}")


def run_single_file():
    config.init('tmp', 10)
    mlir_file = '//workspace/mlir-inconsistent/a.mlir'
    tmp_dir = '//workspace/mlir-inconsistent/tmp'
    if not os.path.exists(tmp_dir):
        try:
            os.makedirs(tmp_dir)
        except Exception as e:
            print(f"Error creating folder {folder_path}: {e}")
    # multiple_run_and_compare(mlir_file)
    lowering_process_with_generalopts(mlir_file, tmp_dir)
    # lowering_process_with_generalopts(mlir_file, tmp_dir)
    # for i in range(500):
    #     print(f'------  Run {i}')
    #     lowering_process_with_generalopts(mlir_file, tmp_dir)

if __name__ == '__main__':
    start_time = time.time()
    run_single_file()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"run_single_file time: {elapsed_time:.4f} s")

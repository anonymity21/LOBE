import json
import random
import os
from tqdm import tqdm 
import shutil
from datetime import datetime


def remove_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            # print(f'File {file_path} has been deleted.')
        except Exception as e:
            print(f'Error: {e}')
    else:
        print(f'File {file_path} not found.')

def move_file(src, dst):
    if not os.path.exists(src):
        print(f"Error moving file {src} to {dst}: file not found")
        return
    directory = os.path.dirname(dst)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    try:
        shutil.move(src, dst)
    except Exception as e:
        print(f"Error moving file {src} to {dst}: {e}")

def mkdir_dir(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating folder {dir}: {e}")

def remove_dir(dir):
    if os.path.exists(dir):
        try:
            shutil.rmtree(dir)
        except Exception as e:
            print(f"Error deleting folder {dir}: {e}")
def get_content_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        result_dict = json.load(json_file)
    return result_dict

def get_file_content(file_path):
    f = open(file_path)
    return f.read();

def append_content_to_file(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()

def random_file_prefix(prefix_name):
    random_bytes = os.urandom(8)  # Generate 8 random bytes
    random_str = random_bytes.hex()
    temp_filename = f"{prefix_name}.{random_str}.mlir"
    return temp_filename


def execmd(cmd, timeout_time=10):
    # print('[execmd] ' + cmd)
    timeout_cmd = ' '.join(['timeout', str(timeout_time), cmd])
    try:
        pipe = os.popen(timeout_cmd)
        reval = pipe.read()
        pipe.close()
        return reval
    except BlockingIOError:
        print("[execmd] trigger BlockingIOError")
        return "None"

def get_all_files_in_directory(directory,postfix='.mlir'):
    files = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path) and item.endswith(postfix):
            files.append(full_path)
    return files

def random_int(min_val, max_val):
    return random.randint(min_val, max_val)

from enum import Enum
class LoweringResult(Enum):
    CONVERT_ERROR = 1
    TIMEOUT = 2
    EXECUTE_ERROR = 3
    NORESULT = 4
    NORMAL = 0

class Configuration:
    def __init__(self):
        self.project_dir = './'
        self.scan_func = self.project_dir + '/third_party_tools/mlir-scan'
        self.deduplicate_print = self.project_dir + '/third_party_tools/mlir-deduplicate'
        self.mlir_opt = self.project_dir + '/third_party_tools/mlir-opt'
        self.jit_runner = self.project_dir + '/third_party_tools/mlir-cpu-runner'
        self.jit_arg = (f'-e main -entry-point-result=void --shared-libs={self.project_dir}/third_party_tools/libmlir_c_runner_utils.so'
        f' --shared-libs={self.project_dir}/third_party_tools/libmlir_runner_utils.so' 
        f' --shared-libs={self.project_dir}/third_party_tools/libmlir_async_runtime.so')
        
        self.convert_opt_file = self.project_dir + '/options/mlir_conversion.json'
        self.general_opt_file = self.project_dir + '/options/mlir_generalopt.json'
        self.specific_opt_file = self.project_dir + '/options/mlir_specificopt.json'
        self.op_priority_file = 'options/op_priority_equal.json'
        # self.stacktrace = {}


       
    def init(self,seed_name, run_times, cov=False, execution_time=1, op_priority_option=False, deduplcate=False, strategy=''):
        print('[Config.init] Init all options and mlir seeds.')
        self.seed_dir = os.path.join(self.project_dir, seed_name)
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.project_dir, 'experiment_log', seed_name, f'{strategy}_{current_datetime}')
        self.cov_dir = os.path.join(self.project_dir, 'cov_collection', seed_name, f'{strategy}_{current_datetime}')
        os.makedirs(self.cov_dir, exist_ok=True)
        self.tmp_dir = self.project_dir + '/sample_tmp/'
        self.crash_dir = self.project_dir + '/crash_file/'
        self.run_times = run_times
        self.execution_time = execution_time * 3600
        self.opt_maxnum = 7
        self.opt_minnum = 1
        self.max_applied_optnnum = 25
        self.cov = cov
        self.deduplicateprint_option = deduplcate
        if self.deduplicateprint_option:
            self.log_dir = os.path.join(self.project_dir, 'experiment_log', seed_name, f'{strategy}_{current_datetime}_deduplicate')

        self.op_priority_option = op_priority_option
        if op_priority_option:
            self.log_dir = os.path.join(self.project_dir, 'experiment_log', seed_name, f'{strategy}_{current_datetime}_priority')

        if self.deduplicateprint_option and op_priority_option:
            self.log_dir = os.path.join(self.project_dir, 'experiment_log', seed_name, f'{strategy}_{current_datetime}_deduplicate_priority')

        self.tensor_config = {"linalg.op.tensor": "linalg.op", "arith.op.tensor": "arith.op"}    
        print(f'[Config.init] self.op_priority: {self.op_priority_file}')


        self.op_convert_dict = get_content_from_json(self.convert_opt_file)
        self.op_priority = get_content_from_json(self.op_priority_file)
        self.general_opts = get_content_from_json(self.general_opt_file)
        self.specific_opts = get_content_from_json(self.specific_opt_file )
        self.test_opts = get_content_from_json(self.test_opt_file)
        # self.stacktrace = get_content_from_json(self.stacktrace_file)
        self.all_mlirfiles =  get_all_files_in_directory(self.seed_dir)
        self.unsupported_ops =['vector.transfer_write', 'vector.transfer_read', 'cf.br', 'cf.cond_br', 'x86vector.avx.rsqrt']
        self.unconverted_ops = ['x86vector.avx.rsqrt', 'ub.poison']

    def process_stacktrace(self, stacktrace):
        new_stacktrace = []
        for line in stacktrace:
            line = line.strip()
            if not line.startswith('#'):
                continue
            line = line.split(' ', 2)[2].split(f'({self.mlir_opt}')[0]
            new_stacktrace.append(line)
        return '\n'.join(new_stacktrace)


def delete_mlir_files(directory='/tmp'):
    print(f"[delete_mlir_files]: Start delete the mlir files in /tmp")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith('.mlir'):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


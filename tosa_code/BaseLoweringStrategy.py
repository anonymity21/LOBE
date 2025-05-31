import os
import util
from util import LoweringResult
from util import Configuration
from abc import ABC, abstractmethod
import random
import time



class LoweringState:
    def __init__(self):
        self.state_graph = {}
        self.seed_pool = {}
    
    def empty_state(self):
        return len(self.seed_pool) == 0
    
    def build_node(self, dialects):
        return (' ').join(sorted(dialects))

    def init(self, mlir_file):
        self.seed_pool[mlir_file] = [] 
    
    def pick_one_state(self):
        process_file = random.choice(list(self.seed_pool.keys()))
        applied_opts = self.seed_pool[process_file]
        return process_file,applied_opts.copy()

    def current_state(self):
        # print('[current lowering state]')
        allnodes = len(self.seed_pool)
        node_set = set()
        for key,value in self.state_graph.items():
            node_set.add(key)
            node_set.update(value)
        # print(f'All nodes from graph: {len(node_set)}, {node_set}')
        # print(f'All nodes: {allnodes}')
        edge_num = sum(len(neighbors) for neighbors in self.state_graph.values())
        # print(f'All edges: {edge_num}')
        return f'All nodes: {len(node_set)}, All edges: {edge_num}'



    def update(self, mlir_file, applied_opts, from_dialects, to_dialects):
        from_node = self.build_node(from_dialects)
        to_node = self.build_node(to_dialects)
        if from_node == to_node:
            return
        if from_node not in self.state_graph:
            self.state_graph[from_node] = set()
        if to_node not in self.state_graph[from_node]:
            self.state_graph[from_node].add(to_node)
            self.seed_pool[mlir_file] = applied_opts.copy()


class BaseLoweringStrategy(ABC):
    def __init__(self, config, mlir_file, tmp_dir, log_file, cov_file):
        self.config = config
        self.mlir_file = mlir_file
        self.tmp_dir = tmp_dir # the directory to save the intermediate files
        self.log_file = log_file
        self.cov_file = cov_file
        self.applied_opts = [] # all passes in lowering process
        self.stacktrace = {}
    
    @abstractmethod
    def lowering(self,lowering_state):
        """Each strategy class will implement this method."""
        pass

    def log_info(self, message):
        """Common function to log information to a log file."""
        util.append_content_to_file(self.log_file, message + '\n')

    def is_empty(self, file_path):
        """Check if a file is empty."""
        content = util.get_file_content(file_path).strip()
        if len(content) == 0 or content =='module {\n}' or content == '\n':
            return True
        return False

    def scan_mlir(self, tmp_mlir_file):
        """Obtain all ops in a tmp_mlir_file."""
        ops = set()
        dialects = set()
        cov_ops = set()
        cov_dialects = set()
        cmd = (' ').join([self.config.scan_func, tmp_mlir_file])
        res = util.execmd(cmd)
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
                self.log_info(f'[scan_mlir]: scan file {tmp_mlir_file},line: {line} has error')
                op = ''
            if dialect == 'builtin' or dialect == 'llvm':
                continue
            ops.add(op)
            if not dialect == 'func':
                dialects.add(dialect)
        # self.log_info(f'[scan_mlir] current ops: {ops}')
        if self.config.cov:
            util.append_content_to_file(self.cov_file, f'dialects: {(" ").join(cov_dialects)}\n')
            util.append_content_to_file(self.cov_file, f'ops: {(" ").join(cov_ops)}\n')
        return ops,dialects


    def update_op_priority(self, op, step=1):
        if op in self.config.op_priority:
            self.config.op_priority[op] = max(self.config.op_priority[op] - step, 0)
            if self.config.tensor_config.get(op, None) is not None:
                self.config.op_priority[self.config.tensor_config[op]] = max(self.config.op_priority[self.config.tensor_config[op]] - step, 0)
        else:
            parts = op.split(".")
            if len(parts) == 2:
                dialect = parts[0] + ".op"
                if dialect in self.config.op_priority:
                    self.config.op_priority[dialect] = max(self.config.op_priority[dialect] - step, 0)
                else:
                    self.config.op_priority[op] = -1
            else:
                self.config.op_priority[op] = -1

    # lowering(conversion) must follow the priority
    def pick_one_op(self, ops):
        assert len(ops) > 0, '[pick_one_op] ops is empty!'
        if isinstance(ops, set):
            ops = list(ops)
        op_with_priority = []
        for op in ops:
            priority = self.config.op_priority.get(op, None)
            if priority is None:
                parts = op.split(".")
                if len(parts) == 2:
                    dialect = parts[0] + ".op"
                    priority = self.config.op_priority.get(dialect, -1)
                else:
                    priority = -1
            op_with_priority.append((op, priority))
        self.log_info(f'[pick_one_op], op_with_priority: {op_with_priority}')
        # print(f'[pick_one_op], op_with_priority: {op_with_priority}')
        max_priority = max(priority for _, priority in op_with_priority)
        top_ops = [op for op, priority in op_with_priority if priority == max_priority]
        return random.choice(top_ops)

        # assert len(ops) > 0,'[pick_one_op] ops is empty!'
        # priority_list = [
        # ["tosa.conv2d", "tosa.matmul", "tosa.transpose", "tosa.conv3d", "tosa.fully_connected", 
        #     "tosa.depthwise_conv", "tosa.maxpool_2d", "tosa.avg_pool2d"],
        # ["tosa.cond_if", "tosa.while_loop", "tosa.yield"],
        # ['tosa']
        # ]
        # if isinstance(ops, set):
        #     ops = list(ops)
        # # Prioritize converting dialects in the priority_list.
        # for priority_ops in priority_list:
        #     available_ops = [op for op in ops if any(op.startswith(p) for p in priority_ops)]
        #     if available_ops:
        #         return random.choice(available_ops)
        # # Convert other dialects
        # available_dialect_ops = [
        #     op for op in ops
        #     if not (op.startswith('func') or op.startswith('cf')) or op in ['func.op.tensor', 'func.func.tensor']
        # ]        
        # if available_dialect_ops:
        #     return random.choice(available_dialect_ops)
        # # Last step, convert func.op  and cf.op
        # available_func_ops = [op for op in ops if op.startswith('func') or op.startswith('cf')]
        # if available_func_ops:
        #     return random.choice(available_func_ops)
        # return random.choice(ops)

    def construct_func_option(self, pass_name):
        if pass_name.startswith('-'):
            return pass_name
        else:
            return f'-pass-pipeline="builtin.module(func.func({pass_name}))"'


    def pick_one_conversion_pass(self, op, op_pass_dict):
        dialect,op_name = op.split('.',1)
        all_op = f'{dialect}.op'
        if op in op_pass_dict:
            return self.construct_func_option(random.choice(op_pass_dict[op]))
        elif all_op in op_pass_dict:
            return self.construct_func_option(random.choice(op_pass_dict[all_op]))
        elif op in self.config.unconverted_ops:
            return ''
        else:
            assert False, f'[pick_one_conversion_pass Warning] need add pass for op {op}'


    def pick_optimization_pass(self, ops):
        """Pick opt_num optimization passes. They contain specific and general optimization."""
        assert len(ops) > 0,'[pick_optimization_pass] ops is empty!'
        current_opt = {}
        results = []
        for op in ops: 
            dialect = op.split('.',1)[0]
            # if dialect in ['affine', 'linalg']:
            #     continue
            new_op = f'{dialect}.op'
            if new_op in self.config.specific_opts:
                current_opt.update(self.config.specific_opts[new_op])
        current_opt.update(self.config.general_opts)
        if len(current_opt) == 0:
            return (' ').join(results)
        # opt_num = 1
        # opt_num = util.random_int(self.config.opt_minnum, self.config.opt_maxnum)
        opt_num = util.random_int(self.config.opt_minnum, len(current_opt))
        # self.log_info(f'[pick_optimization_pass] pick {opt_num} optimization options while {len(current_opt)} options are available')
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
        
    def apply_optimization(self, result_file, process_file, ops):
        next_generalopt = self.pick_optimization_pass(ops)
        if next_generalopt.strip() == '':
            return process_file
        # print(f'[apply_optimization] pick optimization options: {next_generalopt}')
        cmd = (' ').join([self.config.mlir_opt, process_file, next_generalopt, '1>', f'{result_file}', '2>', f'{result_file}.err' ])
        self.log_info(f'[apply_optimization] {next_generalopt}')
        self.log_info(f'[apply_optimization] {cmd}')
        start_time = time.time()
        util.execmd(cmd)
        end_time = time.time()
        total_time = end_time - start_time
        # self.log_info(f"Command executed in {total_time:.3f} seconds")
        if self.is_empty(f'{result_file}.err'):
            if not self.is_empty(result_file):
                # print(f'[apply_optimization] {result_file} is empty after {next_generalopt}')
                self.applied_opts.append(next_generalopt)
                process_file = result_file
        else:
            self.log_dump(f'{result_file}.err',next_generalopt, process_file )
        return process_file

    
    def apply_conversion(self, result_file, process_file, ops):
        # pick one op and one op conversion pass
        next_op = self.pick_one_op(ops)
        next_pass = self.pick_one_conversion_pass(next_op,self.config.op_convert_dict)
        # if next_op == 'cf.br' or next_op == 'cf.cond_br':
        #     next_pass = random.choice(['-convert-cf-to-llvm', '-convert-func-to-llvm'])
        if next_pass == '':
            # print(f'[Warning] There are ops that cannot be converted, current ops: {next_op}')
            return process_file
        cmd = (' ').join([self.config.mlir_opt, process_file, next_pass, '1>', result_file, '2>', f'{result_file}.err' ])
        self.log_info(f'[apply_conversion] {cmd}')
        self.log_info(f'[apply_conversion] {next_op} {next_pass}')
        util.execmd(cmd)
        if self.is_empty(f'{result_file}.err'):
            self.applied_opts.append(next_pass)
            process_file = result_file
        else:     
            self.log_dump(f'{result_file}.err',next_pass, process_file )
            if self.config.op_priority_option:
                self.update_op_priority(next_op)
            if self.config.deduplicateprint_option:
                if self.should_deduplicate(f'{result_file}.err'):
                    process_file = self.remove_duplicate_cast_and_print(process_file)
                    self.log_info(f'after remove duplicate cast and print, process_file: {process_file}')
        return process_file

    def should_deduplicate(self, error_file):
        error_message = util.get_file_content(error_file)
        return "'tensor.cast' op not bufferizable under the given constraints: cannot avoid RaW conflict" in error_message
    
    def remove_duplicate_cast_and_print(self, mid_file):
        # remove the duplicate cast and print to bufferize the tensor
        file_name = os.path.basename(self.mlir_file)
        deduplicated_mlir_file = os.path.join(self.tmp_dir, util.random_file_prefix(file_name))
        cmd = (' ').join([self.config.deduplicate_print, mid_file, '1>', f'{deduplicated_mlir_file}', '2>', f'{deduplicated_mlir_file}.err' ])
        self.log_info(f'[remove_duplicate_cast_and_print] {cmd}')
        util.execmd(cmd)
        if not self.is_empty(f'{deduplicated_mlir_file}.err'):
            self.log_info(f'[remove_duplicate_cast_and_print Failed] {util.get_file_content(f"{mid_file}.err")}')
            return mid_file
        return deduplicated_mlir_file

    def log_dump(self, result_error_file, pass_name, process_file):
        if self.is_empty(result_error_file):
            return 
        error_message = util.get_file_content(result_error_file)
        if not 'PLEASE submit a bug report' in error_message:
            return
        crash_key = self.get_crash_key(error_message)
        # if not crash_key in self.config.stacktrace:
        save_process_file = self.config.crash_dir + os.path.basename(process_file)
        util.append_content_to_file(save_process_file, util.get_file_content(process_file) )
        self.log_info('[Stack Dump] ' + error_message + '\n')
        self.log_info( f'[crashed options] {pass_name}\n')
        self.log_info('[crashed file] ' + save_process_file  + '\n')
        if crash_key == "":
            self.log_info('[crash key] None {error_message}')
        else:
            self.stacktrace.setdefault(crash_key, [save_process_file]).append(save_process_file)
    
    def get_crash_key(self, error_message):
        lines = error_message.split('\n')
        for line in lines:
            if 'mlir-opt:' in line:
                return line.replace('Testing: ', '')
        return self.config.process_stacktrace(lines)



    def runner_dump(self, result_error_file):
        if not self.is_empty(result_error_file):
            error_message = util.get_file_content(result_error_file)
            if 'Program arguments: //MLIR/llvm-release/llvm-project/build/bin/mlir-cpu-runner' in error_message:
                return True
                    
    def log_execution_cmd(self, lowering_result = LoweringResult.NORMAL):
        all_cmd = (' ').join([self.config.mlir_opt,self.mlir_file, (f' | {self.config.mlir_opt} ').join(self.applied_opts), '|', 'timeout 10', self.config.jit_runner, self.config.jit_arg ])
        self.log_info( f'LoweringResult: {lowering_result}\n Total cmd: {all_cmd}')
        return all_cmd
        

    def execute_mlir(self, process_file):
        file_name = os.path.basename(self.mlir_file)
        self.applied_opts.append('-reconcile-unrealized-casts')
        res_file = os.path.join(self.tmp_dir, file_name+'.re')
        conv_err_file = os.path.join(self.tmp_dir, file_name+'.conv.err')
        err_file = os.path.join(self.tmp_dir, file_name+'.err')
        cmd = (' ').join([self.config.mlir_opt, process_file, ' -reconcile-unrealized-casts', '2>', conv_err_file, '|', 'timeout 10', self.config.jit_runner, self.config.jit_arg, '1>', res_file, '2>', err_file  ])
        # self.log_info(f'[execute_mlir] {cmd}')
        util.execmd(cmd)
        if self.is_empty(res_file):
            if not self.is_empty(conv_err_file):
                self.log_execution_cmd(LoweringResult.CONVERT_ERROR)
                return LoweringResult.CONVERT_ERROR
            elif self.is_empty(err_file):
                self.log_execution_cmd(LoweringResult.TIMEOUT)
                return LoweringResult.TIMEOUT
            elif not self.is_empty(err_file):
                if self.runner_dump(err_file):
                    self.log_execution_cmd(LoweringResult.EXECUTE_ERROR)
                    return LoweringResult.EXECUTE_ERROR
                else:
                    self.log_execution_cmd(LoweringResult.CONVERT_ERROR)
                    return LoweringResult.CONVERT_ERROR
            else:
                self.log_execution_cmd(LoweringResult.NORESULT)
                return LoweringResult.NORESULT
        final_result = util.get_file_content(res_file)
        final_err = util.get_file_content(err_file)
        all_cmd = self.log_execution_cmd()
        self.log_info(f'final_result: {final_result}\n final_error: {final_err}')
        return LoweringResult.NORMAL,final_result, all_cmd




    

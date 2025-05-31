from BaseLoweringStrategy import BaseLoweringStrategy
from BaseLoweringStrategy import LoweringState
import os
import util
from util import LoweringResult
import random



'''
1. Optimization and Conversion alternatly performed.
2. if conversion, first select an op, then select the conversion pass
3. if optimization, collect all ops and corresponding optimization pass, randomly select optimization passes
4. repeat this process util all ops have been converted to llvm or the number of passes has reach the limit.
'''
class StatefulLowering(BaseLoweringStrategy):
    def lowering(self, lowering_state):
        process_file = self.mlir_file
        # randomly select the start point
        if lowering_state.empty_state():
            lowering_state.init(process_file)
            # lowering_state.current_state()
        else:
            process_file,self.applied_opts = lowering_state.pick_one_state()
            self.log_info(f'lowering start from {process_file}, the applied_opts is {self.applied_opts}')
            # print(f'lowering start from {process_file}, the applied_opts is {self.applied_opts}')
        ops,dialects = self.scan_mlir(process_file)
        from_dialects = dialects
        file_name = os.path.basename(self.mlir_file)
        while len(ops) != 0 and len(self.applied_opts) < self.config.max_applied_optnnum:
            opt_result_mlir_file = os.path.join(self.tmp_dir, util.random_file_prefix(file_name))
            process_file = self.apply_optimization(opt_result_mlir_file,process_file, ops)
            ops,to_dialects = self.scan_mlir(process_file)
            # self.log_info(f'[StatefulLowering] Current applied opts: {len(self.applied_opts)}')
            from_dialects = to_dialects
            if len(ops) == 0:
                print(f'Already convert all ops.')
                break
            else:
                conv_result_mlir_file = os.path.join(self.tmp_dir, util.random_file_prefix(file_name))
                process_file = self.apply_conversion(conv_result_mlir_file, process_file, ops)
                ops,to_dialects = self.scan_mlir(process_file)
                lowering_state.update(process_file,self.applied_opts, from_dialects, to_dialects)
                # lowering_state.current_state()
                from_dialects = to_dialects
        # self.log_info(lowering_state.current_state())
        if len(ops) != 0 and len(self.applied_opts) >= self.config.max_applied_optnnum:
            self.log_execution_cmd(LoweringResult.CONVERT_ERROR)
            return LoweringResult.CONVERT_ERROR, process_file
        for op in ops:
            if op in self.config.unsupported_ops:
                # print(f'[Warning] There exits op that cannot be converted, unsupported op: {op}')
                self.log_execution_cmd(LoweringResult.CONVERT_ERROR)
                return LoweringResult.CONVERT_ERROR, process_file
        return LoweringResult.NORMAL, process_file
        pass


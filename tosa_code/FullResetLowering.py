from BaseLoweringStrategy import BaseLoweringStrategy
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
class FullResetLowering(BaseLoweringStrategy):
    def lowering(self,lowering_state):
        process_file = self.mlir_file
        ops,_ = self.scan_mlir(process_file)
        # self.log_info(f'[FullResetLowering] Current ops: {ops}')
        file_name = os.path.basename(self.mlir_file)
        conversion_step = 0
        while len(ops) != 0 and conversion_step < self.config.max_applied_optnnum:
            # self.log_info(f'[FullResetLowering] Current applied opts: {len(self.applied_opts)}')
            opt_result_mlir_file = os.path.join(self.tmp_dir, util.random_file_prefix(file_name))
            process_file = self.apply_optimization(opt_result_mlir_file,process_file, ops)
            ops,_ = self.scan_mlir(process_file)
            # self.log_info(f'[FullResetLowering] Current ops: {ops}')
            if len(ops) == 0:
                # print(f'Already convert all ops.')
                break
            else:
                conversion_step += 1
                # self.log_info(f'[FullResetLowering] Current ops: {ops}')
                conv_result_mlir_file = os.path.join(self.tmp_dir, util.random_file_prefix(file_name))
                process_file = self.apply_conversion(conv_result_mlir_file, process_file, ops)
                ops,_ = self.scan_mlir(process_file)

        if len(ops) != 0 and conversion_step >= self.config.max_applied_optnnum:
            # self.log_info(f'[FullResetLowering] Current applied opts: {len(self.applied_opts)}')
            self.log_info('[FullResetLowering] Already reach the max applied opts.')
            self.log_execution_cmd(LoweringResult.CONVERT_ERROR)
            return LoweringResult.CONVERT_ERROR, process_file
        for op in ops:
            if op in self.config.unsupported_ops:
                print(f'[Warning] There exits op that cannot be converted, unsupported op: {op}')
                self.log_execution_cmd(LoweringResult.CONVERT_ERROR)
                return LoweringResult.CONVERT_ERROR, process_file
        return LoweringResult.NORMAL, process_file
        pass


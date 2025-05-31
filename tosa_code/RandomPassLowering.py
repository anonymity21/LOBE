from BaseLoweringStrategy import BaseLoweringStrategy
import os
import util
from util import LoweringResult


'''
1. randomly decide to select conversion or optimization.
2. if conversion, first select an op, then select the conversion pass
3. if optimization, collect all ops and corresponding optimization pass, randomly select optimization passes
4. repeat this process util all ops have been converted to llvm or the number of passes has reach the limit.
'''
class RandomPassLowering(BaseLoweringStrategy):
    def lowering(self,lowernig_record):
        process_file = self.mlir_file
        ops = self.scan_mlir(process_file)
        file_name = os.path.basename(self.mlir_file)
        while len(ops) != 0 and len(self.applied_opts) < self.config.max_applied_optnnum:
            result_mlir_file = os.path.join(self.tmp_dir, util.random_file_prefix(file_name+'.'))
            which_pass = util.random_int(0,1)
            # perform optimization
            if which_pass == 0:
                process_file = self.apply_optimization(result_mlir_file,process_file, ops)
                ops = self.scan_mlir(process_file)
            else:
                conv_result_mlir_file = os.path.join(self.tmp_dir, util.random_file_prefix(file_name+'.'))
                process_file = self.apply_conversion(conv_result_mlir_file, process_file, ops)
                ops = self.scan_mlir(process_file)
        for op in ops:
            if op in self.config.unsupported_ops:
                print(f'[Warning] There exits op that cannot be converted, unsupported op: {op}')
                return LoweringResult.CONVERT_ERROR, process_file
        return LoweringResult.NORMAL, process_file
        pass

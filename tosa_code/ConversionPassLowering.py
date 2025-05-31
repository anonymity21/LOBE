from BaseLoweringStrategy import BaseLoweringStrategy
import os
import util
from util import LoweringResult


'''
1. Lowering a mlir file with only conversion pass
2. First randomly select an op, then select the conversion pass
4. Repeat this process util all ops have been converted to llvm or the number of passes has reach the limit.
'''
class ConversionPassLowering(BaseLoweringStrategy):
    def lowering(self,lowerig_record):
        process_file = self.mlir_file
        ops,_ = self.scan_mlir(process_file)
        self.log_info(f'[ConversionLowering] Current ops: {ops}')

        file_name = os.path.basename(self.mlir_file)
        while len(ops) != 0 and len(self.applied_opts) < self.config.max_applied_optnnum:
            conv_result_mlir_file = os.path.join(self.tmp_dir, util.random_file_prefix(file_name+'.'))
            process_file = self.apply_conversion(conv_result_mlir_file, process_file, ops)
            ops,_ = self.scan_mlir(process_file) 
            self.log_info(f'[ConversionLowering] Current ops: {ops}')

        for op in ops:
            if op in self.config.unsupported_ops:
                print(f'[Warning] There exits op that cannot be converted, unsupported op: {op}')
                self.log_execution_cmd(LoweringResult.CONVERT_ERROR)
                return LoweringResult.CONVERT_ERROR, process_file
        return LoweringResult.NORMAL, process_file
        pass

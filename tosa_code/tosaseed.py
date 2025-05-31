import os
import tempfile
from tqdm import tqdm 
import random
import util

def main():
    seed_dir = '//workspace/mlir-inconsistent/tosa_seed_v5_old'
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)
    for _ in tqdm(range(24), desc="Processing", unit="%", ncols=100): 
        mlir_file = os.path.join(seed_dir, util.random_file_prefix('tosa')) 
        func_num = 1
        iter_times = util.random_int(0,7)
        print_values = util.random_int(1,7)
        cmd = (' ').join(['//workspace/mlir-inconsistent/third_party_tools/tosa-smith-old', f'--f={func_num}', '--enable-control-flow', f'--max-loop-iter-times={iter_times}', f'-v={print_values}', '2>', mlir_file])
        # cmd = (' ').join(['//workspace/mlir-inconsistent/third_party_tools/tosa-smith', f'--f={func_num}',  f'-v={print_values}', '2>', mlir_file])
        util.execmd(cmd)

if __name__ == '__main__':
    main()

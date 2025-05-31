import os
from tqdm import tqdm 
import random
from multiprocessing import Pool
import subprocess
import argparse



dialects = ["arithsem", "linalggeneric", "tensor"]

parser = argparse.ArgumentParser()
parser.add_argument('--seed_dir', required=True, help='Directory to store generated MLIR files')
parser.add_argument('--total', type=int, default=8, help='Number of programs to generate for each dialect')
args = parser.parse_args()

os.makedirs(args.seed_dir, exist_ok=True)

def append_content_to_file(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()


def execmd(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Failed: ", e)
        return None


def random_file_prefix(prefix_name):
    random_bytes = os.urandom(8)  # Generate 8 random bytes
    random_str = random_bytes.hex()
    temp_filename = f"{prefix_name}.{random_str}.mlir"
    return temp_filename

def process_output(output):
    if not output:
        return
    lines = output.splitlines()
    expected_start = lines.index("--== Expected output:")
    expected_end = lines.index("--== End of output.")

    before_expected = "\n".join(lines[:expected_start])
    expected_content = "\n".join(lines[expected_start + 1:expected_end])
    return before_expected,expected_content

def generate_mlir(dialect):
    seed_dir = args.seed_dir
    total_num = args.total
    for i in  tqdm(range(total_num), desc=f"Generating for {dialect}", unit="%", ncols=100): 
        mlir_file = os.path.join(seed_dir, random_file_prefix(dialect)) 
        cmd = (' ').join([f'cabal run mlir-quickcheck -- -d={dialect}'])
        output = execmd(cmd)
        if not output:
            print("Error: No output from command.")
            continue
        before_expected, expected_content = process_output(output)
        if expected_content:
            append_content_to_file(mlir_file, before_expected)
            append_content_to_file(f'{mlir_file}.res', expected_content)

# python3 ratteseed4semantics.py --seed_dir=ratte_seed_v3 --total=8
def main():
    with Pool(processes=len(dialects)) as pool:
        pool.map(generate_mlir, dialects)

if __name__ == '__main__':
    main()
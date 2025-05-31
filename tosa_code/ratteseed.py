import os
from tqdm import tqdm 
import random
import subprocess


dialects = ["arith", "linalg", "tensor"]


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


def main():
    seed_dir = 'ratte_seed_v1'
    for dialect in dialects:
        for _ in tqdm(range(8), desc="Processing", unit="%", ncols=100): 
            mlir_file = os.path.join(seed_dir, random_file_prefix('ratte')) 
            cmd = (' ').join([f'cabal run mlir-quickcheck -- -d={dialect}'])
            output = execmd(cmd)
            if not output:
                print("Error: No output from command.")
                continue
            before_expected, expected_content = process_output(output)
            # append_content_to_file(mlir_file, output)
            append_content_to_file(mlir_file, before_expected)
            # if expected_content:
            #     append_content_to_file(f'{mlir_file}.res', expected_content)



if __name__ == '__main__':
    main()
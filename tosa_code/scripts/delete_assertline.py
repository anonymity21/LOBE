def remove_cf_assert_lines(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            if not line.strip().startswith('cf.assert'):
                output_file.write(line)

input_path = '//workspace/mlir-inconsistent/a.mlir' 
output_path = '//workspace/mlir-inconsistent/a.before.mlir' 

remove_cf_assert_lines(input_path, output_path)

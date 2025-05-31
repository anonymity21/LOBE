import json

convert_opt_file = '//workspace/mlir-inconsistent/mlir_conversion.json'
general_opt_file = '//workspace/mlir-inconsistent/mlir_generalopt.json'
specific_opt_file = '//workspace/mlir-inconsistent/mlir_specificopt.json'
test_opt_file = '//workspace/mlir-inconsistent/mlir_testopt.json'

all_opt_file = '//workspace/mlir-inconsistent/cmd/transform.json'
current_opts = set()
all_opts = set()

remain_opts = {}

# load the options in conversion.json
with open(convert_opt_file, 'r') as json_file:
    op_convert_dict = json.load(json_file)
for value in op_convert_dict.values():
    current_opts.add(value[0])    

# load the options in generalopt.json

with open(general_opt_file, 'r') as json_file:
    general_opts = json.load(json_file)

for key in general_opts.keys():
    current_opts.add(key)  

# load the options in specificopt.json
with open(specific_opt_file, 'r') as json_file:
    specific_opts = json.load(json_file)
for value_dict in specific_opts.values():
    for key in value_dict.keys():   
        current_opts.add(key)  

# load the options in transform.json
with open(all_opt_file, 'r') as json_file:
    all_opts_dict = json.load(json_file)
for key in all_opts_dict.keys():
    all_opts.add(key)  


# find the options that are in transform.json but not in the first three jsons

for item in all_opts:
    if item not in current_opts:
        remain_opts[item] = all_opts_dict[item]




with open(test_opt_file, 'w') as json_file:
    json.dump(remain_opts,json_file, indent=4)
import json

def load_bug_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"{json_path} contains {len(data)} bugs.")
    return data

# 加载完整数据（dict）
data_a = load_bug_data('//workspace/mlir-inconsistent/cov_collection/mergebug/tosa.json')
data_b = load_bug_data('//workspace/mlir-inconsistent/cov_collection/mergebug/ratte.json')

# 提取 key 集合
set_a = set(data_a.keys())
set_b = set(data_b.keys())

only_a = set_a - set_b
only_b = set_b - set_a
both = set_a & set_b

# print("\n[Only in TOSA]")
# print(only_a)
# print("\n[Only in Ratte]")
# print(only_b)

print("\n[In Both]")
for key in both:
    print(f"Key: {key}")
    print(f"  Value in TOSA : {data_a.get(key)}")
    print(f"  Value in Ratte: {data_b.get(key)}")

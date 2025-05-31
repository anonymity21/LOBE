import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

def load_bug_keys(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))
    return set(data.keys())


set_a = load_bug_keys('//workspace/mlir-inconsistent/cov_collection/mergebug/tosa.json')
set_b = load_bug_keys('//workspace/mlir-inconsistent/cov_collection/mergebug/ratte.json')

plt.figure(figsize=(2, 2))  
venn = venn2([set_a, set_b], set_labels=('', ''))  

patch_ids = ['10', '01', '11']
colors = ['#a6bddb', '#b2e2e2', '#9ecae1']
for pid, color in zip(patch_ids, colors):
    patch = venn.get_patch_by_id(pid)
    if patch:
        patch.set_color(color)
        patch.set_alpha(0.8)

for text in venn.subset_labels:
    if text:
        text.set_fontsize(20) 
# plt.title("Venn Diagram of Bug Detection")
plt.savefig("venn_bug_detection.png", dpi=300)

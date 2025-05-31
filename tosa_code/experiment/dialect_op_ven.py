import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

def load_bug_keys(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"{json_path} contains {len(data)} items.")
    return set(data)

def save_pic(set_a, set_b, set_c, category):
    # plt.figure(figsize=(2,2))
    # venn = venn3([set_a, set_b, set_c],set_labels=('', '', ''))

    only_in_a = set_a - set_b - set_c
    print(f"Only in set_a: {only_in_a}")
    only_in_b = set_b  - set_c
    print(f"Only in set_b: {only_in_b}")

    # patch_ids = ['100', '010', '110', '001', '101', '011', '111']
    # colors = ['#cbd5e8', '#f4cae4', '#e6f5c9', '#fddbc7', '#d9f0d3', '#ece2f0', '#f2f2f2']  
    # # patch_ids = ['100', '010', '110', '001', '101', '011', '111']
    # # colors = ['#f2d7d5', '#d4e6f1', '#d1f2eb', '#d6eaf8', '#fdebd0', '#e5e8e8', '#ebdef0']  
    # for pid, color in zip(patch_ids, colors):
    #     patch = venn.get_patch_by_id(pid)
    #     if patch:
    #         patch.set_color(color)
    #         patch.set_alpha(0.8)
    #         patch.set_edgecolor('gray')       # 设置边界颜色
    #         patch.set_linewidth(0.8)          # 设置边界粗细


    # for text in venn.subset_labels:
    #     if text:
    #         text.set_fontsize(14)
    #         text.set_horizontalalignment('center')
    #         text.set_verticalalignment('center')

    # plt.tight_layout()
    # plt.savefig(f"venn_{category}.png", dpi=300)


for category in ['dialects', 'ops']:
    set_a = load_bug_keys(f"//workspace/mlir-inconsistent/cov_collection/mregedialectop/tosasmith_{category}.json")
    set_b = load_bug_keys(f"//workspace/mlir-inconsistent/cov_collection/mregedialectop/ratte_{category}.json")
    set_c = load_bug_keys(f"//workspace/mlir-inconsistent/cov_collection/mregedialectop/ratte_semantics_{category}.json")
    print(f"set_a: {len(set_a)}, set_b: {len(set_b)}, set_c: {len(set_c)}")
    save_pic(set_a, set_b, set_c, category)
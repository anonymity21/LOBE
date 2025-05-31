import matplotlib.pyplot as plt
from matplotlib_venn import venn2

venn_counts = (11, 3, 18)  # 左右不重合区域 & 中间交集

plt.figure(figsize=(5, 4))
venn = venn2(subsets=venn_counts, set_labels=('', ''))

for text in venn.subset_labels:
    if text:
        text.set_fontsize(16)


venn.get_patch_by_id('10').set_color('#bdd7e7')  # 蓝
venn.get_patch_by_id('01').set_color('#c7e9c0')  # 绿
venn.get_patch_by_id('11').set_color('#a6d96a')  # 中间交集
for pid in ('10', '01', '11'):
    venn.get_patch_by_id(pid).set_alpha(0.5)

plt.text(-0.7, -0.5, r'$\it{DESIL}_{smith}$', fontsize=14, style='italic')
plt.text(0.45, -0.5, r'$\it{DESIL}_{smith}^{w/o\ opt}$', fontsize=14, style='italic')

plt.axis('off')
plt.tight_layout()
plt.savefig("venn_desil_smith.png", dpi=300)
plt.show()

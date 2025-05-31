from graphviz import Digraph
import networkx as nx
import json
import os

def draw_lowering_tree(paths, output_file="lowering_tree"):
    dot = Digraph(comment="Dialect Lowering Tree", format="png")
    dot.attr(rankdir='TB')  # Top-to-Bottom layout
    dot.attr('node', shape='box', style='filled', color='lightgray')

    for src, dst, label in paths:
        dot.node(src)
        dot.node(dst)
        dot.edge(src, dst, label=label)

    # 输出 png 图像和 dot 文件
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dot.render(output_file, cleanup=True)
    print(f"✅ 图已生成: {output_file}.png")


dialect_edges = [
    ("tosa.op", "arith.op", "-tosa-to-arith"),
    ("tosa.op", "linalg.op", "-tosa-to-linalg"),
    ("tosa.op", "scf.op", "-tosa-to-scf"),
    ("tosa.op", "tensor.op", "-tosa-to-tensor"),
    ("linalg.op", "scf.op", "-convert-linalg-to-loops"),
    ("linalg.op", "affine.op", "-convert-linalg-to-affine"),
    ("arith.op", "llvm", "-convert-arith-to-llvm"),
    ("scf.op", "cf.op", "-convert-scf-to-cf"),
    ("cf.op", "llvm", "-convert-cf-to-llvm"),
    ("vector.op", "llvm", "-convert-vector-to-llvm"),
    ("vector.op", "scf.op", "-convert-vector-to-scf"),
    # ("memref.op", "affine.op", "-convert-memref-to-affine"),
    ("memref.op", "llvm", "-expand-strided-metadata"),
    ("affine.op", "scf.op", "-lower-affine"),
    ("affine.op", "arith.op", "-lower-affine"),
    ("tensor.op", "memref.op", "-one-shot-bufferize"),
    ("bufferization.op", "memref.op", "-one-shot-bufferize"),
    ("index.op", "llvm", "-convert-index-to-llvm"),
    ("math.op", "llvm", "-convert-math-to-llvm")
    ]

operation_edges = [
    ("tosa.const", "arith.op", "-tosa-to-arith"),
    ("tosa.apply_scale", "arith.op", "-tosa-to-arith"),
    ("tosa.cond_if", "tosa.op", "-tosa-to-scf"),
    ("tosa.while_loop", "tosa.op", "-tosa-to-scf"),
    ("tosa.yield", "tosa.op", "-tosa-to-scf"),
    ("tosa.reshape", "tensor.op", "-tosa-to-tensor"),
    ("tosa.slice", "tensor.op", "-tosa-to-tensor"),
    ("tosa.pad", "tensor.op", "-tosa-to-tensor"),
    ("tosa.concat", "tensor.op", "-tosa-to-tensor"),
    # ("tosa.matmul", "linalg.op", "tosa-to-linalg-named"),
    # ("tosa.transpose", "linalg.op", "tosa-to-linalg-named"),
    ("tosa.op", "linalg.op", "tosa-to-linalg"),
    ("arith.op.tensor", "arith.op", "-one-shot-bufferize"),
    ("arith.op", "llvm", "-convert-arith-to-llvm"),
    ("scf.op", "cf.op", "-convert-scf-to-cf"),
    ("cf.op", "llvm", "-convert-cf-to-llvm"),
    ("vector.op", "llvm", "-convert-vector-to-llvm"),
    ("vector.transfer_write", "scf.op", "-convert-vector-to-scf"),
    ("vector.transfer_read", "scf.op", "-convert-vector-to-scf"),
    ("memref.op", "llvm", "-finalize-memref-to-llvm"),
    ("memref.subview", "affine.apply", "-expand-strided-metadata"),
    ("memref.collapse_shape", "affine.apply", "-expand-strided-metadata"),
    ("memref.expand_shape", "affine.apply", "-expand-strided-metadata"),
    ("linalg.op", "scf.op", "-convert-linalg-to-loops"),
    ("linalg.op", "affine.op", "-convert-linalg-to-affine-loops"),
    ("linalg.op", "scf.op", "-convert-linalg-to-parallel-loops"),
    ("linalg.op.tensor", "linalg.op", "-one-shot-bufferize"),
    ("affine.op", "scf.op", "-lower-affine"),
    ("affine.apply", "arith.op", "lower-affine"),
    ("tensor.op", "memref.op", "-one-shot-bufferize"),
    ("tensor.pad", "linalg.op", "-convert-tensor-to-linalg"),
    # ("bufferization.op", "memref.op", "-one-shot-bufferize"),
    # ("func.op.tensor", "llvm", "-one-shot-bufferize"),
    ("index.op", "llvm", "-convert-index-to-llvm"),
    ("math.op", "llvm", "-convert-math-to-llvm"),
    ("math.erf", "math.op", "-test-math-polynomial-approximation")
]


def assign_priorities(edges):
    G = nx.DiGraph()
    for src, dst, _ in edges:
        G.add_edge(src, dst)

    reverse_G = G.reverse(copy=True)
    priority = {node: 0 for node in reverse_G.nodes}
    for node in nx.topological_sort(reverse_G):
        for succ in reverse_G.successors(node):
            priority[succ] = max(priority[succ], priority[node] + 1)
    return priority

# def assign_priorities(edges):
#     G = nx.DiGraph()
#     for src, dst, _ in edges:
#         G.add_edge(src, dst)

#     reverse_G = G.reverse(copy=True)
    
#     priority = {}
#     queue = [("llvm", 0)]
    
#     while queue:
#         node, level = queue.pop(0)
#         if node in priority and priority[node] <= level:
#             continue
#         priority[node] = level
#         for neighbor in reverse_G.neighbors(node):
#             queue.append((neighbor, level + 1))

#     return priority

def save_to_json(data, filename="options/dialect_op_priority.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ results saves to {filename}")


def main():
    # draw_lowering_tree(dialect_edges, output_file="lowering_tree")
    dialect_priority_map = assign_priorities(dialect_edges)
    operation_priority_map = assign_priorities(operation_edges)

    sorted_priority = dict(sorted(operation_priority_map.items(), key=lambda x: -x[1]))
    save_to_json(sorted_priority, "options/op_priority.json")

    sorted_priority = dict(sorted(dialect_priority_map.items(), key=lambda x: -x[1]))
    save_to_json(sorted_priority, "options/dialect_priority.json")             



    final_op_priority = {}
    for op, priority in operation_priority_map.items():
        dialect = op.split('.')[0]
        dialect_prio = dialect_priority_map.get(f'{dialect}.op', -1)
        combined = float(f"{dialect_prio}.{priority}")
        final_op_priority[op] = combined

    sorted_priority = dict(sorted(final_op_priority.items(), key=lambda x: -x[1]))
    save_to_json(sorted_priority)

if __name__ == '__main__':
    main()

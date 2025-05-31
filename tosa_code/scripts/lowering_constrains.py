from graphviz import Digraph

def draw_lowering_tree(paths, output_file="lowering_tree"):
    dot = Digraph(comment="Dialect Lowering Tree", format="png")
    dot.attr(rankdir='TB')  # Top-to-Bottom layout
    dot.attr('node', shape='box', style='filled', color='lightgray')

    edges = set()
    for path in paths:
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge not in edges:
                dot.edge(*edge)
                edges.add(edge)

    dot.render(output_file, cleanup=True)
    print(f"✅ Tree diagram saved to {output_file}.png")

if __name__ == "__main__":
    paths = [
        ["TOSA", "Tensor", "Linalg", "MemRef", "LLVM"],
        ["TOSA", "Tensor", "Arith", "MemRef", "LLVM"],
        ["SCF", "CFG", "LLVM"]
    ]
    draw_lowering_tree(paths)

from tree_sitter import Language, Parser
from pathlib import Path

CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')

def init_parser():
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)
    return parser

def extract_function_info(root_node, code_bytes):
    """
    Extract the mapping of parameter types to template call types from the AST root node.
    """
    result_dict = {}

    def visit_node(node):
        if node.type == 'function_definition':
            name_node = node.child_by_field_name('declarator')
            if name_node and b"matchAndRewrite" in code_bytes[name_node.start_byte:name_node.end_byte]:
                # 1. extract parameter types (ignoring PatternRewriter)
                param_types = set()
                params_node = name_node.child_by_field_name('parameters')
                if params_node:
                    for param in params_node.children:
                        if param.type == 'parameter_declaration':
                            type_node = param.child_by_field_name('type')
                            if type_node:
                                type_str = code_bytes[type_node.start_byte:type_node.end_byte].decode().strip()
                                if "PatternRewriter" in type_str or "OpAdaptor" in type_str:
                                    continue
                                param_types.add(type_str)

                # 2. extract called template types
                #    (e.g., create<...> or replaceOpWithNewOp<...>)
                called_types = set()
                def extract_templates(subnode):
                    if subnode.type == 'call_expression':
                        func_node = subnode.child_by_field_name('function')
                        if func_node:
                            func_text = code_bytes[func_node.start_byte:func_node.end_byte].decode()
                            if 'create<' in func_text or 'replaceOpWithNewOp<' in func_text:
                                start = func_text.find('<') + 1
                                end = func_text.find('>')
                                if start > 0 and end > start:
                                    tmpl_type = func_text[start:end].strip()
                                    called_types.add(tmpl_type)
                    for child in subnode.children:
                        extract_templates(child)
                extract_templates(node)

                # 3. construct the mapping of param_type → called_type
                for param in param_types:
                    if param not in result_dict:
                        result_dict[param] = set()
                    result_dict[param].update(called_types)

        for child in node.children:
            visit_node(child)

    visit_node(root_node)
    return result_dict

def parse_cpp_file(file_path: Path, parser):
    code_bytes = file_path.read_bytes()
    tree = parser.parse(code_bytes)
    root = tree.root_node
    return extract_function_info(root, code_bytes)

def parse_directory(file_list):
    parser = init_parser()

    for cpp_file in file_list:
        print(f"\n📄 Parsing file: {cpp_file}")
        file_result = parse_cpp_file(cpp_file, parser)
        print_result_per_file(cpp_file, file_result)

def print_result_per_file(file_path, result_dict):
    if not result_dict:
        print("  (No matchAndRewrite found)")
        return
    for param_type, templates in sorted(result_dict.items()):
        print(f"  {param_type}:")
        for t in sorted(templates):
            print(f"    - {t}")


if __name__ == "__main__":
    # target_dir = Path("//workspace/llvm-release/llvm-project/mlir/lib/Conversion")
    cpp_files = [
        Path("//workspace/llvm-release/llvm-project/mlir/lib/Conversion/AffineToStandard/AffineToStandard.cpp"),
        Path("//workspace/llvm-release/llvm-project/mlir/lib/Conversion/VectorToSCF/VectorToSCF.cpp"),
        Path("//workspace/llvm-release/llvm-project/mlir/lib/Conversion/TosaToLinalg/TosaToLinalgNamed.cpp"),
        Path("//workspace/llvm-release/llvm-project/mlir/lib/Conversion/TosaToLinalg/TosaToLinalg.cpp"),
        Path("//workspace/llvm-release/llvm-project/mlir/lib/Conversion/TosaToSCF/TosaToSCF.cpp"),
        Path("//workspace/llvm-release/llvm-project/mlir/lib/Conversion/TosaToArith/TosaToArith.cpp"),
        Path("//workspace/llvm-release/llvm-project/mlir/lib/Conversion/TosaToTensor/TosaToTensor.cpp"),
    ]
       
    parse_directory(cpp_files)


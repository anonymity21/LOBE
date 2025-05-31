def filter_log_blocks(input_log, output_log):
    with open(input_log, 'r') as f:
        lines = f.readlines()

    blocks = []
    current_block = []
    for line in lines:
        if line.startswith("---- Process"):
            if current_block:
                blocks.append(current_block)
                current_block = []
        current_block.append(line)
    if current_block:
        blocks.append(current_block)  # 添加最后一个块

    selected_blocks = []
    for block in blocks:
        has_not_same = any("Not Same" in line for line in block)
        has_same = any("Same Results!" in line for line in block)
        if has_not_same and not has_same:
            selected_blocks.append(block)

    # 写入输出文件
    with open(output_log, 'w') as out:
        for block in selected_blocks:
            out.writelines(block)
            out.write("\n")  # 每个块之间加空行可读性更好

    print(f"✅ Done. Kept {len(selected_blocks)} block(s). Written to: {output_log}")

if __name__ == "__main__":
    from pathlib import Path

    log_dir = Path("//workspace/mlir-inconsistent/multiple_log/ratte_seed_v1/2025-04-02_19-46-59")  # 替换为你的目录路径
    log_files = list(log_dir.glob("*.log"))

    for log_file in log_files:
        filter_log_blocks(log_file, f"{log_file}_filtered.log")

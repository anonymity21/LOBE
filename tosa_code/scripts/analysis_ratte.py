def parse_log_and_compare(log_path):
    with open(log_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    blocks = []
    current_block = []
    extracting = False

    for line in lines:
        if line.startswith("Not Same in"):
            extracting = True
            current_block = [line]
        elif line.startswith("---- Process") and extracting:
            extracting = False
            blocks.append(current_block)
        elif extracting:
            current_block.append(line)
    i =0
    for block_id, block in enumerate(blocks):
        print(f"\n🔍 Block #{block_id + 1}")
        print("\n".join(block))
        print("===================================================")
        i += 1
        if i > 1:
            break

        # # 1. 提取 baseline
        # baseline = ""
        # for line in block:
        #     if line.startswith("results[0]:"):
        #         baseline = line.split("results[0]:", 1)[1].strip()
        #         break
        # baseline_lines = baseline.split("\\n")  # 假设 baseline 是字符串带换行符 `\n`

        # # 2. 提取每个 result（每个 11111 下一行）
        # results = []
        # i = 0
        # while i < len(block):
        #     if block[i] == "11111" and i + 1 < len(block):
        #         result_str = block[i + 1].strip()
        #         results.append(result_str)
        #         i += 2
        #     else:
        #         i += 1

        # # 3. 对每个结果与 baseline 按行比较
        # for idx, result_str in enumerate(results):
        #     result_lines = result_str.split("\\n")
        #     print(f"\n🧪 Comparing Result {idx + 1} with baseline:")

        #     max_len = max(len(result_lines), len(baseline_lines))
        #     for i in range(max_len):
        #         baseline_line = baseline_lines[i] if i < len(baseline_lines) else "<(missing)>"
        #         result_line = result_lines[i] if i < len(result_lines) else "<(missing)>"

        #         if baseline_line != result_line:
        #             print(f"❌ Line {i + 1} differs:")
        #             print(f"   Baseline → {baseline_line}")
        #             print(f"   Result   → {result_line}")
        #             break
        #         else:
        #             print(f"✅ Line {i + 1} matches")

if __name__ == "__main__":
    parse_log_and_compare("//workspace/mlir-inconsistent/multiple_log/ratte_seed_v1/2025-04-02_19-46-59/9911.log_filtered.log")  # 修改为你的路径

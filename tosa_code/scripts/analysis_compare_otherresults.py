def parse_and_save_inconsistent_results(log_path, output_path="inconsistent_final_results.log"):
    with open(log_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]

    blocks = []
    current_block = []
    for line in lines:
        if line.startswith("---- Process"):
            if current_block:
                blocks.append(current_block)
                current_block = []
        current_block.append(line)
    if current_block:
        blocks.append(current_block)

    
    
    for block in blocks:
        all_results = []  # 每项是 (process_header, result_str)
        process_header = block[0] if block else "---- Process (unknown)"
        extracting = False
        result_lines = []

        for line in block:
            if line.startswith("final_result:"):
                extracting = True
                result_lines = [line.split("final_result:", 1)[1].strip()]
            elif line.strip().startswith("final_error:") and extracting:
                extracting = False
                result_str = "\n".join(result_lines)
                all_results.append((process_header, result_str))
            elif extracting:
                result_lines.append(line)

        if not all_results:
            print("❗ No final results found.")
            return

        # 检查是否一致
        baseline = all_results[0][1].splitlines()
        inconsistent = []
        for hdr, res in all_results:
            res_lines = res.splitlines()
            diff_lines = []

            max_len = max(len(baseline), len(res_lines))
            for i in range(max_len):
                base_line = baseline[i] if i < len(baseline) else "<missing>"
                res_line = res_lines[i] if i < len(res_lines) else "<missing>"
                if base_line != res_line:
                    diff_lines.append(f"[Line {i+1}] Baseline: {base_line}\n           Result  : {res_line}")

            if diff_lines:
                inconsistent.append((hdr, diff_lines))

        if not inconsistent:
            print("✅ All final results are consistent.")
        else:
            print(f"❌ Found {len(inconsistent)} inconsistent result(s). Writing to {output_path}")
            with open(output_path, 'a+') as out:
                out.write("Baseline (from first process):\n")
                out.write("\n".join(baseline) + "\n\n")
                for i, (header, diff_lines) in enumerate(inconsistent):
                    out.write(f"{header}\n")
                    out.write(f"Result #{i+1} (differences):\n")
                    out.write("\n".join(diff_lines))
                    out.write("\n\n")

if __name__ == "__main__":

    from pathlib import Path
    log_dir = Path("//workspace/mlir-inconsistent/multiple_log/ratte_seed_v1/2025-04-02_19-46-59")  # 替换为你的目录路径
    log_files = list(log_dir.glob("*.log_otherresult.log"))

    for log_file in log_files:
        parse_and_save_inconsistent_results(log_file, f"{log_file}_notsame.log")
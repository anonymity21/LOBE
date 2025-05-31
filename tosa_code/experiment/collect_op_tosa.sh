#!/bin/bash

# 启动信息
echo "[INFO] Starting tosa batch runs..."

# 每个命令运行一次
python3 tosa_code/main_tosa.py --strategy=fullreset --multiprocess=24 --executiontime=0.5 --seedname=tosa_seed_v1 --deduplicateprint --priority --cov
python3 tosa_code/main_tosa.py --strategy=fullreset --multiprocess=24 --executiontime=0.5 --seedname=tosa_seed_v1 --deduplicateprint

python3 tosa_code/main_tosa.py --strategy=fullreset --multiprocess=24 --executiontime=0.5 --seedname=tosa_seed_v2 --deduplicateprint --priority --cov
python3 tosa_code/main_tosa.py --strategy=fullreset --multiprocess=24 --executiontime=0.5 --seedname=tosa_seed_v2 --deduplicateprint

python3 tosa_code/main_tosa.py --strategy=fullreset --multiprocess=24 --executiontime=0.5 --seedname=tosa_seed_v3 --deduplicateprint --priority --cov
python3 tosa_code/main_tosa.py --strategy=fullreset --multiprocess=24 --executiontime=0.5 --seedname=tosa_seed_v3 --deduplicateprint

python3 tosa_code/main_tosa.py --strategy=fullreset --multiprocess=24 --executiontime=0.5 --seedname=tosa_seed_v4 --deduplicateprint --priority --cov
python3 tosa_code/main_tosa.py --strategy=fullreset --multiprocess=24 --executiontime=0.5 --seedname=tosa_seed_v4 --deduplicateprint

python3 tosa_code/main_tosa.py --strategy=fullreset --multiprocess=24 --executiontime=0.5 --seedname=tosa_seed_v5 --deduplicateprint --priority --cov
python3 tosa_code/main_tosa.py --strategy=fullreset --multiprocess=24 --executiontime=0.5 --seedname=tosa_seed_v5 --deduplicateprint

echo "[INFO] All runs completed."

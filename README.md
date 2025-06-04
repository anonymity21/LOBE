# LOBE: Finding Miscompilations in MLIR via Lowering Space Exploration

## Directory Structure

```
├── cov_collection # Experiment log files (e.g., bugs and coverage)
├── data # Collected experiment results
│ ├── rq2/ # Data for RQ2
│ └── rq3/ # Data for RQ3
├── options # conversion and optimization passes used in project
├── seed # Directory of seeds
├── third_party_tools # External tools used in this project
│ ├── tosasmith/ # Third-party source code for the TOSA generator
│ └── scan-function/ # Tool for extracting MLIR function coverage
├── tosa_code # Source code of LOBE (core logic of the testing framework)
│   ├── experiment/   # Experiment scripts for RQ2 and RQ3
│   └── parser/   # Tools for extracting operation-to-pass mappings from MLIR conversion passes
└── tosasmith # Source code of external tools
```

### 1. Compile tosasmith

```
cd tosasmith
mkdir build && cd build

cmake ../llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DLLVM_CCACHE_BUILD=ON \
  -DCMAKE_INSTALL_PREFIX="../install"

ninja
```

After compilation, copy the necessary binaries to third_party_tools/:

```
cp ./build/bin/mlir-opt $path_to/third_party_tools/
cp ./build/bin/tosasmith $path_to/third_party_tools/
cp ./build/bin/mlir-scan $path_to/third_party_tools/
cp ./build/bin/mlir-deduplicate $path_to/third_party_tools/
cp ./build/lib/libmlir_c_runner_utils.so $path_to/third_party_tools/
cp ./build/lib/libmlir_runner_utils.so $path_to/third_party_tools/
cp ./build/lib/libmlir_async_runtime.so $path_to/third_party_tools/
```

### 2. Configure Python Environment

```
pip install -r requirements.txt
```


### 3. Run LOBE

You can run the LOBE testing framework using the following command:

```
python3 tosa_code/main_tosa.py --strategy=fullreset --executiontime=0.5 --priority --seedname=tosa_seed_v1
```
#### Arguments
- `--strategy`: Specifies the lowering strategy.
  - `fullreset`: Alternates between conversion and optimization passes.
  - `conversion`: Applies only conversion passes.
- `--executiontime`: Sets the execution timeout (in seconds) for each test case.
- `--priority`: Enable the feedback-based scheduling mechanism 
- `--seedname`: Specifies the seed directory name containing input MLIR programs.

#### Output

If any bugs (e.g., crashes or miscompilations) are found during testing, bugs will be saved in the `cov_collection/` directory for further investigation.

### 4. experiment code

#### RQ2: 

- crash bugs: tosa_code/experiment/crash_bug_calculate.py
- miscompilations: tosa_code/experiment/inconsistent_bug_calculate.py
- dialect/operation coverage: tosa_code/experiment/dialect_op_calculate.py
- line/branch coverage:  tosa_code/experiment/merge.py (To enable line coverage collection, you must use an instrumented build of `mlir-opt` with coverage instrumentation enabled.)

#### RQ3: 

- maxStep: tosa_code/experiment/result_calculate_limitedstep.py
- the number of success path and  lowering success rate: tosa_code/experiment/result_calculate.py

### 5. experiment results

#### RQ2:

- `bug_num/`: Contains details about discovered crash bugs and miscompilations, including input seeds and relevant traces.
- `code_coverage/`: Stores line and branch coverage statistics collected during the experiments.
- `dialect_opts/`: Records the optimization and conversion passes applied per dialect during the lowering process.

#### RQ3:

- `ablation.txt`: Presents results from the ablation study, comparing configurations **with** and **without** atomic lowering rules and feedback-driven scheduling.
- `maxStep.txt`: Shows the impact of varying the `maxStep` parameter on testing effectiveness, such as bug detection rate and coverage progression.






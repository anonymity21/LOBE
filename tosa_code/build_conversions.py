import json
class MLIRPassConfig:
    def __init__(self, dialect, dependencies):
        self.dialect = dialect
        self.dependencies = dependencies

    def add_dependency(self, ops, passes):
        self.dependencies.append({"OPS": ops, "PASS": passes})

    def to_dict(self):
        return {
            "dialect": self.dialect,
            "dependency": self.dependencies
        }

def build_conversion():
    all_config = []
    tosa_dependencies = [
        {"OPS": ["tosa.apply_scale"], "PASS": ["-tosa-to-arith=\"include-apply-rescale=true\""]},
        {"OPS": ["tosa.const"], "PASS": ["-tosa-to-arith"]},
        {"OPS": ["tosa.scatter", "tosa.cond_if", "tosa.while_loop", "tosa.yield"], "PASS": ["-tosa-to-scf"]},
        {"OPS": ["tosa.reshape", "tosa.slice", "tosa.pad", "tosa.concat"], "PASS": ["-tosa-to-tensor"]},
        {"OPS": ["tosa.conv2d", "tosa.matmul", "tosa.transpose", "tosa.conv3d", "tosa.fully_connected", 
        "tosa.depthwise_conv", "tosa.maxpool_2d", "tosa.avg_pool2d"], "PASS": ["tosa-to-linalg-named"]},
        {"OPS": ["tosa.add", "tosa.sub", "tosa.mul", "tosa.reciprocal", "tosa.bitwise_or", "tosa.logical_right_shift",
        "tosa.logical_left_shift", "tosa.bitwise_xor", "tosa.bitwise_and", "tosa.negate", "tosa.abs",
         "tosa.arithmetic_right_shift","tosa.int_div", "tosa.bitwise_not", "tosa.clz", "tosa.logical_and", "tosa.logical_not", "tosa.logical_or", "tosa.logical_xor",
         "tosa.pow", "tosa.rsqrt", "tosa.sin", "tosa.cos", "tosa.tanh","tosa.erf", "tosa.clamp", "tosa.sigmoid",
         "tosa.floor","tosa.ceil", "tosa.cast", "tosa.equal", "tosa.select", "tosa.maximum", "tosa.minimum", 
         "tosa.reduce_sum", "tosa.reduce_product", "tosa.reduce_min", "tosa.reduce_max",  "tosa.reduce_all", "tosa.reduce_any", "tosa.argmax",
         "tosa.exp", "tosa.log", "tosa.greater","tosa.greater_equal","tosa.tile","tosa.reverse","tosa.gather" ], "PASS": ["tosa-to-linalg"]}
    ]
    tosa_config = MLIRPassConfig(dialect="tosa", dependencies=tosa_dependencies)
    all_config.append(tosa_config.to_dict())

    tensor_dependencies = [
        {"OPS":[], "PASS":["-one-shot-bufferize=\"bufferize-function-boundaries\""]},
        {"OPS":["tensor.pad"], "PASS":["-convert-tensor-to-linalg"]}
    ]
    tensor_config = MLIRPassConfig(dialect="tensor", dependencies=tensor_dependencies)
    all_config.append(tensor_config.to_dict())

    buffer_dependencies = [
        {"OPS":[], "PASS":["-one-shot-bufferize=\"bufferize-function-boundaries\""]}
    ]
    buffer_config = MLIRPassConfig(dialect="bufferization", dependencies=buffer_dependencies)
    all_config.append(buffer_config.to_dict())


    func_dependencies = [
        {"OPS":[], "PASS":["-convert-func-to-llvm"]},
        {"OPS":["func.func.tensor"], "PASS":["-one-shot-bufferize=\"bufferize-function-boundaries\""]}

    ]
    func_config = MLIRPassConfig(dialect="func", dependencies=func_dependencies)
    all_config.append(func_config.to_dict())

    async_dependencies = [
        {"OPS":[], "PASS":["-convert-async-to-llvm"]}
    ]
    async_config = MLIRPassConfig(dialect="async", dependencies=async_dependencies)
    all_config.append(async_config.to_dict())

    scf_dependencies = [
        {"OPS":[], "PASS":["-convert-scf-to-cf"]}
    ]
    scf_config = MLIRPassConfig(dialect="scf", dependencies=scf_dependencies)
    all_config.append(scf_config.to_dict())

    cf_dependencies = [
        {"OPS":[], "PASS":["-convert-cf-to-llvm"]}
    ]
    cf_config = MLIRPassConfig(dialect="cf", dependencies=cf_dependencies)
    all_config.append(cf_config.to_dict())

    memref_dependencies = [
        {"OPS":[], "PASS":["-finalize-memref-to-llvm"]},
        {"OPS":["memref.subview", "memref.collapse_shape", "memref.expand_shape"], "PASS":["-expand-strided-metadata"]}
    ]
    memref_config = MLIRPassConfig(dialect="memref", dependencies=memref_dependencies)
    all_config.append(memref_config.to_dict())

    linalg_dependencies = [
        {"OPS":[], "PASS":["-convert-linalg-to-loops","-convert-linalg-to-affine-loops","-convert-linalg-to-parallel-loops"]},
        {"OPS":["linalg.op.tensor"], "PASS":["-one-shot-bufferize=\"bufferize-function-boundaries\""]}
        # {"OPS":["linalg.matmul", "linalg.batch_matmul", "linalg.batch_matmul_transpose_a",
        #  "linalg.batch_matmul_transpose_b", "linalg.matmul_transpose_b", "linalg.matmul_transpose_a"], "PASS":["-linalg-block-pack-matmul"]}
    ]
    linalg_config = MLIRPassConfig(dialect="linalg", dependencies=linalg_dependencies)
    all_config.append(linalg_config.to_dict())

    affine_dependencies = [
        {"OPS":[], "PASS":["-lower-affine"]}
    ]
    affine_config = MLIRPassConfig(dialect="affine", dependencies=affine_dependencies)
    all_config.append(affine_config.to_dict())


    arith_dependencies = [
        {"OPS":[], "PASS":["-convert-arith-to-llvm"]},
        {"OPS":["arith.op.tensor"], "PASS":["-one-shot-bufferize=\"bufferize-function-boundaries\""]}
    ]
    arith_config = MLIRPassConfig(dialect="arith", dependencies=arith_dependencies)
    all_config.append(arith_config.to_dict())

    index_dependencies = [
        {"OPS":[], "PASS":["-convert-index-to-llvm"]}
    ]
    index_config = MLIRPassConfig(dialect="index", dependencies=index_dependencies)
    all_config.append(index_config.to_dict())

    vector_dependencies = [
        {"OPS":[], "PASS":["-convert-vector-to-llvm"]},
        {"OPS":["vector.print.vector", "vector.transfer_write", "vector.transfer_read","vector.contract"], "PASS":["-convert-vector-to-scf"]}
    ]
    vector_config = MLIRPassConfig(dialect="vector", dependencies=vector_dependencies)
    all_config.append(vector_config.to_dict())

    math_dependencies = [
        {"OPS":[], "PASS":["-convert-math-to-llvm"]},
        {"OPS":["math.erf"], "PASS":["-test-math-polynomial-approximation"]}
    ]
    math_config = MLIRPassConfig(dialect="math", dependencies=math_dependencies)
    all_config.append(math_config.to_dict())

    op_pass_dict = {}
    for config in all_config:
        for dependency in config['dependency']:
            if len(dependency['OPS']) == 0:
                op_pass_dict[f'{config["dialect"]}.op'] = dependency['PASS']
            else:
                for op in dependency["OPS"]:
                    op_pass_dict[op] = dependency['PASS']
    # op_pass_dict['cf.br'] = ['-convert-func-to-llvm']
    with open('./options/mlir_conversion.json', 'w') as json_file:
        json.dump(op_pass_dict, json_file, indent=4)

    # with open('mlir_config.json', 'w') as json_file:
    #     json.dump(all_config, json_file, indent=4)


def build_specific_opt():
    specific_opt_dict = {}
    affine_opt = {  "--affine-data-copy-generate": [
        "generate-dma=false fast-mem-space=0 skip-non-unit-stride-loops",
        "generate-dma=false fast-mem-space=0 fast-mem-capacity=1",
        "generate-dma fast-mem-space=2 skip-non-unit-stride-loops",
        "generate-dma fast-mem-capacity=16 fast-mem-space=2"
    ],
    "--test-constant-fold": [],
    "--affine-loop-fusion": [
        "fusion-maximal",
        "mode=producer",
        "fusion-maximal mode=sibling",
        "mode=producer fusion-maximal",
        "fusion-compute-tolerance=0"
    ],
    "--affine-loop-tile": [
        "tile-size=32",
        "cache-size=512"
    ],
    "--affine-loop-unroll": [
        "unroll-full",
        "unroll-full unroll-full-threshold=2",
        "unroll-factor=4",
        "unroll-factor=2"
    ],
    "--affine-loop-unroll-jam": [
        "unroll-jam-factor=2"
    ],
    "--affine-parallelize": [
        "max-nested=1",
        "parallel-reductions=1"
    ],
    "--affine-super-vectorize": [
        "virtual-vector-size=128",
        "virtual-vector-size=32,256 test-fastest-varying=1,0",
        "virtual-vector-size=128 test-fastest-varying=0 vectorize-reductions=true"
    ],
    "--affine-super-vectorizer-test": [
        "forward-slicing=true",
        "backward-slicing=true",
        "slicing=true",
        "compose-maps",
        "vector-shape-ratio=4,8",
        "vector-shape-ratio=2,5,2",
        "vectorize-affine-loop-nest"
    ],
      "--test-affine-access-analysis": [],
    "--test-affine-data-copy": [
        "for-memref-region",
        "memref-filter"
    ],
    "--test-affine-loop-unswitch": [],
    "--test-affine-parametric-tile": [],
    "--test-affine-reify-value-bounds": [
        "reify-to-func-args",
        "use-arith-ops"
    ],
    "--test-loop-fusion": [
        "test-loop-fusion-dependence-check",
        "test-loop-fusion-slice-computation",
        "test-loop-fusion-transformation"
    ],
    "--test-loop-permutation": [
        "permutation-map=2,0,1"
    ],
    "--affine-scalrep": [],
    "--test-affine-walk": [],
    "--test-decompose-affine-ops": [],
    "--affine-expand-index-ops": [],
    "--affine-expand-index-ops-as-affine": [],
    "--affine-loop-coalescing": [],
    "--affine-loop-invariant-code-motion": [],
    "--affine-loop-normalize": [],
    "--affine-pipeline-data-transfer": [],
    "--affine-simplify-structures": []
    }
    specific_opt_dict['affine.op'] = affine_opt

    arith_opt = {    "--arith-expand": [
        "include-bf16"
    ],
    "--arith-unsigned-when-equivalent": [],   "--arith-emulate-unsupported-floats": [
        "source-types=bf16,f8E4M3FNUZ target-type=f32"
    ],
    "--arith-emulate-wide-int": [
        "widest-int-supported=32"
    ],
    "--test-arith-emulate-wide-int": [
        "widest-int-supported=8"
    ],
    "--test-emulate-narrow-int": [
        "arith-compute-bitwidth=8",
        "memref-load-bitwidth=8",
        "memref-load-bitwidth=32",
        "arith-compute-bitwidth=1 memref-load-bitwidth=32"
    ],
    "--arith-int-range-narrowing": [
        "int-bitwidths-supported=32"
    ],"--int-range-optimizations": []}
    specific_opt_dict['arith.op'] = arith_opt

    func_opt = {
        "--duplicate-function-elimination": [],
         "--test-func-erase-arg": [],
    "--test-func-erase-result": [],
    "--test-func-insert-arg": [],
    "--test-func-insert-result": [],
    "--test-func-set-type": [],
    "--test-function-pass": []
    }
    specific_opt_dict['func.op'] = func_opt
    linalg_opt = {
        "--convert-elementwise-to-linalg": [],
        # "--linalg-block-pack-matmul": [
        # "block-factors=32,16,64 lhs-transpose-outer-blocks=true lhs-transpose-inner-blocks=true rhs-transpose-outer-blocks=false rhs-transpose-inner-blocks=false",
        # "block-factors=32,16,64 allow-padding=1",
        # "block-factors=32,16,64 allow-padding=1 mnk-padded-multiples=256,512,384"
        # ],
        # "linalg-detensorize": [
        #     "aggressive-mode"
        # ],
        # "| //MLIR/llvm-release/llvm-project/build/bin/mlir-opt -pass-pipeline=\"builtin.module(func.func(linalg-detensorize{aggressive-mode}))\" | //MLIR/llvm-release/llvm-project/build/bin/mlir-opt" : [],
        "--linalg-fold-unit-extent-dims": [
            "use-rank-reducing-slices"
        ],
        "--linalg-fuse-elementwise-ops": [],
        "--linalg-generalize-named-ops": [],
        "--linalg-inline-scalar-operands": [],
        "--linalg-named-op-conversion": [],
        "--linalg-specialize-generic-ops": [],
         "--test-linalg-data-layout-propagation": [],
    "--test-linalg-decompose-ops": [
        "remove-dead-args-and-results"
    ],
    "--test-linalg-drop-unit-dims": [],
    "--test-linalg-greedy-fusion": [],
    "--test-linalg-pad-fusion": [],
    "--test-linalg-rank-reduce-contraction-ops": [],
    "--test-linalg-elementwise-fusion-patterns": [
        "collapse-dimensions-control=2,3",
        "fuse-with-reshape-by-collapsing",
        "fuse-with-reshape-by-collapsing-control",
        "fuse-generic-ops-control",
        "fuse-multiuse-producer",
        "control-fusion-by-expansion",
        "fuse-with-reshape-by-expansion"
    ],
    "--test-linalg-transform-patterns": [
        "test-bubble-up-extract-slice-op-pattern",
        "test-erase-unused-operands-and-results",
        "test-erase-unnecessary-inputs",
        "test-vector-transfer-forwarding-patterns",
        "test-generalize-pad-tensor",
        "test-generalize-tensor-unpack",
        "test-swap-subtensor-padtensor",
        "test-swap-extract-slice-with-fill-pattern",
        "test-patterns",
        "test-linalg-to-vector-patterns",
        "test-decompose-winograd-ops",
        "test-winograd-conv2d"
    ]
    }
    specific_opt_dict['linalg.op'] = linalg_opt

    llvm_opt = {
        "--ensure-debug-info-scope-on-llvm-func": [],
        "--llvm-add-comdats": [],
        "--llvm-legalize-for-export": [],
        "--llvm-optimize-for-nvvm-target": [],
        "--llvm-request-c-wrappers": []
    }
    specific_opt_dict['llvm.op'] = llvm_opt

    math_opt = {
        "--math-extend-to-supported-types" : [],
        "--math-uplift-to-fma": [],
         "--test-math-algebraic-simplification": [],
    "--test-math-polynomial-approximation": [
        "enable-avx2"
    ],
    "--test-expand-math": [],
        "--test-math-to-vcix": []
    }
    specific_opt_dict['math.op'] = math_opt

    memref_opt = {
         "--expand-realloc": [
        "emit-deallocs"
    ],
    "--expand-strided-metadata": [],
    "--fold-memref-alias-ops": [],
    "--memref-emulate-wide-int": [
        "widest-int-supported=32"
    ],
    "--memref-expand": [],
    "--normalize-memrefs": [],
    "--resolve-ranked-shaped-type-result-dims": [],
    "--resolve-shaped-type-result-dims": [],
    "--test-memref-bound-check": [],
    "--test-memref-dependence-check": [],
    "--test-memref-stride-calculation": []
    }
    specific_opt_dict['memref.op'] = memref_opt

    scf_opt = {
    # "--scf-bufferize": [],
    "--scf-for-loop-canonicalization": [],
    "--scf-for-loop-peeling": [
        "peel-front",
        "skip-partial"
    ],
    "--scf-for-loop-range-folding": [],
    "--scf-for-loop-specialization": [],
    "--scf-for-to-while": [],
    "--scf-forall-to-for": [],
    "--scf-forall-to-parallel": [],
    "--scf-parallel-loop-fusion": [],
    "--scf-parallel-loop-specialization": [],
    "--test-scf-parallel-loop-collapsing": [
        "collapsed-indices-0=1  collapsed-indices-2=2",
        "collapsed-indices-0=0  collapsed-indices-1=2"
    ],
     "--test-scf-for-utils": [
        "test-replace-with-new-yields"
    ],
    "--test-scf-if-utils": [],
    "--test-scf-pipelining": [
        "annotate",
        "no-epilogue-peeling"
    ],
    "--test-scf-uplift-while-to-for": [],
    "--test-scf-while-op-builder": [],
    "--scf-parallel-loop-tiling": [
        "parallel-loop-tile-sizes=0,0",
        "parallel-loop-tile-sizes=1,4 no-min-max-bounds=true"
    ]
    }
    specific_opt_dict['scf.op'] = scf_opt
    
    tensor_opt = {
        "--fold-tensor-subset-ops": [],
         "--test-tensor-copy-insertion": [
        "allow-return-allocs-from-loops",
        "bufferize-function-boundaries",
        "must-infer-memory-space"
    ],
    "--test-tensor-transform-patterns": [
        "test-drop-redundant-insert-slice-rank-expansion",
        "test-fold-consecutive-insert-extract-slice",
        "test-fold-constant-extract-slice",
        "test-fold-into-pack-and-unpack",
        "test-reassociative-reshape-folding",
        "test-rewrite-extract-slice-from-collapse-shape",
        "test-simplify-pack-unpack-patterns",
        "test-tracking-listener",
        "use-foreach"
    ]
    }
    specific_opt_dict['tensor.op'] = tensor_opt

    tosa_opt = {
    "--tosa-infer-shapes": [],
    "--tosa-layerwise-constant-fold": [
        "aggressive-reduce-constant"
    ],
    "--tosa-make-broadcastable": [],
    "--tosa-optional-decompositions": [],
    "--tosa-reduce-transposes": [],
    "--tosa-test-quant-utils": [],
       "--tosa-validate": [
        "profile=bi,mi,mt strict-op-spec-alignment",
        "profile=bi,mi,mt"
    ]
    }
    specific_opt_dict['tosa.op'] = tosa_opt

    async_opt = {
        "-async-func-to-async-runtime": [],
        "--async-parallel-for": [
        "async-dispatch=true",
        "async-dispatch=false",
        "async-dispatch=false num-workers=20 min-task-size=1",
        "num-workers=-1"
    ],
        "--async-runtime-policy-based-ref-counting": [],
        "--async-runtime-ref-counting-opt": [],
        "--async-runtime-ref-counting-opt": [],
        "--async-to-async-runtime": []
    }
    specific_opt_dict['async.op'] = async_opt

    vector_opt = {
        "-lower-vector-mask": [],
        "-lower-vector-multi-reduction": ["lowering-strategy=inner-reduction", "lowering-strategy=inner-parallel"],
        "--test-create-vector-broadcast": [],
        "--test-vector-extract-strided-slice-lowering": [],
        "--test-vector-unrolling-patterns": [
        "unroll-based-on-type unroll-order=2,0,1",
        "unroll-based-on-type",
        "unroll-based-on-type unroll-order=0,3,1,2"
        ],
        "--test-vector-transferop-opt": [],
        "--test-vector-contraction-prepare-for-mmt-lowering": [],
        "--test-vector-chained-reduction-folding-patterns": [],
        "--test-vector-transfer-collapse-inner-most-dims": [],
        "--test-scalar-vector-transfer-lowering": [
        "allow-multiple-uses"
        ],
        "--test-vector-linearize": [
        "target-vector-bitwidth=128",
        "target-vector-bitwidth=0"
        ],
        "--test-vector-to-vector-lowering": [
        "unroll"
        ],
        "--test-vector-gather-lowering": []
    }
    specific_opt_dict['vector.op'] = vector_opt


    with open('./options/mlir_specificopt.json', 'w') as json_file:
        json.dump(specific_opt_dict, json_file, indent=4)

def main():
    build_conversion()
    build_specific_opt()


if __name__ == '__main__':
    main()

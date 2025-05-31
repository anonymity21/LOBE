# LOBE
Finding Miscompilations in MLIR via Lowering Space Exploration




1. compile tosa-smith

 cmake ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_BUILD_EXAMPLES=ON  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_CCACHE_BUILD=On -DCMAKE_INSTALL_PREFIX="../install"

 ninja



 pip install -r requirements.txt
 
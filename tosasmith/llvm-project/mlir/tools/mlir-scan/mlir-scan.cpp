#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"



#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>
#include <random>
#include <filesystem>

using namespace mlir;

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));


  bool handleVectorPrint(Operation* subOp) {  
    bool hasVector = llvm::any_of(subOp->getOperandTypes(), [](Type t) {
      return llvm::isa<mlir::VectorType>(t);
    });
  
    if (!hasVector) {
      hasVector = llvm::any_of(subOp->getResultTypes(), [](Type t) {
        return llvm::isa<mlir::VectorType>(t);
      });
    }
    return hasVector;
  }
bool hasTensorInputOrOutput(mlir::Operation *op) {
  for (auto operand : op->getOperands()) {
    if (mlir::isa<mlir::TensorType>(operand.getType()))
      return true;
  }
  for (auto result : op->getResults()) {
    if (mlir::isa<mlir::TensorType>(result.getType()))
      return true;
  }
  return false;
}

bool hasTensorArgs(mlir::func::FuncOp funcOp) {
  auto funcType = funcOp.getFunctionType();
  for (auto type : funcType.getInputs()) {
    if (mlir::isa<mlir::TensorType>(type))
      return true;
  }
  for (auto type : funcType.getResults()) {
    if (mlir::isa<mlir::TensorType>(type))
      return true;
  }
  return false;
}

int main(int argc, char **argv) {
  // regist the options
  cl::ParseCommandLineOptions(argc, argv, "scan mlir\n");
  llvm::InitLLVM(argc, argv);

  // Load the mlir file
  MLIRContext context;
  registerAllDialects(context);
  context.loadAllAvailableDialects();
  OwningOpRef<ModuleOp> module;
  llvm::SourceMgr sourceMgr;

  if (!llvm::sys::fs::exists(inputFilename)) {
    llvm::errs() << "Error: Input file '" << inputFilename << "' does not exist.\n";
    return 1; 
    }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);

  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "[main] Error can't load file " << inputFilename << "\n";
    return 1;
  }
  llvm::outs() << "[main] Load file " << inputFilename << " success!\n";
  Operation* op = module.get();

  std::string fileName = inputFilename;
  std::filesystem::path pathObj(fileName);
  std::string filenameWithoutExtension = pathObj.stem().string();

  op->walk([&](Operation* subOp) {
    Dialect* dialect = subOp->getDialect();
    StringRef dialectNamespace = dialect->getNamespace();
    StringRef opName = subOp->getName().getStringRef();

    if (dialectNamespace == "tosa" || dialectNamespace == "tensor" || dialectNamespace == "scf") {
      llvm::outs() << dialectNamespace << " " << opName << "\n";
    } else if (opName == "vector.print") {
      if (handleVectorPrint(subOp)) {
        llvm::outs() << dialectNamespace << " " << opName << ".vector\n";
      } else {
        llvm::outs() << dialectNamespace << " " << opName << "\n";
      }
    } else if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(subOp)) {
      if (hasTensorArgs(funcOp)) {
        llvm::outs() << dialectNamespace << " " << opName  << ".tensor\n";
      } else {
        llvm::outs() << dialectNamespace << " " << opName << "\n";
      }
    } else if (hasTensorInputOrOutput(subOp)) {
      llvm::outs() << dialectNamespace << " " << dialectNamespace << ".op" << ".tensor\n";
    } else {
      llvm::outs() << dialectNamespace << " " << opName << "\n";
    }
  });
    llvm::outs() << "[main] Success traverse!\n";

  return 0;

}
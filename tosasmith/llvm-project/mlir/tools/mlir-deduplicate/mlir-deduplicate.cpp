#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h"
#include <set>
#include <map>
#include <unordered_set>

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input .mlir file>"), llvm::cl::Required);

bool isRedundantCast(Operation *op, std::set<std::string> &castSeen) {
  if (op->getName().getStringRef() != "tensor.cast") return false;
  std::string src = std::to_string(reinterpret_cast<uintptr_t>(op->getOperand(0).getImpl()));
  if (castSeen.count(src)) return true;
  castSeen.insert(src);
  return false;
}

bool isRedundantCall(Operation *op, std::set<std::pair<std::string, std::string>> &callSeen) {
  if (op->getName().getStringRef() != "func.call") return false;
  auto calleeAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!calleeAttr) return false;
  std::string callee = calleeAttr.getValue().str();
  std::string arg = std::to_string(reinterpret_cast<uintptr_t>(op->getOperand(0).getImpl()));
  std::pair<std::string, std::string> key = {callee, arg};
  if (callSeen.count(key)) return true;
  callSeen.insert(key);
  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Redundant Cast/Call Cleaner\n");

  MLIRContext context;
  registerAllDialects(context);
  context.loadAllAvailableDialects();

  llvm::SourceMgr sourceMgr;
  if (!llvm::sys::fs::exists(inputFilename)) {
    llvm::errs() << "Error: Input file '" << inputFilename << "' does not exist.\n";
    return 1;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "[main] Error: can't parse input file " << inputFilename << "\n";
    return 1;
  }

//   llvm::outs() << "[main] Loaded " << inputFilename << " successfully.\n";

  std::set<std::string> castSeen;
  std::set<std::pair<std::string, std::string>> callSeen;
  std::unordered_set<Operation*> castToRemove;
  std::unordered_set<Operation*> callToRemove;

  module->walk([&](Operation *op) {
    if (isRedundantCast(op, castSeen)) {
      castToRemove.insert(op);
    } else if (isRedundantCall(op, callSeen)) {
      callToRemove.insert(op);
    }
  });

  // Ensure calls using redundant casts are also removed
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "func.call") return;
    Value arg = op->getOperand(0);
    if (auto *defOp = arg.getDefiningOp()) {
      if (castToRemove.count(defOp)) {
        callToRemove.insert(op);
      }
    }
  });
  OpBuilder builder(&context);
  for (Operation *op : callToRemove) {
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();

    auto i64Type = builder.getIntegerType(64);
    auto constValue = builder.getIntegerAttr(i64Type, 1111111111);
    Value placeholder = builder.create<arith::ConstantOp>(loc, i64Type, constValue);
    builder.create<vector::PrintOp>(loc, placeholder);
    op->erase();
  }

  for (Operation *op : castToRemove)
    op->erase();

  module->print(llvm::outs());
  return 0;
}

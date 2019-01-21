//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn/CodeGen/Cuda/ASTStencilBody.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/Cuda/ASTStencilFunctionParamVisitor.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"
#include <string>

namespace dawn {
namespace codegen {
namespace cuda {

ASTStencilBody::ASTStencilBody(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const std::unordered_map<int, Array3i>& fieldIndexMap,
    const std::unique_ptr<iir::MultiStage>& ms, const CacheProperties& cacheProperties,
    Array3ui blockSizes)
    : ASTCodeGenCXX(), instantiation_(stencilInstantiation), offsetPrinter_("+", "", "", true),
      fieldIndexMap_(fieldIndexMap), ms_(ms), cacheProperties_(cacheProperties),
      blockSizes_(blockSizes) {}

ASTStencilBody::~ASTStencilBody() {}

std::string ASTStencilBody::getName(const std::shared_ptr<Expr>& expr) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
}

std::string ASTStencilBody::getName(const std::shared_ptr<Stmt>& stmt) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<ReturnStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  ss_ << "return ";

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}

void ASTStencilBody::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
}

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
  dawn_unreachable("stencil functions not allows in cuda backend");
}

void ASTStencilBody::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
  dawn_unreachable("stencil functions not allows in cuda backend");
}

void ASTStencilBody::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  std::string name = getName(expr);
  int accessID = instantiation_->getAccessIDFromExpr(expr);

  if(instantiation_->isGlobalVariable(accessID)) {
    ss_ << "globals_." << name;
  } else {
    ss_ << name;

    if(expr->isArrayAccess()) {
      ss_ << "[";
      expr->getIndex()->accept(*this);
      ss_ << "]";
    }
  }
}

void ASTStencilBody::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  int accessID = instantiation_->getAccessIDFromExpr(expr);
  if(cacheProperties_.isIJCached(accessID)) {
    CodeGeneratorHelper::derefIJCache(ss_, cacheProperties_,
                                      instantiation_->getAccessIDFromExpr(expr), blockSizes_,
                                      expr->getOffset());
    return;
  }
  if(cacheProperties_.isKCached(accessID)) {
    derefKCache(expr);
    return;
  }

  CodeGeneratorHelper::generateFieldAccessDeref(ss_, ms_, instantiation_, accessID, fieldIndexMap_,
                                                expr->getOffset());
}

void ASTStencilBody::derefKCache(const std::shared_ptr<FieldAccessExpr>& expr) {
  int accessID = instantiation_->getAccessIDFromExpr(expr);
  std::string accessName = cacheProperties_.getCacheName(accessID);
  auto vertExtent = ms_->getKCacheVertExtent(accessID);

  const int kcacheCenterOffset = cacheProperties_.getKCacheCenterOffset(accessID);

  DAWN_ASSERT((expr->getOffset()[0] == 0) && (expr->getOffset()[1] == 0));
  DAWN_ASSERT((expr->getOffset()[2] <= vertExtent.Plus) &&
              (expr->getOffset()[2] >= vertExtent.Minus));

  int index = kcacheCenterOffset;

  auto offset = expr->getOffset();
  if(offset[2] != 0)
    index += offset[2];
  ss_ << accessName << "[" + std::to_string(index) + "]";
}

} // namespace cuda
} // namespace codegen
} // namespace dawn

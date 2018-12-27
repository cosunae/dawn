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

std::string ASTStencilBody::getCodeAndResetStream() {
  activateLocalField_ = false;
  return ASTCodeGenCXX::getCodeAndResetStream();
}
std::string ASTStencilBody::getName(const std::shared_ptr<Expr>& expr) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
}

std::string ASTStencilBody::getName(const std::shared_ptr<Stmt>& stmt) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<AssignmentExpr>& expr) {
  expr->getLeft()->accept(*this);
  ss_ << " " << expr->getOp() << " ";
  activateLocalField_ = true;
  expr->getRight()->accept(*this);
}

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

void ASTStencilBody::visit(const std::shared_ptr<VarDeclStmt>& stmt) {

  activateLocalField_ = true;
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  const auto& type = stmt->getType();
  if(type.isConst())
    ss_ << "const ";
  if(type.isVolatile())
    ss_ << "volatile ";

  if(type.isBuiltinType())
    ss_ << ASTCodeGenCXX::builtinTypeIDToCXXType(type.getBuiltinTypeID(), true);
  else
    ss_ << type.getName();
  ss_ << " " << getName(stmt) << "_" << suf_;

  if(stmt->isArray())
    ss_ << "[" << stmt->getDimension() << "]";

  if(stmt->hasInit()) {
    ss_ << " " << stmt->getOp() << " ";
    if(!stmt->isArray())
      stmt->getInitList().front()->accept(*this);
    else {
      ss_ << "{";
      int numInit = stmt->getInitList().size();
      for(int i = 0; i < numInit; ++i) {
        stmt->getInitList()[i]->accept(*this);
        ss_ << ((i != (numInit - 1)) ? ", " : "");
      }
      ss_ << "}";
    }
  }
  ss_ << ";\n";
}
void ASTStencilBody::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  std::string name = getName(expr);
  int accessID = instantiation_->getAccessIDFromExpr(expr);

  if(instantiation_->isGlobalVariable(accessID)) {
    ss_ << "globals_." << name;
  } else {
    ss_ << name << "_" << suf_;

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
    derefIJCache(expr);
    return;
  }
  if(cacheProperties_.isKCached(accessID)) {
    derefKCache(expr);
    return;
  }

  auto offsets = expr->getOffset();
  if(offsets[0] % 2 != 0) {
    if(suf_ == "x") {
      offsets[0] = (offsets[0] - 1) / 2;
      CodeGeneratorHelper::generateFieldAccessDeref(ss_, id_, ms_, instantiation_, accessID,
                                                    fieldIndexMap_, offsets, activateLocalField_,
                                                    "y");
    } else if(suf_ == "y") {
      offsets[0] = (offsets[0] + 1) / 2;
      CodeGeneratorHelper::generateFieldAccessDeref(ss_, id_, ms_, instantiation_, accessID,
                                                    fieldIndexMap_, offsets, activateLocalField_,
                                                    "x");
    }
  } else {
    offsets[0] /= 2;
    CodeGeneratorHelper::generateFieldAccessDeref(ss_, id_, ms_, instantiation_, accessID,
                                                  fieldIndexMap_, offsets, activateLocalField_,
                                                  suf_);
  }
}

void ASTStencilBody::setFieldSuffix(std::string suf) { suf_ = suf; }

void ASTStencilBody::derefIJCache(const std::shared_ptr<FieldAccessExpr>& expr) {
  int accessID = instantiation_->getAccessIDFromExpr(expr);
  std::string accessName = cacheProperties_.getCacheName(accessID);

  std::string index;
  if(cacheProperties_.isCommonCache(accessID)) {
    index = cacheProperties_.getCommonCacheIndexName(iir::Cache::CacheTypeKind::IJ);
  } else {
    index = "iblock - " + std::to_string(cacheProperties_.getOffsetBeginIJCache(accessID, 0)) +
            " (jblock - " + std::to_string(cacheProperties_.getOffsetBeginIJCache(accessID, 1)) +
            ")*" + std::to_string(cacheProperties_.getStride(accessID, 1, blockSizes_));
  }
  DAWN_ASSERT(expr->getOffset()[2] == 0);

  auto offset = expr->getOffset();
  std::string offsetStr;
  if(offset[0] != 0)
    offsetStr += std::to_string(offset[0]);
  if(offset[1] != 0)
    offsetStr += ((offsetStr != "") ? "+" : "") + std::to_string(offset[1]) + "*" +
                 std::to_string(cacheProperties_.getStride(accessID, 1, blockSizes_));
  ss_ << accessName
      << (offsetStr.empty() ? "[" + index + "]" : ("[" + index + "+" + offsetStr + "]")) << "."
      << suf_;
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
  ss_ << accessName << "[" + std::to_string(index) + "]." << suf_;
}

LocalDecler::LocalDecler(const int id, MemberFunction& cudaKernel,
                         const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                         const std::unordered_map<int, Array3i>& fieldIndexMap,
                         const std::unique_ptr<iir::MultiStage>& ms,
                         const CacheProperties& cacheProperties, Array3ui blockSizes)
    : id_(id), cudaKernel_(cudaKernel), instantiation_(stencilInstantiation),
      fieldIndexMap_(fieldIndexMap), ms_(ms), cacheProperties_(cacheProperties),
      blockSizes_(blockSizes) {}

LocalDecler::~LocalDecler() {}

void LocalDecler::visit(const std::shared_ptr<AssignmentExpr>& expr) {
  expr->getLeft()->accept(*this);
  activateLocalField_ = true;
  expr->getRight()->accept(*this);
}
void LocalDecler::visit(const std::shared_ptr<VarDeclStmt>& stmt) {
  activateLocalField_ = true;
  if(stmt->hasInit()) {
    if(!stmt->isArray())
      stmt->getInitList().front()->accept(*this);
    else {
      int numInit = stmt->getInitList().size();
      for(int i = 0; i < numInit; ++i) {
        stmt->getInitList()[i]->accept(*this);
      }
    }
  }
}

void LocalDecler::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  if(!activateLocalField_) {
    return;
  }

  int accessID = instantiation_->getAccessIDFromExpr(expr);
  if(cacheProperties_.isIJCached(accessID)) {
    return;
  }
  if(cacheProperties_.isKCached(accessID) &&
     !cacheProperties_.requiresFill(ms_->getCache(accessID))) {
    return;
  }

  auto offsets = expr->getOffset();
  offsets[0] /= 2;
  std::string name = CodeGeneratorHelper::getLocaDerefName(instantiation_, id_, accessID, offsets);

  if(!loaded_.count(name)) {
    loaded_.insert(name);
    std::stringstream ss;
    CodeGeneratorHelper::generateFieldAccessLoad(ss, ms_, instantiation_, accessID, fieldIndexMap_,
                                                 offsets);

    cudaKernel_.addStatement("gridtools::clang::float_type2 " + name + "=" + ss.str());
  }
  if((expr->getOffset()[0] % 2 != 0)) {
    std::stringstream ss;
    offsets = expr->getOffset();
    offsets[0] = (offsets[0] > 0) ? (offsets[0] + 1) / 2 : (offsets[0] - 1) / 2;

    name = CodeGeneratorHelper::getLocaDerefName(instantiation_, id_, accessID, offsets);

    if(!loaded_.count(name)) {
      loaded_.insert(name);

      CodeGeneratorHelper::generateFieldAccessLoad(ss, ms_, instantiation_, accessID,
                                                   fieldIndexMap_, offsets);

      cudaKernel_.addStatement("gridtools::clang::float_type2 " + name + "=" + ss.str());
    }
  }
  return;
}

} // namespace cuda
} // namespace codegen
} // namespace dawn

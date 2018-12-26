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

#ifndef DAWN_CODEGEN_CUDA_ASTSTENCILBODY_H
#define DAWN_CODEGEN_CUDA_ASTSTENCILBODY_H

#include "dawn/CodeGen/ASTCodeGenCXX.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/CodeGen/Cuda/CacheProperties.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/IIR/Interval.h"
#include "dawn/Support/StringUtil.h"
#include <stack>
#include <unordered_map>

namespace dawn {

namespace iir {
class StencilInstantiation;
class StencilFunctionInstantiation;
}

namespace codegen {
namespace cuda {

/// @brief ASTVisitor to generate C++ naive code for the stencil and stencil function bodies
/// @ingroup cuda
class ASTStencilBody : public ASTCodeGenCXX {
protected:
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;
  RangeToString offsetPrinter_;
  const std::unordered_map<int, Array3i>& fieldIndexMap_;
  const std::unique_ptr<iir::MultiStage>& ms_;
  const CacheProperties& cacheProperties_;
  const Array3ui blockSizes_;
  int id_;
  bool activateLocalField_ = false;

public:
  using Base = ASTCodeGenCXX;

  /// @brief constructor
  ASTStencilBody(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                 const std::unordered_map<int, Array3i>& fieldIndexMap,
                 const std::unique_ptr<iir::MultiStage>& ms, const CacheProperties& cacheProperties,
                 Array3ui blockSizes);

  virtual ~ASTStencilBody() override;
  void setId(const int id) { id_ = id; }
  void setFieldSuffix(std::string suf);

  virtual std::string getCodeAndResetStream() override;

  /// @name Statement implementation
  /// @{
  virtual void visit(const std::shared_ptr<VarDeclStmt>& expr) override;
  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<AssignmentExpr>& expr) override;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override;
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;
  /// @}

  /// @brief Mapping of VarDeclStmt and Var/FieldAccessExpr to their name
  std::string getName(const std::shared_ptr<Expr>& expr) const override;
  std::string getName(const std::shared_ptr<Stmt>& stmt) const override;

private:
  void derefIJCache(const std::shared_ptr<FieldAccessExpr>& expr);
  void derefKCache(const std::shared_ptr<FieldAccessExpr>& expr);
};

class LocalDecler : public ASTVisitorForwarding {
protected:
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;
  RangeToString offsetPrinter_;
  const std::unordered_map<int, Array3i>& fieldIndexMap_;
  const std::unique_ptr<iir::MultiStage>& ms_;
  const CacheProperties& cacheProperties_;
  const Array3ui blockSizes_;
  MemberFunction& cudaKernel_;

  const int id_;

  std::set<std::string> loaded_;
  bool activateLocalField_ = false;

public:
  using Base = ASTCodeGenCXX;

  /// @brief constructor
  LocalDecler(const int id, MemberFunction& cudaKernel,
              const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
              const std::unordered_map<int, Array3i>& fieldIndexMap,
              const std::unique_ptr<iir::MultiStage>& ms, const CacheProperties& cacheProperties,
              Array3ui blockSizes);

  virtual ~LocalDecler() override;

  /// @name Statement implementation
  /// @{
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<AssignmentExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;
  /// @}
};

} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif

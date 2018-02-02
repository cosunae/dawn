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

#include "dawn/Optimizer/PassTemporaryToStencilFunction.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/Stencil.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"

namespace dawn {

namespace {
// TODO just have one interval class, we dont need two
sir::Interval intervalToSIRInterval(Interval interval) {
  return sir::Interval(interval.lowerLevel(), interval.upperLevel(), interval.lowerOffset(),
                       interval.upperOffset());
}

template <class T>
struct aggregate_adapter : public T {
  template <class... Args>
  aggregate_adapter(Args&&... args) : T{std::forward<Args>(args)...} {}
};

class TmpAssignment : public ASTVisitorPostOrder, public NonCopyable {
protected:
  std::shared_ptr<StencilInstantiation> instantiation_;
  sir::Interval interval_;
  std::shared_ptr<sir::StencilFunction> tmpFunction_;

  // TODO remove, not used
  std::shared_ptr<std::vector<std::shared_ptr<FieldAccessExpr>>> tmpComputationArgs_;
  //  std::unordered_set<std::string> insertedFields_;
  int accessID_ = -1;
  std::shared_ptr<FieldAccessExpr> tmpFieldAccessExpr_;

public:
  TmpAssignment(std::shared_ptr<StencilInstantiation> instantiation, sir::Interval const& interval)
      : instantiation_(instantiation), interval_(interval), tmpComputationArgs_(nullptr),
        tmpFieldAccessExpr_(nullptr) {}

  virtual ~TmpAssignment() {}

  int temporaryFieldAccessID() const { return accessID_; }

  std::shared_ptr<std::vector<std::shared_ptr<FieldAccessExpr>>> temporaryComputationArgs() {
    return tmpComputationArgs_;
  }

  std::shared_ptr<FieldAccessExpr> getTemporaryFieldAccessExpr() { return tmpFieldAccessExpr_; }

  std::shared_ptr<sir::StencilFunction> temporaryStencilFunction() { return tmpFunction_; }

  bool foundTemporaryToReplace() { return (tmpFunction_ != nullptr); }
  /// @name Statement implementation
  /// @{
  //  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) override;
  virtual bool preVisitNode(const std::shared_ptr<ExprStmt> stmt) override {
    return true;
    std::cout << "This is expr " << std::endl;

    //    stmt->getExpr()->accept(*this);
  }
  //  virtual void visit(const std::shared_ptr<IfStmt>& stmt) override;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual bool preVisitNode(std::shared_ptr<FieldAccessExpr> expr) override {
    DAWN_ASSERT(tmpFunction_);
    for(int idx : expr->getArgumentMap()) {
      DAWN_ASSERT(idx == -1);
    }
    for(int off : expr->getArgumentOffset())
      DAWN_ASSERT(off == 0);

    std::cout << "IIIIIIIIIII" << std::endl;
    //    if(insertedFields_.count(expr->getName()))
    //      return;
    //    insertedFields_.emplace(expr->getName());

    if(tmpComputationArgs_ == nullptr)
      tmpComputationArgs_ = std::make_shared<std::vector<std::shared_ptr<FieldAccessExpr>>>();

    tmpComputationArgs_->push_back(expr);
    expr->setArgumentMap({0, 1, 2});
    expr->setArgumentOffset({1, 1, 1});
    tmpFunction_->Args.push_back(std::make_shared<sir::Field>(expr->getName(), SourceLocation{}));
    return true;
  }

  /// @name Expression implementation
  /// @{

  virtual bool preVisitNode(const std::shared_ptr<AssignmentExpr> expr) override {
    std::cout << "KKK " << std::endl;
    if(isa<FieldAccessExpr>(*(expr->getLeft()))) {
      tmpFieldAccessExpr_ = std::dynamic_pointer_cast<FieldAccessExpr>(expr->getLeft());
      accessID_ = instantiation_->getAccessIDFromExpr(expr->getLeft());

      std::cout << " ACC " << accessID_ << instantiation_->getNameFromAccessID(accessID_)
                << std::endl;
      if(!instantiation_->isTemporaryField(accessID_))
        return false;

      std::string tmpFieldName = instantiation_->getNameFromAccessID(accessID_);
      std::cout << " TTT " << accessID_ << instantiation_->getNameFromAccessID(accessID_)
                << std::endl;

      tmpFunction_ = std::make_shared<sir::StencilFunction>();

      tmpFunction_->Name = tmpFieldName + "_OnTheFly";
      tmpFunction_->Loc = expr->getSourceLocation();
      // TODO cretae a interval->sir::interval converter
      tmpFunction_->Intervals.push_back(std::make_shared<sir::Interval>(interval_));
      //              aggregate_adapter<sir::Interval>>(
      //          interval_.lowerLevel(), interval_.upperLevel(), interval_.lowerOffset(),
      //          interval_.upperOffset()));

      tmpFunction_->Args.push_back(std::make_shared<sir::Offset>("iOffset"));
      tmpFunction_->Args.push_back(std::make_shared<sir::Offset>("jOffset"));
      tmpFunction_->Args.push_back(std::make_shared<sir::Offset>("kOffset"));
      return true;
    }
    return false;
  }
  virtual std::shared_ptr<Expr> postVisitNode(const std::shared_ptr<AssignmentExpr> expr) override {
    if(isa<FieldAccessExpr>(*(expr->getLeft()))) {
      tmpFieldAccessExpr_ = std::dynamic_pointer_cast<FieldAccessExpr>(expr->getLeft());
      accessID_ = instantiation_->getAccessIDFromExpr(expr->getLeft());
      if(!instantiation_->isTemporaryField(accessID_))
        return expr;

      DAWN_ASSERT(tmpFunction_);

      auto functionExpr = expr->getRight()->clone();

      auto retStmt = std::make_shared<ReturnStmt>(functionExpr);

      std::shared_ptr<BlockStmt> root = std::make_shared<BlockStmt>();
      root->push_back(retStmt);
      std::shared_ptr<AST> ast = std::make_shared<AST>(root);
      tmpFunction_->Asts.push_back(ast);

      return std::make_shared<NOPExpr>();
    }
    return expr;
  }
  //  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override;
  //  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override = 0;
  /// @}
};

class TmpReplacement : public ASTVisitorPostOrder, public NonCopyable {
protected:
  std::shared_ptr<StencilInstantiation> instantiation_;
  std::unordered_map<std::shared_ptr<FieldAccessExpr>, std::shared_ptr<StencilFunCallExpr>> const&
      temporaryFieldExprToFunctionCall_;
  std::shared_ptr<std::vector<std::shared_ptr<FieldAccessExpr>>> tmpComputationArgs_;

public:
  TmpReplacement(std::shared_ptr<StencilInstantiation> instantiation,
                 std::unordered_map<std::shared_ptr<FieldAccessExpr>,
                                    std::shared_ptr<StencilFunCallExpr>> const&
                     temporaryFieldExprToFunctionCall)
      : instantiation_(instantiation),
        temporaryFieldExprToFunctionCall_(temporaryFieldExprToFunctionCall) {}

  virtual ~TmpReplacement() {}

  /// @name Expression implementation
  /// @{
  virtual bool preVisitNode(std::shared_ptr<FieldAccessExpr> expr) override {
    if(!temporaryFieldExprToFunctionCall_.count(expr))
      return false;
    // TODO we need to version to tmp function generation, in case tmp is recomputed multiple times
    std::string callee = expr->getName() + "_OnTheFly";
    StencilFunCallExpr stencilFnCallExpr(callee);

    //    instantiation_->getStencilFunctionInstantiation()
    // TODO coming from stencil functions is not yet supported
    for(int idx : expr->getArgumentMap()) {
      DAWN_ASSERT(idx == -1);
    }
    for(int off : expr->getArgumentOffset())
      DAWN_ASSERT(off == 0);

    // TODO need to provide proper argument index of stencil fun arg expr for nested functions
    StencilFunArgExpr iOff(0, expr->getOffset()[0], 0);
    StencilFunArgExpr jOff(1, expr->getOffset()[0], 0);
    StencilFunArgExpr kOff(2, expr->getOffset()[0], 0);

    stencilFnCallExpr.insertArgument(std::make_shared<StencilFunArgExpr>(iOff));
    stencilFnCallExpr.insertArgument(std::make_shared<StencilFunArgExpr>(jOff));
    stencilFnCallExpr.insertArgument(std::make_shared<StencilFunArgExpr>(kOff));

    stencilFnCallExpr.insertArguments(tmpComputationArgs_->begin(), tmpComputationArgs_->end());
    return true;
  }

  //  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override;
  //  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override = 0;
  /// @}
};

} // anonymous namespace

PassTemporaryToStencilFunction::PassTemporaryToStencilFunction()
    : Pass("PassTemporaryToStencilFunction") {}

bool PassTemporaryToStencilFunction::run(
    std::shared_ptr<StencilInstantiation> stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  std::cout << "RUNNING " << std::endl;
  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    Stencil& stencil = *stencilPtr;

    // Iterate multi-stages backwards
    int stageIdx = stencil.getNumStages() - 1;
    for(auto multiStage : stencil.getMultiStages()) {
      std::shared_ptr<std::vector<std::shared_ptr<FieldAccessExpr>>> temporaryComputationArgs;
      std::unordered_map<std::shared_ptr<FieldAccessExpr>, std::shared_ptr<StencilFunCallExpr>>
          temporaryFieldExprToFunction;

      for(const auto& stagePtr : multiStage->getStages()) {

        for(const auto& doMethodPtr : stagePtr->getDoMethods()) {
          for(const auto& stmtAccessPair : doMethodPtr->getStatementAccessesPairs()) {
            const Statement& stmt = *(stmtAccessPair->getStatement());
            std::cout << "INDO " << std::endl;

            if(stmt.ASTStmt->getKind() != Stmt::SK_ExprStmt)
              continue;
            std::cout << "AFTER EXPR " << std::endl;

            //            if(temporaryFieldAccessExpr == nullptr)
            // TODO catch a temp expr
            {
              const Interval& interval = doMethodPtr->getInterval();
              const sir::Interval sirInterval = intervalToSIRInterval(interval);

              TmpAssignment tmpAssignment(stencilInstantiation, sirInterval);
              stmt.ASTStmt->acceptAndReplace(tmpAssignment);
              if(tmpAssignment.foundTemporaryToReplace()) {
                std::cout << "FOUND TMP " << tmpAssignment.temporaryStencilFunction()->Name
                          << std::endl;
                temporaryComputationArgs = tmpAssignment.temporaryComputationArgs();
                DAWN_ASSERT(temporaryComputationArgs != nullptr);

                std::shared_ptr<sir::StencilFunction> stencilFunction =
                    tmpAssignment.temporaryStencilFunction();
                std::shared_ptr<AST> ast = stencilFunction->getASTOfInterval(sirInterval);

                DAWN_ASSERT(ast);
                DAWN_ASSERT(stencilFunction);

                std::shared_ptr<StencilFunCallExpr> stencilFunCallExpr =
                    std::make_shared<StencilFunCallExpr>(stencilFunction->Name);

                temporaryFieldExprToFunction.emplace(tmpAssignment.getTemporaryFieldAccessExpr(),
                                                     stencilFunCallExpr);
                for(auto it : stencilFunction->Args) {
                  std::cout << "CHECK " << it << std::endl;
                }
                auto stencilFun = stencilInstantiation->makeStencilFunctionInstantiation(
                    stencilFunCallExpr, stencilFunction, ast, sirInterval, nullptr);
                //////////
              }
              TmpReplacement tmpReplacement(stencilInstantiation, temporaryFieldExprToFunction);
              stmt.ASTStmt->acceptAndReplace(tmpReplacement);
              //                auto& function = scope_.top()->FunctionInstantiation;
              //                auto stencilFun =
              //                getCurrentCandidateScope()->FunctionInstantiation;
              //                auto& argumentIndex = getCurrentCandidateScope()->ArgumentIndex;
              //                bool needsLazyEval = expr->getArgumentIndex() != -1;

              //                if(stencilFun->isArgOffset(argumentIndex)) {
              //                  // Argument is an offset
              //                  stencilFun->setCallerOffsetOfArgOffset(
              //                      argumentIndex, needsLazyEval
              //                                         ?
              //                                         function->getCallerOffsetOfArgOffset(expr->getArgumentIndex())
              //                                         : Array2i{{expr->getDimension(),
              //                                         expr->getOffset()}});
              //                } else {
              //                  // Argument is a direction
              //                  stencilFun->setCallerDimensionOfArgDirection(
              //                      argumentIndex, needsLazyEval
              //                                         ?
              //                                         function->getCallerDimensionOfArgDirection(expr->getArgumentIndex())
              //                                         : expr->getDimension());
              //                }
              //                argumentIndex += 1;

              //                stencilFun->closeFunctionBindings();
            }
            //            if(!temporaryFieldAccessExpr.empty()) {
            //              //              TmpReplacement tmpReplacement(stencilInstantiation,
            //              //              temporaryFieldAccessExpr,
            //              //                                            temporaryComputationArgs);
            //            }
            stencilInstantiation->removeUncompleteStencilFunctionInstantations();
          }
        }
        //        for(const Field& field : stagePtr->getFields()) {
        //          // This is caching non-temporary fields
        //          if(!instantiation_->isTemporaryField(field.getAccessID()))
        //            continue;
        //        }
      }
      //      std::shared_ptr<DependencyGraphAccesses> newGraph, oldGraph;
      //      newGraph = std::make_shared<DependencyGraphAccesses>(stencilInstantiation.get());

      //      // Iterate stages bottom -> top
      //      for(auto stageRit = multiStage.getStages().rbegin(),
      //               stageRend = multiStage.getStages().rend();
      //          stageRit != stageRend; ++stageRit) {
      //        Stage& stage = (**stageRit);
      //        DoMethod& doMethod = stage.getSingleDoMethod();

      //        // Iterate statements bottom -> top
      //        for(int stmtIndex = doMethod.getStatementAccessesPairs().size() - 1; stmtIndex >= 0;
      //            --stmtIndex) {
      //          oldGraph = newGraph->clone();

      //          auto& stmtAccessesPair = doMethod.getStatementAccessesPairs()[stmtIndex];
      //          newGraph->insertStatementAccessesPair(stmtAccessesPair);

      //          // Try to resolve race-conditions by using double buffering if necessary
      //          auto rc =
      //              fixRaceCondition(newGraph.get(), stencil, doMethod, loopOrder, stageIdx,
      //              stmtIndex);

      //          if(rc == RCKind::RK_Unresolvable)
      //            // Nothing we can do ... bail out
      //            return false;
      //          else if(rc == RCKind::RK_Fixed) {
      //            // We fixed a race condition (this means some fields have changed and our
      //            current graph
      //            // is invalid)
      //            newGraph = oldGraph;
      //            newGraph->insertStatementAccessesPair(stmtAccessesPair);
      //          }
      //        }
      //      }
      //      stageIdx--;
    }
  }

  //  if(context->getOptions().ReportPassTemporaryToStencilFunction && numRenames_ == 0)
  //    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
  //              << ": no rename\n";
  return true;
}

} // namespace dawn

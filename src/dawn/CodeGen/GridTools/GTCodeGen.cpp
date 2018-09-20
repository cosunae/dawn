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

#include "dawn/CodeGen/GridTools/GTCodeGen.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/GridTools/ASTStencilBody.h"
#include "dawn/CodeGen/GridTools/ASTStencilDesc.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringUtil.h"
#include <boost/optional.hpp>
#include <unordered_map>

namespace dawn {
namespace codegen {
namespace gt {

namespace {
class BCFinder : public ASTVisitorForwarding {
public:
  using Base = ASTVisitorForwarding;
  BCFinder() : BCsFound_(0) {}
  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
    BCsFound_++;
    Base::visit(stmt);
  }
  void resetFinder() { BCsFound_ = 0; }

  int reportBCsFound() { return BCsFound_; }

private:
  int BCsFound_;
};
}

GTCodeGen::GTCodeGen(OptimizerContext* context) : CodeGen(context), mplContainerMaxSize_(20) {}

GTCodeGen::~GTCodeGen() {}

GTCodeGen::IntervalDefinitions::IntervalDefinitions(const iir::Stencil& stencil) : Axis{0, 0} {
  auto intervals = stencil.getIntervals();
  std::transform(intervals.begin(), intervals.end(),
                 std::inserter(intervalProperties_, intervalProperties_.begin()),
                 [](iir::Interval const& i) { return iir::IntervalProperties{i}; });

  DAWN_ASSERT(!intervalProperties_.empty());

  // Add intervals for the stencil functions
  for(const auto& stencilFun : stencil.getStencilInstantiation().getStencilFunctionInstantiations())
    intervalProperties_.insert(stencilFun->getInterval());

  // Compute axis and populate the levels
  // Notice we dont take into account caches in order to build the axis
  Axis = intervalProperties_.begin()->interval_;
  for(const auto& intervalP : intervalProperties_) {
    const auto& interval = intervalP.interval_;
    Levels.insert(interval.lowerLevel());
    Levels.insert(interval.upperLevel());
    Axis.merge(interval);
  }

  // inserting the intervals of the caches
  for(const auto& mss : stencil.getChildren()) {
    for(const auto& cachePair : mss->getCaches()) {
      auto const& cache = cachePair.second;
      const boost::optional<iir::Interval> interval = cache.getInterval();
      if(interval.is_initialized())
        intervalProperties_.insert(*interval);

      // for the kcaches with fill, the interval could span beyond the axis of the do methods.
      // We need to extent the axis, to make sure that at least on interval will trigger the begin
      // of the kcache interval
      if(cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::fill) {
        DAWN_ASSERT(interval.is_initialized());
        Levels.insert(interval->lowerLevel());
        Levels.insert(interval->upperLevel());
        Axis.merge(*interval);
      }
    }
  }

  // GT HACK DOMETHOD: Compute the intervals required by each stage. Note that each stage needs to
  // have Do-Methods
  // for the entire axis, this means we may need to add empty Do-Methods
  // See https://github.com/eth-cscs/gridtools/issues/330
  int numStages = stencil.getNumStages();
  for(int i = 0; i < numStages; ++i) {
    const std::unique_ptr<iir::Stage>& stagePtr = stencil.getStage(i);

    auto gapIntervals = iir::Interval::computeGapIntervals(Axis, stagePtr->getIntervals());
    StageIntervals.emplace(stagePtr->getStageID(), gapIntervals);
    for(auto const& interval : gapIntervals) {
      intervalProperties_.insert(interval);
    }
  }
}

/// @brief The StencilFunctionAsBCGenerator class parses a stencil function that is used as a
/// boundary
/// condition into it's stringstream. In order to use stencil_functions as boundary conditions, we
/// need them to be members of the stencil-wrapper class. The goal is to template the function s.t
/// every field is a template argument.
class StencilFunctionAsBCGenerator : public ASTCodeGenCXX {
private:
  std::shared_ptr<sir::StencilFunction> function;
  const std::shared_ptr<iir::StencilInstantiation> instantiation_;

public:
  using Base = ASTCodeGenCXX;
  StencilFunctionAsBCGenerator(
      const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
      const std::shared_ptr<sir::StencilFunction>& functionToAnalyze)
      : function(functionToAnalyze), instantiation_(stencilInstantiation) {}

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) {
    auto printOffset = [](const Array3i& argumentoffsets) {
      std::string retval = "";
      std::array<std::string, 3> dims{"i", "j", "k"};
      for(int i = 0; i < 3; ++i) {
        retval +=
            dims[i] + (argumentoffsets[i] != 0 ? " + " + std::to_string(argumentoffsets[i]) + ", "
                                               : (i < 2 ? ", " : ""));
      }
      return retval;
    };
    expr->getName();
    auto getArgumentIndex = [&](const std::string& name) {
      size_t pos =
          std::distance(function->Args.begin(),
                        std::find_if(function->Args.begin(), function->Args.end(),
                                     [&](const std::shared_ptr<sir::StencilFunctionArg>& arg) {
                                       return arg->Name == name;
                                     }));

      DAWN_ASSERT_MSG(pos < function->Args.size(), "");
      return pos;
    };
    ss_ << dawn::format("data_field_%i(%s)", getArgumentIndex(expr->getName()),
                        printOffset(expr->getOffset()));
  }
  void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
  }
  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
  }
  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
  }
  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
    DAWN_ASSERT_MSG(0, "StencilFunCallExpr not allowed in this context");
  }
  void visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
    DAWN_ASSERT_MSG(0, "StencilFunArgExpr not allowed in this context");
  }

  void visit(const std::shared_ptr<ReturnStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "ReturnStmt not allowed in this context");
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) {
    if(instantiation_->isGlobalVariable(instantiation_->getAccessIDFromExpr(expr)))
      ss_ << "globals::get().";

    ss_ << getName(expr);

    if(expr->isArrayAccess()) {
      ss_ << "[";
      expr->getIndex()->accept(*this);
      ss_ << "]";
    }
  }

  std::string getName(const std::shared_ptr<Stmt>& stmt) const {
    return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
  }

  std::string getName(const std::shared_ptr<Expr>& expr) const {
    return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
  }
};

std::string GTCodeGen::cacheWindowToString(iir::Cache::window const& cacheWindow) {
  return std::string("window<") + std::to_string(cacheWindow.m_m) + "," +
         std::to_string(cacheWindow.m_p) + ">";
}

void GTCodeGen::generateSyncStorages(
    MemberFunction& method,
    IndexRange<std::unordered_map<int, iir::Stencil::FieldInfo>> const& stencilFields) const {
  // synchronize storages method
  for(auto fieldIt : stencilFields) {
    method.addStatement((*fieldIt).second.Name + ".sync()");
  }
}

void GTCodeGen::buildPlaceholderDefinitions(
    MemberFunction& function, std::unordered_map<int, iir::Stencil::FieldInfo> const& stencilFields,
    std::vector<std::string> const& stencilGlobalVariables,
    std::vector<std::string> const& stencilConstructorTemplates,
    const sir::GlobalVariableMap& globalsMap) const {

  const int numFields = stencilFields.size();

  int accessorIdx = 0, nonTempFieldId = 0;
  for(const auto& fieldInfoPair : stencilFields) {
    const auto& fieldInfo = fieldInfoPair.second;
    // Fields
    function.addTypeDef("p_" + fieldInfo.Name)
        .addType(c_gt() + (fieldInfo.IsTemporary ? "tmp_arg" : "arg"))
        .addTemplate(Twine(accessorIdx))
        .addTemplate(fieldInfo.IsTemporary ? "storage_t"
                                           : stencilConstructorTemplates[nonTempFieldId]);
    if(!fieldInfo.IsTemporary) {
      nonTempFieldId++;
    }
    ++accessorIdx;
  }

  if(!globalsMap.empty()) {
    function.addTypeDef("p_globals")
        .addType(c_gt() + "arg")
        .addTemplate(Twine(accessorIdx))
        .addTemplate("decltype(m_globals_gp_)");
  }
}

std::string GTCodeGen::generateStencilInstantiation(
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW, ssMS, tss;

  Namespace gridtoolsNamespace("gridtools", ssSW);

  // K-Cache branch changes the signature of Do-Methods
  const char* DoMethodArg = "Evaluation& eval";

  Class StencilWrapperClass(stencilInstantiation->getName(), ssSW);
  StencilWrapperClass.changeAccessibility(
      "public"); // The stencils should technically be private but nvcc doesn't like it ...

  bool isEmpty = true;
  // Functions for boundary conditions
  for(auto usedBoundaryCondition : stencilInstantiation->getBoundaryConditions()) {
    for(const auto& sf : stencilInstantiation->getSIR()->StencilFunctions) {
      if(sf->Name == usedBoundaryCondition.second->getFunctor()) {

        Structure BoundaryCondition = StencilWrapperClass.addStruct(Twine(sf->Name));
        std::string templatefunctions = "typename Direction ";
        std::string functionargs = "Direction ";

        // A templated datafield for every function argument
        for(int i = 0; i < usedBoundaryCondition.second->getFields().size(); i++) {
          templatefunctions += dawn::format(",typename DataField_%i", i);
          functionargs += dawn::format(", DataField_%i &data_field_%i", i, i);
        }
        functionargs += ", int i , int j, int k";
        auto BC = BoundaryCondition.addMemberFunction(
            Twine("GT_FUNCTION void"), Twine("operator()"), Twine(templatefunctions));
        BC.isConst(true);
        BC.addArg(functionargs);
        BC.startBody();
        StencilFunctionAsBCGenerator reader(stencilInstantiation, sf);
        sf->Asts[0]->accept(reader);
        std::string output = reader.getCodeAndResetStream();
        BC << output;
        BC.commit();
        break;
      }
    }
  }

  const auto& globalsMap = *(stencilInstantiation->getSIR()->GlobalVariableMap);

  // Generate stencils
  const auto& stencils = stencilInstantiation->getStencils();
  for(std::size_t stencilIdx = 0; stencilIdx < stencils.size(); ++stencilIdx) {
    std::string stencilType;
    const iir::Stencil& stencil = *stencils[stencilIdx];

    std::unordered_map<int, iir::Stencil::FieldInfo> StencilFields = stencil.getFields();

    auto nonTempFields = makeRange(
        StencilFields,
        std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>([](
            std::pair<int, iir::Stencil::FieldInfo> const& f) { return !f.second.IsTemporary; }));
    auto tempFields = makeRange(
        StencilFields,
        std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
            [](std::pair<int, iir::Stencil::FieldInfo> const& f) { return f.second.IsTemporary; }));

    if(stencil.isEmpty())
      continue;

    isEmpty = false;
    Structure StencilClass = StencilWrapperClass.addStruct(Twine("stencil_") + Twine(stencilIdx));
    std::string StencilName = StencilClass.getName();

    if(!globalsMap.empty()) {
      StencilClass.addMember("const globals&", "m_globals");
      StencilClass.addMember("decltype(backend_t::make_global_parameter(m_globals))",
                             "m_globals_gp_");
    }

    //
    // Interval typedefs
    //
    StencilClass.addComment("Intervals");
    IntervalDefinitions intervalDefinitions(stencil);

    std::size_t maxLevel = intervalDefinitions.Levels.size() - 1;

    auto makeLevelName = [&](int level, int offset) {
      clear(tss);
      int gt_level =
          (level == sir::Interval::End ? maxLevel
                                       : std::distance(intervalDefinitions.Levels.begin(),
                                                       intervalDefinitions.Levels.find(level)));
      int gt_offset =
          (level != sir::Interval::End) ? offset + 1 : (offset <= 0) ? offset - 1 : offset;
      tss << "gridtools::level<" << gt_level << ", " << gt_offset << ">";

      return tss.str();
    };

    // Generate typedefs for the individual intervals
    auto codeGenInterval = [&](std::string const& name, iir::Interval const& interval) {
      StencilClass.addTypeDef(name)
          .addType(c_gt() + "interval")
          .addTemplates(
              makeArrayRef({makeLevelName(interval.lowerLevel(), interval.lowerOffset()),
                            makeLevelName(interval.upperLevel(), interval.upperOffset())}));
    };

    for(const auto& intervalProperties : intervalDefinitions.intervalProperties_) {
      codeGenInterval(intervalProperties.name_, intervalProperties.interval_);
    }

    ASTStencilBody stencilBodyCGVisitor(stencilInstantiation.get(),
                                        intervalDefinitions.intervalProperties_);

    // Generate typedef for the axis
    const iir::Interval& axis = intervalDefinitions.Axis;
    StencilClass.addTypeDef(Twine("axis_") + StencilName)
        .addType(c_gt() + "interval")
        .addTemplates(makeArrayRef(
            {makeLevelName(axis.lowerLevel(), axis.lowerOffset() - 2),
             makeLevelName(axis.upperLevel(),
                           (axis.upperOffset() + 1) == 0 ? 1 : (axis.upperOffset() + 1))}));

    // Generate typedef of the grid
    StencilClass.addTypeDef(Twine("grid_") + StencilName)
        .addType(c_gt() + "grid")
        .addTemplate(Twine("axis_") + StencilName);

    //
    // Generate stencil functions code for stencils instantiated by this stencil
    //
    std::unordered_set<std::string> generatedStencilFun;
    for(const auto& stencilFun : stencilInstantiation->getStencilFunctionInstantiations()) {
      std::string stencilFunName = iir::StencilFunctionInstantiation::makeCodeGenName(*stencilFun);
      if(generatedStencilFun.emplace(stencilFunName).second) {
        Structure StencilFunStruct = StencilClass.addStruct(stencilFunName);

        // Field declaration
        const auto& fields = stencilFun->getCalleeFields();
        std::vector<std::string> arglist;

        if(fields.empty() && !stencilFun->hasReturn()) {
          DiagnosticsBuilder diag(DiagnosticsKind::Error, stencilFun->getStencilFunction()->Loc);
          diag << "no storages referenced in stencil function '" << stencilFun->getName()
               << "', this would result in invalid gridtools code";
          context_->getDiagnostics().report(diag);
          return "";
        }

        // If we have a return argument, we generate a special `__out` field
        int accessorID = 0;
        if(stencilFun->hasReturn()) {
          StencilFunStruct.addStatement("using __out = gridtools::accessor<0, "
                                        "gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, "
                                        "0, 0>>");
          arglist.push_back("__out");
          accessorID++;
        }
        // Generate field declarations
        for(std::size_t m = 0; m < fields.size(); ++m, ++accessorID) {
          std::string paramName =
              stencilFun->getOriginalNameFromCallerAccessID(fields[m].getAccessID());

          // Generate parameter of stage
          codegen::Type extent(c_gt() + "extent", clear(tss));
          for(auto& e : fields[m].getExtents().getExtents())
            extent.addTemplate(Twine(e.Minus) + ", " + Twine(e.Plus));

          StencilFunStruct.addTypeDef(paramName)
              .addType(c_gt() + "accessor")
              .addTemplate(Twine(accessorID))
              .addTemplate(c_gt_enum() +
                           ((fields[m].getIntend() == iir::Field::IK_Input) ? "in" : "inout"))
              .addTemplate(extent);

          arglist.push_back(std::move(paramName));
        }

        // Global accessor declaration
        for(auto accessID : stencilFun->getAccessIDSetGlobalVariables()) {
          std::string paramName = stencilFun->getNameFromAccessID(accessID);
          StencilFunStruct.addTypeDef(paramName)
              .addType(c_gt() + "global_accessor")
              .addTemplate(Twine(accessorID));
          accessorID++;

          arglist.push_back(std::move(paramName));
        }

        // Generate arglist
        StencilFunStruct.addTypeDef("arg_list").addType("boost::mpl::vector").addTemplates(arglist);
        mplContainerMaxSize_ = std::max(mplContainerMaxSize_, arglist.size());

        // Generate Do-Method
        auto doMethod = StencilFunStruct.addMemberFunction("GT_FUNCTION static void", "Do",
                                                           "typename Evaluation");

        DAWN_ASSERT_MSG(intervalDefinitions.intervalProperties_.count(stencilFun->getInterval()),
                        "non-existing interval");
        auto intervalIt = intervalDefinitions.intervalProperties_.find(stencilFun->getInterval());

        doMethod.addArg(DoMethodArg);
        doMethod.addArg(intervalIt->name_);
        doMethod.startBody();

        stencilBodyCGVisitor.setCurrentStencilFunction(stencilFun);
        stencilBodyCGVisitor.setIndent(doMethod.getIndent());
        for(const auto& statementAccessesPair : stencilFun->getStatementAccessesPairs()) {
          statementAccessesPair->getStatement()->ASTStmt->accept(stencilBodyCGVisitor);
          doMethod.indentStatment();
          doMethod << stencilBodyCGVisitor.getCodeAndResetStream();
        }

        doMethod.commit();
      }
    }

    // Done generating stencil functions ...
    stencilBodyCGVisitor.setCurrentStencilFunction(nullptr);

    std::vector<std::string> makeComputation;

    //
    // Generate code for stages and assemble the `make_computation`
    //
    std::size_t multiStageIdx = 0;
    for(auto multiStageIt = stencil.getChildren().begin(),
             multiStageEnd = stencil.getChildren().end();
        multiStageIt != multiStageEnd; ++multiStageIt, ++multiStageIdx) {
      const iir::MultiStage& multiStage = **multiStageIt;

      // Generate `make_multistage`
      ssMS << "gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::";
      if(!context_->getOptions().UseParallelEP &&
         multiStage.getLoopOrder() == iir::LoopOrderKind::LK_Parallel)
        ssMS << iir::LoopOrderKind::LK_Forward << " /*parallel*/ ";
      else
        ssMS << multiStage.getLoopOrder();
      ssMS << ">(),";

      // Add the MultiStage caches
      if(!multiStage.getCaches().empty()) {
        ssMS << RangeToString(", ", "gridtools::define_caches(", "),")(
            multiStage.getCaches(),
            [&](const std::pair<int, iir::Cache>& AccessIDCachePair) -> std::string {
              auto const& cache = AccessIDCachePair.second;
              DAWN_ASSERT(cache.getInterval().is_initialized() ||
                          cache.getCacheIOPolicy() == iir::Cache::local);

              std::string intervalName;
              if(cache.getInterval().is_initialized()) {
                DAWN_ASSERT(intervalDefinitions.intervalProperties_.count(*(cache.getInterval())));
                intervalName =
                    intervalDefinitions.intervalProperties_.find(*(cache.getInterval()))->name_;
              }
              return (c_gt() + "cache<" +
                      // Type: IJ or K
                      c_gt() + cache.getCacheTypeAsString() + ", " +
                      // IOPolicy: local, fill, bpfill, flush, epflush or flush_and_fill
                      c_gt() + "cache_io_policy::" + cache.getCacheIOPolicyAsString() +
                      // Interval: if IOPolicy is not local, we need to provide the interval
                      (cache.getCacheIOPolicy() != iir::Cache::local ? ", " + intervalName
                                                                     : std::string()) +
                      // cache window if policy is bpfill
                      ((cache.requiresWindow()) ? "," + cacheWindowToString(*(cache.getWindow()))
                                                : std::string()) +
                      // Placeholder which will be cached
                      ">(p_" + stencilInstantiation->getNameFromAccessID(AccessIDCachePair.first) +
                      "())")
                  .str();
            });
      }

      std::size_t stageIdx = 0;
      for(auto stageIt = multiStage.childrenBegin(), stageEnd = multiStage.childrenEnd();
          stageIt != stageEnd; ++stageIt, ++stageIdx) {
        const auto& stagePtr = *stageIt;
        const iir::Stage& stage = *stagePtr;

        Structure StageStruct =
            StencilClass.addStruct(Twine("stage_") + Twine(multiStageIdx) + "_" + Twine(stageIdx));

        ssMS << "gridtools::make_stage_with_extent<" << StageStruct.getName() << ", extent< ";
        auto extents = stage.getExtents().getExtents();
        ssMS << extents[0].Minus << ", " << extents[0].Plus << ", " << extents[1].Minus << ", "
             << extents[1].Plus << "> >(";

        const auto& fields = stage.getFields();

        // Field declaration
        std::vector<std::string> arglist;
        if(fields.empty()) {
          DiagnosticsBuilder diag(DiagnosticsKind::Error,
                                  stencilInstantiation->getSIRStencil()->Loc);
          diag << "no storages referenced in stencil '" << stencilInstantiation->getName()
               << "', this would result in invalid gridtools code";
          context_->getDiagnostics().report(diag);
          return "";
        }

        std::size_t accessorIdx = 0;
        for(const auto& fieldPair : fields) {
          const auto& field = fieldPair.second;
          const int accessID = fieldPair.first;

          std::string paramName = stencilInstantiation->getNameFromAccessID(accessID);

          // Generate parameter of stage
          codegen::Type extent(c_gt() + "extent", clear(tss));
          for(auto& e : field.getExtents().getExtents())
            extent.addTemplate(Twine(e.Minus) + ", " + Twine(e.Plus));

          StageStruct.addTypeDef(paramName)
              .addType(c_gt() + "accessor")
              .addTemplate(Twine(accessorIdx))
              .addTemplate(c_gt_enum() +
                           ((field.getIntend() == iir::Field::IK_Input) ? "in" : "inout"))
              .addTemplate(extent);

          // Generate placeholder mapping of the field in `make_stage`
          ssMS << "p_" << paramName << "()"
               << ((!stage.hasGlobalVariables() && (accessorIdx == fields.size() - 1)) ? "" : ", ");

          arglist.push_back(std::move(paramName));
          ++accessorIdx;
        }

        std::size_t maxAccessors = fields.size();
        if(stage.hasGlobalVariables()) {
          StageStruct.addTypeDef("globals")
              .addType(c_gt() + "global_accessor")
              .addTemplate(Twine(accessorIdx));

          ssMS << "p_"
               << "globals"
               << "()";
          arglist.push_back("globals");
        }

        ssMS << ")" << ((stageIdx != multiStage.getChildren().size() - 1) ? "," : ")");

        // Generate arglist
        StageStruct.addTypeDef("arg_list").addType("boost::mpl::vector").addTemplates(arglist);
        mplContainerMaxSize_ = std::max(mplContainerMaxSize_, arglist.size());

        // Generate Do-Method
        for(const auto& doMethodPtr : stage.getChildren()) {
          const iir::DoMethod& doMethod = *doMethodPtr;

          auto DoMethodCodeGen =
              StageStruct.addMemberFunction("GT_FUNCTION static void", "Do", "typename Evaluation");
          DoMethodCodeGen.addArg(DoMethodArg);
          DAWN_ASSERT(intervalDefinitions.intervalProperties_.count(
              iir::IntervalProperties{doMethod.getInterval()}));
          DoMethodCodeGen.addArg(
              intervalDefinitions.intervalProperties_.find(doMethod.getInterval())->name_);
          DoMethodCodeGen.startBody();

          stencilBodyCGVisitor.setIndent(DoMethodCodeGen.getIndent());
          for(const auto& statementAccessesPair : doMethod.getChildren()) {
            statementAccessesPair->getStatement()->ASTStmt->accept(stencilBodyCGVisitor);
            DoMethodCodeGen << stencilBodyCGVisitor.getCodeAndResetStream();
          }
        }

        // Generate empty Do-Methods
        // See https://github.com/eth-cscs/gridtools/issues/330
        const auto& stageIntervals = stage.getIntervals();
        for(const auto& interval : intervalDefinitions.StageIntervals[stagePtr->getStageID()]) {
          if(std::find(stageIntervals.begin(), stageIntervals.end(), interval) ==
             stageIntervals.end()) {
            StageStruct.addMemberFunction("GT_FUNCTION static void", "Do", "typename Evaluation")
                .addArg(DoMethodArg)
                .addArg(iir::Interval::makeCodeGenName(interval));
          }
        }
      }

      makeComputation.push_back(ssMS.str());
      clear(ssMS);
    }

    //
    // Generate constructor/destructor and methods of the stencil
    //
    std::vector<std::string> StencilGlobalVariables = stencil.getGlobalVariables();
    std::size_t numFields = StencilFields.size();

    mplContainerMaxSize_ = std::max(mplContainerMaxSize_, numFields);

    std::vector<std::string> StencilConstructorTemplates;
    for(int i = 0; i < nonTempFields.size(); ++i) {
      StencilConstructorTemplates.push_back("S" + std::to_string(i));
    }

    // Generate constructor
    auto StencilConstructor = StencilClass.addConstructor(RangeToString(", ", "", "")(
        StencilConstructorTemplates, [](const std::string& str) { return "class " + str; }));

    StencilConstructor.addArg("const gridtools::clang::domain& dom");
    if(!globalsMap.empty()) {
      StencilConstructor.addArg("const globals& globals");
    }
    int index = 0;
    for(auto field : nonTempFields) {
      StencilConstructor.addArg(StencilConstructorTemplates[index] + " " + (*field).second.Name);
      index++;
    }

    if(!globalsMap.empty()) {
      StencilConstructor.addInit("m_globals(globals)");
      StencilConstructor.addInit(
          "m_globals_gp_(gridtools::clang::backend_t::make_global_parameter(m_globals))");
    }
    StencilConstructor.startBody();

    int numTemporaries = tempFields.size();

    // Add static asserts to check halos against extents
    StencilConstructor.addComment("Check if extents do not exceed the halos");
    int nonTempFieldId = 0;
    for(const auto& fieldPair : StencilFields) {
      const auto& fieldInfo = fieldPair.second;
      if(!fieldInfo.IsTemporary) {
        auto const& ext = fieldInfo.field.getExtentsRB();
        // ===-----------------------------------------------------------------------------------===
        // PRODUCTIONTODO: [BADSTATICASSERTS]
        // Offset-Computation in K is currently broken and hence turned off. Remvove the -1 once it
        // is resolved
        // https://github.com/MeteoSwiss-APN/dawn/issues/110
        // ===-----------------------------------------------------------------------------------===
        for(int dim = 0; dim < ext.getSize() - 1; ++dim) {
          std::string at_call = "template at<" + std::to_string(dim) + ">()";
          std::string storage = StencilConstructorTemplates[nonTempFieldId];
          // assert for + accesses
          // ===---------------------------------------------------------------------------------===
          // PRODUCTIONTODO: [STAGGERING]
          // we need the staggering offset in K in order to have valid production code
          // https://github.com/MeteoSwiss-APN/dawn/issues/108
          // ===---------------------------------------------------------------------------------===
          std::string staggeringoffset = (dim == 2) ? " - 1" : "";
          StencilConstructor.addStatement(
              "static_assert((static_cast<int>(" + storage + "::storage_info_t::halo_t::" +
              at_call + ") >= " + std::to_string(ext[dim].Plus) + staggeringoffset + ") || " + "(" +
              storage + "::storage_info_t::layout_t::" + at_call + " == -1)," +
              "\"Used extents exceed halo limits.\")");
          // assert for - accesses
          StencilConstructor.addStatement("static_assert(((-1)*static_cast<int>(" + storage +
                                          "::storage_info_t::halo_t::" + at_call + ") <= " +
                                          std::to_string(ext[dim].Minus) + ") || " + "(" + storage +
                                          "::storage_info_t::layout_t::" + at_call + " == -1)," +
                                          "\"Used extents exceed halo limits.\")");
        }
        ++nonTempFieldId;
      }
    }

    // Generate domain
    buildPlaceholderDefinitions(StencilConstructor, StencilFields, StencilGlobalVariables,
                                StencilConstructorTemplates, globalsMap);

    std::vector<std::string> ArglistPlaceholders;
    for(const auto& field : StencilFields)
      ArglistPlaceholders.push_back("p_" + field.second.Name);
    if(!globalsMap.empty()) {
      ArglistPlaceholders.push_back("p_globals");
    }
    StencilConstructor.addTypeDef("domain_arg_list")
        .addType("boost::mpl::vector")
        .addTemplates(ArglistPlaceholders);

    // Placeholders to map the real storages to the placeholders (no temporaries)
    std::vector<std::string> DomainMapPlaceholders;
    for(auto field : nonTempFields) {
      DomainMapPlaceholders.push_back(
          std::string("(p_" + (*field).second.Name + "() = " + (*field).second.Name + ")"));
    }
    if(stencil.hasGlobalVariables()) {
      DomainMapPlaceholders.push_back("(p_globals() = m_globals_gp_)");
    }

    // Generate grid
    StencilConstructor.addComment("Grid");
    StencilConstructor.addStatement("gridtools::halo_descriptor di = {dom.iminus(), dom.iminus(), "
                                    "dom.iplus(), dom.isize() - 1 - dom.iplus(), dom.isize()}");
    StencilConstructor.addStatement("gridtools::halo_descriptor dj = {dom.jminus(), dom.jminus(), "
                                    "dom.jplus(), dom.jsize() - 1 - dom.jplus(), dom.jsize()}");

    auto getLevelSize = [](int level) -> std::string {
      switch(level) {
      case sir::Interval::Start:
        return "dom.kminus()";
      case sir::Interval::End:
        return "dom.ksize() == 0 ? 0 : dom.ksize() - dom.kplus()";
      default:
        return std::to_string(level);
      }
    };

    StencilConstructor.addStatement("auto grid_ = grid_stencil_" + std::to_string(stencilIdx) +
                                    "(di, dj)");

    int levelIdx = 0;
    // notice we skip the first level since it is kstart and not included in the GT grid definition
    for(auto it = intervalDefinitions.Levels.begin(), end = intervalDefinitions.Levels.end();
        it != end; ++it, ++levelIdx)
      StencilConstructor.addStatement("grid_.value_list[" + std::to_string(levelIdx) + "] = " +
                                      getLevelSize(*it));

    // generate sync storage calls
    generateSyncStorages(StencilConstructor, nonTempFields);

    // Generate make_computation
    StencilConstructor.addComment("Computation");

    // This is a memory leak.. but nothing we can do ;)
    StencilConstructor.addStatement(
        Twine("m_stencil = gridtools::make_computation<gridtools::clang::backend_t>(grid_, " +
              RangeToString(", ", "", "")(DomainMapPlaceholders) +
              RangeToString(", ", ", ", ")")(makeComputation)));
    StencilConstructor.commit();

    StencilClass.addComment("Members");
    stencilType = "computation<void>";
    StencilClass.addMember(stencilType, "m_stencil");

    // Generate stencil getter
    StencilClass.addMemberFunction(stencilType + "*", "get_stencil")
        .addStatement("return &m_stencil");
  }

  if(isEmpty) {
    DiagnosticsBuilder diag(DiagnosticsKind::Error, stencilInstantiation->getSIRStencil()->Loc);
    diag << "empty stencil '" << stencilInstantiation->getName()
         << "', this would result in invalid gridtools code";
    context_->getDiagnostics().report(diag);
    return "";
  }

  //
  // Generate constructor/destructor and methods of the stencil wrapper
  //
  if(!stencilInstantiation->getBoundaryConditions().empty())
    StencilWrapperClass.addComment("Fields that require Boundary Conditions");
  // add all fields that require a boundary condition as members since they need to be called from
  // this class and not from individual stencils
  std::unordered_set<std::string> memberfields;
  for(auto usedBoundaryCondition : stencilInstantiation->getBoundaryConditions()) {
    for(const auto& field : usedBoundaryCondition.second->getFields()) {
      memberfields.emplace(field->Name);
    }
  }
  for(const auto& field : memberfields) {
    StencilWrapperClass.addMember(Twine("storage_t"), Twine(field));
  }

  StencilWrapperClass.addComment("Stencil-Data");

  // Define allocated memebers if necessary
  if(stencilInstantiation->hasAllocatedFields()) {
    StencilWrapperClass.addMember(c_gtc() + "meta_data_t", "m_meta_data");

    for(int AccessID : stencilInstantiation->getAllocatedFieldAccessIDs())
      StencilWrapperClass.addMember(c_gtc() + "storage_t",
                                    "m_" + stencilInstantiation->getNameFromAccessID(AccessID));
  }

  // Stencil members
  StencilWrapperClass.addMember("const " + c_gtc() + "domain&", "m_dom");

  StencilWrapperClass.addMember("static constexpr const char* s_name =",
                                Twine("\"") + StencilWrapperClass.getName() + Twine("\""));

  // Stencil members
  StencilWrapperClass.addComment("Members representing all the stencils that are called");
  std::vector<std::string> stencilMembers;
  for(std::size_t i = 0; i < stencils.size(); ++i) {
    StencilWrapperClass.addMember("stencil_" + Twine(i), "m_stencil_" + Twine(i));
    stencilMembers.emplace_back("m_stencil_" + std::to_string(i));
  }

  if(!globalsMap.empty()) {
    StencilWrapperClass.addMember("globals", "m_globals");
  }

  StencilWrapperClass.changeAccessibility("public");
  StencilWrapperClass.addCopyConstructor(Class::Deleted);

  // Generate stencil wrapper constructor
  auto SIRFieldsWithoutTemps = stencilInstantiation->getSIRStencil()->Fields;
  for(auto it = SIRFieldsWithoutTemps.begin(); it != SIRFieldsWithoutTemps.end();)
    if((*it)->IsTemporary)
      it = SIRFieldsWithoutTemps.erase(it);
    else
      ++it;

  std::vector<std::pair<std::string, std::string>> StencilWrapperConstructorArguments;
  for(int accessorIdx = 0; accessorIdx < SIRFieldsWithoutTemps.size(); ++accessorIdx) {
    std::string storageType = getStorageType(*(SIRFieldsWithoutTemps[accessorIdx]));
    StencilWrapperConstructorArguments.emplace_back(storageType,
                                                    SIRFieldsWithoutTemps[accessorIdx]->Name);
  }

  std::vector<std::string> StencilWrapperConstructorTemplates;
  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i)
    StencilWrapperConstructorTemplates.push_back("S" + std::to_string(i + 1));

  auto StencilWrapperConstructor = StencilWrapperClass.addConstructor();
  //      }));

  StencilWrapperConstructor.addArg("const " + c_gtc() + "domain& dom");
  for(const auto& FieldStorage : StencilWrapperConstructorArguments) {
    StencilWrapperConstructor.addArg(FieldStorage.first + " " + FieldStorage.second);
  }

  // Initialize allocated fields
  if(stencilInstantiation->hasAllocatedFields()) {
    std::vector<std::string> tempFields;
    for(auto accessID : stencilInstantiation->getAllocatedFieldAccessIDs()) {
      tempFields.push_back(stencilInstantiation->getNameFromAccessID(accessID));
    }
    addTmpStorageInit_wrapper(StencilWrapperConstructor, stencils, tempFields);
  }
  StencilWrapperConstructor.addInit("m_dom(dom)");
  // Initialize storages that require boundary conditions
  for(const auto& memberfield : memberfields) {
    StencilWrapperConstructor.addInit(memberfield + "(" + memberfield + ")");
  }

  // Initialize stencils
  for(size_t i = 0; i < stencils.size(); ++i) {
    const auto& fields = stencils[i]->getFields();

    std::vector<iir::Stencil::FieldInfo> nonTempFields;

    for(const auto& field : fields) {
      if(!field.second.IsTemporary) {
        nonTempFields.push_back(field.second);
      }
    }

    std::string initctr = "(dom, ";
    if(!globalsMap.empty()) {
      initctr = initctr + "m_globals,";
    }
    StencilWrapperConstructor.addInit(
        "m_stencil_" + Twine(i) +
        RangeToString(", ", initctr.c_str(),
                      ")")(nonTempFields, [&](const iir::Stencil::FieldInfo& fieldInfo) {
          if(stencilInstantiation->isAllocatedField(fieldInfo.field.getAccessID()))
            return "m_" + fieldInfo.Name;
          else
            return fieldInfo.Name;
        }));
  }

  StencilWrapperConstructor.commit();

  // Create the StencilID -> stencil name map
  std::unordered_map<int, std::vector<std::string>> stencilIDToStencilNameMap;
  std::unordered_map<int, std::string> stencilIDToRunArguments;

  for(std::size_t i = 0; i < stencils.size(); ++i) {
    stencilIDToStencilNameMap[stencils[i]->getStencilID()].emplace_back("stencil_" +
                                                                        std::to_string(i));

    const auto& fields = stencils[i]->getFields();
    std::vector<iir::Stencil::FieldInfo> nonTempFields;

    for(const auto& field : fields) {
      if(!field.second.IsTemporary) {
        nonTempFields.push_back(field.second);
      }
    }
    stencilIDToRunArguments[stencils[i]->getStencilID()] =
        "m_dom," +
        RangeToString(", ", "", "")(nonTempFields, [&](const iir::Stencil::FieldInfo& fieldInfo) {
          if(stencilInstantiation->isAllocatedField(fieldInfo.field.getAccessID()))
            return "m_" + fieldInfo.Name;
          else
            return fieldInfo.Name;
        });
  }

  StencilWrapperConstructor.commit();

  // Generate the run method by generate code for the stencil description AST
  MemberFunction RunMethod = StencilWrapperClass.addMemberFunction("void", "run");
  RunMethod.startBody();

  // Create the StencilID -> stencil name map
  stencilIDToStencilNameMap.clear();
  for(std::size_t i = 0; i < stencils.size(); ++i)
    stencilIDToStencilNameMap[stencils[i]->getStencilID()].emplace_back(stencilMembers[i]);

  ASTStencilDesc stencilDescCGVisitor(stencilInstantiation, stencilIDToStencilNameMap,
                                      stencilIDToRunArguments);
  stencilDescCGVisitor.setIndent(RunMethod.getIndent());
  for(const auto& statement : stencilInstantiation->getStencilDescStatements()) {
    statement->ASTStmt->accept(stencilDescCGVisitor);
    RunMethod << stencilDescCGVisitor.getCodeAndResetStream();
  }

  RunMethod.commit();

  // Generate name getter
  StencilWrapperClass.addMemberFunction("std::string", "get_name")
      .isConst(true)
      .addStatement("return std::string(s_name)");

  if(!globalsMap.empty()) {
    generateGlobalsAPI(StencilWrapperClass, globalsMap);
  }

  // Generate stencil getter
  MemberFunction stencilGetter =
      StencilWrapperClass.addMemberFunction("std::vector<computation<void>*>", "getStencils");
  stencilGetter.addStatement(
      "return " +
      RangeToString(", ", "std::vector<gridtools::computation<void>*>({", "})")(
          stencilMembers, [](const std::string& member) { return member + ".get_stencil()"; }));
  stencilGetter.commit();

  MemberFunction clearMeters = StencilWrapperClass.addMemberFunction("void", "reset_meters");
  clearMeters.startBody();
  std::string s = RangeToString("\n", "", "")(stencilMembers, [](const std::string& member) {
    return member + ".get_stencil()->reset_meter();";
  });
  clearMeters << s;
  clearMeters.commit();

  StencilWrapperClass.commit();

  gridtoolsNamespace.commit();

  // Remove trailing ';' as this is retained by Clang's Rewriter
  std::string str = ssSW.str();
  str[str.size() - 2] = ' ';

  return str;
}

// std::string GTCodeGen::generateGlobals(std::shared_ptr<SIR> const& Sir) {
//  using namespace codegen;

//  const auto& globalsMap = *(Sir->GlobalVariableMap);
//  if(globalsMap.empty())
//    return "";

//  std::stringstream ss;

//  Namespace gridtoolsNamespace("gridtools", ss);

//  std::string StructName = "globals";
//  std::string BaseName = "gridtools::clang::globals_impl<" + StructName + ">";

//  Struct GlobalsStruct(StructName + ": public " + BaseName, ss);
//  GlobalsStruct.addTypeDef("base_t").addType("gridtools::clang::globals_impl<globals>");

//  for(const auto& globalsPair : globalsMap) {
//    sir::Value& value = *globalsPair.second;
//    std::string Name = globalsPair.first;
//    std::string Type = sir::Value::typeToString(value.getType());
//    std::string AdapterBase = std::string("base_t::variable_adapter_impl") + "<" + Type + ">";

//    Structure AdapterStruct = GlobalsStruct.addStructMember(Name + "_adapter", Name, AdapterBase);
//    AdapterStruct.addConstructor().addArg("").addInit(
//        AdapterBase + "(" + Type + "(" + (value.empty() ? std::string() : value.toString()) +
//        "))");

//    auto AssignmentOperator =
//        AdapterStruct.addMemberFunction(Name + "_adapter&", "operator=", "class ValueType");
//    AssignmentOperator.addArg("ValueType&& value");
//    if(value.isConstexpr())
//      AssignmentOperator.addStatement(
//          "throw std::runtime_error(\"invalid assignment to constant variable '" + Name + "'\")");
//    else
//      AssignmentOperator.addStatement("get_value() = value");
//    AssignmentOperator.addStatement("return *this");
//    AssignmentOperator.commit();
//  }

//  GlobalsStruct.commit();

//  // Add the symbol for the singleton
//  codegen::Statement(ss) << "template<> " << StructName << "* " << BaseName
//                         << "::s_instance = nullptr";

//  gridtoolsNamespace.commit();

//  // Remove trailing ';' as this is retained by Clang's Rewriter
//  std::string str = ss.str();
//  str[str.size() - 2] = ' ';

//  return str;
//}

std::unique_ptr<TranslationUnit> GTCodeGen::generateCode() {
  mplContainerMaxSize_ = 30;
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_->getStencilInstantiationMap()) {
    std::string code = generateStencilInstantiation(nameStencilCtxPair.second);

    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  // Generate globals
  std::string globals = generateGlobals(context_->getSIR(), "gridtools");

  // If we need more than 20 elements in boost::mpl containers, we need to increment to the nearest
  // multiple of ten
  // http://www.boost.org/doc/libs/1_61_0/libs/mpl/doc/refmanual/limit-vector-size.html
  if(mplContainerMaxSize_ > 20) {
    mplContainerMaxSize_ += (10 - mplContainerMaxSize_ % 10);
    DAWN_LOG(INFO) << "increasing boost::mpl template limit to " << mplContainerMaxSize_;
  }

  DAWN_ASSERT_MSG(mplContainerMaxSize_ % 10 == 0,
                  "boost::mpl template limit needs to be multiple of 10");

  std::vector<std::string> ppDefines;
  auto makeDefine = [](std::string define, int value) {
    return "#define " + define + " " + std::to_string(value);
  };

  ppDefines.push_back(makeDefine("GRIDTOOLS_CLANG_GENERATED", 1));
  ppDefines.push_back("#define GRIDTOOLS_CLANG_BACKEND_T GT");

  CodeGen::addMplIfdefs(ppDefines, mplContainerMaxSize_, context_->getOptions().MaxHaloPoints);

  BCFinder finder;
  for(const auto& stencilInstantiation : context_->getStencilInstantiationMap()) {
    for(const auto& stmt : stencilInstantiation.second->getStencilDescStatements()) {
      stmt->ASTStmt->accept(finder);
    }
  }
  if(finder.reportBCsFound()) {
    ppDefines.push_back("#ifdef __CUDACC__\n#include "
                        "<boundary-conditions/apply_gpu.hpp>\n#else\n#include "
                        "<boundary-conditions/apply.hpp>\n#endif\n");
  }

  DAWN_LOG(INFO) << "Done generating code";

  return make_unique<TranslationUnit>(context_->getSIR()->Filename, std::move(ppDefines),
                                      std::move(stencils), std::move(globals));
}

std::vector<std::string> GTCodeGen::buildFieldTemplateNames(
    IndexRange<std::vector<iir::Stencil::FieldInfo>> const& stencilFields) const {
  std::vector<std::string> templates;
  for(int i = 0; i < stencilFields.size(); ++i)
    templates.push_back("S" + std::to_string(i + 1));

  return templates;
}

} // namespace gt
} // namespace codegen
} // namespace dawn

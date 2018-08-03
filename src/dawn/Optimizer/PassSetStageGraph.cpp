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

#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/STLExtras.h"

namespace dawn {

/// @brief Compute the dependency between the stages `from` and `to`
/// @return `true` if the stage `from` depends on `to`, `false` otherwise
static bool depends(const iir::Stage& fromStage, const iir::Stage& toStage) {
  if(!fromStage.overlaps(toStage))
    return false;

  bool intervalsMatch = fromStage.getEnclosingInterval() == toStage.getEnclosingInterval();

  for(const auto& fromFieldPair : fromStage.getFields()) {
    const Field& fromField = fromFieldPair.second;
    for(const auto& toFieldPair : toStage.getFields()) {
      const Field& toField = toFieldPair.second;
      if(fromField.getAccessID() != toField.getAccessID())
        continue;

      Field::IntendKind fromFieldIntend = fromField.getIntend();
      Field::IntendKind toFieldIntend = toField.getIntend();

      switch(fromFieldIntend) {
      case Field::IK_Output:
        if(!intervalsMatch || toFieldIntend == Field::IK_Input ||
           toFieldIntend == Field::IK_InputOutput)
          return true;
        break;
      case Field::IK_InputOutput:
        return true;
      case Field::IK_Input:
        if(toFieldIntend == Field::IK_Output || toFieldIntend == Field::IK_InputOutput)
          return true;
        break;
      }
    }
  }
  return false;
}

PassSetStageGraph::PassSetStageGraph() : Pass("PassSetStageGraph") {
  dependencies_.push_back("PassSetStageName");
}

bool PassSetStageGraph::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();
  int stencilIdx = 0;

  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;
    int numStages = stencil.getNumStages();

    auto stageDAG = std::make_shared<iir::DependencyGraphStage>(stencilInstantiation);

    // Build DAG of stages (backward sweep)
    for(int i = numStages - 1; i >= 0; --i) {
      const auto& fromStagePtr = stencil.getStage(i);
      stageDAG->insertNode(fromStagePtr->getStageID());
      int curStageID = fromStagePtr->getStageID();

      for(int j = i - 1; j >= 0; --j) {
        const auto& toStagePtr = stencil.getStage(j);
        if(depends(*fromStagePtr, *toStagePtr))
          stageDAG->insertEdge(curStageID, toStagePtr->getStageID());
      }
    }

    if(context->getOptions().DumpStageGraph)
      stageDAG->toDot("stage_" + stencilInstantiation->getName() + "_s" +
                      std::to_string(stencilIdx) + ".dot");

    stencil.setStageDependencyGraph(std::move(stageDAG));
  }

  return true;
}

} // namespace dawn

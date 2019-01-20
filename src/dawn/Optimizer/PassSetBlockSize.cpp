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

#include "dawn/Optimizer/PassSetBlockSize.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/Optimizer/OptimizerContext.h"

namespace dawn {

PassSetBlockSize::PassSetBlockSize() : Pass("PassSetBlockSize") {
  dependencies_.push_back("PassSetStageName");
}

bool PassSetBlockSize::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  const auto& IIR = stencilInstantiation->getIIR();

  std::array<unsigned int, 3> blockSize{0, 0, 0};
  if(!context->getOptions().block_size.empty()) {
    std::string blockSizeStr = context->getOptions().block_size;
    std::istringstream idomain_size(blockSizeStr);
    std::string arg;
    getline(idomain_size, arg, ',');
    unsigned int iBlockSize = std::stoi(arg);
    getline(idomain_size, arg, ',');
    unsigned int jBlockSize = std::stoi(arg);
    getline(idomain_size, arg, ',');
    unsigned int kBlockSize = std::stoi(arg);

    blockSize = {iBlockSize, jBlockSize, kBlockSize};
  } else if(context->getOptions().Backend == "cuda") {
    bool verticalPattern = true;
    for(const auto& stage : iterateIIROver<iir::Stage>(*IIR)) {
      if(!stage->getExtents().isHorizontalPointwise()) {
        verticalPattern = false;
      }
    }
    if(verticalPattern) {
      IIR->setBlockSize({32, 1, 4});
    } else {
      IIR->setBlockSize({32, 4, 4});
    }
  }
  // Other backends not supported yet, and gridtools currently ignores the value for block size

  return true;
}

} // namespace dawn

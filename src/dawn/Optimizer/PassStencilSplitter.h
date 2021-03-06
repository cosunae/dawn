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

#ifndef DAWN_OPTIMIZER_PASSSTENCILSPLITTER_H
#define DAWN_OPTIMIZER_PASSSTENCILSPLITTER_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief Pass for splitting stencils due to software limitations i.e stencils are too large
///
/// This Pass depends on `PassSetStageGraph`.
///
/// @ingroup optimizer
class PassStencilSplitter : public Pass {
public:
  PassStencilSplitter();

  /// @brief Maximum number of allowed fields per stencil
  ///
  /// This is the threshold for the splitting pass to be invoked.
  static constexpr int MaxFieldPerStencil = 40;

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace dawn

#endif

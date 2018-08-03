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

#ifndef DAWN_IIR_IIR_H
#define DAWN_IIR_IIR_H

#include "dawn/IIR/Stencil.h"

namespace dawn {

namespace iir {

/// @brief A Stencil is represented by a collection of MultiStages
/// @ingroup optimizer
class IIR : public IIRNode<void, IIR, Stencil, std::shared_ptr, impl::Vector> {

public:
  using StencilSmartPtr_t = child_smartptr_t<Stencil>;

  IIR() = default;
  IIR(const IIR&) = default;
  IIR(IIR&&) = default;

  IIR& operator=(const IIR&) = default;
  IIR& operator=(IIR&&) = default;
  /// @}
};
} // namespace iir
} // namespace dawn

#endif
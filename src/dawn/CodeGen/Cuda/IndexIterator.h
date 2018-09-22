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

#ifndef DAWN_CODEGEN_CUDA_INDEXITERATOR_H
#define DAWN_CODEGEN_CUDA_INDEXITERATOR_H

#include <string>
#include "dawn/Support/Array.h"

namespace dawn {
namespace codegen {
namespace cuda {

struct IndexIterator {
  static std::string name(Array3i dims) {
    std::string n_ = "";
    for(const int i : dims) {
      n_ = n_ + std::to_string(i);
    }
    return n_;
  }
};

} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
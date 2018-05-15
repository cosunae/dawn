//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil stencil {
  storage b, a;
  var tmp;

  Do {
    vertical_region(k_start, k_start) {
      tmp = a;
    }
    vertical_region(k_start+1, k_end-1) {
      tmp = tmp[k-1];
    }
    vertical_region(k_end, k_end) {
      tmp = tmp[k-1]+a;
    }

    vertical_region(k_start, k_start) {
      b = tmp;
    }
    vertical_region(k_start+1, k_end-4) {
      b = tmp[k-1,i+1];
    }
    vertical_region(k_end-3, k_end) {
      b = tmp[k-1,i+1]+a;
    }
  }
};

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

#include "dawn/CodeGen/GridTools/CodeGenUtils.h"

namespace dawn {
namespace codegen {
namespace gt {

std::vector<std::string>
CodeGenUtils::buildPlaceholderList(const std::map<int, iir::Stencil::FieldInfo>& stencilFields,
                                   const sir::GlobalVariableMap& globalsMap, bool buildPair) {
  std::vector<std::string> plchdrs;
  for(const auto& fieldInfoPair : stencilFields) {
    const auto& fieldInfo = fieldInfoPair.second;
    if(buildPair) {
      plchdrs.push_back("p_" + fieldInfo.Name + "{} = " + fieldInfo.Name);
    } else {
      plchdrs.push_back("p_" + fieldInfo.Name);
    }
  }

  if(!globalsMap.empty()) {
    plchdrs.push_back("p_globals");
  }
  return plchdrs;
}

} // namespace gt
} // namespace codegen
} // namespace dawn

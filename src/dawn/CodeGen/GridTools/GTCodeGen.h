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

#ifndef DAWN_CODEGEN_GRIDTOOLS_GTCODEGEN_H
#define DAWN_CODEGEN_GRIDTOOLS_GTCODEGEN_H

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/Optimizer/Interval.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace dawn {

class StencilInstantiation;
class OptimizerContext;
class Stage;
class Stencil;

namespace codegen {
namespace gt {

/// @brief GridTools C++ code generation for the gridtools_clang DSL
/// @ingroup gt
class GTCodeGen : public CodeGen {
public:
  GTCodeGen(OptimizerContext* context);
  virtual ~GTCodeGen();

  virtual std::unique_ptr<TranslationUnit> generateCode() override;

  /// @brief Definitions of the gridtools::intervals
  struct IntervalDefinitions {
    IntervalDefinitions(const Stencil& stencil);

    /// Intervals of the stencil
    std::unordered_set<IntervalProperties> intervalProperties_;

    /// Axis of the stencil (i.e the interval which spans accross all other intervals)
    Interval Axis;

    /// Levels of the axis
    std::set<int> Levels;

    /// Intervals of the Do-Methods of each stage
    std::unordered_map<std::shared_ptr<Stage>, std::vector<Interval>> StageIntervals;
  };

private:
  std::string generateStencilInstantiation(const StencilInstantiation* stencilInstantiation);
  std::string generateGlobals(const std::shared_ptr<SIR>& Sir);
  std::string cacheWindowToString(boost::optional<Cache::window> const& cacheWindow);
  std::string buildMakeComputation(std::vector<std::string> const& DomainMapPlaceholders,
                                   std::vector<std::string> const& makeComputation,
                                   const std::__cxx11::string& gridName) const;
  void
  buildPlaceholderDefinitions(MemberFunction& function,
                              std::vector<Stencil::FieldInfo> const& stencilFields,
                              std::vector<std::string> const& stencilGlobalVariables,
                              std::vector<std::string> const& stencilConstructorTemplates) const;

  std::string getFieldName(std::shared_ptr<sir::Field> f) const { return f->Name; }

  std::string getFieldName(Stencil::FieldInfo const& f) const { return f.Name; }

  bool isTemporary(std::shared_ptr<sir::Field> f) const { return f->IsTemporary; }

  // TODO we should eliminate the redundancy on FieldInfos
  bool isTemporary(Stencil::FieldInfo const& f) const { return f.IsTemporary; }

  /// code generate sync methods statements for all the fields passed
  void generateSyncStorages(MemberFunction& method,
                            const IndexRange<std::vector<Stencil::FieldInfo>>& stencilFields) const;

  /// construct a string of template parameters for storages
  template <typename TFieldInfo>
  std::vector<std::string>
  buildFieldTemplateNames(IndexRange<std::vector<TFieldInfo>> const& stencilFields) const {
    std::vector<std::string> templates;
    for(int i = 0; i < stencilFields.size(); ++i)
      templates.push_back("S" + std::to_string(i + 1));

    return templates;
  }

  template <typename TFieldInfo>
  int computeNumTemporaries(std::vector<TFieldInfo> const& stencilFields) const {
    int numTemporaries = 0;
    for(auto const& f : stencilFields)
      numTemporaries += (isTemporary(f) ? 1 : 0);
    return numTemporaries;
  }

  template <typename TFieldInfo>
  MemberFunction
  createStorageTemplateMethod(Structure& structure, std::string const& returnType,
                              std::string const& functionName,
                              IndexRange<std::vector<TFieldInfo>> const& nonTempFields) const {
    auto storageTemplates = buildFieldTemplateNames(nonTempFields);

    auto function = structure.addMemberFunction(
        returnType, functionName,
        RangeToString(", ", "", "")(storageTemplates,
                                    [](const std::string& str) { return "class " + str; }));
    int i = 0;
    for(auto const& field : nonTempFields) {
      function.addArg(storageTemplates[i] + " " + getFieldName(field));
      ++i;
    }

    return function;
  }

  /// Maximum needed vector size of boost::fusion containers
  std::size_t mplContainerMaxSize_;
};

} // namespace gt
} // namespace codegen
} // namespace dawn

#endif

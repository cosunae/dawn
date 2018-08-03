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

#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace dawn {

std::ostream& operator<<(std::ostream& os, const iir::Stencil::StagePosition& position) {
  return (os << "(" << position.MultiStageIndex << ", " << position.StageOffset << ")");
}

std::ostream& operator<<(std::ostream& os, const iir::Stencil::StatementPosition& position) {
  return (os << "(Stage=" << position.StagePos << ", DoMethod=" << position.DoMethodIndex
             << ", Statement=" << position.StatementIndex << ")");
}

std::ostream& operator<<(std::ostream& os, const iir::Stencil::Lifetime& lifetime) {
  return (os << "[Begin=" << lifetime.Begin << ", End=" << lifetime.End << "]");
}

// TODO solve with iterators
std::ostream& operator<<(std::ostream& os, const iir::Stencil& stencil) {
  int multiStageIdx = 0;
  for(const auto& MS : stencil.getChildren()) {
    os << "MultiStage " << (multiStageIdx++) << ": (" << MS->getLoopOrder() << ")\n";
    for(const auto& stage : MS->getChildren())
      os << "  " << stencil.getStencilInstantiation().getNameFromStageID(stage->getStageID()) << " "
         << RangeToString()(stage->getFields(), [&](const std::pair<int, Field>& fieldPair) {
              return stencil.getStencilInstantiation().getNameFromAccessID(fieldPair.first);
            }) << "\n";
  }
  return os;
}

namespace iir {

bool Stencil::StagePosition::operator<(const Stencil::StagePosition& other) const {
  return MultiStageIndex < other.MultiStageIndex ||
         (MultiStageIndex == other.MultiStageIndex && StageOffset < other.StageOffset);
}

bool Stencil::StagePosition::operator==(const Stencil::StagePosition& other) const {
  return MultiStageIndex == other.MultiStageIndex && StageOffset == other.StageOffset;
}

bool Stencil::StagePosition::operator!=(const Stencil::StagePosition& other) const {
  return !(*this == other);
}

bool Stencil::StatementPosition::operator<(const Stencil::StatementPosition& other) const {
  return StagePos < other.StagePos ||
         (StagePos == other.StagePos && DoMethodIndex == other.DoMethodIndex &&
          StatementIndex < other.StatementIndex);
}

bool Stencil::StatementPosition::operator<=(const Stencil::StatementPosition& other) const {
  return operator<(other) || operator==(other);
}

bool Stencil::StatementPosition::operator==(const Stencil::StatementPosition& other) const {
  return StagePos == other.StagePos && DoMethodIndex == other.DoMethodIndex &&
         StatementIndex == other.StatementIndex;
}

bool Stencil::StatementPosition::operator!=(const Stencil::StatementPosition& other) const {
  return !(*this == other);
}

bool Stencil::StatementPosition::inSameDoMethod(const Stencil::StatementPosition& other) const {
  return StagePos == other.StagePos && DoMethodIndex == other.DoMethodIndex;
}

bool Stencil::Lifetime::overlaps(const Stencil::Lifetime& other) const {
  // Note: same stage but different Do-Method are treated as overlapping!

  bool lowerBoundOverlap = false;
  if(Begin.StagePos == other.End.StagePos && Begin.DoMethodIndex != other.End.DoMethodIndex)
    lowerBoundOverlap = true;
  else
    lowerBoundOverlap = Begin <= other.End;

  bool upperBoundOverlap = false;
  if(other.Begin.StagePos == End.StagePos && other.Begin.DoMethodIndex != End.DoMethodIndex)
    upperBoundOverlap = true;
  else
    upperBoundOverlap = other.Begin <= End;

  return lowerBoundOverlap && upperBoundOverlap;
}

Stencil::Stencil(StencilInstantiation& stencilInstantiation,
                 const std::shared_ptr<sir::Stencil>& SIRStencil, int StencilID,
                 const std::shared_ptr<DependencyGraphStage>& stageDependencyGraph)
    : stencilInstantiation_(stencilInstantiation), SIRStencil_(SIRStencil), StencilID_(StencilID),
      stageDependencyGraph_(stageDependencyGraph) {}

std::unordered_set<Interval> Stencil::getIntervals() const {
  std::unordered_set<Interval> intervals;
  for(const auto& multistage : children_)
    for(const auto& stage : multistage->getChildren())
      for(const auto& doMethod : stage->getChildren())
        intervals.insert(doMethod->getInterval());

  return intervals;
}

std::vector<Stencil::FieldInfo> Stencil::getFields(bool withTemporaries) const {
  std::set<int> fieldAccessIDs;
  for(const auto& multistage : children_) {
    for(const auto& stage : multistage->getChildren()) {
      for(const auto& fieldPair : stage->getFields()) {
        // TODO redo transform
        fieldAccessIDs.insert(fieldPair.first);
      }
    }
  }

  std::vector<FieldInfo> fields;

  for(const auto& AccessID : fieldAccessIDs) {
    std::string name = stencilInstantiation_.getNameFromAccessID(AccessID);
    bool isTemporary = stencilInstantiation_.isTemporaryField(AccessID);
    Array3i specifiedDimension = stencilInstantiation_.getFieldDimensionsMask(AccessID);

    if(isTemporary) {
      if(withTemporaries) {
        fields.insert(fields.begin(), FieldInfo{isTemporary, name, AccessID, specifiedDimension});
      }
    } else {
      fields.emplace_back(FieldInfo{isTemporary, name, AccessID, specifiedDimension});
    }
  }

  return fields;
}

std::vector<std::string> Stencil::getGlobalVariables() const {
  std::set<int> globalVariableAccessIDs;
  for(const auto& multistage : children_) {
    for(const auto& stage : multistage->getChildren()) {
      globalVariableAccessIDs.insert(stage->getAllGlobalVariables().begin(),
                                     stage->getAllGlobalVariables().end());
    }
  }

  std::vector<std::string> globalVariables;
  for(const auto& AccessID : globalVariableAccessIDs)
    globalVariables.push_back(stencilInstantiation_.getNameFromAccessID(AccessID));

  return globalVariables;
}

int Stencil::getNumStages() const {
  return std::accumulate(childrenBegin(), childrenEnd(), int(0),
                         [](int numStages, const Stencil::MultiStageSmartPtr_t& MS) {
                           return numStages + MS->getChildren().size();
                         });
}

void Stencil::forEachStatementAccessesPair(
    std::function<void(ArrayRef<std::unique_ptr<StatementAccessesPair>>)> func, bool updateFields) {
  forEachStatementAccessesPairImpl(func, 0, getNumStages(), updateFields);
}

void Stencil::forEachStatementAccessesPair(
    std::function<void(ArrayRef<std::unique_ptr<StatementAccessesPair>>)> func,
    const Stencil::Lifetime& lifetime, bool updateFields) {
  int startStageIdx = getStageIndexFromPosition(lifetime.Begin.StagePos);
  int endStageIdx = getStageIndexFromPosition(lifetime.End.StagePos);
  forEachStatementAccessesPairImpl(func, startStageIdx, endStageIdx + 1, updateFields);
}

void Stencil::forEachStatementAccessesPairImpl(
    std::function<void(ArrayRef<std::unique_ptr<StatementAccessesPair>>)> func, int startStageIdx,
    int endStageIdx, bool updateFields) {
  for(int stageIdx = startStageIdx; stageIdx < endStageIdx; ++stageIdx) {
    const auto& stage = getStage(stageIdx);
    for(const auto& doMethodPtr : stage->getChildren())
      func(doMethodPtr->getChildren());

    if(updateFields)
      stage->update();
  }
}

void Stencil::updateFields(const Stencil::Lifetime& lifetime) {
  int startStageIdx = getStageIndexFromPosition(lifetime.Begin.StagePos);
  int endStageIdx = getStageIndexFromPosition(lifetime.End.StagePos);
  updateFieldsImpl(startStageIdx, endStageIdx + 1);
}

void Stencil::updateFields() { updateFieldsImpl(0, getNumStages()); }

void Stencil::updateFieldsImpl(int startStageIdx, int endStageIdx) {
  for(int stageIdx = startStageIdx; stageIdx < endStageIdx; ++stageIdx)
    getStage(stageIdx)->update();
}

std::unordered_map<int, Field> Stencil::getFields2() const {
  std::unordered_map<int, Field> fields;

  for(const auto& mssPtr : children_) {
    for(const auto& fieldPair : mssPtr->getFields()) {
      const Field& field = fieldPair.second;
      auto it = fields.find(field.getAccessID());
      if(it != fields.end()) {
        // Adjust the Intend
        if(it->second.getIntend() == Field::IK_Input && field.getIntend() == Field::IK_Output)
          it->second.setIntend(Field::IK_InputOutput);
        else if(it->second.getIntend() == Field::IK_Output && field.getIntend() == Field::IK_Input)
          it->second.setIntend(Field::IK_InputOutput);

        // Merge the Extent
        it->second.mergeReadExtents(field.getReadExtents());
        it->second.mergeWriteExtents(field.getWriteExtents());

        it->second.extendInterval(field.getInterval());
      } else
        fields.emplace(field.getAccessID(), field);
    }
  }

  return fields;
}

void Stencil::setStageDependencyGraph(const std::shared_ptr<DependencyGraphStage>& stageDAG) {
  stageDependencyGraph_ = stageDAG;
}

const std::shared_ptr<DependencyGraphStage>& Stencil::getStageDependencyGraph() const {
  return stageDependencyGraph_;
}

const std::unique_ptr<MultiStage>&
Stencil::getMultiStageFromMultiStageIndex(int multiStageIdx) const {
  DAWN_ASSERT_MSG(multiStageIdx < children_.size(), "invalid multi-stage index");
  auto msIt = children_.begin();
  std::advance(msIt, multiStageIdx);
  return *msIt;
}

const std::unique_ptr<MultiStage>& Stencil::getMultiStageFromStageIndex(int stageIdx) const {
  return getMultiStageFromMultiStageIndex(getPositionFromStageIndex(stageIdx).MultiStageIndex);
}

Stencil::StagePosition Stencil::getPositionFromStageIndex(int stageIdx) const {
  DAWN_ASSERT(!children_.empty());
  if(stageIdx == -1)
    return StagePosition(0, -1);

  int curIdx = 0, multiStageIdx = 0;
  for(const auto& MS : children_) {

    // Is our stage in this multi-stage?
    int numStages = MS->getChildren().size();
    if((curIdx + numStages) <= stageIdx) {
      curIdx += numStages;
      multiStageIdx++;
      continue;
    } else {
      int stageOffset = stageIdx - curIdx;
      DAWN_ASSERT_MSG(stageOffset < numStages, "invalid stage index");
      return StagePosition(multiStageIdx, stageOffset);
    }
  }
  dawn_unreachable("invalid stage index");
}

int Stencil::getStageIndexFromPosition(const Stencil::StagePosition& position) const {
  auto curMSIt = children_.begin();
  std::advance(curMSIt, position.MultiStageIndex);

  // Count the number of stages in the multistages before our current MS
  int numStagesInMSBeforeCurMS = std::accumulate(
      childrenBegin(), curMSIt, int(0), [&](int numStages, const MultiStageSmartPtr_t& MS) {
        return numStages + MS->getChildren().size();
      });

  // Add the current stage offset
  return numStagesInMSBeforeCurMS + position.StageOffset;
}

const std::unique_ptr<Stage>& Stencil::getStage(const StagePosition& position) const {
  // Get the multi-stage ...
  DAWN_ASSERT_MSG(position.MultiStageIndex < children_.size(), "invalid multi-stage index");
  auto msIt = children_.begin();
  std::advance(msIt, position.MultiStageIndex);
  const auto& MS = *msIt;

  // ... and the requested stage inside the given multi-stage
  DAWN_ASSERT_MSG(position.StageOffset == -1 || position.StageOffset < MS->getChildren().size(),
                  "invalid stage offset");
  auto stageIt = MS->childrenBegin();
  std::advance(stageIt, position.StageOffset == -1 ? 0 : position.StageOffset);
  return *stageIt;
}

const std::unique_ptr<Stage>& Stencil::getStage(int stageIdx) const {
  int curIdx = 0;
  for(const auto& MS : children_) {

    // Is our stage in this multi-stage?
    int numStages = MS->getChildren().size();

    if((curIdx + numStages) <= stageIdx) {
      // No... continue
      curIdx += numStages;
      continue;
    } else {
      // Yes... advance to our stage
      int stageOffset = stageIdx - curIdx;

      DAWN_ASSERT_MSG(stageOffset < MS->getChildren().size(), "invalid stage index");
      auto stageIt = MS->childrenBegin();
      std::advance(stageIt, stageOffset);

      return *stageIt;
    }
  }
  dawn_unreachable("invalid stage index");
}

void Stencil::insertStage(const StagePosition& position, std::unique_ptr<Stage>&& stage) {

  // Get the multi-stage ...
  DAWN_ASSERT_MSG(position.MultiStageIndex < children_.size(), "invalid multi-stage index");
  auto msIt = children_.begin();
  std::advance(msIt, position.MultiStageIndex);
  const auto& MS = *msIt;

  // ... and the requested stage inside the given multi-stage
  DAWN_ASSERT_MSG(position.StageOffset == -1 || position.StageOffset < MS->getChildren().size(),
                  "invalid stage offset");
  auto stageIt = MS->childrenBegin();

  // A stage offset of -1 indicates *before* the first element (thus nothing to do).
  // Otherwise we advance one beyond the requested stage as we insert *after* the specified
  // stage and `std::list::insert` inserts *before*.
  if(position.StageOffset != -1) {
    std::advance(stageIt, position.StageOffset);
    if(stageIt != MS->childrenEnd())
      stageIt++;
  }

  MS->insertChild(stageIt, std::move(stage));
}

Interval Stencil::getAxis(bool useExtendedInterval) const {
  int numStages = getNumStages();
  DAWN_ASSERT_MSG(numStages, "need atleast one stage");

  Interval axis = getStage(0)->getEnclosingExtendedInterval();
  for(int stageIdx = 1; stageIdx < numStages; ++stageIdx)
    axis.merge(useExtendedInterval ? getStage(stageIdx)->getEnclosingExtendedInterval()
                                   : getStage(stageIdx)->getEnclosingInterval());
  return axis;
}

void Stencil::renameAllOccurrences(int oldAccessID, int newAccessID) {
  for(const auto& multistage : getChildren()) {
    multistage->renameAllOccurrences(oldAccessID, newAccessID);
  }
}

std::unordered_map<int, Stencil::Lifetime>
Stencil::getLifetime(const std::unordered_set<int>& AccessIDs) const {
  std::unordered_map<int, Lifetime> lifetimeMap;
  for(int AccessID : AccessIDs) {
    lifetimeMap.emplace(AccessID, getLifetime(AccessID));
  }

  return lifetimeMap;
}

Stencil::Lifetime Stencil::getLifetime(const int AccessID) const {
  // use make_optional(false, ...) just to avoid a gcc warning
  boost::optional<StatementPosition> Begin = boost::make_optional(false, StatementPosition{});
  StatementPosition End;

  int multiStageIdx = 0;
  for(const auto& multistagePtr : children_) {

    int stageOffset = 0;
    for(const auto& stagePtr : multistagePtr->getChildren()) {

      int doMethodIndex = 0;
      for(const auto& doMethodPtr : stagePtr->getChildren()) {
        DoMethod& doMethod = *doMethodPtr;

        int statementIdx = 0;
        for(const auto& stmtAccessPair : doMethod.getChildren()) {
          const Accesses& accesses = *stmtAccessPair->getAccesses();

          auto processAccessMap = [&](const std::unordered_map<int, Extents>& accessMap) {
            if(!accessMap.count(AccessID))
              return;

            StatementPosition pos(StagePosition(multiStageIdx, stageOffset), doMethodIndex,
                                  statementIdx);

            if(!Begin.is_initialized())
              Begin = boost::make_optional(pos);
            End = pos;
          };

          processAccessMap(accesses.getWriteAccesses());
          processAccessMap(accesses.getReadAccesses());
          statementIdx++;
        }

        doMethodIndex++;
      }

      stageOffset++;
    }

    multiStageIdx++;
  }

  DAWN_ASSERT(Begin.is_initialized());

  return Lifetime(*Begin, End);
}

bool Stencil::isEmpty() const {
  for(const auto& MS : getChildren())
    for(const auto& stage : MS->getChildren())
      for(const auto& doMethod : stage->getChildren())
        if(!doMethod->childrenEmpty())
          return false;
  return true;
}

boost::optional<Interval> Stencil::getEnclosingIntervalTemporaries() const {
  boost::optional<Interval> tmpInterval;
  for(const auto& mss : getChildren()) {
    auto mssInterval = mss->getEnclosingAccessIntervalTemporaries();
    if(!mssInterval.is_initialized())
      continue;
    if(tmpInterval.is_initialized()) {
      tmpInterval->merge(*mssInterval);
    } else {
      tmpInterval = mssInterval;
    }
  }
  return tmpInterval;
}

const std::shared_ptr<sir::Stencil> Stencil::getSIRStencil() const { return SIRStencil_; }

void Stencil::accept(ASTVisitor& visitor) {
  for(const auto& multistagePtr : children_)
    for(const auto& stagePtr : multistagePtr->getChildren())
      for(const auto& doMethodPtr : stagePtr->getChildren())
        for(const auto& stmtAcessesPairPtr : doMethodPtr->getChildren())
          stmtAcessesPairPtr->getStatement()->ASTStmt->accept(visitor);
}

std::unordered_map<int, Extents> const Stencil::computeEnclosingAccessExtents() const {
  std::unordered_map<int, Extents> maxExtents_;
  // iterate through multistages
  for(const auto& MS : children_) {
    // iterate through stages
    for(const auto& stage : MS->getChildren()) {
      for(const auto& fieldPair : stage->getFields()) {
        const auto& field = fieldPair.second;
        // TODO recover
        const int accessID = fieldPair.first;
        // add the stage extent to the field extent
        Extents e = field.getExtents();
        e.add(stage->getExtents());
        // merge with the current minimum/maximum extent for the given field
        auto finder = maxExtents_.find(accessID);
        if(finder != maxExtents_.end()) {
          finder->second.merge(e);
        } else {
          maxExtents_.emplace(accessID, e);
        }
      }
    }
  }
  return maxExtents_;
}

} // namespace iir
} // namespace dawn
//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Support/Twine.h"
#include "gsl/Support/SmallString.h"
#include <iostream>
#include <sstream>

namespace gsl {

std::string Twine::str() const {
  // If we're storing only a std::string, just return it.
  if(LHSKind == StdStringKind && RHSKind == EmptyKind)
    return *LHS.stdString;

  // Otherwise, flatten and copy the contents first.
  SmallString<256> Vec;
  return toStringRef(Vec).str();
}

void Twine::toVector(SmallVectorImpl<char>& Out) const {
  std::stringstream OS;
  print(OS);
  std::string outStr = OS.str();
  Out.append(outStr.begin(), outStr.end());
}

StringRef Twine::toNullTerminatedStringRef(SmallVectorImpl<char>& Out) const {
  if(isUnary()) {
    switch(getLHSKind()) {
    case CStringKind:
      // Already null terminated, yay!
      return StringRef(LHS.cString);
    case StdStringKind: {
      const std::string* str = LHS.stdString;
      return StringRef(str->c_str(), str->size());
    }
    default:
      break;
    }
  }
  toVector(Out);
  Out.push_back(0);
  Out.pop_back();
  return StringRef(Out.data(), Out.size());
}

void Twine::printOneChild(std::ostream& OS, Child Ptr, NodeKind Kind) const {
  switch(Kind) {
  case Twine::NullKind:
    break;
  case Twine::EmptyKind:
    break;
  case Twine::TwineKind:
    Ptr.twine->print(OS);
    break;
  case Twine::CStringKind:
    OS << Ptr.cString;
    break;
  case Twine::StdStringKind:
    OS << *Ptr.stdString;
    break;
  case Twine::StringRefKind:
    OS << Ptr.stringRef->str();
    break;
  case Twine::SmallStringKind:
    OS << StringRef(Ptr.smallString->data(), Ptr.smallString->size()).str();
    break;
  case Twine::CharKind:
    OS << Ptr.character;
    break;
  case Twine::DecUIKind:
    OS << Ptr.decUI;
    break;
  case Twine::DecIKind:
    OS << Ptr.decI;
    break;
  case Twine::DecULKind:
    OS << *Ptr.decUL;
    break;
  case Twine::DecLKind:
    OS << *Ptr.decL;
    break;
  case Twine::DecULLKind:
    OS << *Ptr.decULL;
    break;
  case Twine::DecLLKind:
    OS << *Ptr.decLL;
    break;
  case Twine::UHexKind:
    OS << *Ptr.uHex;
    break;
  }
}

void Twine::printOneChildRepr(std::ostream& OS, Child Ptr, NodeKind Kind) const {
  switch(Kind) {
  case Twine::NullKind:
    OS << "null";
    break;
  case Twine::EmptyKind:
    OS << "empty";
    break;
  case Twine::TwineKind:
    OS << "rope:";
    Ptr.twine->printRepr(OS);
    break;
  case Twine::CStringKind:
    OS << "cstring:\"" << Ptr.cString << "\"";
    break;
  case Twine::StdStringKind:
    OS << "std::string:\"" << Ptr.stdString << "\"";
    break;
  case Twine::StringRefKind:
    OS << "stringref:\"" << Ptr.stringRef->str() << "\"";
    break;
  case Twine::SmallStringKind:
    OS << "smallstring:\"" << StringRef(Ptr.smallString->data(), Ptr.smallString->size()).str()
       << "\"";
    break;
  case Twine::CharKind:
    OS << "char:\"" << Ptr.character << "\"";
    break;
  case Twine::DecUIKind:
    OS << "decUI:\"" << Ptr.decUI << "\"";
    break;
  case Twine::DecIKind:
    OS << "decI:\"" << Ptr.decI << "\"";
    break;
  case Twine::DecULKind:
    OS << "decUL:\"" << *Ptr.decUL << "\"";
    break;
  case Twine::DecLKind:
    OS << "decL:\"" << *Ptr.decL << "\"";
    break;
  case Twine::DecULLKind:
    OS << "decULL:\"" << *Ptr.decULL << "\"";
    break;
  case Twine::DecLLKind:
    OS << "decLL:\"" << *Ptr.decLL << "\"";
    break;
  case Twine::UHexKind:
    OS << "uhex:\"" << Ptr.uHex << "\"";
    break;
  }
}

void Twine::print(std::ostream& OS) const {
  printOneChild(OS, LHS, getLHSKind());
  printOneChild(OS, RHS, getRHSKind());
}

void Twine::printRepr(std::ostream& OS) const {
  OS << "(Twine ";
  printOneChildRepr(OS, LHS, getLHSKind());
  OS << " ";
  printOneChildRepr(OS, RHS, getRHSKind());
  OS << ")";
}

void Twine::dump() const { print(std::cerr); }

void Twine::dumpRepr() const { printRepr(std::cerr); }

} // namespace gsl

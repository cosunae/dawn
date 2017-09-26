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
//
// This file includes the headers of the tinyformat library.
// See: https://github.com/c42f/tinyformat
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_SUPPORT_FORMAT_H
#define GSL_SUPPORT_FORMAT_H

#include "gsl/Support/Assert.h"

#define TINYFORMAT_ERROR(reason) GSL_ASSERT_MSG(0, reason)
#define TINYFORMAT_USE_VARIADIC_TEMPLATES
#include "gsl/Support/External/tinyformat/tinyformat.h"

namespace gsl {

/// @fn format
/// @brief Format list of arguments according to the given format string and return the result as a
/// string
///
/// Signature:
/// @code
///   template<typename... Args>
///   std::string format(const char* fmt, const Args&... args);
/// @endcode
///
/// @see https://github.com/c42f/tinyformat
/// @ingroup support
using tfm::format;

} // namespace gsl

#endif

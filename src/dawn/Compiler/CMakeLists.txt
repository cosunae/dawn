##===------------------------------------------------------------------------------*- CMake -*-===##
##                          _                      
##                         | |                     
##                       __| | __ ___      ___ ___  
##                      / _` |/ _` \ \ /\ / / '_  | 
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

dawn_add_library(
  NAME DawnCompiler
  SOURCES DawnCompiler.h
          DawnCompiler.cpp
          DiagnosticsEngine.cpp
          DiagnosticsEngine.h
          DiagnosticsMessage.h
          DiagnosticsQueue.cpp
          DiagnosticsQueue.h
          Options.h
          Options.inc
  OBJECT
)

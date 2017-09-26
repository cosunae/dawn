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

#ifndef GSL_OPTIMIZER_CACHE_H
#define GSL_OPTIMIZER_CACHE_H

#include "gsl/Support/HashCombine.h"
#include <string>

namespace gsl {

/// @brief Cache specification of gridtools
/// @ingroup optimizer
class Cache {
public:
  /// @brief Available cache types
  enum CacheTypeKind {
    IJ,  ///< IJ caches require synchronization capabilities, as different (i,j) grid points are
         ///  processed by parallel cores. GPU backend keeps them in shared memory
    K,   ///< Processing of all the K elements is done by same thread, so resources for K caches can
         ///  be private and do not require synchronization. GPU backend uses registers.
    IJK, ///< IJK caches is an extension to 3rd dimension of IJ caches. GPU backend uses shared
         ///  memory
    bypass ///< bypass the cache for read only parameters
  };

  /// @brief IO policies of the cache
  enum CacheIOPolicy {
    unknown,        ///< Not yet set
    fill_and_flush, ///< Read values from the cached field and write the result back
    fill,           ///< Read values form the cached field but do not write back
    flush,          ///< Write values back the the cached field but do not read in
    epflush,        ///< End point cache flush: indicates a flush only at the end point
                    ///  of the interval being cached
    bpfill,         ///< Begin point cache fill: indicates a fill only at the begin point
                    ///  of the interval being cached
    local           ///< Local only cache, neither read nor write the the cached field
  };

  Cache(CacheTypeKind type, CacheIOPolicy policy, int AccessID);

  /// @brief Get the AccessID of the field
  int getCachedFieldAccessID() const;

  /// @brief Get the type of cache
  CacheTypeKind getCacheType() const;
  std::string getCacheTypeAsString() const;

  /// @brief Get the I/O policy of the cache
  CacheIOPolicy getCacheIOPolicy() const;
  std::string getCacheIOPolicyAsString() const;

  /// @name Comparison operator
  /// @{
  bool operator==(const Cache& other) const {
    return (AccessID_ == other.AccessID_ && type_ == other.type_ && policy_ == other.policy_);
  }
  bool operator!=(const Cache& other) const { return !(*this == other); }
  /// @}

private:
  CacheTypeKind type_;
  CacheIOPolicy policy_;
  int AccessID_;
};

} // namespace gsl

namespace std {

template <>
struct hash<gsl::Cache> {
  size_t operator()(const gsl::Cache& cache) const {
    std::size_t hash = 0;
    gsl::hash_combine(hash, cache.getCachedFieldAccessID(),
                      static_cast<int>(cache.getCacheIOPolicy()),
                      static_cast<int>(cache.getCacheType()));
    return hash;
  }
};

} // namespace std

#endif

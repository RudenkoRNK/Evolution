#pragma once
#include "Evolution/ArgumentTraits.hpp"
#include "tbb/enumerable_thread_specific.h"

namespace Evolution {

template <class FG> struct GeneratorTraits final {
  auto constexpr static isGenerator = ArgumentTraits<FG>::nArguments == 0;

  using Function =
      std::conditional_t<isGenerator,
                         typename ArgumentTraits<FG>::template Type<0>, FG>;

  using ThreadSpecific = tbb::enumerable_thread_specific<Function>;

  using ThreadSpecificOrGlobalFunction =
      std::conditional_t<isGenerator, ThreadSpecific, Function>;

  using ThreadSpecificOrGlobalRefFunction =
      std::conditional_t<isGenerator, ThreadSpecific, Function &>;

  static ThreadSpecificOrGlobalRefFunction
  GetThreadSpecificOrGlobal(FG &&fg) noexcept(!isGenerator) {
    if constexpr (!isGenerator)
      return fg;
    else
      return ThreadSpecific(std::forward<FG>(fg));
  }

  static Function &
  GetFunction(ThreadSpecificOrGlobalFunction &ThreadSpecificOrGlobal) noexcept(
      !isGenerator) {
    if constexpr (!isGenerator)
      return ThreadSpecificOrGlobal;
    else
      return ThreadSpecificOrGlobal.local();
  }
};
} // namespace Evolution

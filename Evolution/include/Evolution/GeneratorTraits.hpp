#pragma once
#include "Utility/TypeTraits.hpp"
#include <tbb/enumerable_thread_specific.h>

namespace Utility {

struct GeneratorTraits final {
  template <typename FG>
  auto constexpr static isGenerator =
      ArgumentTraits<std::remove_reference_t<FG>>::nArguments == 0;

  template <typename FG>
  using Function = std::conditional_t<
      isGenerator<FG>,
      typename ArgumentTraits<std::remove_reference_t<FG>>::template Type<0>,
      std::remove_reference_t<FG>>;

  template <typename FG>
  using ThreadSpecific = tbb::enumerable_thread_specific<Function<FG>>;

  template <typename FG>
  using ThreadGeneratorOrFunction =
      std::conditional_t<isGenerator<FG>, ThreadSpecific<FG>, Function<FG>>;

  template <typename FG>
  auto static GetThreadGeneratorOrFunction(FG &&fg) noexcept(!isGenerator<FG>)
      -> std::conditional_t<isGenerator<FG>, ThreadSpecific<FG>, FG &&> {
    if constexpr (!isGenerator<FG>)
      return std::forward<FG>(fg);
    else
      return ThreadSpecific<FG>(std::forward<FG>(fg));
  }

  template <typename FG>
  auto static GetFunction(ThreadGeneratorOrFunction<FG> &
                              ThreadSpecificOrGlobal) noexcept(!isGenerator<FG>)
      -> std::add_lvalue_reference_t<Function<FG>> {
    if constexpr (!isGenerator<FG>)
      return ThreadSpecificOrGlobal;
    else
      return ThreadSpecificOrGlobal.local();
  }

  template <typename FG>
  auto static GetFunctionForSingleThread(FG &&fg) noexcept(!isGenerator<FG>)
      -> std::conditional_t<isGenerator<FG>, Function<FG>, FG &&> {
    if constexpr (!isGenerator<FG>)
      return std::forward<FG>(fg);
    else
      return fg();
  }
};
} // namespace Utility

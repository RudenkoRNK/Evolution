#pragma once
#include "Evolution/ArgumentTraits.hpp"
#include "tbb/enumerable_thread_specific.h"

namespace Evolution {

struct GeneratorTraits final {
  template <class FG>
  auto constexpr static isGenerator =
      ArgumentTraits<std::remove_reference_t<FG>>::nArguments == 0;

  template <class FG>
  using Function = std::conditional_t<
      isGenerator<FG>,
      typename ArgumentTraits<std::remove_reference_t<FG>>::template Type<0>,
      std::remove_reference_t<FG>>;

  template <class FG>
  using ThreadSpecific = tbb::enumerable_thread_specific<Function<FG>>;

  template <class FG>
  using ThreadSpecificOrGlobalFunction =
      std::conditional_t<isGenerator<FG>, ThreadSpecific<FG>, Function<FG>>;

  template <class FG>
  using ThreadSpecificOrGlobalRefFunction =
      std::conditional_t<isGenerator<FG>, ThreadSpecific<FG>,
                         std::add_lvalue_reference_t<Function<FG>>>;

  template <class FG>
  static auto GetThreadSpecificOrGlobal(FG &&fg) noexcept(!isGenerator<FG>)
      -> std::conditional_t<isGenerator<FG>, ThreadSpecific<FG>, FG &&> {
    if constexpr (!isGenerator<FG>)
      return std::forward<FG>(fg);
    else
      return ThreadSpecific<FG>(std::forward<FG>(fg));
  }

  template <class FG>
  static auto GetFunction(ThreadSpecificOrGlobalFunction<FG> &
                              ThreadSpecificOrGlobal) noexcept(!isGenerator<FG>)
      -> std::add_lvalue_reference_t<Function<FG>> {
    if constexpr (!isGenerator<FG>)
      return ThreadSpecificOrGlobal;
    else
      return ThreadSpecificOrGlobal.local();
  }
};
} // namespace Evolution

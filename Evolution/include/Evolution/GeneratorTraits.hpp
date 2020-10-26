#pragma once
#include "Utility/TypeTraits.hpp"
#include <tbb/enumerable_thread_specific.h>

namespace Evolution {

struct GeneratorTraits final {
private:
  template <typename Callable>
  using CallableTraits = typename Utility::CallableTraits<Callable>;
  template <typename Callable>
  using RawCallable = typename std::remove_cvref_t<Callable>;

public:
  template <typename FG>
  auto constexpr static isGenerator = CallableTraits<FG>::nArguments == 0;

private:
  template <typename FG>
  using FunctionObject =
      std::conditional_t<isGenerator<FG>,
                         typename CallableTraits<FG>::template Type<0>, FG>;
  template <typename FG>
  using GeneratorObject =
      std::conditional_t<isGenerator<FG>, FG, std::function<RawCallable<FG>()>>;

public:
  template <typename FG>
  using Function = typename CallableTraits<FunctionObject<FG>>::std_function;
  template <typename FG>
  using Generator = typename CallableTraits<GeneratorObject<FG>>::std_function;
  template <typename FG>
  using TBBGenerator = tbb::enumerable_thread_specific<Function<FG>>;

  template <typename FG>
  using TBBGeneratorOrFunction =
      std::conditional_t<isGenerator<FG>, TBBGenerator<FG>, Function<FG>>;

private:
  template <typename Callable>
  auto constexpr static is_std_function =
      std::is_same_v<RawCallable<Callable>,
                     CallableTraits<Callable>::std_function>;

  template <typename Callable>
  auto static to_std_function(Callable &&callable) noexcept(
      is_std_function<Callable>)
      -> std::conditional_t<is_std_function<Callable>, Callable &&,
                            typename CallableTraits<Callable>::std_function> {
    using std_function = typename CallableTraits<Callable>::std_function;
    if constexpr (is_std_function<Callable>)
      return std::forward<Callable>(callable);
    else
      return std_function(std::forward<Callable>(callable));
  }

public:
  template <typename FG>
  auto static WrapGeneratorOrFunction(FG &&fg)
      -> std::conditional_t<isGenerator<FG> || !is_std_function<FG>,
                            TBBGeneratorOrFunction<FG>, FG &&> {
    if constexpr (!isGenerator<FG>)
      return to_std_function<FG>(std::forward<FG>(fg));
    else
      return TBBGenerator<FG>(std::forward<FG>(fg));
  }

  template <typename FG>
  static constexpr Function<FG> &
  GetFunction(Function<FG> &generatorOrFunction) noexcept {
    return generatorOrFunction;
  }
  template <typename FG>
  static constexpr Function<FG> const &
  GetFunction(Function<FG> const &generatorOrFunction) noexcept {
    return generatorOrFunction;
  }
  template <typename FG>
  auto static GetFunction(Generator<FG> const &generatorOrFunction)
      -> decltype(generatorOrFunction()) {
    return generatorOrFunction();
  }
  template <typename FG>
  auto static GetFunction(TBBGenerator<FG> &generatorOrFunction)
      -> decltype(generatorOrFunction.local()) {
    return generatorOrFunction.local();
  }
};
} // namespace Evolution

#pragma once
#define BOOST_ALLOW_DEPRECATED_HEADERS
#include "Utility/TypeTraits.hpp"
#include <concepts>

namespace Evolution {
template <typename Callable>
using CallableTraits = Utility::CallableTraits<Callable>;

template <typename T> static decltype(auto) _DNADecay() {
  if constexpr (Utility::TypeTraits::isInstanceOf<std::unique_ptr, T>)
    return *(std::declval<T>());
  else
    return *static_cast<std::add_pointer_t<T>>(nullptr);
}
template <typename T>
using DNADecay = std::remove_cvref_t<decltype(_DNADecay<T>())>;

template <typename T> concept DNAConcept = requires() {
  requires std::copyable<T>;
  requires std::is_nothrow_swappable_v<T>;
  requires std::is_nothrow_move_assignable_v<T>;
};

template <typename T> concept GradeConcept = requires() {
  requires std::copyable<T>;
  requires std::is_nothrow_swappable_v<T>;
  requires std::is_nothrow_move_assignable_v<T>;
};

template <typename T> concept EvaluateArgumentConcept = requires() {
  requires DNAConcept<std::remove_cvref_t<T>>;
  requires std::is_const_v<std::remove_reference_t<T>> ||
      !std::is_reference_v<T>;
};

template <typename T> concept EvaluateReturnConcept = requires() {
  requires GradeConcept<T>;
};

template <typename EvaluateFunction>
concept EvaluateFunctionConcept = requires() {
  requires CallableTraits<EvaluateFunction>::nArguments == 1;
  requires CallableTraits<EvaluateFunction>::template isValue<0>;
  requires EvaluateArgumentConcept<
      typename CallableTraits<EvaluateFunction>::template ArgType<0>>;
  requires EvaluateReturnConcept<
      typename CallableTraits<EvaluateFunction>::ReturnType>;
};

template <typename EvaluateGenerator>
concept EvaluateGeneratorConcept = requires() {
  requires CallableTraits<EvaluateGenerator>::nArguments == 0;
  requires EvaluateFunctionConcept<
      typename CallableTraits<EvaluateGenerator>::ReturnType>;
};

template <typename EvaluateFG>
concept EvaluateFunctionOrGeneratorConcept =
    EvaluateGeneratorConcept<EvaluateFG> || EvaluateFunctionConcept<EvaluateFG>;

template <typename T> concept MutateCrossoverArgumentConcept = requires() {
  requires DNAConcept<std::remove_cvref_t<T>>;
  requires std::is_lvalue_reference_v<T> ||
      !std::is_const_v<std::remove_reference_t<T>>;
};

template <typename T> concept MutateCrossoverReturnConcept = requires() {
  requires DNAConcept<DNADecay<T>>;
};

template <typename MutateFunction> concept MutateFunctionConcept = requires() {
  requires CallableTraits<MutateFunction>::nArguments == 1;
  requires MutateCrossoverArgumentConcept<
      typename CallableTraits<MutateFunction>::template ArgType<0>>;
  requires MutateCrossoverReturnConcept<
      typename CallableTraits<MutateFunction>::ReturnType>;
  requires std::is_same_v<
      typename std::remove_cvref_t<
          DNADecay<typename CallableTraits<MutateFunction>::ReturnType>>,
      typename std::remove_cvref_t<
          typename CallableTraits<MutateFunction>::template ArgType<0>>>;
};

template <typename MutateGenerator>
concept MutateGeneratorConcept = requires() {
  requires CallableTraits<MutateGenerator>::nArguments == 0;
  requires MutateFunctionConcept<
      typename CallableTraits<MutateGenerator>::ReturnType>;
};

template <typename MutateFG>
concept MutateFunctionOrGeneratorConcept =
    MutateGeneratorConcept<MutateFG> || MutateFunctionConcept<MutateFG>;

template <typename CrossoverFunction>
concept CrossoverFunctionConcept = requires() {
  requires CallableTraits<CrossoverFunction>::nArguments == 2;
  requires MutateCrossoverArgumentConcept<
      typename CallableTraits<CrossoverFunction>::template ArgType<0>>;
  requires MutateCrossoverArgumentConcept<
      typename CallableTraits<CrossoverFunction>::template ArgType<1>>;
  requires MutateCrossoverReturnConcept<
      typename CallableTraits<CrossoverFunction>::ReturnType>;
  requires std::is_same_v<
      typename std::remove_cvref_t<
          DNADecay<typename CallableTraits<CrossoverFunction>::ReturnType>>,
      typename std::remove_cvref_t<
          typename CallableTraits<CrossoverFunction>::template ArgType<0>>>;
  requires std::is_same_v<
      typename std::remove_cvref_t<
          DNADecay<typename CallableTraits<CrossoverFunction>::ReturnType>>,
      typename std::remove_cvref_t<
          typename CallableTraits<CrossoverFunction>::template ArgType<1>>>;
};

template <typename CrossoverGenerator>
concept CrossoverGeneratorConcept = requires() {
  requires CallableTraits<CrossoverGenerator>::nArguments == 0;
  requires CrossoverFunctionConcept<
      typename CallableTraits<CrossoverGenerator>::ReturnType>;
};

template <typename CrossoverFG>
concept CrossoverFunctionOrGeneratorConcept =
    CrossoverGeneratorConcept<CrossoverFG> ||
    CrossoverFunctionConcept<CrossoverFG>;

} // namespace Evolution

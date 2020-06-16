#pragma once
#include <functional>
#include <tuple>
#include <type_traits>

namespace Evolution {
template <class Function> struct ArgumentTraits final {
private:
  template <class Function_> struct FunctionArgTypes;
  template <class Function_, class... Args>
  struct FunctionArgTypes<Function_(Args...)> {
    using Types = typename std::tuple<Function_, Args...>;
  };
  template <class ParenthesisOperator> struct LambdaArgTypes;
  template <class ParenthesisOperator, class Result, class... Args>
  struct LambdaArgTypes<Result (ParenthesisOperator::*)(Args...) const> {
    using Types = typename std::tuple<Result, Args...>;
  };
  template <class ParenthesisOperator, class Result, class... Args>
  struct LambdaArgTypes<Result (ParenthesisOperator::*)(Args...)> {
    using Types = typename std::tuple<Result, Args...>;
  };

  template <class Function_, bool isFunction> struct ArgTypes;
  template <class Function_> struct ArgTypes<Function_, true> {
    using Types = typename FunctionArgTypes<Function_>::Types;
  };
  template <class Function_> struct ArgTypes<Function_, false> {
    using Types =
        typename LambdaArgTypes<decltype(&Function_::operator())>::Types;
  };

  using Types =
      typename ArgTypes<Function, std::is_function_v<Function>>::Types;

public:
  auto constexpr static nArguments = std::tuple_size_v<Types> - 1;
  template <size_t n> using Type = std::tuple_element_t<n, Types>;
  template <size_t n>
  auto constexpr static isLValueReference = std::is_lvalue_reference_v<Type<n>>;
  template <size_t n>
  auto constexpr static isRValueReference = std::is_rvalue_reference_v<Type<n>>;
  template <size_t n>
  auto constexpr static isReference = std::is_reference_v<Type<n>>;
  template <size_t n> auto constexpr static isValue = !isReference<n>;
  template <size_t n>
  auto constexpr static isConst =
      std::is_const_v<std::remove_reference_t<Type<n>>>;
};
} // namespace Evolution

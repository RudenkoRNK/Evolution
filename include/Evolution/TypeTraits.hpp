#pragma once
#include <functional>
#include <tuple>
#include <type_traits>

namespace Evolution {
class TypeTraits final {
private:
  template <typename T, template <typename...> class Template>
  struct isInstanceOf_ : std::false_type {};
  template <template <typename...> class Template, typename... Args>
  struct isInstanceOf_<Template<Args...>, Template> : std::true_type {};

public:
  template <class Instance, template <typename...> class Template>
  auto constexpr static isInstanceOf =
      isInstanceOf_<std::remove_reference_t<Instance>, Template>::value;
};

template <class Callable> struct ArgumentTraits final {
private:
  enum class CallableType { Function, Method, Lambda };

  template <class Callable_> struct FunctionArgTypes;
  template <class Callable_, class... Args>
  struct FunctionArgTypes<Callable_(Args...)> {
    using Types = typename std::tuple<Callable_, Args...>;
  };
  template <class CallableAddress> struct LambdaOrMethodArgTypes;
  template <class CallableAddress, class Result, class... Args>
  struct LambdaOrMethodArgTypes<Result (CallableAddress::*)(Args...) const> {
    using Types = typename std::tuple<Result, Args...>;
    auto constexpr static isCallableConst = true;
  };
  template <class CallableAddress, class Result, class... Args>
  struct LambdaOrMethodArgTypes<Result (CallableAddress::*)(Args...)> {
    using Types = typename std::tuple<Result, Args...>;
    auto constexpr static isCallableConst = false;
  };

  template <class Callable_, CallableType> struct ArgTypes;
  template <class Callable_>
  struct ArgTypes<Callable_, CallableType::Function> {
    using Types = typename FunctionArgTypes<Callable_>::Types;
    auto constexpr static isCallableConst = false;
  };
  template <class Callable_> struct ArgTypes<Callable_, CallableType::Lambda> {
    using Types = typename LambdaOrMethodArgTypes<decltype(
        &Callable_::operator())>::Types;
    auto constexpr static isCallableConst = LambdaOrMethodArgTypes<decltype(
        &Callable_::operator())>::isCallableConst;
  };
  template <class Callable_> struct ArgTypes<Callable_, CallableType::Method> {
    using Types = typename LambdaOrMethodArgTypes<Callable_>::Types;
    auto constexpr static isCallableConst =
        LambdaOrMethodArgTypes<Callable_>::isCallableConst;
  };

  auto constexpr static ThisCallableType = std::is_function_v<Callable>
                                               ? CallableType::Function
                                               : std::is_class_v<Callable>
                                                     ? CallableType::Lambda
                                                     : CallableType::Method;

  using ArgTypes_ = typename ArgTypes<Callable, ThisCallableType>;
  using Types = typename ArgTypes_::Types;

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
  auto constexpr static isCallableConst = ArgTypes_::isCallableConst;
};
} // namespace Evolution

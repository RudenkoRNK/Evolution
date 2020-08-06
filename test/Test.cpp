#define BOOST_TEST_MODULE Test

#include "Evolution/Environment.hpp"
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlow.hpp"
#include "Evolution/Utility.hpp"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/flow_graph.h"
#include <boost/test/included/unit_test.hpp>
#include <random>

using namespace Evolution;

BOOST_AUTO_TEST_CASE(first_test) {
  auto sf = StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  BOOST_TEST(!sf.Verify());
  sf.SetEvaluate(s0);
  BOOST_TEST(sf.Verify());
}

BOOST_AUTO_TEST_CASE(second_test) {
  auto sf = StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  auto s1 = sf.GetOrAddInitialState(1);
  auto s2 = sf.AddCrossover(s0, s1);
  sf.SetEvaluate(s0);
  sf.SetEvaluate(s2);

  auto Evaluate = [](int x) { return x * 1.0; };
  auto Mutate = [](int x) { return x + 1; };
  auto Crossover = [](int x, int y) { return x + y; };
  auto population = std::vector<int>{1, 1};

  auto env =
      Environment(Evaluate, Mutate, Crossover, sf, true, std::move(population));
  env.Run();
  env.Run();
  env.Run();
  population = env.GetPopulation();
  auto grades = env.GetGrades();
  BOOST_TEST(population.at(0) == 5);
  BOOST_TEST(population.at(1) == 3);
  BOOST_TEST(grades.at(0) == 5);
  BOOST_TEST(grades.at(1) == 3);
}

BOOST_AUTO_TEST_CASE(arg_traits_test) {
  auto lambda1 = [](std::string const &) { return 0; };
  using T = ArgumentTraits<decltype(lambda1)>::Type<1>;
  auto lambda2 = [&](T t) { return lambda1(t); };

  BOOST_TEST(ArgumentTraits<decltype(lambda2)>::isConst<1>);
  BOOST_TEST(ArgumentTraits<decltype(lambda2)>::isLValueReference<1>);
}

BOOST_AUTO_TEST_CASE(perm_test) {
  auto perm = std::vector<size_t>{5, 2, 3, 0, 1, 4};
  auto v = std::vector<size_t>(perm.size());
  std::iota(v.begin(), v.end(), 0);
  Permute(v, perm);
  BOOST_TEST(v == perm);
}

BOOST_AUTO_TEST_CASE(quadratic_equation) {
  // Try to solve quadratic equation with guessing
  // x^2 + bx + c = 0
  // b = -6, c = 9, x0 = x1 = 3
  double b = -6;
  double c = 9;
  auto Evaluate = [b, c](double x) { return -(x * x + b * x + c); };
  auto MutateGen = []() {
    auto gen = std::mt19937(0);
    auto rand = std::uniform_real_distribution<>();
    return [gen, rand](double x) mutable { return x * (rand(gen) * 4 - 2); };
  };
  auto Crossover = [](double x, double y) { return (x + y) / 2; };
  auto Generator = []() -> double {
    auto gen = std::mt19937(0);
    auto rand = std::uniform_real_distribution<>();
    return (rand(gen) - 0.5) * 1000000 - 10000;
  };

  auto N = 100;
  using Env =
      Environment<decltype(Evaluate), decltype(MutateGen), decltype(Crossover)>;

  auto population = Env::GeneratePopulation(N, Generator);
  auto sf = Env::GenerateStateFlow(N);
  auto env = Environment(Evaluate, MutateGen, Crossover, sf, true,
                         std::move(population));

  for (auto i = size_t{0}; i < 500; ++i)
    env.Run();

  sf = StateFlow{};
  for (auto i = size_t{0}; i < N; ++i)
    sf.SetEvaluate(sf.GetOrAddInitialState(i));
  env.SetStateFlow(std::move(sf));
  env.Run();

  BOOST_TEST(std::abs(env.GetPopulation().at(0) - 3) < 0.000001);
}

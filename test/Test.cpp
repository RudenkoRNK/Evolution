#define BOOST_TEST_MODULE Test

#include "Evolution/Environment.hpp"
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlow.hpp"
#include "Evolution/Utility.hpp"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/flow_graph.h"
#include <boost/test/included/unit_test.hpp>
#include <functional>
#include <random>

BOOST_AUTO_TEST_CASE(first_test) {
  auto sf = Evolution::StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  BOOST_TEST(sf.IsNotReady().has_value());
  sf.SetEvaluate(s0);
  BOOST_TEST(!sf.IsNotReady().has_value());
}

BOOST_AUTO_TEST_CASE(second_test) {
  auto sf = Evolution::StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  auto s1 = sf.GetOrAddInitialState(1);
  auto s2 = sf.AddCrossover(s0, s1);
  sf.SetEvaluate(s0);
  sf.SetEvaluate(s2);

  auto Evaluate = [](int x) { return x * 1.0; };
  auto Mutate = [](int x) { return x + 1; };
  auto Crossover = [](int x, int y) { return x + y; };
  auto generator = []() -> int { return 1; };

  auto env =
      Evolution::Environment(generator, Evaluate, Mutate, Crossover, sf, true);
  env.Run();
  env.Run();
  env.Run();
  auto population = env.GetPopulation();
  auto grades = env.GetGrades();
  BOOST_TEST(population.at(0) == 5);
  BOOST_TEST(population.at(1) == 3);
  BOOST_TEST(grades.at(0) == 5);
  BOOST_TEST(grades.at(1) == 3);
}

BOOST_AUTO_TEST_CASE(arg_traits_test) {
  auto lambda1 = [](std::string const &) { return 0; };
  using T = Utility::ArgumentTraits<decltype(lambda1)>::Type<1>;
  auto lambda2 = [&](T t) { return lambda1(t); };

  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda1)>::isConst<1>);
  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda1)>::isLValueReference<1>);
  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda2)>::isConst<1>);
  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda2)>::isLValueReference<1>);
}

BOOST_AUTO_TEST_CASE(arg_traits_test_2) {
  auto x = 0;
  auto lambda1 = [x](std::string const &) mutable {
    ++x;
    return 0;
  };
  auto const lambda2 = [x](std::string const &) mutable {
    ++x;
    return 0;
  };

  auto lambda3 = [&](std::string const &) {
    ++x;
    return 0;
  };
  auto const lambda4 = [&](std::string const &) {
    ++x;
    return 0;
  };

  BOOST_TEST(!Utility::ArgumentTraits<decltype(lambda1)>::isCallableConst);
  BOOST_TEST(!Utility::ArgumentTraits<decltype(lambda2)>::isCallableConst);
  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda3)>::isCallableConst);
  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda4)>::isCallableConst);
}

BOOST_AUTO_TEST_CASE(perm_test) {
  auto perm = std::vector<size_t>{5, 2, 3, 0, 1, 4};
  auto v = std::vector<size_t>(perm.size());
  std::iota(v.begin(), v.end(), 0);
  Utility::Permute(v, perm);
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
  auto maxRand = 1000000;
  auto Generator = [&]() -> double {
    auto gen = std::mt19937(0);
    auto rand = std::uniform_real_distribution<>();
    return (rand(gen) - 0.5) * maxRand - 10000;
  };

  auto N = 100;
  using Env = Evolution::Environment<decltype(Evaluate), decltype(MutateGen),
                                     decltype(Crossover)>;

  auto sf = Env::GenerateStateFlow(N);
  auto env = Evolution::Environment(Generator, Evaluate, MutateGen, Crossover,
                                    sf, true);

  for (auto i = size_t{0}; i < 500; ++i)
    env.Run();

  auto sf2 = Evolution::StateFlow{};
  for (auto i = size_t{0}; i < N; ++i)
    sf2.SetEvaluate(sf2.GetOrAddInitialState(i));
  env.SetStateFlow(std::move(sf2));
  env.Run();

  BOOST_TEST(std::abs(env.GetPopulation().at(0) - 3) < 0.000001);

  auto env2 = Evolution::Environment(Generator, Evaluate, MutateGen, Crossover,
                                     sf, true);
  auto absd = 0.00001;
  auto Optimize = [&]() {
    auto nGens = size_t{0};
    env2.Run();
    auto best = env2.GetGrades();
    auto diff = absd + 1;
    while (diff > absd) {
      env2.Run();
      auto const &g2 = env2.GetGrades();
      diff = std::inner_product(best.begin(), best.end(), g2.begin(), 0.0,
                                std::plus{}, [](auto x1, auto x2) {
                                  if (x2 < x1)
                                    return 0.0;
                                  return x2 - x1;
                                });
      for (auto i = size_t{0}; i < g2.size(); ++i)
        if (g2.at(i) > best.at(i))
          best.at(i) = g2.at(i);
      ++nGens;
    }
    return nGens;
  };

  auto gens1 = Optimize();
  env2.RegeneratePopulation();
  // Check that giving an answer reduces the number of iterations
  env2.SetPopulation({3});
  auto gens2 = Optimize();
  BOOST_TEST(gens2 < gens1);
}

BOOST_AUTO_TEST_CASE(swap_args_test) {
  auto sf = Evolution::StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  auto s1 = sf.GetOrAddInitialState(1);
  auto s4 = sf.AddCrossover(s0, s1);
  auto s6 = sf.AddMutate(s1);
  sf.SetEvaluate(s4);
  sf.SetEvaluate(s6);

  auto sf2 = sf;
  sf2.SetSwapArgumentsAllowedInCrossover();

  auto copyCounter = std::atomic<size_t>{0};
  struct DNA {
    std::atomic<size_t> &copyCounter;
    DNA(std::atomic<size_t> &copyCounter) : copyCounter(copyCounter) {}
    DNA(DNA const &dna) noexcept : copyCounter(dna.copyCounter) {
      ++copyCounter;
    }
    DNA(DNA &&dna) noexcept : copyCounter(dna.copyCounter) {}
    DNA &operator=(DNA const &) {
      ++copyCounter;
      return *this;
    }
    DNA &operator=(DNA &&) noexcept { return *this; }
  };

  auto Evaluate = [](DNA const &x) { return 1.0; };
  auto Mutate = [](DNA x) { return x; };
  auto Crossover = [](DNA const &x, DNA y) {
    auto z = DNA(x);
    return y;
  };
  auto generator = [&]() -> DNA { return {copyCounter}; };
  auto env = Evolution::Environment(generator, Evaluate, Mutate, Crossover, sf);
  BOOST_TEST(copyCounter == 0);
  env.Run();
  auto ctr1 = size_t{copyCounter};
  BOOST_TEST(ctr1 <= 5);
  copyCounter = 0;
  env.SetStateFlow(std::move(sf2));
  env.Run();
  auto ctr2 = size_t{copyCounter};
  BOOST_TEST(ctr2 <= 4);
  BOOST_TEST(ctr1 > ctr2);
}

BOOST_AUTO_TEST_CASE(grades_preserve_test) {
  // This test asserts that initial evaluated are not reevaluated
  auto sf = Evolution::StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  auto s1 = sf.GetOrAddInitialState(1);
  auto s2 = sf.GetOrAddInitialState(2);
  auto s3 = sf.GetOrAddInitialState(3);
  auto s4 = sf.AddCrossover(s0, s1);
  auto s5 = sf.AddCrossover(s3, s2);
  auto s6 = sf.AddCrossover(s4, s5);
  auto s7 = sf.AddCrossover(s5, s4);
  sf.SetEvaluate(s6);
  sf.SetEvaluate(s7);
  sf.SetEvaluate(s1);
  sf.SetEvaluate(s2);

  auto Evaluate = [](int x) {
    auto gen = std::mt19937(0);
    auto rand = std::uniform_int_distribution<>();
    return rand(gen);
  };
  auto Mutate = [](int x) { return x + 1; };
  auto Crossover = [](int x, int y) { return x + y; };
  auto generator = []() -> int { return 1; };
  auto env = Evolution::Environment(generator, Evaluate, Mutate, Crossover, sf);
  auto grades = env.GetGrades();
  auto g1 = grades.at(1);
  auto g2 = grades.at(2);
  env.Run(10);
  grades = env.GetGrades();
  BOOST_TEST(g1 == grades.at(1));
  BOOST_TEST(g2 == grades.at(2));
}

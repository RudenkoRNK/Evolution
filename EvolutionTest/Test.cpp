#define BOOST_TEST_MODULE Test

#include "Evolution/Environment.hpp"
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlow.hpp"
#include "Evolution/Utils.hpp"
#include <boost/test/included/unit_test.hpp>
#include <functional>
#include <random>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/flow_graph.h>

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

  auto sf = Evolution::GenerateStateFlow(N);
  auto env = Evolution::Environment(Generator, Evaluate, MutateGen, Crossover,
                                    sf, true);

  for (auto i = size_t{0}; i != 500; ++i)
    env.Run();

  auto sf2 = Evolution::StateFlow{};
  for (auto i = size_t{0}; i != N; ++i)
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
      for (auto i = size_t{0}; i != g2.size(); ++i)
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
    // NOLINTNEXTLINE
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
  env.Run(size_t{10});
  grades = env.GetGrades();
  BOOST_TEST(g1 == grades.at(1));
  BOOST_TEST(g2 == grades.at(2));
}

BOOST_AUTO_TEST_CASE(random_flow_test) {
  auto Evaluate = [](double x) {
    auto static thread_local gen = std::mt19937(412);
    auto rand = std::uniform_real_distribution<>();
    return rand(gen);
  };
  auto Mutate = [](double x) {
    auto static thread_local gen = std::mt19937(433);
    auto rand = std::uniform_real_distribution<>();
    return rand(gen);
  };
  auto Crossover = [](double x, double y) {
    auto static thread_local gen = std::mt19937(444);
    auto rand = std::uniform_real_distribution<>();
    return rand(gen);
  };
  auto Generator = []() -> double {
    auto static thread_local gen = std::mt19937(100);
    auto rand = std::uniform_real_distribution<>();
    return rand(gen);
  };

  auto gen = std::mt19937(500);
  auto rand = std::uniform_int_distribution<>();
  auto nextInd = 0;
  auto GetRandomState = [&](Evolution::StateFlow &sf) {
    auto &&[sb, se] = sf.GetStates();
    auto i = rand(gen) % (se - sb);
    return *(sb + i);
  };
  auto AddInitial = [&](Evolution::StateFlow &sf) {
    if (rand(gen) % 3 == 0)
      ++nextInd;
    sf.GetOrAddInitialState(++nextInd);
  };
  auto AddMutate = [&](Evolution::StateFlow &sf) {
    return sf.AddMutate(GetRandomState(sf));
  };
  auto AddCrossover = [&](Evolution::StateFlow &sf) {
    auto s1 = GetRandomState(sf);
    auto s2 = GetRandomState(sf);
    if (s1 == s2)
      s2 = sf.AddMutate(s2);
    return sf.AddCrossover(s1, s2);
  };
  auto AddEvaluate = [&](Evolution::StateFlow &sf) {
    sf.SetEvaluate(GetRandomState(sf));
  };
  auto AddEvaluateOnLeaves = [&](Evolution::StateFlow &sf) {
    auto &&[sb, se] = sf.GetStates();
    for (auto si = sb; si != se; ++si)
      if (sf.IsLeaf(*si))
        sf.SetEvaluate(*si);
  };
  auto SFGen = [&]() {
    nextInd = 0;
    auto rand = std::uniform_int_distribution<>();
    auto sf = Evolution::StateFlow{};
    sf.GetOrAddInitialState(++nextInd);
    sf.GetOrAddInitialState(++nextInd);
    for (auto i = 0; i != 1000; ++i) {
      auto dice = rand(gen);
      if (dice % 10 == 0)
        AddInitial(sf);
      else if (dice % 2 == 0)
        AddMutate(sf);
      else
        AddCrossover(sf);
    };
    for (auto i = 0; i != 200; ++i)
      AddEvaluate(sf);
    AddEvaluateOnLeaves(sf);
    auto nEvs = int(sf.GetNEvaluates());
    auto minEvals = int(std::max(sf.GetIndex(sf.GetMaxIndexState()) + 1,
                                 sf.GetInitialStates().size()));
    for (auto i = nEvs; i < minEvals; ++i)
      sf.SetEvaluate(AddMutate(sf));

    return sf;
  };

  auto env = Evolution::Environment(Generator, Evaluate, Mutate, Crossover,
                                    SFGen(), true);
  for (auto i = 0; i != 10; ++i) {
    env.RegeneratePopulation();
    env.SetStateFlow(Evolution::StateFlow(SFGen()));
    env.Run(size_t{5});
  }
}

BOOST_AUTO_TEST_CASE(ctor_test) {
  struct DNA {
    int x;
    int y;
    DNA() = delete;
    DNA(int x, int y) : x(x), y(y) {}
  };
  using Grade = DNA;

  auto Evaluate = [](DNA x) { return x; };
  auto Mutate = [](DNA x) { return x; };
  auto Crossover = [](DNA x, DNA y) { return x; };
  auto generator = []() -> DNA { return DNA(1, 2); };

  auto env = Evolution::Environment(generator, Evaluate, Mutate, Crossover,
                                    Evolution::GenerateStateFlow(10), false,
                                    [](auto const &x, auto const &y) {
                                      return Utility::GetIndices(x.size());
                                    });
  env.Run();
}

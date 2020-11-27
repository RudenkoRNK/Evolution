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
  auto opts = Evolution::EnvironmentOptions{};

  auto env =
      Evolution::Environment(generator, Evaluate, Mutate, Crossover, sf, opts);
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

BOOST_AUTO_TEST_CASE(empty_sf_test) {
  auto cnt = std::atomic_int{0};
  auto Evaluate = [&](int x) {
    ++cnt;
    return x;
  };
  auto Mutate = [&](int x) {
    ++cnt;
    return x + 1;
  };
  auto Crossover = [&](int x, int y) {
    ++cnt;
    return x + y;
  };
  auto generator = [&]() -> int {
    ++cnt;
    return 1;
  };
  auto sf = Evolution::StateFlow{};
  BOOST_TEST(!sf.IsNotReady());
  auto opts = Evolution::EnvironmentOptions{};
  auto env =
      Evolution::Environment(generator, Evaluate, Mutate, Crossover, sf, opts);
  env.Run(size_t{10});
  BOOST_TEST(cnt <= 7);
  cnt = 0;
  opts.isEvaluateLightweight = Utility::AutoOption::False();
  opts.isMutateLightweight = Utility::AutoOption::False();
  opts.isCrossoverLightweight = Utility::AutoOption::False();
  auto env2 =
      Evolution::Environment(generator, Evaluate, Mutate, Crossover, sf, opts);
  env2.Run(size_t{10});
  BOOST_TEST(cnt == 0);
}

BOOST_AUTO_TEST_CASE(sf_generator_test) {
  for (auto i = size_t{0}; i < size_t{100}; ++i) {
    auto sf = Evolution::GenerateStateFlow(i);
    BOOST_TEST(!sf.IsNotReady());
  }
}

int Evaluate11(int x) { return x; }

BOOST_AUTO_TEST_CASE(env_ctor_test) {
  auto sf = Evolution::GenerateStateFlow(2);

  auto a = 1;
  auto Evaluate1 = [](int x) { return x; };
  auto Evaluate2 = [&](int x) { return a; };
  auto Evaluate3 = [a](int x) { return a; };
  auto Evaluate4 = [a](int x) mutable {
    return a++ /*not thread safe but does not matter*/;
  };
  auto Evaluate5 = std::function(Evaluate1);
  auto Evaluate6 = std::function(Evaluate2);
  auto Evaluate7 = std::function(Evaluate3);
  auto Evaluate8 = std::function(Evaluate4);
  struct Evaluate9T {
    int operator()(int x) { return x; }
  };
  struct Evaluate10T {
    int operator()(int x) const { return x; }
  };
  auto Evaluate9 = Evaluate9T{};
  auto Evaluate10 = Evaluate10T{};
  auto Evaluate12 = []() {
    auto gen = std::mt19937(0);
    auto rand = std::uniform_real_distribution<>();
    return
        [gen, rand](int x) mutable { return static_cast<int>(rand(gen) * 10); };
  };
  auto Evaluate13 = []() { return [](int x) { return x; }; };
  auto Evaluate14 = []() { return Evaluate9T{}; };
  auto Evaluate15 = []() { return Evaluate10T{}; };

  auto Mutate = [](int x) { return x + 1; };
  auto Crossover = [](int x, int y) { return x + y; };
  auto generator = []() -> int { return 1; };
  auto opts = Evolution::EnvironmentOptions{};

  Evolution::Environment(generator, Evaluate1, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate2, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate3, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate4, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate5, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate6, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate7, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate8, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate9, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate10, Mutate, Crossover, sf, opts);
#pragma warning(disable : 4180)
  Evolution::Environment(generator, Evaluate11, Mutate, Crossover, sf, opts);
#pragma warning(default : 4180)
  Evolution::Environment(generator, Evaluate12, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate13, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate14, Mutate, Crossover, sf, opts);
  Evolution::Environment(generator, Evaluate15, Mutate, Crossover, sf, opts);
}

BOOST_AUTO_TEST_CASE(env_ctor_test2) {
  auto sf = Evolution::GenerateStateFlow(10);
  auto Mutate1 = [](std::string x) -> std::string { return x; };
  auto Mutate2 = [](std::string &x) -> std::string { return x; };
  auto Mutate3 = [](std::string const &x) -> std::string { return x; };
  auto Mutate4 = [](std::string &&x) -> std::string { return x; };
  auto Mutate5 = [](std::string &x) -> std::string & { return x; };
  auto Mutate6 = [](std::string &x) -> std::string const & { return x; };
  auto Mutate7 = [](std::string & x) -> std::string && { return std::move(x); };
  auto Mutate8 = [](std::string &x) -> std::string { return x; };
  auto Mutate9 = [](std::string &x) -> std::unique_ptr<std::string> {
    return std::make_unique<std::string>(std::move(x));
  };

  auto Evaluate = [](std::string const &x) {
    auto gen = std::mt19937(0);
    auto rand = std::uniform_int_distribution<>(0, 255);
    return rand(gen);
  };
  auto Crossover = [](std::string const &x,
                      std::string const &y) -> decltype(auto) { return x + y; };
  auto generator = []() -> std::string {
    auto gen = std::mt19937(0);
    auto rand = std::uniform_int_distribution<>(0, 255);
    return std::string(1, static_cast<char>(rand(gen)));
  };
  auto opts = Evolution::EnvironmentOptions{};

  Evolution::Environment(generator, Evaluate, Mutate1, Crossover, sf, opts)
      .Run(size_t{3});
  Evolution::Environment(generator, Evaluate, Mutate2, Crossover, sf, opts)
      .Run(size_t{3});
  Evolution::Environment(generator, Evaluate, Mutate3, Crossover, sf, opts)
      .Run(size_t{3});
  Evolution::Environment(generator, Evaluate, Mutate4, Crossover, sf, opts)
      .Run(size_t{3});
  Evolution::Environment(generator, Evaluate, Mutate5, Crossover, sf, opts)
      .Run(size_t{3});
  Evolution::Environment(generator, Evaluate, Mutate6, Crossover, sf, opts)
      .Run(size_t{3});
  Evolution::Environment(generator, Evaluate, Mutate7, Crossover, sf, opts)
      .Run(size_t{3});
  Evolution::Environment(generator, Evaluate, Mutate8, Crossover, sf, opts)
      .Run(size_t{3});
  Evolution::Environment(generator, Evaluate, Mutate9, Crossover, sf, opts)
      .Run(size_t{3});
  static_assert(Evolution::MutateFunctionOrGeneratorConcept<decltype(Mutate9)>);
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
  auto opts = Evolution::EnvironmentOptions{};

  auto N = 100;

  auto sf = Evolution::GenerateStateFlow(N);
  auto env = Evolution::Environment(Generator, Evaluate, MutateGen, Crossover,
                                    sf, opts);

  for (auto i = size_t{0}; i != 500; ++i)
    env.Run();

  auto sf2 = Evolution::StateFlow{};
  for (auto i = size_t{0}; i != N; ++i)
    sf2.SetEvaluate(sf2.GetOrAddInitialState(i));
  env.SetStateFlow(std::move(sf2));
  env.Run();

  BOOST_TEST(std::abs(env.GetPopulation().at(0) - 3) < 0.000001);

  auto env2 = Evolution::Environment(Generator, Evaluate, MutateGen, Crossover,
                                     sf, opts);
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
  auto opts = Evolution::EnvironmentOptions{};
  opts.isEvaluateLightweight = Utility::AutoOption::True();
  opts.isMutateLightweight = Utility::AutoOption::True();
  opts.isCrossoverLightweight = Utility::AutoOption::True();
  auto env =
      Evolution::Environment(generator, Evaluate, Mutate, Crossover, sf, opts);
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
  auto opts = Evolution::EnvironmentOptions{};
  opts.isEvaluateLightweight = Utility::AutoOption::True();
  opts.isMutateLightweight = Utility::AutoOption::True();
  opts.isCrossoverLightweight = Utility::AutoOption::True();
  auto env =
      Evolution::Environment(generator, Evaluate, Mutate, Crossover, sf, opts);
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
    auto &&inits = sf.GetInitialStates();
    auto maxState =
        *std::max_element(inits.begin(), inits.end(), [&](auto s1, auto s2) {
          return sf.GetIndex(s1) < sf.GetIndex(s2);
        });
    auto minEvals =
        int(std::max(sf.GetIndex(maxState) + 1, sf.GetInitialStates().size()));
    for (auto i = nEvs; i < minEvals; ++i)
      sf.SetEvaluate(AddMutate(sf));

    return sf;
  };
  auto opts = Evolution::EnvironmentOptions{};

  auto env = Evolution::Environment(Generator, Evaluate, Mutate, Crossover,
                                    SFGen(), opts);
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
  auto opts = Evolution::EnvironmentOptions{};
  opts.isEvaluateLightweight = Utility::AutoOption::True();
  opts.isMutateLightweight = Utility::AutoOption::True();
  opts.isCrossoverLightweight = Utility::AutoOption::True();

  auto env = Evolution::Environment(generator, Evaluate, Mutate, Crossover,
                                    Evolution::GenerateStateFlow(10), opts,
                                    [](auto const &x, auto const &y) {
                                      return Utility::GetIndices(x.size());
                                    });
  env.Run();
}

BOOST_AUTO_TEST_CASE(copy_test) {
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

  auto sf = Evolution::StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  auto s1 = sf.GetOrAddInitialState(1);
  auto s2 = sf.AddMutate(s0);
  auto s3 = sf.AddCrossover(s2, s1);
  sf.SetEvaluate(s2);
  sf.SetEvaluate(s3);

  auto orig = DNA{copyCounter};
  auto generator = [&]() -> DNA { return DNA(orig); };
  auto Evaluate = [](DNA const &x) { return 0; };
  auto Mutate1 = [](DNA const &x) { return DNA(x); };
  auto Crossover1 = [](DNA const &x, DNA const &y) { return DNA(x); };
  auto opts = Evolution::EnvironmentOptions{};
  opts.allowMoveFromPopulation = true;
  opts.isEvaluateLightweight = Utility::AutoOption::Auto();
  opts.isMutateLightweight = Utility::AutoOption::Auto();
  opts.isCrossoverLightweight = Utility::AutoOption::Auto();

  auto opts2 = Evolution::EnvironmentOptions{};
  opts2.allowMoveFromPopulation = false;
  opts2.isEvaluateLightweight = Utility::AutoOption::Auto();
  opts2.isMutateLightweight = Utility::AutoOption::Auto();
  opts2.isCrossoverLightweight = Utility::AutoOption::Auto();

  auto env1 = Evolution::Environment(generator, Evaluate, Mutate1, Crossover1,
                                     sf, opts);
  BOOST_TEST(copyCounter <= 7);
  copyCounter = 0;

  env1.Run();
  BOOST_TEST(copyCounter <= 4);
  copyCounter = 0;

  auto env5 = Evolution::Environment(generator, Evaluate, Mutate1, Crossover1,
                                     sf, opts2);
  BOOST_TEST(copyCounter <= 7);
  copyCounter = 0;

  env5.Run();
  BOOST_TEST(copyCounter <= 4);
  copyCounter = 0;

  auto Mutate2 = [](DNA &&x) -> DNA & { return x; };
  auto Crossover2 = [](DNA &&x, DNA &&y) -> DNA & { return x; };

  auto env2 = Evolution::Environment(generator, Evaluate, Mutate2, Crossover2,
                                     sf, opts);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env2.Run();
  BOOST_TEST(copyCounter <= 3);
  copyCounter = 0;

  auto env6 = Evolution::Environment(generator, Evaluate, Mutate2, Crossover2,
                                     sf, opts2);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env6.Run();
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  auto Mutate3 = [](DNA x) -> DNA { return x; };
  auto Crossover3 = [](DNA x, DNA y) -> DNA { return x; };

  auto env3 = Evolution::Environment(generator, Evaluate, Mutate3, Crossover3,
                                     sf, opts);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env3.Run();
  BOOST_TEST(copyCounter <= 3);
  copyCounter = 0;

  auto env7 = Evolution::Environment(generator, Evaluate, Mutate3, Crossover3,
                                     sf, opts2);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env7.Run();
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  auto Mutate4 = [&](DNA const &x) -> DNA & { return orig; };
  auto Crossover4 = [&](DNA const &x, DNA const &y) -> DNA & { return orig; };

  auto env4 = Evolution::Environment(generator, Evaluate, Mutate4, Crossover4,
                                     sf, opts);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env4.Run(size_t{100});
  BOOST_TEST(copyCounter == 0);

  auto env8 = Evolution::Environment(generator, Evaluate, Mutate4, Crossover4,
                                     sf, opts2);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env8.Run(size_t{100});
  BOOST_TEST(copyCounter <= 200);
}

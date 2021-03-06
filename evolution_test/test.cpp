#define BOOST_TEST_MODULE Test
#define NOMINMAX

#include "evolution/environment.hpp"
#include "evolution/state_flow.hpp"
#include "evolution/task_flow.hpp"
#include "evolution/utils.hpp"
#include <boost/test/included/unit_test.hpp>
#include <functional>
#include <random>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/flow_graph.h>

auto static constexpr verbose = false;
using namespace Evolution;

BOOST_AUTO_TEST_CASE(first_test) {
  auto sf = StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  BOOST_TEST(sf.IsNotReady().has_value());
  sf.SetEvaluate(s0);
  BOOST_TEST(!sf.IsNotReady().has_value());
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
  auto generator = []() -> int { return 1; };
  auto opts = EnvironmentOptions{};

  auto env = Environment(generator, Evaluate, Mutate, Crossover, sf, opts);
  using Env = std::remove_cvref_t<decltype(env)>;
  static_assert(std::copyable<Env>);

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

BOOST_AUTO_TEST_CASE(swap_copy_env_test) {
  using std::swap;
  auto sf1 = GenerateStateFlow(10);
  auto sf2 = GenerateStateFlow(20);

  auto Evaluate = [](int x) { return x * 1.0; };
  auto Mutate = [](int x) { return x + 1; };
  auto Crossover = [](int x, int y) { return x + y; };
  auto generator = []() -> int { return 1; };
  auto opts = EnvironmentOptions{};

  auto env1 = Environment(generator, Evaluate, Mutate, Crossover, sf1, opts);
  auto env2 = Environment(generator, Evaluate, Mutate, Crossover, sf2, opts);
  using Env = std::remove_cvref_t<decltype(env1)>;
  static_assert(std::is_nothrow_swappable_v<Env>);
  static_assert(noexcept(swap(env1, env2)));

  env1.Run();
  env2.Run();
  swap(env1, env2);
  env1.Run();
  env2.Run();
  auto env3 = env1;
  auto env4 = env2;
  env1.Run();
  env2.Run();
  env3.Run();
  env4.Run();
  env1 = env4;
  env2 = env3;
  env1.Run();
  env2.Run();
  env3.Run();
  env4.Run();
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
  auto sf = StateFlow{};
  BOOST_TEST(!sf.IsNotReady());
  auto opts = EnvironmentOptions{};
  auto env = Environment(generator, Evaluate, Mutate, Crossover, sf, opts);
  env.Run(size_t{10});
  BOOST_TEST(cnt <= 7);
  cnt = 0;
  opts.isEvaluateLightweight = Utility::AutoOption::False();
  opts.isMutateLightweight = Utility::AutoOption::False();
  opts.isCrossoverLightweight = Utility::AutoOption::False();
  auto env2 = Environment(generator, Evaluate, Mutate, Crossover, sf, opts);
  env2.Run(size_t{10});
  BOOST_TEST(cnt == 0);
}

BOOST_AUTO_TEST_CASE(sf_generator_test) {
  for (auto i = size_t{0}; i < size_t{100}; ++i) {
    auto sf = GenerateStateFlow(i);
    BOOST_TEST(!sf.IsNotReady());
  }
}

int Evaluate11(int x) { return x; }

BOOST_AUTO_TEST_CASE(env_ctor_test) {
  auto sf = GenerateStateFlow(2);

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
  auto opts = EnvironmentOptions{};

  Environment(generator, Evaluate1, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate2, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate3, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate4, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate5, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate6, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate7, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate8, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate9, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate10, Mutate, Crossover, sf, opts);
#pragma warning(disable : 4180)
  Environment(generator, Evaluate11, Mutate, Crossover, sf, opts);
#pragma warning(default : 4180)
  Environment(generator, Evaluate12, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate13, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate14, Mutate, Crossover, sf, opts);
  Environment(generator, Evaluate15, Mutate, Crossover, sf, opts);
}

BOOST_AUTO_TEST_CASE(env_ctor_test2) {
  auto sf = GenerateStateFlow(10);
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
  auto opts = EnvironmentOptions{};

  Environment(generator, Evaluate, Mutate1, Crossover, sf, opts).Run(size_t{3});
  Environment(generator, Evaluate, Mutate2, Crossover, sf, opts).Run(size_t{3});
  Environment(generator, Evaluate, Mutate3, Crossover, sf, opts).Run(size_t{3});
  Environment(generator, Evaluate, Mutate4, Crossover, sf, opts).Run(size_t{3});
  Environment(generator, Evaluate, Mutate5, Crossover, sf, opts).Run(size_t{3});
  Environment(generator, Evaluate, Mutate6, Crossover, sf, opts).Run(size_t{3});
  Environment(generator, Evaluate, Mutate7, Crossover, sf, opts).Run(size_t{3});
  Environment(generator, Evaluate, Mutate8, Crossover, sf, opts).Run(size_t{3});
  Environment(generator, Evaluate, Mutate9, Crossover, sf, opts).Run(size_t{3});
  static_assert(MutateFunctionOrGeneratorConcept<decltype(Mutate9)>);
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
    return [rand, gen](double x) mutable { return x * (rand(gen) * 4 - 2); };
  };
  auto Crossover = [](double x, double y) { return (x + y) / 2; };
  auto maxRand = 1000000;
  auto Generator = [&]() -> double { return maxRand; };
  auto opts = EnvironmentOptions{};

  auto N = 100;

  auto sf = GenerateStateFlow(N);
  using EnvironmentT =
      typename Environment<decltype(Evaluate), decltype(MutateGen),
                           decltype(Crossover)>;
  static_assert(std::copyable<EnvironmentT>);

  auto envPtr = std::unique_ptr<EnvironmentT>(nullptr);
  auto copyTime = std::chrono::nanoseconds{};
  auto copyCtorTime = std::chrono::nanoseconds{};
  auto moveTime = std::chrono::nanoseconds{};
  auto moveCtorTime = std::chrono::nanoseconds{};
  {
    auto envT =
        Environment(Generator, Evaluate, MutateGen, Crossover, sf, opts);
    auto envT2 = envT;
    auto envT3 = envT;
    copyCtorTime = Utility::Benchmark([&]() { auto envT2 = envT; }, 100);
    copyTime = Utility::Benchmark([&]() { envT2 = envT; }, 100);
    moveTime = Utility::Benchmark([&]() { envT2 = std::move(envT); }, 100);
    moveCtorTime =
        Utility::Benchmark([&]() { auto envT3 = std::move(envT); }, 100);
    envPtr = std::make_unique<EnvironmentT>(std::move(envT3));
  }
  auto env = *envPtr;
  for (auto i = size_t{0}; i != 500; ++i)
    env.Run();

  auto sf2 = StateFlow{};
  for (auto i = size_t{0}; i != N; ++i)
    sf2.SetEvaluate(sf2.GetOrAddInitialState(i));
  env.SetStateFlow(std::move(sf2));
  env.Run();

  BOOST_TEST(std::abs(env.GetPopulation().at(0) - 3) < 0.000001);
  auto moveTimeC = moveTime.count();
  auto moveCtorTimeC = moveCtorTime.count();
  auto copyTimeC = copyTime.count();
  auto copyCtorTimeC = copyCtorTime.count();

  BOOST_TEST(moveTimeC * 10 < copyCtorTimeC);
  BOOST_TEST(moveTimeC * 10 < copyTimeC);
  BOOST_TEST(moveCtorTimeC * 10 < copyTimeC);

  auto env2 = Environment(Generator, Evaluate, MutateGen, Crossover, sf, opts);
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
  auto sf = StateFlow{};
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
  auto opts = EnvironmentOptions{};
  opts.isEvaluateLightweight = Utility::AutoOption::True();
  opts.isMutateLightweight = Utility::AutoOption::True();
  opts.isCrossoverLightweight = Utility::AutoOption::True();
  auto env = Environment(generator, Evaluate, Mutate, Crossover, sf, opts);
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
  auto sf = StateFlow{};
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
  auto opts = EnvironmentOptions{};
  opts.isEvaluateLightweight = Utility::AutoOption::True();
  opts.isMutateLightweight = Utility::AutoOption::True();
  opts.isCrossoverLightweight = Utility::AutoOption::True();
  auto env = Environment(generator, Evaluate, Mutate, Crossover, sf, opts);
  auto grades = env.GetGrades();
  auto g1 = grades.at(1);
  auto g2 = grades.at(2);
  env.Run(size_t{10});
  grades = env.GetGrades();
  BOOST_TEST(g1 == grades.at(1));
  BOOST_TEST(g2 == grades.at(2));
}

BOOST_AUTO_TEST_CASE(random_flow_test) {
  auto thread_local gen = std::mt19937(412);
  auto Evaluate = [&](double x) {
    auto rand = std::uniform_real_distribution<>();
    return rand(gen);
  };
  auto Mutate = [&](double x) {
    auto rand = std::uniform_real_distribution<>();
    return rand(gen);
  };
  auto Crossover = [&](double x, double y) {
    auto rand = std::uniform_real_distribution<>();
    return rand(gen);
  };
  auto Generator = [&]() -> double {
    auto rand = std::uniform_real_distribution<>();
    return rand(gen);
  };

  auto rand = std::uniform_int_distribution<>();
  auto nextInd = 0;
  auto GetRandomState = [&](StateFlow &sf) {
    auto &&[sb, se] = sf.GetStates();
    auto i = rand(gen) % (se - sb);
    return *(sb + i);
  };
  auto AddInitial = [&](StateFlow &sf) {
    if (rand(gen) % 3 == 0)
      ++nextInd;
    sf.GetOrAddInitialState(++nextInd);
  };
  auto AddMutate = [&](StateFlow &sf) {
    return sf.AddMutate(GetRandomState(sf));
  };
  auto AddCrossover = [&](StateFlow &sf) {
    auto s1 = GetRandomState(sf);
    auto s2 = GetRandomState(sf);
    if (s1 == s2)
      s2 = sf.AddMutate(s2);
    return sf.AddCrossover(s1, s2);
  };
  auto AddEvaluate = [&](StateFlow &sf) { sf.SetEvaluate(GetRandomState(sf)); };
  auto AddEvaluateOnLeaves = [&](StateFlow &sf) {
    auto &&[sb, se] = sf.GetStates();
    for (auto si = sb; si != se; ++si)
      if (sf.IsLeaf(*si))
        sf.SetEvaluate(*si);
  };
  auto SFGen = [&]() {
    nextInd = 0;
    auto rand = std::uniform_int_distribution<>();
    auto sf = StateFlow{};
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
  auto opts = EnvironmentOptions{};

  auto env = Environment(Generator, Evaluate, Mutate, Crossover, SFGen(), opts);
  for (auto i = 0; i != 10; ++i) {
    env.RegeneratePopulation();
    env.SetStateFlow(StateFlow(SFGen()));
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

  struct Grade {
    int x;
    int y;
    Grade() = delete;
    Grade(int x, int y) : x(x), y(y) {}
  };
  auto Evaluate = [](DNA x) { return Grade{x.x, x.y}; };
  auto Mutate = [](DNA x) { return DNA{x}; };
  auto Crossover = [](DNA x, DNA y) { return DNA{x}; };
  auto generator = []() -> DNA { return DNA{1, 2}; };
  auto opts = EnvironmentOptions{};
  opts.isEvaluateLightweight = Utility::AutoOption::True();
  opts.isMutateLightweight = Utility::AutoOption::True();
  opts.isCrossoverLightweight = Utility::AutoOption::True();

  auto env =
      Environment(generator, Evaluate, Mutate, Crossover, GenerateStateFlow(10),
                  opts, [](auto const &x, auto const &y) {
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

  auto sf = StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  auto s1 = sf.GetOrAddInitialState(1);
  auto s2 = sf.AddMutate(s0);
  auto s3 = sf.AddCrossover(s2, s1);
  sf.SetEvaluate(s2);
  sf.SetEvaluate(s3);

  auto orig = DNA{copyCounter};
  auto generator = [&]() -> DNA { return DNA{orig}; };
  auto Evaluate = [](DNA const &x) { return 0; };
  auto Mutate1 = [](DNA const &x) { return DNA{x}; };
  auto Crossover1 = [](DNA const &x, DNA const &y) { return DNA{x}; };
  auto opts = EnvironmentOptions{};
  opts.allowMoveFromPopulation = true;
  opts.isEvaluateLightweight = Utility::AutoOption::Auto();
  opts.isMutateLightweight = Utility::AutoOption::Auto();
  opts.isCrossoverLightweight = Utility::AutoOption::Auto();

  auto opts2 = EnvironmentOptions{};
  opts2.allowMoveFromPopulation = false;
  opts2.isEvaluateLightweight = Utility::AutoOption::Auto();
  opts2.isMutateLightweight = Utility::AutoOption::Auto();
  opts2.isCrossoverLightweight = Utility::AutoOption::Auto();

  auto env1 = Environment(generator, Evaluate, Mutate1, Crossover1, sf, opts);
  BOOST_TEST(copyCounter <= 7);
  copyCounter = 0;

  env1.Run();
  BOOST_TEST(copyCounter <= 4);
  copyCounter = 0;

  auto env5 = Environment(generator, Evaluate, Mutate1, Crossover1, sf, opts2);
  BOOST_TEST(copyCounter <= 7);
  copyCounter = 0;

  env5.Run();
  BOOST_TEST(copyCounter <= 4);
  copyCounter = 0;

  auto Mutate2 = [](DNA &&x) -> DNA & { return x; };
  auto Crossover2 = [](DNA &&x, DNA &&y) -> DNA & { return x; };

  auto env2 = Environment(generator, Evaluate, Mutate2, Crossover2, sf, opts);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env2.Run();
  BOOST_TEST(copyCounter <= 3);
  copyCounter = 0;

  auto env6 = Environment(generator, Evaluate, Mutate2, Crossover2, sf, opts2);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env6.Run();
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  auto Mutate3 = [](DNA x) -> DNA { return x; };
  auto Crossover3 = [](DNA x, DNA y) -> DNA { return x; };

  auto env3 = Environment(generator, Evaluate, Mutate3, Crossover3, sf, opts);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env3.Run();
  BOOST_TEST(copyCounter <= 3);
  copyCounter = 0;

  auto env7 = Environment(generator, Evaluate, Mutate3, Crossover3, sf, opts2);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env7.Run();
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  auto Mutate4 = [&](DNA const &x) -> DNA & { return orig; };
  auto Crossover4 = [&](DNA const &x, DNA const &y) -> DNA & { return orig; };

  auto env4 = Environment(generator, Evaluate, Mutate4, Crossover4, sf, opts);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env4.Run(size_t{100});
  BOOST_TEST(copyCounter == 0);

  auto env8 = Environment(generator, Evaluate, Mutate4, Crossover4, sf, opts2);
  BOOST_TEST(copyCounter <= 5);
  copyCounter = 0;

  env8.Run(size_t{100});
  BOOST_TEST(copyCounter <= 200);
}

BOOST_AUTO_TEST_CASE(tbb_exception_test) {
  using T = std::string;
  using Ptr = std::shared_ptr<T>;
  using InputNode = tbb::flow::function_node<T, Ptr>;
  using EvaluateNode = tbb::flow::function_node<Ptr, int>;
  using CrossoverNode = tbb::flow::function_node<std::tuple<Ptr, Ptr>, Ptr>;
  using CrossoverJoinNode = tbb::flow::join_node<std::tuple<Ptr, Ptr>>;
  tbb::flow::graph g;
  InputNode i1(g, tbb::flow::concurrency::serial,
               [](T x) { return std::make_shared<T>(std::move(x)); });
  InputNode i2 = i1;
  bool isEvaluateThrow = false;
  bool isCrossoverThrow = false;
  EvaluateNode e1(g, tbb::flow::concurrency::serial, [&](Ptr x) -> int {
    if (isEvaluateThrow)
      throw std::runtime_error("");
    return static_cast<int>(x->size());
  });
  EvaluateNode e2 = e1;
  CrossoverNode c1(g, tbb::flow::concurrency::serial,
                   [&](std::tuple<Ptr, Ptr> x) {
                     if (isCrossoverThrow)
                       throw std::runtime_error("");
                     return std::get<0>(x);
                   });
  CrossoverNode c2 = c1;
  CrossoverJoinNode j1(g);
  CrossoverJoinNode j2(g);
  tbb::flow::make_edge(i1, input_port<0>(j1));
  tbb::flow::make_edge(i2, input_port<1>(j1));
  tbb::flow::make_edge(i1, input_port<0>(j2));
  tbb::flow::make_edge(i2, input_port<1>(j2));
  tbb::flow::make_edge(j1, c1);
  tbb::flow::make_edge(j2, c2);
  tbb::flow::make_edge(c1, e1);
  tbb::flow::make_edge(c2, e2);
  isEvaluateThrow = true;
  isCrossoverThrow = false;
  for (size_t i = 0; i != 10; ++i) {
    try {
      i1.try_put("1");
      i2.try_put("2");
      auto reset = Utility::RAII([]() {}, [&]() noexcept { g.reset(); });
      g.wait_for_all();
    } catch (...) {
    }
    isEvaluateThrow = false;
    isCrossoverThrow = false;
    i1.try_put("1");
    i2.try_put("2");
    g.wait_for_all();
  }
}

BOOST_AUTO_TEST_CASE(exception_test) {
  using Population = std::vector<std::string>;
  class CustomException final : public std::runtime_error {
  public:
    CustomException() : std::runtime_error("") {}
  };
  // Fail with N>5
  auto N = 5;
  auto isThrow = std::vector<bool>(5);
  auto population = Population{};
  for (auto i = size_t{0}; i != N; ++i) {
    population.push_back(std::string(1, static_cast<char>(i)));
  }
  auto sf = GenerateStateFlow(N);
  auto opts1 = EnvironmentOptions{};
  opts1.isEvaluateLightweight = Utility::AutoOption::Auto();
  opts1.isMutateLightweight = Utility::AutoOption::Auto();
  opts1.isCrossoverLightweight = Utility::AutoOption::Auto();
  opts1.allowMoveFromPopulation = false;

  auto opts2 = EnvironmentOptions{};
  opts2.isEvaluateLightweight = Utility::AutoOption::Auto();
  opts2.isMutateLightweight = Utility::AutoOption::Auto();
  opts2.isCrossoverLightweight = Utility::AutoOption::Auto();
  opts2.allowMoveFromPopulation = true;

  auto generator = [&]() -> std::string {
    if (isThrow[0])
      throw CustomException{};
    return "generator";
  };
  auto Evaluate = [&](std::string const &x) {
    if (isThrow[1])
      throw CustomException{};
    return x.size();
  };
  auto Mutate = [&](std::string const &x) -> std::string {
    if (isThrow[2])
      throw CustomException{};
    return "mutate";
  };
  auto Crossover = [&](std::string const &x,
                       std::string const &y) -> std::string {
    if (isThrow[3])
      throw CustomException{};
    return "crossover";
  };
  auto Sort = [&](Population const &, std::vector<size_t> const &grades) {
    if (isThrow[4])
      throw CustomException{};
    auto permutation = Utility::GetIndices(grades.size());
    std::sort(permutation.begin(), permutation.end(),
              [&](size_t index0, size_t index1) {
                return grades[index0] > grades[index1];
              });
    return permutation;
  };
  auto Cycle = [&](auto &&callable, std::vector<bool> thrown) {
    isThrow = std::vector<bool>(5);
    for (auto i = size_t{0}, e = isThrow.size(); i != e; ++i) {
      isThrow.at(i) = true;
      auto catched = false;
      try {
        callable();
      } catch (CustomException &) {
        catched = true;
      }
      BOOST_TEST(catched == thrown.at(i));
      isThrow.at(i) = false;
      callable();
    }
  };

  auto CheckPopulation = [&](Population const &pop,
                             std::vector<std::string> const &allowedWords,
                             bool isAllowOrigin) {
    for (auto i = size_t{0}; i != pop.size(); ++i) {
      auto const &e = pop[i];
      auto found = false;
      found |= std::find(allowedWords.begin(), allowedWords.end(), e) !=
               allowedWords.end();
      if (isAllowOrigin)
        found |= population[i] == e;
      BOOST_TEST(found);
    }
  };

  Cycle(
      [&]() {
        Environment(generator, Evaluate, Mutate, Crossover, sf, opts1, Sort);
      },
      {true, true, true, true, true});
  Cycle(
      [&]() {
        Environment(generator, Evaluate, Mutate, Crossover, sf, opts2, Sort);
      },
      {true, true, true, true, true});

  auto env1 =
      Environment(generator, Evaluate, Mutate, Crossover, sf, opts1, Sort);
  auto env2 =
      Environment(generator, Evaluate, Mutate, Crossover, sf, opts2, Sort);

  Cycle([&]() { env1.SetPopulation(Population(population)); },
        {false, true, false, false, true});
  Cycle([&]() { env2.SetPopulation(Population(population)); },
        {false, true, false, false, true});

  auto isCatch = std::vector<bool>{false, true, true, true, true};
  for (auto it = size_t{0}, e = isThrow.size(); it != e; ++it) {
    env1.SetPopulation(Population(population));
    isThrow[it] = true;
    auto isCatched = false;
    try {
      env1.Run();
    } catch (CustomException &) {
      isCatched = true;
    }
    BOOST_TEST(isCatched == isCatch[it]);
    auto const &pop = env1.GetPopulation();
    if (it == 0 || it == 4)
      CheckPopulation(pop, {"mutate", "crossover"}, true);
    else if (it == 1)
      CheckPopulation(pop, {}, true);
    else if (it == 2)
      CheckPopulation(pop, {"crossover"}, true);
    else if (it == 3)
      CheckPopulation(pop, {"mutate"}, true);
    isThrow[it] = false;
    env1.Run();
  }
}

BOOST_AUTO_TEST_CASE(perf_test) {
  // Perfomance can be compared to similar example of openGA library
  // https://github.com/Arash-codedev/openGA/tree/master/examples/so-1

  struct MySolution {
    double x;
    double y;

    std::string to_string() const {
      return "{x:" + std::to_string(x) + ", y:" + std::to_string(y) + "}";
    }
  };
  auto static constexpr invroot2 = 1.4142135 / 2;
  auto Evaluate = [](MySolution const &s) {
    auto const &[x, y] = s;
    double predictable_noise =
        30.0 * sin(x * 100.0 * sin(y) + y * 100.0 * cos(x));
    auto cost_distance2 = x * x + y * y + predictable_noise;
    auto cost_sqsin =
        125 + 45.0 * sqrt(x + y) * sin((15.0 * (x + y)) / (x * x + y * y));
    return -cost_distance2 - cost_sqsin;
  };

  auto Mutate = [](MySolution s) -> MySolution {
    auto &&[x, y] = s;
    auto dx = invroot2 * (x - y);
    auto dy = invroot2 * (x + y);
    auto rand = std::uniform_real_distribution(-1.5, 1.5);
    auto &&gen = Utility::GetRandomGenerator();
    dy *= std::pow(2, rand(gen));
    dx += dy * std::pow(2, rand(gen)) * (rand(gen) > 0 ? 1 : -1);
    auto s1 = MySolution{(dx + dy) * invroot2, (dy - dx) * invroot2};
    assert(s.x + s.y > -0.0001);
    return s1;
  };
  auto Crossover = [](MySolution &&s1, MySolution const &s2) -> MySolution & {
    s1.x = (s1.x + s2.x) / 2;
    s1.y = (s1.y + s2.y) / 2;
    return s1;
  };
  auto Generator = []() -> MySolution {
    auto &&gen = Utility::GetRandomGenerator();
    auto rand = std::uniform_int_distribution(0, 20);
    auto dx = rand(gen) - 10;
    auto dy = rand(gen);
    auto s = MySolution{(dx + dy) * invroot2, (dy - dx) * invroot2};
    assert(s.x + s.y > -0.0001);
    return s;
  };

  auto nGens = 30;
  auto opts = EnvironmentOptions{};
  opts.isEvaluateLightweight = Utility::AutoOption::True();
  opts.isMutateLightweight = Utility::AutoOption::True();
  opts.isCrossoverLightweight = Utility::AutoOption::True();

  auto env = Environment(Generator, Evaluate, Mutate, Crossover,
                         GenerateStateFlow(20), opts);

  auto tot = Utility::Benchmark([&]() {
    for (auto i = 0; i != nGens; ++i) {
      auto gent = Utility::Benchmark([&]() { env.Run(); });
      auto const &pop = env.GetPopulation();
      auto const &grade = env.GetGrades();
      auto avg =
          std::accumulate(grade.begin(), grade.end(), 0.0) / grade.size();
      if constexpr (verbose) {
        std::cout << "Generation [" << i << "], "
                  << "Best=" << grade[0] << ", "
                  << "Avg=" << avg << ", "
                  << "Best genes=(" << pop[0].to_string() << ")"
                  << ", "
                  << "Exe_time=" << gent.count() / 1000 / 1000 << " ms"
                  << std::endl;
      }
    }
  });
  if constexpr (verbose) {
    std::cout << "Total " << tot.count() / 1000 / 1000 << " ms" << std::endl;
  }
#ifndef NDEBUG
  BOOST_TEST(tot.count() / nGens / 1000 / 1000 < 50);
#else
  BOOST_TEST(tot.count() / nGens / 1000 / 1000 < 4);
#endif // !NDEBUG
}

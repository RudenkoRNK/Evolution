#pragma once

#include "evolution/concepts.hpp"
#include "evolution/state_flow.hpp"

namespace Evolution {

StateFlow inline GenerateStateFlow(size_t populationSize) {
  if (populationSize == 0)
    return StateFlow{};
  if (populationSize == 1) {
    auto sf = StateFlow{};
    sf.SetEvaluate(sf.GetOrAddInitialState(0));
    return sf;
  }
  // Save top 10%
  // mutate once top 30%,
  // Crossover top 5% with next-top 5%
  // Crossover next-top 5% with next-next-top 5%
  // Crossover top 10% with next-top 10%
  // Crossover top 20% with next-top 20%
  // Crossover top 20% with (70-90)% range
  auto nLeft = populationSize;
  auto percentsLeft = size_t{100};
  auto GetAbsValue = [&](size_t percentage) -> size_t {
    auto ret = size_t{nLeft * percentage / percentsLeft};
    percentsLeft -= percentage;
    nLeft -= ret;
    assert(percentsLeft >= 0);
    assert(nLeft >= 0);
    return ret;
  };

  auto nSaves = GetAbsValue(10);
  auto nMutates = GetAbsValue(30);
  auto nCrossovers0 = GetAbsValue(5);
  auto nCrossovers1 = GetAbsValue(5);
  auto nCrossovers2 = GetAbsValue(10);
  auto nCrossovers3 = GetAbsValue(20);
  auto nCrossovers4 = GetAbsValue(20);
  assert(percentsLeft == 0);
  assert(nLeft == 0);
  assert(nCrossovers0 * 2 <= populationSize);
  assert(nCrossovers1 * 3 <= populationSize);
  assert(nCrossovers2 * 2 <= populationSize);
  assert(nCrossovers3 * 2 <= populationSize);

  auto sf = StateFlow{};
  for (auto i = size_t{0}; i != nSaves; ++i)
    sf.SetEvaluate(sf.GetOrAddInitialState(i));
  for (auto i = size_t{0}; i != nMutates; ++i)
    sf.SetEvaluate(sf.AddMutate(sf.GetOrAddInitialState(i)));
  for (auto i = size_t{0}; i != nCrossovers0; ++i) {
    auto j = 2 * nCrossovers0 - i - 1;
    sf.SetEvaluate(sf.AddCrossover(sf.GetOrAddInitialState(i),
                                   sf.GetOrAddInitialState(j)));
  }
  for (auto i = nCrossovers1; i != nCrossovers1 * 2; ++i) {
    auto j = 3 * nCrossovers1 - (i - nCrossovers1) - 1;
    sf.SetEvaluate(sf.AddCrossover(sf.GetOrAddInitialState(i),
                                   sf.GetOrAddInitialState(j)));
  }
  for (auto i = size_t{0}; i != nCrossovers2; ++i) {
    auto j = 2 * nCrossovers2 - i - 1;
    sf.SetEvaluate(sf.AddCrossover(sf.GetOrAddInitialState(i),
                                   sf.GetOrAddInitialState(j)));
  }
  for (auto i = size_t{0}; i != nCrossovers3; ++i) {
    auto j = 2 * nCrossovers3 - i - 1;
    sf.SetEvaluate(sf.AddCrossover(sf.GetOrAddInitialState(i),
                                   sf.GetOrAddInitialState(j)));
  }
  for (auto i = size_t{0}; i != nCrossovers4; ++i) {
    auto j = (populationSize - populationSize / 10) - i - 1;
    if (i == j)
      j--;
    sf.SetEvaluate(sf.AddCrossover(sf.GetOrAddInitialState(i),
                                   sf.GetOrAddInitialState(j)));
  }
  sf.SetSwapArgumentsAllowedInCrossover();

  assert(sf.GetNEvaluates() == populationSize);
  assert(!sf.IsNotReady());
  return sf;
}

template <typename FG, typename... Args>
bool inline IsFGLightweight(FG const &Func, Args &&... args) {
  auto constexpr static maxLightweightClocks = size_t{1000000};
  auto FG_ = std::function(Func);
  auto &&Func_ = GeneratorTraits::GetFunction<decltype(Func)>(FG_);
  auto freq = 2; // clocks per nanosecond
  auto time = Utility::Benchmark(std::forward<decltype(Func_)>(Func_),
                                 std::forward<Args>(args)...);
  auto nanosecs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(time).count();
  auto clocks = freq * nanosecs;
  return clocks < maxLightweightClocks;
}

} // namespace Evolution
#pragma once
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlow.hpp"

namespace Evolution {

template <class EvaluateFG, class MutateFG, class CrossoverFG>
class Environment final {
private:
  using TaskFlowInst = TaskFlow<EvaluateFG, MutateFG, CrossoverFG>;

public:
  using DNA = typename TaskFlowInst::DNA;
  using Population = typename TaskFlowInst::Population;
  using Grades = typename TaskFlowInst::Grades;

private:
  using PopulationActionFunction = std::function<void(Population &, Grades &)>;
  using DNAGeneratorFunctionInst = std::function<DNA()>;
  DNAGeneratorFunctionInst DNAGenerator;
  TaskFlowInst taskFlow;
  Population population;
  Grades grades;
  PopulationActionFunction SortPopulation_;

public:
  template <class DNAGeneratorFunction>
  Environment(DNAGeneratorFunction const &DNAGenerator,
              EvaluateFG const &Evaluate, MutateFG const &Mutate,
              CrossoverFG const &Crossover, StateFlow const &stateFlow,
              bool isBenchmarkFunctions = false)
      : DNAGenerator(DNAGeneratorFunctionInst(DNAGenerator)),
        taskFlow(Evaluate, Mutate, Crossover, stateFlow,
                 isBenchmarkFunctions && TaskFlowInst::IsEvaluateLightweight(
                                             Evaluate, this->DNAGenerator()),
                 isBenchmarkFunctions && TaskFlowInst::IsMutateLightweight(
                                             Mutate, this->DNAGenerator()),
                 isBenchmarkFunctions && TaskFlowInst::IsCrossoverLightweight(
                                             Crossover, this->DNAGenerator(),
                                             this->DNAGenerator())) {
    static_assert(
        std::is_convertible_v<DNAGeneratorFunction, DNAGeneratorFunctionInst>);
    ResizePopulation(stateFlow.GetNEvaluates());
  }

  Population const &GetPopulation() const noexcept { return population; }
  Grades const &GetGrades() const noexcept { return grades; }

  void Run(size_t n = 1) {
    Run(n, [](Population &, Grades &) {});
  }

  template <class GenerationActionFunction>
  void Run(size_t n, GenerationActionFunction &&GenerationAction) {
    static_assert(std::is_convertible_v<GenerationActionFunction,
                                        PopulationActionFunction>);
    if (n == 0)
      return;
    auto population_ = population;
    auto grades_ = grades;
    for (auto gen = size_t{0}; gen < n; ++gen) {
      taskFlow.Run(population_, grades_);
      SortPopulation(population_, grades_);
      GenerationAction(population_, grades_);
    }
    population = std::move(population_);
    grades = std::move(grades_);
  }

  Grades EvaluatePopulation(Population const &population) {
    return taskFlow.EvaluatePopulation(population);
  }

  void SetPopulation(Population &&population) {
    assert(population.size() == taskFlow.GetStateFlow().GetNEvaluates());
    auto grades = EvaluatePopulation(population);
    SetPopulation(std::move(population), std::move(grades));
  }

  template <class SortPopulationFunction>
  void SetSortPopulationFunction(SortPopulationFunction &&SortPopulation) {
    static_assert(std::is_convertible_v<SortPopulationFunction,
                                        PopulationActionFunction>);
    this->SortPopulation_ = PopulationActionFunction(
        std::forward<SortPopulationFunction>(SortPopulation));
    SortPopulation(population, grades);
  }

  void SetStateFlow(StateFlow const &stateFlow) {
    taskFlow.SetStateFlow(StateFlow(stateFlow));
    ResizePopulation(stateFlow.GetNEvaluates());
  }

  StateFlow static GenerateStateFlow(size_t populationSize) {
    // Save top 10%
    // mutate once top 30%,
    // Crossover top 5% with next-top 5%
    // Crossover next-top 5% with next-next-top 5%
    // Crossover top 10% with next-top 10%
    // Crossover top 20% with next-top 20%
    // Crossover top 20% with (70-90)% range
    assert(populationSize >= 2);
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
    for (auto i = size_t{0}; i < nSaves; ++i)
      sf.SetEvaluate(sf.GetOrAddInitialState(i));
    for (auto i = size_t{0}; i < nMutates; ++i)
      sf.SetEvaluate(sf.AddMutate(sf.GetOrAddInitialState(i)));
    for (auto i = size_t{0}; i < nCrossovers0; ++i) {
      auto j = 2 * nCrossovers0 - i - 1;
      sf.SetEvaluate(sf.AddCrossover(sf.GetOrAddInitialState(i),
                                     sf.GetOrAddInitialState(j)));
    }
    for (auto i = nCrossovers1; i < nCrossovers1 * 2; ++i) {
      auto j = 3 * nCrossovers1 - (i - nCrossovers1) - 1;
      sf.SetEvaluate(sf.AddCrossover(sf.GetOrAddInitialState(i),
                                     sf.GetOrAddInitialState(j)));
    }
    for (auto i = size_t{0}; i < nCrossovers2; ++i) {
      auto j = 2 * nCrossovers2 - i - 1;
      sf.SetEvaluate(sf.AddCrossover(sf.GetOrAddInitialState(i),
                                     sf.GetOrAddInitialState(j)));
    }
    for (auto i = size_t{0}; i < nCrossovers3; ++i) {
      auto j = 2 * nCrossovers3 - i - 1;
      sf.SetEvaluate(sf.AddCrossover(sf.GetOrAddInitialState(i),
                                     sf.GetOrAddInitialState(j)));
    }
    for (auto i = size_t{0}; i < nCrossovers4; ++i) {
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

private:
  void ResizePopulation(size_t newSize) {
    if (population.size() < newSize) {
      auto diff = newSize - population.size();
      auto next = Population{};
      next.reserve(diff);
      std::generate_n(std::back_inserter(next), diff, DNAGenerator);
      auto nextGrades = EvaluatePopulation(next);
      auto newPop = Population{};
      auto newGrades = Grades{};
      newPop.reserve(newSize);
      newGrades.reserve(newSize);
      newPop.insert(newPop.end(), population.begin(), population.end());
      std::move(next.begin(), next.end(), std::back_inserter(newPop));
      newGrades.insert(newGrades.end(), grades.begin(), grades.end());
      newGrades.insert(newGrades.end(), nextGrades.begin(), nextGrades.end());
      SetPopulation(std::move(newPop), std::move(newGrades));
    } else {
      population.resize(newSize, DNAGenerator());
      grades.resize(newSize);
    }
  }

  void SetPopulation(Population &&population, Grades &&grades) {
    SortPopulation(population, grades);
    this->population = std::move(population);
    this->grades = std::move(grades);
  }

  void SortPopulation(Population &population, Grades &grades) {
    if (SortPopulation_) {
      auto size = population.size();
      SortPopulation_(population, grades);
      assert(population.size() == size);
      assert(grades.size() == size);
      return;
    }
    auto permutation = GetIndices(population.size());
    std::sort(permutation.begin(), permutation.end(),
              [&](size_t index0, size_t index1) {
                return grades.at(index0) > grades.at(index1);
              });
    // buffer for exception-safety
    auto buffer = std::vector<size_t>(population.size());
    static_assert(
        noexcept(Permute(population, permutation, std::identity{}, buffer)));
    static_assert(
        noexcept(Permute(grades, permutation, std::identity{}, buffer)));
    Permute(population, permutation, std::identity{}, buffer);
    Permute(grades, permutation, std::identity{}, buffer);
  }
};

} // namespace Evolution

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
  using SortPopulationFunctionInst =
      std::function<void(Population &, Grades &)>;
  TaskFlowInst taskFlow;
  Population population;
  Grades grades;
  SortPopulationFunctionInst SortPopulationF;

public:
  Environment(EvaluateFG const &Evaluate, MutateFG const &Mutate,
              CrossoverFG const &Crossover, StateFlow const &stateFlow,
              bool isSwapArgumentsAllowedInCrossover, Population &&population)
      : taskFlow(Evaluate, Mutate, Crossover, stateFlow,
                 isSwapArgumentsAllowedInCrossover) {
    SetPopulation(std::move(population));
  }

  Population const &GetPopulation() const noexcept { return population; }
  Grades const &GetGrades() const noexcept { return grades; }

  void Run() {
    taskFlow.Run(population, grades);
    SortPopulation(population, grades);
  }

  void SetPopulation(Population &&population) {
    assert(population.size() == taskFlow.GetStateFlow().GetNEvaluates());
    auto grades_ = taskFlow.EvaluatePopulation(population);
    SortPopulation(population, grades_);
    this->population = std::move(population);
    this->grades = std::move(grades_);
  }

  template <class SortPopulationFunction>
  void SetSortPopulationFunction(SortPopulationFunction &&SortPopulation) {
    static_assert(std::is_convertible_v<SortPopulationFunction,
                                        SortPopulationFunctionInst>);
    SortPopulationF = SortPopulationFunctionInst(
        std::forward<SortPopulationFunction>(SortPopulation));
    SortPopulation(population, grades);
  }

  void SetStateFlow(StateFlow &&stateFlow,
                    bool isSwapArgumentsAllowedInCrossover = false) {
    assert(population.size() >= stateFlow.GetNEvaluates());
    taskFlow.SetStateFlow(std::move(stateFlow),
                          isSwapArgumentsAllowedInCrossover);
    population.resize(stateFlow.GetNEvaluates());
    grades.resize(stateFlow.GetNEvaluates());
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

    assert(sf.GetNEvaluates() == populationSize);
    assert(sf.Verify());
    return sf;
  }

  template <class DNAGeneratorFunction>
  Population static GeneratePopulation(size_t populationSize,
                                       DNAGeneratorFunction &DNAGenerator) {
    auto population = Population{};
    population.reserve(populationSize);
    std::generate_n(std::back_inserter(population), populationSize,
                    DNAGenerator);
    return population;
  }

private:
  void SortPopulation(Population &population, Grades &grades) {
    if (SortPopulationF) {
      auto size = population.size();
      SortPopulationF(population, grades);
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
    Permute(population, permutation, std::identity{}, buffer);
    Permute(grades, permutation, std::identity{}, buffer);
  }
};

} // namespace Evolution

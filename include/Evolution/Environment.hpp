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
      std::function<void(Population const &, Grades const &)>;
  TaskFlowInst taskFlow;
  Population population;
  Grades grades;
  SortPopulationFunctionInst SortPopulationF;

public:
  Environment(EvaluateFG &&Evaluate, MutateFG &&Mutate, CrossoverFG &&Crossover,
              Population &&population, Grades &&grades,
              StateFlow const &stateFlow,
              bool isSwapArgumentsAllowedInCrossover = false)
      : taskFlow(std::forward<EvaluateFG>(Evaluate),
                 std::forward<MutateFG>(Mutate),
                 std::forward<CrossoverFG>(Crossover), stateFlow,
                 isSwapArgumentsAllowedInCrossover) {
    assert(population.size() == stateFlow.GetNEvaluates());
    this->population = std::move(population);
    this->grades = std::move(grades);
  }

  Population const &GetPopulation() const noexcept { return population; }
  Grades const &GetGrades() const noexcept { return grades; }

  void Run() {
    taskFlow.Run(population, grades);
    SortPopulation();
  }

  void SetPopulation(Population &&population, Grades &&grades) noexcept {
    assert(population.size() = this->population.size());
    assert(grades.size() = this->population.size());
    this->population = std::move(population);
    this->grades = std::move(grades);
  }

  // To actually sort population the SortPopulation should invoke
  // PermutePopulation inside itself
  template <class SortPopulationFunction>
  void SetSortPopulationFunction(SortPopulationFunction &&SortPopulation) {
    static_assert(std::is_convertible_v<SortPopulationFunction,
                                        SortPopulationFunctionInst>);
    SortPopulationF = SortPopulationFunctionInst(
        std::forward<SortPopulationFunction>(SortPopulation));
  }

  void PermutePopulation(std::vector<size_t> &permutation) {
    // buffer for exception-safety
    auto buffer = std::vector<size_t>(population.size());
    Permute(population, permutation, std::identity{}, buffer);
    Permute(grades, permutation, std::identity{});
  }

  template <class Indexer, class IndexFunction>
  void PermutePopulation(std::vector<Indexer> &permutation,
                         IndexFunction &Index) {
    // buffer for exception-safety
    auto buffer = std::vector<size_t>(population.size());
    Permute(population, permutation, Index, buffer);
    Permute(grades, permutation, Index, buffer);
  }

  static StateFlow GenerateStateFlow(size_t populationSize) {
    // Save top 10%
    // mutate once top 40%,
    // Crossover top 5% with next-top 5%
    // Crossover next-top 5% with next-next-top 5%
    // Crossover top 10% with next-top 10%
    // Crossover top 20% with next-top 20%
    // Crossover top 10% with low 10%
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
    auto nMutates = GetAbsValue(40);
    auto nCrossovers0 = GetAbsValue(5);
    auto nCrossovers1 = GetAbsValue(5);
    auto nCrossovers2 = GetAbsValue(10);
    auto nCrossovers3 = GetAbsValue(20);
    auto nCrossovers4 = GetAbsValue(10);
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
      auto j = 3 * nCrossovers1 - i - 1;
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
      auto j = populationSize - i - 1;
      sf.SetEvaluate(sf.AddCrossover(sf.GetOrAddInitialState(i),
                                     sf.GetOrAddInitialState(j)));
    }

    assert(sf.GetNEvaluates() == populationSize);
    assert(sf.Verify());
    return sf;
  }

  template <class DNAGeneratorFunction>
  static Population GeneratePopulation(size_t populationSize,
                                       DNAGeneratorFunction &DNAGenerator,
                                       bool isParallel = false) {
    auto population = Population{};
    population.reserve(populationSize);

    if (isParallel) {
      auto populationConcurrent = tbb::concurrent_vector<DNA>{};
      populationConcurrent.reserve(populationSize);
      std::generate_n(std::execution::par_unseq,
                      std::back_inserter(populationConcurrent), populationSize,
                      DNAGenerator);
      for (auto &&dna : populationConcurrent)
        population.push_back(std::move(dna));
    } else
      std::generate_n(std::execution::seq, std::back_inserter(population),
                      populationSize, DNAGenerator);
    return population;
  }

  static Grades EvaluatePopulation(Population const &population,
                                   EvaluateFG &&Evaluate) {
    auto &&EvaluateThreadSpecificOrGlobal =
        GeneratorTraits<EvaluateFG>::GetThreadSpecificOrGlobal(
            std::forward<EvaluateFG>(Evaluate));
    auto indices = GetIndices(population.size());
    auto grades = Grades(population.size());
    tbb::parallel_for_each(indices.begin(), indices.end(), [&](size_t index) {
      auto &&Evaluate = GeneratorTraits<EvaluateFG>::GetFunction(
          EvaluateThreadSpecificOrGlobal);
      grades.at(index) = Evaluate(population.at(index));
    });
    return grades;
  }

private:
  void SortPopulation() {
    if (SortPopulationF) {
      SortPopulationF(population, grades);
      return;
    }
    auto permutation = GetIndices(population.size());
    std::sort(permutation.begin(), permutation.end(),
              [&](size_t index0, size_t index1) {
                return grades.at(index0) > grades.at(index1);
              });
    PermutePopulation(permutation);
  }
};

} // namespace Evolution

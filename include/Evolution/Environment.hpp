#pragma once
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlow.hpp"
#include "Utility/Misc.hpp"

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
  using SortPopulationFunction =
      std::function<std::vector<size_t>(Population const &, Grades const &)>;
  using GenerationActionFunction =
      std::function<bool(Population const &, Grades const &)>;
  using DNAGeneratorFunction = std::function<DNA()>;
  DNAGeneratorFunction DNAGenerator;
  TaskFlowInst taskFlow;
  Population population;
  Grades grades;
  SortPopulationFunction SortPopulation_;

public:
  template <class DNAGeneratorFunctionT>
  Environment(DNAGeneratorFunctionT const &DNAGenerator,
              EvaluateFG const &Evaluate, MutateFG const &Mutate,
              CrossoverFG const &Crossover, StateFlow const &stateFlow,
              bool isBenchmarkFunctions = false)
      : DNAGenerator(DNAGeneratorFunction(DNAGenerator)),
        taskFlow(Evaluate, Mutate, Crossover, stateFlow,
                 isBenchmarkFunctions && TaskFlowInst::IsEvaluateLightweight(
                                             Evaluate, this->DNAGenerator()),
                 isBenchmarkFunctions && TaskFlowInst::IsMutateLightweight(
                                             Mutate, this->DNAGenerator()),
                 isBenchmarkFunctions && TaskFlowInst::IsCrossoverLightweight(
                                             Crossover, this->DNAGenerator(),
                                             this->DNAGenerator())) {
    static_assert(
        std::is_convertible_v<DNAGeneratorFunctionT, DNAGeneratorFunction>);
    ResizePopulation(stateFlow.GetNEvaluates());
  }

  Population const &GetPopulation() const { return population; }
  Grades const &GetGrades() const { return grades; }

  void Run(size_t n = 1) {
    if (n < 1)
      return;
    Run([&](Population const &, Grades const &) { return --n > 0; });
  }

  template <class GenerationActionFunctionT>
  void Run(GenerationActionFunctionT &&GenerationAction) {
    static_assert(std::is_convertible_v<GenerationActionFunctionT,
                                        GenerationActionFunction>);
    do {
      assert(Verify(population, grades));
      taskFlow.Run(population, grades);
      SortPopulation(population, grades);
    } while (GenerationAction(population, grades));
  }

  Grades EvaluatePopulation(Population const &population) {
    return taskFlow.EvaluatePopulation(population);
  }

  // Also can be used to provide DNA examples
  void SetPopulation(Population &&population) {
    AppendPopulation(std::move(population), population.size());
  }

  void RegeneratePopulation() {
    auto newPop = Population{};
    newPop.reserve(population.size());
    std::generate_n(std::back_inserter(newPop), population.size(),
                    DNAGenerator);
    SetPopulation(std::move(newPop));
  }

  template <class SortPopulationFunctionT>
  void SetSortPopulationFunction(SortPopulationFunctionT &&SortPopulation_) {
    static_assert(
        std::is_convertible_v<SortPopulationFunctionT, SortPopulationFunction>);
    this->SortPopulation_ = SortPopulationFunction(
        std::forward<SortPopulationFunctionT>(SortPopulation_));
    SortPopulation(population, grades);
  }

  void SetStateFlow(StateFlow &&stateFlow) {
    taskFlow.SetStateFlow(std::move(stateFlow));
    ResizePopulation(stateFlow.GetNEvaluates());
  }

private:
  void ResizePopulation(size_t newSize) {
    if (population.size() < newSize) {
      auto diff = newSize - population.size();
      auto next = Population{};
      next.reserve(diff);
      std::generate_n(std::back_inserter(next), diff, DNAGenerator);
      AppendPopulation(std::move(next));
    } else {
      grades.resize(newSize);
      population.resize(newSize, DNAGenerator());
    }
    assert(Verify(population, grades));
  }

  void AppendPopulation(Population &&pop, size_t backOffset = 0) {
    auto grd = EvaluatePopulation(pop);
    assert(population.size() >= backOffset);
    auto diff = pop.size() > backOffset ? pop.size() - backOffset : 0;
    auto newSize = population.size() + diff;

    population.reserve(newSize);
    grades.reserve(newSize);
    std::move(pop.begin(), pop.end() - diff, population.end() - backOffset);
    std::move(pop.end() - diff, pop.end(), std::back_inserter(population));
    std::move(grd.begin(), grd.end() - diff, grades.end() - backOffset);
    std::move(grd.end() - diff, grd.end(), std::back_inserter(grades));
    SortPopulation(population, grades);
  }

  void SortPopulation(Population &population, Grades &grades) const {
    auto permutation = std::vector<size_t>{};
    if (SortPopulation_)
      permutation = SortPopulation_(population, grades);
    else {
      permutation = Utility::GetIndices(population.size());
      std::sort(permutation.begin(), permutation.end(),
                [&](size_t index0, size_t index1) {
                  return grades.at(index0) > grades.at(index1);
                });
    }
    Utility::Permute(population, permutation, std::identity{});
    Utility::Permute(grades, permutation, std::identity{});
  }

  size_t GetPopulationSize() const {
    return taskFlow.GetStateFlow().GetNEvaluates();
  }

  bool Verify(Population const &population, Grades const &grades) const
      noexcept {
    auto verified = true;
    verified &= population.size() == GetPopulationSize();
    verified &= grades.size() == GetPopulationSize();
    return verified;
  }
};

} // namespace Evolution

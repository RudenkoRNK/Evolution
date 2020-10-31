#pragma once
#include "Evolution/Concepts.hpp"
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlow.hpp"
#include "Utility/Misc.hpp"

namespace Evolution {

template <EvaluateFunctionOrGeneratorConcept EvaluateFG,
          MutateFunctionOrGeneratorConcept MutateFG,
          CrossoverFunctionOrGeneratorConcept CrossoverFG>
class Environment final {
private:
  using TaskFlowInst = TaskFlow<EvaluateFG, MutateFG, CrossoverFG>;

public:
  using DNA = typename TaskFlowInst::DNA;
  using Grade = typename TaskFlowInst::Grade;
  using Population = typename TaskFlowInst::Population;
  using Grades = typename TaskFlowInst::Grades;

private:
  using SortPopulationFunction =
      std::function<std::vector<size_t>(Population const &, Grades const &)>;
  using GenerationActionFunction =
      std::function<bool(Population const &, Grades const &)>;
  using DNAGeneratorFunction = std::function<DNA()>;
  DNAGeneratorFunction DNAGenerator;
  SortPopulationFunction SortPopulation_;
  TaskFlowInst taskFlow;
  Population population;
  Grades grades;

public:
  Environment(DNAGeneratorFunction const &DNAGenerator,
              EvaluateFG const &Evaluate, MutateFG const &Mutate,
              CrossoverFG const &Crossover, StateFlow const &stateFlow,
              bool isBenchmarkFunctions = false,
              SortPopulationFunction const &SortPopulation_ = GetSortFunction())
      : DNAGenerator(DNAGenerator), SortPopulation_(SortPopulation_),
        taskFlow(Evaluate, Mutate, Crossover, stateFlow,
                 isBenchmarkFunctions && TaskFlowInst::IsEvaluateLightweight(
                                             Evaluate, this->DNAGenerator()),
                 isBenchmarkFunctions && TaskFlowInst::IsMutateLightweight(
                                             Mutate, this->DNAGenerator()),
                 isBenchmarkFunctions && TaskFlowInst::IsCrossoverLightweight(
                                             Crossover, this->DNAGenerator(),
                                             this->DNAGenerator())) {
    RegeneratePopulation();
  }

  Population const &GetPopulation() const { return population; }
  Grades const &GetGrades() const { return grades; }

  void Run(size_t n = 1) {
    if (n < 1)
      return;
    Run([&](Population const &, Grades const &) { return --n > 0; });
  }

  template <typename GenerationActionFunctionT>
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
  void SetPopulation(Population &&pop) {
    if (pop.size() > GetPopulationSize())
      throw std::length_error(
          "For setting larger population, first extend stateFlow");
    auto grd = EvaluatePopulation(pop);
    AppendPopulation(std::move(pop), std::move(grd), pop.size());
    SortPopulation(population, grades);
    assert(Verify(population, grades));
  }

  void RegeneratePopulation() {
    ReservePopulation(GetPopulationSize());
    auto pop = GeneratePopulation(GetPopulationSize());
    auto grd = EvaluatePopulation(pop);
    SortPopulation(pop, grd);
    AppendPopulation(std::move(pop), std::move(grd), population.size());
    assert(Verify(population, grades));
  }

  void SetStateFlow(StateFlow &&stateFlow) {
    auto newSize = stateFlow.GetNEvaluates();
    auto oldSize = GetPopulationSize();
    auto pop = Population{};
    auto grd = Grades{};
    if (newSize > oldSize) {
      ReservePopulation(newSize);
      pop = GeneratePopulation(newSize - oldSize);
      grd = EvaluatePopulation(pop);
    }
    taskFlow.SetStateFlow(std::move(stateFlow));
    if (newSize > oldSize) {
      AppendPopulation(std::move(pop), std::move(grd));
      SortPopulation(population, grades);
    }
    else
      ShrinkPopulation(newSize);
    assert(Verify(population, grades));
  }

private:
  void ReservePopulation(size_t size) {
    population.reserve(size);
    grades.reserve(size);
  }

  void ShrinkPopulation(size_t newSize) noexcept {
    assert(population.size() >= newSize);
    grades.erase(grades.begin() + newSize, grades.end());
    population.erase(population.begin() + newSize, population.end());
  }

  Population GeneratePopulation(size_t size) {
    auto pop = Population{};
    pop.reserve(size);
    std::generate_n(std::back_inserter(pop), size, DNAGenerator);
    return pop;
  }

  void AppendPopulation(Population &&pop, Grades &&grd,
                        size_t backOffset = 0) noexcept {
    assert(population.size() >= backOffset);
    auto diff = pop.size() > backOffset ? pop.size() - backOffset : 0;
    auto newSize = population.size() + diff;
    assert(population.capacity() >= newSize);
    assert(grades.capacity() >= newSize);

    std::move(pop.begin(), pop.end() - diff, population.end() - backOffset);
    std::move(pop.end() - diff, pop.end(), std::back_inserter(population));
    std::move(grd.begin(), grd.end() - diff, grades.end() - backOffset);
    std::move(grd.end() - diff, grd.end(), std::back_inserter(grades));
  }

  void SortPopulation(Population &population, Grades &grades) const {
    auto permutation = SortPopulation_(population, grades);
    Utility::Permute(population, permutation, std::identity{});
    Utility::Permute(grades, permutation, std::identity{});
  }

  size_t GetPopulationSize() const {
    return taskFlow.GetStateFlow().GetNEvaluates();
  }

  bool Verify(Population const &population, Grades const &grades) const {
    auto verified = true;
    verified &= population.size() == GetPopulationSize();
    verified &= grades.size() == GetPopulationSize();
    return verified;
  }

  auto static GetSortFunction() {
    static_assert(std::is_arithmetic_v<Grade>,
                  "For custom Grades provide custom SortPopulation function");
    return [](Population const &population, Grades const &grades) {
      auto permutation = Utility::GetIndices(grades.size());
      std::sort(permutation.begin(), permutation.end(),
                [&](size_t index0, size_t index1) {
                  return grades[index0] > grades[index1];
                });
      return permutation;
    };
  }
};

} // namespace Evolution

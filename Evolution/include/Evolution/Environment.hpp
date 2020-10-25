#pragma once
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlow.hpp"
#include "Utility/Misc.hpp"

namespace Evolution {

template <typename EvaluateFG, typename MutateFG, typename CrossoverFG>
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
    ResizePopulation(stateFlow.GetNEvaluates());
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
  void SetPopulation(Population &&population) {
    if (population.size() > GetPopulation().size())
      throw std::length_error(
          "For setting larger population, first extend stateFlow");
    AppendPopulation(std::move(population), population.size());
  }

  void RegeneratePopulation() {
    auto newPop = Population{};
    newPop.reserve(population.size());
    std::generate_n(std::back_inserter(newPop), population.size(),
                    DNAGenerator);
    SetPopulation(std::move(newPop));
  }

  void SetStateFlow(StateFlow &&stateFlow) {
    taskFlow.SetStateFlow(std::move(stateFlow));
    ResizePopulation(taskFlow.GetStateFlow().GetNEvaluates());
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
      grades.erase(grades.begin() + newSize, grades.end());
      population.erase(population.begin() + newSize, population.end());
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

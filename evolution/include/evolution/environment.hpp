#pragma once
#include "evolution/concepts.hpp"
#include "evolution/state_flow.hpp"
#include "evolution/task_flow_container.hpp"
#include "evolution/utils.hpp"
#include "utility/misc.hpp"

namespace Evolution {

template <EvaluateFunctionOrGeneratorConcept EvaluateFG,
          MutateFunctionOrGeneratorConcept MutateFG,
          CrossoverFunctionOrGeneratorConcept CrossoverFG>
class Environment final {
private:
  using TaskFlowContainerInst =
      typename TaskFlowContainer<EvaluateFG, MutateFG, CrossoverFG>;
  using TaskFlowInst = typename TaskFlowContainerInst::TaskFlowInst;

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
  TaskFlowContainerInst taskFlow;
  Population population;
  Grades grades;

public:
  Environment(DNAGeneratorFunction const &DNAGenerator,
              EvaluateFG const &Evaluate, MutateFG const &Mutate,
              CrossoverFG const &Crossover, StateFlow const &stateFlow,
              EnvironmentOptions const &options = EnvironmentOptions{},
              SortPopulationFunction const &SortPopulation_ = GetSortFunction())
      : DNAGenerator(DNAGenerator), SortPopulation_(SortPopulation_),
        taskFlow(Evaluate, Mutate, Crossover, stateFlow,
                 ResolveAutoOptions(DNAGenerator, Evaluate, Mutate, Crossover,
                                    options)) {
    RegeneratePopulation();
  }

  Population const &GetPopulation() const & { return population; }
  Grades const &GetGrades() const & { return grades; }

  void Run(size_t n = 1) {
    Run([&](Population const &, Grades const &) { return n-- > 0; });
  }

  template <typename GenerationActionFunctionT>
  void Run(GenerationActionFunctionT &&GenerationAction) {
    static_assert(std::is_convertible_v<GenerationActionFunctionT,
                                        GenerationActionFunction>);
    while (GenerationAction(population, grades)) {
      assert(Verify(population, grades));
      taskFlow.Get().Run(population, grades);
      SortPopulation(population, grades);
    };
  }

  Grades EvaluatePopulation(Population const &population) {
    return taskFlow.Get().EvaluatePopulation(population);
  }

  // Also can be used to provide DNA examples
  void SetPopulation(Population &&pop) {
    assert(pop.size() <= GetPopulationSize() &&
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
    taskFlow.Get().SetStateFlow(std::move(stateFlow));
    if (newSize > oldSize) {
      AppendPopulation(std::move(pop), std::move(grd));
      SortPopulation(population, grades);
    } else
      ShrinkPopulation(newSize);
    assert(Verify(population, grades));
  }

private:
  static EnvironmentOptions
  ResolveAutoOptions(DNAGeneratorFunction const &DNAGenerator,
                     EvaluateFG const &Evaluate, MutateFG const &Mutate,
                     CrossoverFG const &Crossover,
                     EnvironmentOptions const &opts) {
    auto options = opts;
    auto dna0 = std::optional<DNA>{};
    auto dna1 = std::optional<DNA>{};
    using MutateF = typename GeneratorTraits::Function<MutateFG>;
    using MutateCT = typename Utility::CallableTraits<MutateF>;
    using CrossoverF = typename GeneratorTraits::Function<CrossoverFG>;
    using CrossoverCT = typename Utility::CallableTraits<CrossoverF>;

    if (opts.isEvaluateLightweight.isAuto()) {
      if (!dna0)
        dna0 = DNAGenerator();
      options.isEvaluateLightweight = IsFGLightweight(Evaluate, dna0.value());
    }
    if (opts.isMutateLightweight.isAuto()) {
      if (!dna0)
        dna0 = DNAGenerator();
      options.isMutateLightweight =
          IsFGLightweight(Mutate, MutateCT::Forward<1>(dna0.value()));
    }
    if (opts.isCrossoverLightweight.isAuto()) {
      dna0 = DNAGenerator();
      dna1 = DNAGenerator();
      options.isCrossoverLightweight =
          IsFGLightweight(Crossover, CrossoverCT::Forward<1>(dna0.value()),
                          CrossoverCT::Forward<2>(dna1.value()));
    }
    return options;
  }

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
    auto eHandler = Utility::ExceptionSaver{};
    if constexpr (std::is_nothrow_default_constructible_v<DNA>) {
      pop.resize(size);
      std::generate(std::execution::par_unseq, pop.begin(), pop.end(),
                    eHandler.Wrap(DNAGenerator));
      eHandler.Rethrow();
    } else {
      auto popPar = tbb::concurrent_vector<std::pair<size_t, DNA>>{};
      popPar.reserve(size);
      auto indices = Utility::GetIndices(size);
      std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                    eHandler.Wrap([&](size_t index) {
                      popPar.emplace_back(index, DNAGenerator());
                    }));
      eHandler.Rethrow();
      std::sort(popPar.begin(), popPar.end(),
                [](auto const &lhs, auto const &rhs) {
                  return lhs.first < rhs.first;
                });
      for (auto &&[index, dna] : popPar) {
        assert(index == pop.size());
        pop.push_back(std::move(dna));
      }
    }
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
    return taskFlow.Get().GetStateFlow().GetNEvaluates();
  }

  bool Verify(Population const &population, Grades const &grades) const {
    auto verified = true;
    verified &= population.size() == GetPopulationSize();
    verified &= grades.size() == GetPopulationSize();
    return verified;
  }

  static auto GetSortFunction() {
    static_assert(std::is_arithmetic_v<Grade>,
                  "For custom Grades provide custom SortPopulation function");
    return [](Population const &population, Grades const &grades) {
      auto permutation = Utility::GetSortPermutation(grades, std::greater<>{});
      return permutation;
    };
  }
};

} // namespace Evolution

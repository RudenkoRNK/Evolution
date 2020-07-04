#pragma once
#define NOMINMAX
#include "ArgumentTraits.hpp"
#include "StateFlow.hpp"
#include "Utility.hpp"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_unordered_set.h"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/flow_graph.h"
#include "tbb/parallel_for_each.h"
#include <algorithm>
#include <cassert>
#include <execution>
#include <functional>
#include <map>
#include <utility>
#include <variant>
#include <vector>

namespace Evolution {
template <class EvaluateFG, class MutateFG, class CrossoverFG>
class TBBFlow final {
private:
  template <class FG>
  auto constexpr static isThreadSpecific = ArgumentTraits<FG>::nArguments == 0;

  template <class FG>
  using Func_ =
      std::conditional_t<isThreadSpecific<FG>,
                         typename ArgumentTraits<FG>::template Type<0>, FG>;

  using EvaluateFunction = Func_<EvaluateFG>;
  using MutateFunction = Func_<MutateFG>;
  using CrossoverFunction = Func_<CrossoverFG>;

public:
  using DNA = std::remove_cvref_t<
      typename ArgumentTraits<EvaluateFunction>::template Type<1>>;

private:
  using DNAGeneratorFunction = std::function<DNA()>;

  // DNA should a modifiable self-sufficient type
  static_assert(!std::is_const_v<DNA>);
  static_assert(!std::is_reference_v<DNA>);

  // Check that arguments and return values of functions are of type DNA
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<typename ArgumentTraits<
                                        EvaluateFunction>::template Type<1>>>);
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<typename ArgumentTraits<
                                        MutateFunction>::template Type<0>>>);
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<typename ArgumentTraits<
                                        MutateFunction>::template Type<1>>>);
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<typename ArgumentTraits<
                                        CrossoverFunction>::template Type<0>>>);
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<typename ArgumentTraits<
                                        CrossoverFunction>::template Type<1>>>);
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<typename ArgumentTraits<
                                        CrossoverFunction>::template Type<2>>>);

  // Evaluate function must not modify its argument
  static_assert(ArgumentTraits<EvaluateFunction>::template isConst<1> ||
                ArgumentTraits<EvaluateFunction>::template isValue<1>);

  // Const qualifier can appear only with r-value reference
  static_assert(ArgumentTraits<MutateFunction>::template isLValueReference<1> ||
                !ArgumentTraits<MutateFunction>::template isConst<1>);
  static_assert(
      ArgumentTraits<CrossoverFunction>::template isLValueReference<1> ||
      !ArgumentTraits<CrossoverFunction>::template isConst<1>);
  static_assert(
      ArgumentTraits<CrossoverFunction>::template isLValueReference<2> ||
      !ArgumentTraits<CrossoverFunction>::template isConst<2>);

  auto constexpr static isMutateInPlace =
      (!ArgumentTraits<MutateFunction>::template isConst<1>);
  auto constexpr static isCrossoverInPlaceFirst =
      !ArgumentTraits<CrossoverFunction>::template isConst<1>;
  auto constexpr static isCrossoverInPlaceSecond =
      !ArgumentTraits<CrossoverFunction>::template isConst<2>;

  using EvaluateGenerator = EvaluateFG;
  using MutateGenerator = MutateFG;
  using CrossoverGenerator = CrossoverFG;

  template <class Function>
  using ThreadSpecific = tbb::enumerable_thread_specific<Function>;

  template <class FG>
  using ThreadSpecificOrGlobalFunction =
      std::conditional_t<isThreadSpecific<FG>, ThreadSpecific<Func_<FG>>,
                         Func_<FG>>;

  using EvaluateThreadSpecificOrGlobalFunction =
      ThreadSpecificOrGlobalFunction<EvaluateFG>;
  using MutateThreadSpecificOrGlobalFunction =
      ThreadSpecificOrGlobalFunction<MutateFG>;
  using CrossoverThreadSpecificOrGlobalFunction =
      ThreadSpecificOrGlobalFunction<CrossoverFG>;

  using State = StateFlow::State;
  using StateSet = StateFlow::StateSet;
  using StateVector = StateFlow::StateVector;
  using Operation = StateFlow::Operation;
  using OperationSet = StateFlow::OperationSet;
  using OperationType = StateFlow::OperationType;
  using IndexSet = StateFlow::IndexSet;

  using DNAPtr = std::shared_ptr<DNA>;

  // TODO: add support for lightweight policy
  using InputNode = tbb::flow::function_node<size_t, DNAPtr>;
  using EvaluateNode = tbb::flow::function_node<DNAPtr, int>;
  using MutateNode = tbb::flow::function_node<DNAPtr, DNAPtr>;
  using CrossoverNode =
      tbb::flow::function_node<std::tuple<DNAPtr, DNAPtr>, DNAPtr>;
  using CrossoverJoinNode = tbb::flow::join_node<std::tuple<DNAPtr, DNAPtr>>;

public:
  using Population = std::vector<DNA>;
  using Grades = std::vector<double>;

  template <class DNAGeneratorFG>
  TBBFlow(EvaluateFG &&Evaluate, MutateFG &&Mutate, CrossoverFG &&Crossover,
          DNAGeneratorFG &&DNAGenerator, StateFlow &&stateFlow);
  void Run();
  void AllowSwapArgumentsInCrossover() noexcept;
  void PermutePopulation(std::vector<size_t> &permutation);
  template <class Indexer, class IndexFunction>
  void PermutePopulation(std::vector<Indexer> &permutation,
                         IndexFunction &&Index);
  void RegeneratePopulation();
  void SetPopulation(Population &&population) noexcept;
  void SetPopulationWithGrades(Population &&population,
                               Grades &&grades) noexcept;

  Population const &GetPopulation() const noexcept;
  Grades const &GetGrades() const noexcept;
  size_t GetPopulationSize() const noexcept;
  StateFlow const &GetStateFlow() const noexcept;

private:
  // Set up in constructor:
  EvaluateThreadSpecificOrGlobalFunction EvaluateThreadSpecificOrGlobal;
  MutateThreadSpecificOrGlobalFunction MutateThreadSpecificOrGlobal;
  CrossoverThreadSpecificOrGlobalFunction CrossoverThreadSpecificOrGlobal;
  ThreadSpecific<DNAGeneratorFunction> DNAGeneratorThreadSpecific;
  DNAGeneratorFunction DNAGenerator;
  StateFlow SF;

  // Set up with additional interface methods:
  Population population;
  Grades grades;
  bool isSwapArgumentsAllowedInCrossover = false;

  // Generated internally:
  size_t const populationSize;
  bool isPopulationGenerated = false;
  bool isPopulationEvaluated = false;
  bool isTaskFlowGenerated = false;
  tbb::flow::graph taskFlow;
  tbb::concurrent_unordered_map<DNAPtr, double, std::hash<DNAPtr>>
      evaluateBuffer;
  std::vector<InputNode> inputNodes;
  std::vector<size_t> inputIndices;
  std::vector<EvaluateNode> evaluateNodes;
  std::vector<MutateNode> mutateNodes;
  std::vector<CrossoverJoinNode> crossoverJoinNodes;
  std::vector<CrossoverNode> crossoverNodes;

  EvaluateFunction &GetEvaluateFunction();
  MutateFunction &GetMutateFunction();
  CrossoverFunction &GetCrossoverFunction();
  DNAGeneratorFunction &GetDNAGeneratorFunction();
  DNAPtr CopyHelper(DNA const &src) const;
  DNAPtr MoveHelper(DNA &&src) const;
  void EvaluateHelper(DNAPtr iSrc, bool isCopy = false);
  DNAPtr MutateHelper(DNAPtr iSrc, bool isCopy);
  DNAPtr CrossoverHelper(std::tuple<DNAPtr, DNAPtr> iSrcs, bool isCopy0,
                         bool isCopy1);
  InputNode &AddInput(size_t index, bool isCopy);
  template <class Node>
  EvaluateNode &AddEvaluate(Node &predecessor, bool isCopy = false);
  template <class Node> MutateNode &AddMutate(Node &predecessor, bool isCopy);
  template <class Node0, class Node1>
  CrossoverNode &AddCrossover(Node0 &predecessor0, Node1 &predecessor1,
                              bool isCopy0, bool isCopy1);

  void RunTaskFlow();
  void ResetTaskFlow();
  void GenerateTaskFlow();
  void ResetPopulation() noexcept;
  Population
  GeneratePopulation(); /* Should be const, except invoking DNAGenerator */
  void EvaluatePopulation();
  void MoveResultsFromBuffer();
};

} // namespace Evolution

namespace Evolution {
template <class EvaluateFG, class MutateFG, class CrossoverFG>
template <class DNAGeneratorFG>
inline TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::TBBFlow(
    EvaluateFG &&Evaluate, MutateFG &&Mutate, CrossoverFG &&Crossover,
    DNAGeneratorFG &&DNAGenerator_, StateFlow &&stateFlow)
    : SF(std::move(stateFlow)),
      EvaluateThreadSpecificOrGlobal(std::forward<EvaluateFG>(Evaluate)),
      MutateThreadSpecificOrGlobal(std::forward<MutateFG>(Mutate)),
      CrossoverThreadSpecificOrGlobal(std::forward<CrossoverFG>(Crossover)),
      populationSize(stateFlow.GetNEvaluates()) {
  assert(SF.Verify());
  if constexpr (std::is_convertible_v<DNAGeneratorFG, DNAGeneratorFunction>)
    DNAGenerator =
        DNAGeneratorFunction(std::forward<DNAGeneratorFG>(DNAGenerator_));
  else {
    static_assert(std::is_convertible_v<
                  typename ArgumentTraits<DNAGeneratorFG>::template Type<0>,
                  DNAGeneratorFunction>);
    DNAGeneratorThreadSpecific = ThreadSpecific<DNAGeneratorFunction>(
        std::forward<DNAGeneratorFG>(DNAGenerator_));
  }
}
template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::Run() {
  if (!isPopulationGenerated)
    RegeneratePopulation();
  if (!isPopulationEvaluated) {
    EvaluatePopulation();
    return;
  }
  if (!isTaskFlowGenerated)
    GenerateTaskFlow();

  RunTaskFlow();
  MoveResultsFromBuffer();
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::Population const &
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::GetPopulation() const noexcept {
  assert(isPopulationGenerated);
  return population;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::Grades const &
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::GetGrades() const noexcept {
  return grades;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::PermutePopulation(
    std::vector<size_t> &permutation) {
  Permute(population, permutation, std::identity{});
  Permute(grades, permutation, std::identity{});
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
template <class Indexer, class IndexFunction>
inline void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::PermutePopulation(
    std::vector<Indexer> &permutation, IndexFunction &&Index) {
  Permute(population, permutation, Index);
  Permute(grades, permutation, Index);
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void TBBFlow<EvaluateFG, MutateFG,
                    CrossoverFG>::AllowSwapArgumentsInCrossover() noexcept {
  isSwapArgumentsAllowedInCrossover = true;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::RegeneratePopulation() {
  auto population_ = GeneratePopulation();
  ResetPopulation();
  population = std::move(population_);
  isPopulationGenerated = true;
  isPopulationEvaluated = false;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::SetPopulation(
    Population &&population_) noexcept {
  assert(SF.GetNEvaluates() == population_.size());
  population = std::move(population_);
  isPopulationGenerated = true;
  isPopulationEvaluated = false;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::SetPopulationWithGrades(
    Population &&population_, Grades &&grades_) {
  assert(SF.GetNEvaluates() == grades_.size());
  SetPopulation(std::move(population_));
  grades = std::move(grades_);
  isPopulationEvaluated = true;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline StateFlow const &
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::GetStateFlow() const noexcept {
  return SF;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::EvaluateFunction &
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::GetEvaluateFunction() {
  if constexpr (isThreadSpecific<EvaluateFG>)
    return EvaluateThreadSpecificOrGlobal.local();
  else
    return EvaluateThreadSpecificOrGlobal;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::MutateFunction &
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::GetMutateFunction() {
  if constexpr (isThreadSpecific<MutateFG>)
    return MutateThreadSpecificOrGlobal.local();
  else
    return MutateThreadSpecificOrGlobal;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::CrossoverFunction &
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::GetCrossoverFunction() {
  if constexpr (isThreadSpecific<CrossoverFG>)
    return CrossoverThreadSpecificOrGlobal.local();
  else
    return CrossoverThreadSpecificOrGlobal;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline
    typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::DNAGeneratorFunction &
    TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::GetDNAGeneratorFunction() {
  if (DNAGenerator)
    return DNAGenerator;
  return DNAGeneratorThreadSpecific.local();
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline size_t
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::GetPopulationSize() const noexcept {
  return populationSize;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::DNAPtr
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::CopyHelper(DNA const &src) const {
  return DNAPtr(new DNA(src));
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::DNAPtr
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::MoveHelper(DNA &&src) const {
  return DNAPtr(new DNA(std::move(src)));
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::EvaluateHelper(DNAPtr iSrc,
                                                                bool isCopy) {
  // Actually this functionality should not be used
  assert(!isCopy);
  if (isCopy)
    iSrc = CopyHelper(*iSrc);
  auto &&Evaluate = GetEvaluateFunction();
  auto grade = Evaluate(*iSrc);
  evaluateBuffer.emplace(iSrc, grade);
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::DNAPtr
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::MutateHelper(DNAPtr iSrc,
                                                         bool isCopy) {
  auto &&Mutate = GetMutateFunction();
#ifndef NDEBUG
  auto iSrcOrig = iSrc;
#endif // !NDEBUG
  auto isCopyHelp = isMutateInPlace && isCopy;
  auto constexpr isMove =
      ArgumentTraits<MutateFunction>::template isRValueReference<1> ||
      ArgumentTraits<MutateFunction>::template isValue<1>;
  auto isPtrMove = !isCopy || isCopyHelp;
  if (isCopyHelp)
    iSrc = CopyHelper(*iSrc);

  auto MutatePtr = [&](DNAPtr iSrc) {
    assert(isCopy || evaluateBuffer.count(iSrcOrig) == 0);
    if constexpr (isMove) {
      assert(!isCopy || iSrcOrig != iSrc);
      return Mutate(std::move(*iSrc));
    } else {
      assert(!isCopy || !isMutateInPlace);
      return Mutate(*iSrc);
    }
  };

  auto &&dst = MutatePtr(iSrc);
  if (!isPtrMove)
    return MoveHelper(std::move(dst));
  *iSrc = std::move(dst);
  return iSrc;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::DNAPtr
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::CrossoverHelper(
    std::tuple<DNAPtr, DNAPtr> iSrcs, bool isCopy0, bool isCopy1) {
  auto &&Crossover = GetCrossoverFunction();
  auto isSwap = ((isCrossoverInPlaceFirst && !isCrossoverInPlaceSecond &&
                  isCopy0 && !isCopy1) ||
                 (isCrossoverInPlaceSecond && !isCrossoverInPlaceFirst &&
                  isCopy1 && !isCopy0)) &&
                isSwapArgumentsAllowedInCrossover;
  auto iSrc0 = std::get<0>(iSrcs);
  auto iSrc1 = std::get<1>(iSrcs);
  if (isSwap) {
    iSrc0.swap(iSrc1);
    std::swap(isCopy0, isCopy1);
  }
#ifndef NDEBUG
  auto iSrcOrig0 = iSrc0;
  auto iSrcOrig1 = iSrc1;
#endif // !NDEBUG

  auto isCopyHelp0 = isCrossoverInPlaceFirst && isCopy0;
  auto isCopyHelp1 = isCrossoverInPlaceSecond && isCopy1;
  auto constexpr isMove0 =
      ArgumentTraits<CrossoverFunction>::template isRValueReference<1> ||
      ArgumentTraits<CrossoverFunction>::template isValue<1>;
  auto constexpr isMove1 =
      ArgumentTraits<CrossoverFunction>::template isRValueReference<2> ||
      ArgumentTraits<CrossoverFunction>::template isValue<2>;
  auto isPtrMove0 = !isCopy0 || isCopyHelp0;
  auto isPtrMove1 = !isCopy1 || isCopyHelp1;

  if (isCopyHelp0)
    iSrc0 = CopyHelper(*iSrc0);
  if (isCopyHelp1)
    iSrc1 = CopyHelper(*iSrc1);

  auto CrossoverPtr = [&](DNAPtr iSrc0, DNAPtr iSrc1) {
    assert(isCopy0 || evaluateBuffer.count(iSrcOrig0) == 0);
    assert(isCopy1 || evaluateBuffer.count(iSrcOrig1) == 0);
    if constexpr (isMove0) {
      assert(!isCopy0 || iSrcOrig0 != iSrc0);
      if constexpr (isMove1) {
        assert(!isCopy1 || iSrcOrig1 != iSrc1);
        return Crossover(std::move(*iSrc0), std::move(*iSrc1));
      } else {
        assert(!isCopy1 || !isCrossoverInPlaceSecond);
        return Crossover(std::move(*iSrc0), *iSrc1);
      }
    } else {
      assert(!isCopy0 || !isCrossoverInPlaceFirst);
      if constexpr (isMove1) {
        assert(!isCopy1 || iSrcOrig1 != iSrc1);
        return Crossover(*iSrc0, std::move(*iSrc1));
      } else {
        assert(!isCopy1 || !isCrossoverInPlaceSecond);
        return Crossover(*iSrc0, *iSrc1);
      }
    }
  };

  auto &&dst = CrossoverPtr(iSrc0, iSrc1);
  if (!isPtrMove0 && !isPtrMove1)
    return MoveHelper(std::move(dst));
  if (isPtrMove0 && !isPtrMove1) {
    *iSrc0 = std::move(dst);
    return iSrc0;
  }
  if (isPtrMove1 && !isPtrMove0) {
    *iSrc1 = std::move(dst);
    return iSrc1;
  }
  if constexpr (isCrossoverInPlaceFirst && !isCrossoverInPlaceSecond) {
    *iSrc0 = std::move(dst);
    return iSrc0;
  }
  if constexpr (isCrossoverInPlaceSecond && !isCrossoverInPlaceFirst) {
    *iSrc1 = std::move(dst);
    return iSrc1;
  }
  *iSrc0 = std::move(dst);
  return iSrc0;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::InputNode &
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::AddInput(size_t index,
                                                     bool isCopy) {
  // Should be preallocated to avoid refs invalidation
  assert(inputNodes.capacity() > inputNodes.size());
  inputIndices.push_back(index);
  inputNodes.push_back(InputNode(
      taskFlow, tbb::flow::concurrency::serial, [&, isCopy](size_t index_) {
        return isCopy ? CopyHelper(population.at(index_))
                      : MoveHelper(std::move(population.at(index_)));
      }));
  assert(inputIndices.size() == inputNodes.size());
  return inputNodes.back();
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
template <class Node>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::EvaluateNode &
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::AddEvaluate(Node &predecessor,
                                                        bool isCopy) {
  // Should be preallocated to avoid refs invalidation
  assert(evaluateNodes.capacity() > evaluateNodes.size());
  evaluateNodes.push_back(EvaluateNode(taskFlow, tbb::flow::concurrency::serial,
                                       [&, isCopy](DNAPtr iSrc) {
                                         EvaluateHelper(iSrc, isCopy);
                                         return 0;
                                       }));
  tbb::flow::make_edge(predecessor, evaluateNodes.back());
  return evaluateNodes.back();
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
template <class Node>
typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::MutateNode &
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::AddMutate(Node &predecessor,
                                                      bool isCopy) {
  // Should be preallocated to avoid refs invalidation
  assert(mutateNodes.capacity() > mutateNodes.size());
  mutateNodes.push_back(MutateNode(
      taskFlow, tbb::flow::concurrency::serial,
      [&, isCopy](DNAPtr iSrc) { return MutateHelper(iSrc, isCopy); }));
  tbb::flow::make_edge(predecessor, mutateNodes.back());
  return mutateNodes.back();
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
template <class Node0, class Node1>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::CrossoverNode &
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::AddCrossover(Node0 &predecessor0,
                                                         Node1 &predecessor1,
                                                         bool isCopy0,
                                                         bool isCopy1) {
  // Should be preallocated to avoid refs invalidation
  assert(crossoverNodes.capacity() > crossoverNodes.size());
  crossoverNodes.push_back(
      CrossoverNode(taskFlow, tbb::flow::concurrency::serial,
                    [&, isCopy0, isCopy1](std::tuple<DNAPtr, DNAPtr> iSrcs) {
                      return CrossoverHelper(iSrcs, isCopy0, isCopy1);
                    }));
  crossoverJoinNodes.push_back(CrossoverJoinNode(taskFlow));
  tbb::flow::make_edge(predecessor0, input_port<0>(crossoverJoinNodes.back()));
  tbb::flow::make_edge(predecessor1, input_port<1>(crossoverJoinNodes.back()));
  tbb::flow::make_edge(crossoverJoinNodes.back(), crossoverNodes.back());
  return crossoverNodes.back();
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline typename TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::Population
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::GeneratePopulation() {
  auto population = Population{};
  auto populationConcurrent = tbb::concurrent_vector<DNA>{};
  populationConcurrent.reserve(GetPopulationSize());

  tbb::parallel_for(tbb::blocked_range<size_t>(0, GetPopulationSize()),
                    [&](tbb::blocked_range<size_t> r) {
                      std::generate_n(std::back_inserter(populationConcurrent),
                                      r.end() - r.begin(),
                                      GetDNAGeneratorFunction());
                    });

  population.reserve(GetPopulationSize());
  for (auto &&dna : populationConcurrent)
    population.push_back(std::move(dna));
  return population;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::EvaluatePopulation() {
  assert(isPopulationGenerated);
  auto indices = std::vector<size_t>(GetPopulationSize());
  std::iota(indices.begin(), indices.end(), size_t{0});
  auto grades_ = Grades(GetPopulationSize());
  tbb::parallel_for_each(indices.begin(), indices.end(), [&](size_t index) {
    auto &&Evaluate = GetEvaluateFunction();
    grades_.at(index) = Evaluate(population.at(index));
  });
  grades = std::move(grades_);
  isPopulationEvaluated = true;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::MoveResultsFromBuffer() {
  auto availableIndices = std::unordered_set<size_t>{};
  for (auto i = size_t{0}; i < population.size(); ++i)
    availableIndices.insert(i);
  for (auto state : SF.GetInitialStates())
    if (SF.IsEvaluate(state))
      availableIndices.erase(SF.GetIndex(state));
  assert(evaluateBuffer.size() == availableIndices.size());
  for (auto &&dnaPair : evaluateBuffer) {
    auto index = *availableIndices.begin();
    availableIndices.erase(index);
    population.at(index) = std::move(*dnaPair.first);
    grades.at(index) = dnaPair.second;
  }
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::RunTaskFlow() {
  assert(population.size() == GetPopulationSize());
  evaluateBuffer.clear();
  for (auto i = size_t{0}; i < inputNodes.size(); ++i)
    inputNodes.at(i).try_put(inputIndices.at(i));
  taskFlow.wait_for_all();
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::ResetTaskFlow() {
  inputNodes.clear();
  evaluateNodes.clear();
  mutateNodes.clear();
  crossoverJoinNodes.clear();
  crossoverNodes.clear();
  inputIndices.clear();
  taskFlow.reset();
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::GenerateTaskFlow() {
  using InputNodeRef = std::reference_wrapper<InputNode>;
  using MutateNodeRef = std::reference_wrapper<MutateNode>;
  using CrossoverNodeRef = std::reference_wrapper<CrossoverNode>;
  auto constexpr InputNodeIndex = std::in_place_index<0>;
  auto constexpr MutateNodeIndex = std::in_place_index<1>;
  auto constexpr CrossoverNodeIndex = std::in_place_index<2>;
  using NodeRef = std::variant<InputNodeRef, MutateNodeRef, CrossoverNodeRef>;

  assert(inputNodes.size() == 0);
  assert(evaluateNodes.size() == 0);
  assert(mutateNodes.size() == 0);
  assert(crossoverJoinNodes.size() == 0);
  assert(crossoverNodes.size() == 0);
  assert(inputIndices.size() == 0);
  assert(!taskFlow.is_cancelled());

  inputNodes.reserve(SF.GetInitialStates().size());
  inputIndices.reserve(SF.GetInitialStates().size());
  evaluateNodes.reserve(SF.GetNEvaluates());
  mutateNodes.reserve(SF.GetNMutates());
  crossoverJoinNodes.reserve(SF.GetNCrossovers());
  crossoverNodes.reserve(SF.GetNCrossovers());
  auto nodes = std::unordered_map<State, NodeRef>{};

#ifndef NDEBUG
  auto maxIterations = SF.GetNStates() + 1;
  auto whileGuard = size_t{0};
  auto evaluateCount = size_t{0};
#endif // !NDEBUG

  auto IsResolved = [&](State state) { return nodes.contains(state); };
  auto IsResolvable = [&](State state) {
    if (SF.IsInitialState(state))
      return true;
    auto op = SF.GetAnyInOperation(state);
    auto parent = SF.GetSource(op);
    return IsResolved(parent) &&
           (!SF.IsCrossover(op) ||
            IsResolved(SF.GetSource(SF.GetCrossoverPair(op))));
  };
  auto IsCopyRequired = [&](Operation operation) {
    auto parent = SF.GetSource(operation);
    if (SF.IsEvaluate(parent))
      return true;
    auto const &[oB, oE] = SF.GetOutOperations(parent);
    if (oE - oB == 1)
      return false;
    // TODO: This is too conservative.
    // There are cases when copy is not needed
    return true;
  };
  auto ResolveInitial = [&](State state) {
    assert(!IsResolved(state) && IsResolvable(state));
    assert(SF.IsInitialState(state));
    auto &&node = AddInput(SF.GetIndex(state), SF.IsEvaluate(state));
    nodes.emplace(state, NodeRef(InputNodeIndex, std::ref(node)));
  };
  auto ResolveMutate = [&](State state) {
    assert(!IsResolved(state) && IsResolvable(state));
    auto op = SF.GetAnyInOperation(state);
    assert(SF.IsMutate(op));
    auto parent = SF.GetSource(op);
    auto &&parentNodeVar = nodes.at(parent);
    std::visit(
        [&](auto parentNodeRef) {
          auto &&node = AddMutate(parentNodeRef.get(), IsCopyRequired(op));
          nodes.emplace(state, NodeRef(MutateNodeIndex, std::ref(node)));
        },
        parentNodeVar);
  };
  auto ResolveCrossover = [&](State state) {
    assert(!IsResolved(state) && IsResolvable(state));
    auto op0 = SF.GetAnyInOperation(state);
    auto op1 = SF.GetCrossoverPair(op0);
    auto parent0 = SF.GetSource(op0);
    auto parent1 = SF.GetSource(op1);
    auto &&parentNodeVar0 = nodes.at(parent0);
    auto &&parentNodeVar1 = nodes.at(parent1);
    std::visit(
        [&](auto parentNodeRef0) {
          std::visit(
              [&](auto parentNodeRef1) {
                auto &&node =
                    AddCrossover(parentNodeRef0.get(), parentNodeRef1.get(),
                                 IsCopyRequired(op0), IsCopyRequired(op1));
                nodes.emplace(state,
                              NodeRef(CrossoverNodeIndex, std::ref(node)));
              },
              parentNodeVar1);
        },
        parentNodeVar0);
  };
  auto ResolveEvaluate = [&](State state) {
    assert(SF.IsEvaluate(state));
#ifndef NDEBUG
    evaluateCount++;
#endif // !NDEBUG
    if (SF.IsInitialState(state))
      return;
    std::visit(
        [&](auto nodeRef) { AddEvaluate(nodeRef.get(), /*isCopy*/ false); },
        nodes.at(state));
  };

  bool modified = true;
  auto Act = [&](State state) {
    if (IsResolved(state))
      return;
    if (!IsResolvable(state))
      return;
    modified = true;
    if (SF.IsInitialState(state))
      ResolveInitial(state);
    else if (SF.IsMutate(SF.GetAnyInOperation(state)))
      ResolveMutate(state);
    else
      ResolveCrossover(state);
    if (SF.IsEvaluate(state))
      ResolveEvaluate(state);
  };

  while (modified) {
    modified = false;
    SF.BreadthFirstSearch(SF.GetInitialStates(), std::move(Act));
    assert(whileGuard++ < maxIterations);
  }
  assert(nodes.size() == SF.GetNStates());
  assert(evaluateCount == SF.GetNEvaluates());
  isTaskFlowGenerated = true;
}

template <class EvaluateFG, class MutateFG, class CrossoverFG>
inline void
TBBFlow<EvaluateFG, MutateFG, CrossoverFG>::ResetPopulation() noexcept {
  population.clear();
  grades.clear();
  isPopulationGenerated = false;
  isPopulationEvaluated = false;
}

} // namespace Evolution

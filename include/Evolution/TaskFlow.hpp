#pragma once
#define NOMINMAX
#include "Evolution/ArgumentTraits.hpp"
#include "Evolution/GeneratorTraits.hpp"
#include "Evolution/StateFlow.hpp"
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
class TaskFlow final {

private:
  using EvaluateFunction = typename GeneratorTraits<EvaluateFG>::Function;
  using MutateFunction = typename GeneratorTraits<MutateFG>::Function;
  using CrossoverFunction = typename GeneratorTraits<CrossoverFG>::Function;

public:
  using DNA = std::remove_cvref_t<
      typename ArgumentTraits<EvaluateFunction>::template Type<1>>;

private:
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

  using State = StateFlow::State;
  using StateSet = StateFlow::StateSet;
  using StateVector = StateFlow::StateVector;
  using Operation = StateFlow::Operation;
  using OperationSet = StateFlow::OperationSet;
  using OperationType = StateFlow::OperationType;
  using IndexSet = StateFlow::IndexSet;

  using DNAPtr = std::shared_ptr<DNA>;

  // TODO: add support for lightweight policy
  using InputNode = tbb::flow::function_node<DNA const *, DNAPtr>;
  using EvaluateNode = tbb::flow::function_node<DNAPtr, int>;
  using MutateNode = tbb::flow::function_node<DNAPtr, DNAPtr>;
  using CrossoverNode =
      tbb::flow::function_node<std::tuple<DNAPtr, DNAPtr>, DNAPtr>;
  using CrossoverJoinNode = tbb::flow::join_node<std::tuple<DNAPtr, DNAPtr>>;
  using EvaluateBuffer = tbb::concurrent_vector<std::pair<DNAPtr, double>>;

public:
  using EvaluateThreadSpecificOrGlobalFunction =
      typename GeneratorTraits<EvaluateFG>::ThreadSpecificOrGlobalFunction;
  using MutateThreadSpecificOrGlobalFunction =
      typename GeneratorTraits<MutateFG>::ThreadSpecificOrGlobalFunction;
  using CrossoverThreadSpecificOrGlobalFunction =
      typename GeneratorTraits<CrossoverFG>::ThreadSpecificOrGlobalFunction;
  using Population = std::vector<DNA>;
  using Grades = std::vector<double>;

  TaskFlow(EvaluateFG &&Evaluate, MutateFG &&Mutate, CrossoverFG &&Crossover,
           StateFlow const &stateFlow,
           bool isSwapArgumentsAllowedInCrossover = false) {
    assert(stateFlow.Verify());
    GenerateGraph(stateFlow, isSwapArgumentsAllowedInCrossover);
    nonEvaluateInitialIndices = EvaluateNonEvaluateInitialIndices(stateFlow);
    EvaluateThreadSpecificOrGlobal =
        GeneratorTraits<EvaluateFG>::GetThreadSpecificOrGlobal(
            std::forward<EvaluateFG>(Evaluate));
    MutateThreadSpecificOrGlobal =
        GeneratorTraits<MutateFG>::GetThreadSpecificOrGlobal(
            std::forward<MutateFG>(Mutate));
    CrossoverThreadSpecificOrGlobal =
        GeneratorTraits<CrossoverFG>::GetThreadSpecificOrGlobal(
            std::forward<CrossoverFG>(Crossover));
  }

  void Run(Population &population, Grades &grades) {
    RunTaskFlow(population);
    MoveResultsFromBuffer(population, grades);
  }

private:
  std::unordered_set<size_t> nonEvaluateInitialIndices;

  // tbb::flow::graph does not have copy or move assignment
  std::unique_ptr<tbb::flow::graph> graphPtr;
  std::vector<size_t> inputIndices;
  std::vector<InputNode> inputNodes;
  std::vector<EvaluateNode> evaluateNodes;
  std::vector<MutateNode> mutateNodes;
  std::vector<CrossoverJoinNode> crossoverJoinNodes;
  std::vector<CrossoverNode> crossoverNodes;
  tbb::concurrent_vector<std::pair<DNAPtr, double>> evaluateBuffer;
  EvaluateThreadSpecificOrGlobalFunction EvaluateThreadSpecificOrGlobal;
  MutateThreadSpecificOrGlobalFunction MutateThreadSpecificOrGlobal;
  CrossoverThreadSpecificOrGlobalFunction CrossoverThreadSpecificOrGlobal;

  EvaluateFunction &GetEvaluateFunction() {
    return GeneratorTraits<EvaluateFG>::GetFunction(
        EvaluateThreadSpecificOrGlobal);
  }
  MutateFunction &GetMutateFunction() {
    return GeneratorTraits<MutateFG>::GetFunction(MutateThreadSpecificOrGlobal);
  }
  CrossoverFunction &GetCrossoverFunction() {
    return GeneratorTraits<CrossoverFG>::GetFunction(
        CrossoverThreadSpecificOrGlobal);
  }

  DNAPtr CopyHelper(DNA const &src) const { return DNAPtr(new DNA(src)); }
  DNAPtr MoveHelper(DNA &&src) const { return DNAPtr(new DNA(std::move(src))); }
  void EvaluateHelper(DNAPtr iSrc, bool isCopy) {
    // Actually this functionality should not be used
    assert(!isCopy);
    if (isCopy)
      iSrc = CopyHelper(*iSrc);
    assert(evaluateBuffer.capacity() > evaluateBuffer.size());
    auto &&Evaluate = GetEvaluateFunction();
    auto grade = Evaluate(*iSrc);
    evaluateBuffer.push_back({iSrc, grade});
  }
  DNAPtr MutateHelper(DNAPtr iSrc, bool isCopy) {
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

    assert(isCopy || std::find_if(evaluateBuffer.begin(), evaluateBuffer.end(),
                                  [&](auto const &pair) {
                                    return pair.first == iSrcOrig;
                                  }) == evaluateBuffer.end());

    auto MutatePtr = [&](DNAPtr iSrc) {
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
  DNAPtr CrossoverHelper(std::tuple<DNAPtr, DNAPtr> iSrcs, bool isCopy0,
                         bool isCopy1, bool isSwapArgumentsAllowed) {
    auto &&Crossover = GetCrossoverFunction();
    auto isSwap = ((isCrossoverInPlaceFirst && !isCrossoverInPlaceSecond &&
                    isCopy0 && !isCopy1) ||
                   (isCrossoverInPlaceSecond && !isCrossoverInPlaceFirst &&
                    isCopy1 && !isCopy0)) &&
                  isSwapArgumentsAllowed;
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

    assert(isCopy0 || std::find_if(evaluateBuffer.begin(), evaluateBuffer.end(),
                                   [&](auto const &pair) {
                                     return pair.first == iSrcOrig0;
                                   }) == evaluateBuffer.end());

    assert(isCopy1 || std::find_if(evaluateBuffer.begin(), evaluateBuffer.end(),
                                   [&](auto const &pair) {
                                     return pair.first == iSrcOrig1;
                                   }) == evaluateBuffer.end());

    auto CrossoverPtr = [&](DNAPtr iSrc0, DNAPtr iSrc1) {
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

  InputNode &AddInput(size_t index) {
    // Should be preallocated to avoid refs invalidation
    assert(inputNodes.capacity() > inputNodes.size());
    inputIndices.push_back(index);
    // Could be implemented with MoveHelper in some cases, but this approach is
    // exception-safe for the Population
    inputNodes.push_back(
        InputNode(*graphPtr, tbb::flow::concurrency::serial,
                  [&](DNA const *dna) { return CopyHelper(*dna); }));
    assert(inputIndices.size() == inputNodes.size());
    return inputNodes.back();
  }
  template <class Node>
  EvaluateNode &AddEvaluate(Node &predecessor, bool isCopy = false) {
    // Should be preallocated to avoid refs invalidation
    assert(evaluateNodes.capacity() > evaluateNodes.size());
    evaluateNodes.push_back(EvaluateNode(
        *graphPtr, tbb::flow::concurrency::serial, [&, isCopy](DNAPtr iSrc) {
          EvaluateHelper(iSrc, isCopy);
          return 0;
        }));
    tbb::flow::make_edge(predecessor, evaluateNodes.back());
    return evaluateNodes.back();
  }
  template <class Node> MutateNode &AddMutate(Node &predecessor, bool isCopy) {
    // Should be preallocated to avoid refs invalidation
    assert(mutateNodes.capacity() > mutateNodes.size());
    mutateNodes.push_back(MutateNode(
        *graphPtr, tbb::flow::concurrency::serial,
        [&, isCopy](DNAPtr iSrc) { return MutateHelper(iSrc, isCopy); }));
    tbb::flow::make_edge(predecessor, mutateNodes.back());
    return mutateNodes.back();
  }

  template <class Node0, class Node1>
  CrossoverNode &AddCrossover(Node0 &predecessor0, Node1 &predecessor1,
                              bool isCopy0, bool isCopy1,
                              bool isSwapArgumentsAllowed) {
    // Should be preallocated to avoid refs invalidation
    assert(crossoverNodes.capacity() > crossoverNodes.size());
    crossoverNodes.push_back(
        CrossoverNode(*graphPtr, tbb::flow::concurrency::serial,
                      [&, isCopy0, isCopy1, isSwapArgumentsAllowed](
                          std::tuple<DNAPtr, DNAPtr> iSrcs) {
                        return CrossoverHelper(iSrcs, isCopy0, isCopy1,
                                               isSwapArgumentsAllowed);
                      }));
    crossoverJoinNodes.push_back(CrossoverJoinNode(*graphPtr));
    assert(crossoverJoinNodes.size() == crossoverNodes.size());
    tbb::flow::make_edge(predecessor0,
                         input_port<0>(crossoverJoinNodes.back()));
    tbb::flow::make_edge(predecessor1,
                         input_port<1>(crossoverJoinNodes.back()));
    tbb::flow::make_edge(crossoverJoinNodes.back(), crossoverNodes.back());
    return crossoverNodes.back();
  }

  void GenerateGraph(StateFlow const &stateFlow,
                     bool isSwapArgumentsAllowedInCrossover) {
    using InputNodeRef = std::reference_wrapper<InputNode>;
    using MutateNodeRef = std::reference_wrapper<MutateNode>;
    using CrossoverNodeRef = std::reference_wrapper<CrossoverNode>;
    auto constexpr InputNodeIndex = std::in_place_index<0>;
    auto constexpr MutateNodeIndex = std::in_place_index<1>;
    auto constexpr CrossoverNodeIndex = std::in_place_index<2>;
    using NodeRef = std::variant<InputNodeRef, MutateNodeRef, CrossoverNodeRef>;

    graphPtr = std::unique_ptr<tbb::flow::graph>(new tbb::flow::graph{});

    inputIndices.reserve(stateFlow.GetInitialStates().size());
    inputNodes.reserve(stateFlow.GetInitialStates().size());
    evaluateNodes.reserve(stateFlow.GetNEvaluates() -
                          stateFlow.GetNInitialEvaluates());
    mutateNodes.reserve(stateFlow.GetNMutates());
    crossoverJoinNodes.reserve(stateFlow.GetNCrossovers());
    crossoverNodes.reserve(stateFlow.GetNCrossovers());
    evaluateBuffer.reserve(stateFlow.GetNEvaluates() -
                           stateFlow.GetNInitialEvaluates());

    auto nodes = std::unordered_map<State, NodeRef>{};

#ifndef NDEBUG
    auto maxIterations = stateFlow.GetNStates() + 1;
    auto whileGuard = size_t{0};
    auto evaluateCount = size_t{0};
#endif // !NDEBUG

    auto IsResolved = [&](State state) { return nodes.contains(state); };
    auto IsResolvable = [&](State state) {
      if (stateFlow.IsInitialState(state))
        return true;
      auto op = stateFlow.GetAnyInOperation(state);
      auto parent = stateFlow.GetSource(op);
      return IsResolved(parent) &&
             (!stateFlow.IsCrossover(op) ||
              IsResolved(stateFlow.GetSource(stateFlow.GetCrossoverPair(op))));
    };
    auto IsCopyRequired = [&](Operation operation) {
      auto parent = stateFlow.GetSource(operation);
      if (stateFlow.IsEvaluate(parent))
        return true;
      auto const &[oB, oE] = stateFlow.GetOutOperations(parent);
      if (oE - oB == 1)
        return false;
      // TODO: This is too conservative.
      // There are cases when copy is not needed
      return true;
    };
    auto ResolveInitial = [&](State state) {
      assert(!IsResolved(state) && IsResolvable(state));
      assert(stateFlow.IsInitialState(state));
      auto &&node = AddInput(stateFlow.GetIndex(state));
      nodes.emplace(state, NodeRef(InputNodeIndex, std::ref(node)));
    };
    auto ResolveMutate = [&](State state) {
      assert(!IsResolved(state) && IsResolvable(state));
      auto op = stateFlow.GetAnyInOperation(state);
      assert(stateFlow.IsMutate(op));
      auto parent = stateFlow.GetSource(op);
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
      auto op0 = stateFlow.GetAnyInOperation(state);
      auto op1 = stateFlow.GetCrossoverPair(op0);
      auto parent0 = stateFlow.GetSource(op0);
      auto parent1 = stateFlow.GetSource(op1);
      auto &&parentNodeVar0 = nodes.at(parent0);
      auto &&parentNodeVar1 = nodes.at(parent1);
      std::visit(
          [&](auto parentNodeRef0) {
            std::visit(
                [&](auto parentNodeRef1) {
                  auto &&node =
                      AddCrossover(parentNodeRef0.get(), parentNodeRef1.get(),
                                   IsCopyRequired(op0), IsCopyRequired(op1),
                                   isSwapArgumentsAllowedInCrossover);
                  nodes.emplace(state,
                                NodeRef(CrossoverNodeIndex, std::ref(node)));
                },
                parentNodeVar1);
          },
          parentNodeVar0);
    };
    auto ResolveEvaluate = [&](State state) {
      assert(stateFlow.IsEvaluate(state));
#ifndef NDEBUG
      evaluateCount++;
#endif // !NDEBUG
      if (stateFlow.IsInitialState(state))
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
      if (stateFlow.IsInitialState(state))
        ResolveInitial(state);
      else if (stateFlow.IsMutate(stateFlow.GetAnyInOperation(state)))
        ResolveMutate(state);
      else
        ResolveCrossover(state);
      if (stateFlow.IsEvaluate(state))
        ResolveEvaluate(state);
    };

    while (modified) {
      modified = false;
      stateFlow.BreadthFirstSearch(stateFlow.GetInitialStates(),
                                   std::move(Act));
      assert(whileGuard++ < maxIterations);
    }
    assert(nodes.size() == stateFlow.GetNStates());
    assert(evaluateCount == stateFlow.GetNEvaluates());
  }

  std::unordered_set<size_t>
  EvaluateNonEvaluateInitialIndices(StateFlow const &stateFlow) const {
    auto availableIndices = std::unordered_set<size_t>{};
    for (auto i = size_t{0}; i < stateFlow.GetNEvaluates(); ++i)
      availableIndices.insert(i);
    for (auto state : stateFlow.GetInitialStates())
      if (stateFlow.IsEvaluate(state))
        availableIndices.erase(stateFlow.GetIndex(state));
    return availableIndices;
  }

  size_t GetPopulationSize() const noexcept {
    return evaluateNodes.size() + nonEvaluateInitialIndices.size();
  }

  void RunTaskFlow(Population const &population) {
    assert(population.size() == GetPopulationSize());
    evaluateBuffer.clear();
    for (auto i = size_t{0}; i < inputNodes.size(); ++i)
      inputNodes.at(i).try_put(&population.at(inputIndices.at(i)));
    (*graphPtr).wait_for_all();
  }

  void MoveResultsFromBuffer(Population &population, Grades &grades) noexcept {
    assert(evaluateBuffer.size() == nonEvaluateInitialIndices.size());
    assert(population.size() == GetPopulationSize());
    assert(grades.size() == GetPopulationSize());
    auto i = size_t{0};
    for (auto index : nonEvaluateInitialIndices) {
      population.at(index) = std::move(*evaluateBuffer.at(i).first);
      grades.at(index) = std::move(evaluateBuffer.at(i).second);
      ++i;
    }
  }
};

} // namespace Evolution
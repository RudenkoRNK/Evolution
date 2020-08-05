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
  using EvaluateFunction = typename GeneratorTraits::Function<EvaluateFG>;
  using MutateFunction = typename GeneratorTraits::Function<MutateFG>;
  using CrossoverFunction = typename GeneratorTraits::Function<CrossoverFG>;

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
  using IndexSet = StateFlow::IndexSet;

  using DNAPtr = std::shared_ptr<DNA>;

  // TODO: add support for lightweight policy
  using InputNode = tbb::flow::function_node<DNA const *, DNAPtr>;
  using EvaluateNode = tbb::flow::function_node<DNAPtr, int>;
  using MutateNode = tbb::flow::function_node<DNAPtr, DNAPtr>;
  using CrossoverNode =
      tbb::flow::function_node<std::tuple<DNAPtr, DNAPtr>, DNAPtr>;
  using CrossoverJoinNode = tbb::flow::join_node<std::tuple<DNAPtr, DNAPtr>>;
  using EvaluateThreadSpecificOrGlobalFunction =
      typename GeneratorTraits::ThreadSpecificOrGlobalFunction<EvaluateFG>;
  using MutateThreadSpecificOrGlobalFunction =
      typename GeneratorTraits::ThreadSpecificOrGlobalFunction<MutateFG>;
  using CrossoverThreadSpecificOrGlobalFunction =
      typename GeneratorTraits::ThreadSpecificOrGlobalFunction<CrossoverFG>;

  struct TBBFlow {
    StateFlow stateFlow;
    // tbb::flow::graph does not have copy or move assignment
    std::unique_ptr<tbb::flow::graph> graphPtr;
    std::vector<size_t> inputIndices;
    std::vector<InputNode> inputNodes;
    std::vector<EvaluateNode> evaluateNodes;
    std::vector<MutateNode> mutateNodes;
    std::vector<CrossoverJoinNode> crossoverJoinNodes;
    std::vector<CrossoverNode> crossoverNodes;
    std::unordered_set<size_t> nonEvaluateInitialIndices;
  };

public:
  using Population = std::vector<DNA>;
  using Grades = std::vector<double>;

  TaskFlow(EvaluateFG const &Evaluate, MutateFG const &Mutate,
           CrossoverFG const &Crossover, StateFlow const &stateFlow,
           bool isSwapArgumentsAllowedInCrossover)
      : tbbFlow(GenerateTBBFlow(StateFlow(stateFlow),
                                isSwapArgumentsAllowedInCrossover)),
        EvaluateThreadSpecificOrGlobal(
            GeneratorTraits::GetThreadSpecificOrGlobal(Evaluate)),
        MutateThreadSpecificOrGlobal(
            GeneratorTraits::GetThreadSpecificOrGlobal(Mutate)),
        CrossoverThreadSpecificOrGlobal(
            GeneratorTraits::GetThreadSpecificOrGlobal(Crossover)) {}

  void Run(Population &population, Grades &grades) {
    RunTaskFlow(population);
    MoveResultsFromBuffer(population, grades);
  }

  void SetStateFlow(StateFlow &&stateFlow,
                    bool isSwapArgumentsAllowedInCrossover) {
    auto tbbFlow_ = GenerateTBBFlow(std::move(stateFlow),
                                    isSwapArgumentsAllowedInCrossover);
    tbbFlow.inputNodes.clear();
    tbbFlow.evaluateNodes.clear();
    tbbFlow.mutateNodes.clear();
    tbbFlow.crossoverJoinNodes.clear();
    tbbFlow.crossoverNodes.clear();
    tbbFlow = std::move(tbbFlow_);
  }

private:
  TBBFlow tbbFlow;
  tbb::concurrent_unordered_map<DNAPtr, double, std::hash<DNAPtr>>
      evaluateBuffer;
  EvaluateThreadSpecificOrGlobalFunction EvaluateThreadSpecificOrGlobal;
  MutateThreadSpecificOrGlobalFunction MutateThreadSpecificOrGlobal;
  CrossoverThreadSpecificOrGlobalFunction CrossoverThreadSpecificOrGlobal;
#ifndef NDEBUG
  tbb::concurrent_unordered_map<DNAPtr, size_t, std::hash<DNAPtr>> workBuffer;
  tbb::concurrent_unordered_set<State> isComputedSet;
  tbb::concurrent_unordered_set<State> isEvaluatedSet;
#endif // !NDEBUG

  EvaluateFunction &GetEvaluateFunction() {
    return GeneratorTraits::GetFunction<EvaluateFG>(
        EvaluateThreadSpecificOrGlobal);
  }
  MutateFunction &GetMutateFunction() {
    return GeneratorTraits::GetFunction<MutateFG>(MutateThreadSpecificOrGlobal);
  }
  CrossoverFunction &GetCrossoverFunction() {
    return GeneratorTraits::GetFunction<CrossoverFG>(
        CrossoverThreadSpecificOrGlobal);
  }

  DNAPtr CopyHelper(DNA const &src) const { return DNAPtr(new DNA(src)); }
  DNAPtr MoveHelper(DNA &&src) const { return DNAPtr(new DNA(std::move(src))); }
  void EvaluateHelper(DNAPtr iSrc, bool isCopy, State state) {
    // Actually this functionality should not be used
    assert(!isCopy);
    CheckRace(iSrc);
    RegisterEvaluate(state);

    if (isCopy)
      iSrc = CopyHelper(*iSrc);
    auto &&Evaluate = GetEvaluateFunction();
    auto grade = Evaluate(*iSrc);
    evaluateBuffer.emplace(iSrc, grade);
  }
  DNAPtr MutateHelper(DNAPtr iSrc, bool isCopy, State state) {
    auto iSrcOrig = iSrc;
    if (!isCopy)
      Register(iSrcOrig);
    else
      CheckRace(iSrcOrig);
    RegisterCompute(state);

    auto isCopyHelp = isMutateInPlace && isCopy;
    auto constexpr isMove =
        ArgumentTraits<MutateFunction>::template isRValueReference<1> ||
        ArgumentTraits<MutateFunction>::template isValue<1>;
    auto isPtrMove = !isCopy || isCopyHelp;
    if (isCopyHelp)
      iSrc = CopyHelper(*iSrc);

    auto &&Mutate = GetMutateFunction();
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

    if (!isCopy)
      Unregister(iSrcOrig);
    return iSrc;
  }
  DNAPtr CrossoverHelper(std::tuple<DNAPtr, DNAPtr> iSrcs, bool isCopy0,
                         bool isCopy1, bool isSwapArgumentsAllowed,
                         State state) {
    auto iSrc0 = std::get<0>(iSrcs);
    auto iSrc1 = std::get<1>(iSrcs);
    auto iSrcOrig0 = iSrc0;
    auto iSrcOrig1 = iSrc1;
    if (!isCopy0)
      Register(iSrcOrig0);
    else
      CheckRace(iSrcOrig0);
    if (!isCopy1)
      Register(iSrcOrig1);
    else
      CheckRace(iSrcOrig1);
    RegisterCompute(state);

    auto isSwap = ((isCrossoverInPlaceFirst && !isCrossoverInPlaceSecond &&
                    isCopy0 && !isCopy1) ||
                   (isCrossoverInPlaceSecond && !isCrossoverInPlaceFirst &&
                    isCopy1 && !isCopy0)) &&
                  isSwapArgumentsAllowed;
    if (isSwap) {
      iSrc0.swap(iSrc1);
      std::swap(isCopy0, isCopy1);
    }
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

    auto &&Crossover = GetCrossoverFunction();
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

    if (!isCopy0)
      Unregister(iSrcOrig0);
    if (!isCopy1)
      Unregister(iSrcOrig1);
    return iSrc0;
  }

  InputNode &AddInput(TBBFlow &tbbFlow, size_t index, State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.inputNodes.capacity() > tbbFlow.inputNodes.size());
    tbbFlow.inputIndices.push_back(index);
    // Could be implemented with MoveHelper in some cases, but this approach is
    // exception-safe for the Population
    tbbFlow.inputNodes.push_back(InputNode(*tbbFlow.graphPtr,
                                           tbb::flow::concurrency::serial,
                                           [&, state](DNA const *dna) {
                                             RegisterCompute(state);
                                             return CopyHelper(*dna);
                                           }));
    assert(tbbFlow.inputIndices.size() == tbbFlow.inputNodes.size());
    return tbbFlow.inputNodes.back();
  }
  template <class Node>
  EvaluateNode &AddEvaluate(TBBFlow &tbbFlow, Node &predecessor, bool isCopy,
                            State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.evaluateNodes.capacity() > tbbFlow.evaluateNodes.size());
    tbbFlow.evaluateNodes.push_back(
        EvaluateNode(*tbbFlow.graphPtr, tbb::flow::concurrency::serial,
                     [&, isCopy, state](DNAPtr iSrc) {
                       EvaluateHelper(iSrc, isCopy, state);
                       return 0;
                     }));
    tbb::flow::make_edge(predecessor, tbbFlow.evaluateNodes.back());
    return tbbFlow.evaluateNodes.back();
  }
  template <class Node>
  MutateNode &AddMutate(TBBFlow &tbbFlow, Node &predecessor, bool isCopy,
                        State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.mutateNodes.capacity() > tbbFlow.mutateNodes.size());
    tbbFlow.mutateNodes.push_back(
        MutateNode(*tbbFlow.graphPtr, tbb::flow::concurrency::serial,
                   [&, isCopy, state](DNAPtr iSrc) {
                     return MutateHelper(iSrc, isCopy, state);
                   }));
    tbb::flow::make_edge(predecessor, tbbFlow.mutateNodes.back());
    return tbbFlow.mutateNodes.back();
  }
  template <class Node0, class Node1>
  CrossoverNode &AddCrossover(TBBFlow &tbbFlow, Node0 &predecessor0,
                              Node1 &predecessor1, bool isCopy0, bool isCopy1,
                              bool isSwapArgumentsAllowed, State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.crossoverNodes.capacity() > tbbFlow.crossoverNodes.size());
    tbbFlow.crossoverNodes.push_back(
        CrossoverNode(*tbbFlow.graphPtr, tbb::flow::concurrency::serial,
                      [&, isCopy0, isCopy1, isSwapArgumentsAllowed,
                       state](std::tuple<DNAPtr, DNAPtr> iSrcs) {
                        return CrossoverHelper(iSrcs, isCopy0, isCopy1,
                                               isSwapArgumentsAllowed, state);
                      }));
    tbbFlow.crossoverJoinNodes.push_back(CrossoverJoinNode(*tbbFlow.graphPtr));
    assert(tbbFlow.crossoverJoinNodes.size() == tbbFlow.crossoverNodes.size());
    tbb::flow::make_edge(predecessor0,
                         input_port<0>(tbbFlow.crossoverJoinNodes.back()));
    tbb::flow::make_edge(predecessor1,
                         input_port<1>(tbbFlow.crossoverJoinNodes.back()));
    tbb::flow::make_edge(tbbFlow.crossoverJoinNodes.back(),
                         tbbFlow.crossoverNodes.back());
    return tbbFlow.crossoverNodes.back();
  }

  TBBFlow GenerateTBBFlow(StateFlow &&stateFlow,
                          bool isSwapArgumentsAllowedInCrossover) {
    assert(stateFlow.Verify());
    auto tbbFlow = GenerateGraph(stateFlow, isSwapArgumentsAllowedInCrossover);
    tbbFlow.nonEvaluateInitialIndices =
        EvaluateNonEvaluateInitialIndices(stateFlow);
    tbbFlow.stateFlow = std::move(stateFlow);
    return tbbFlow;
  }
  TBBFlow GenerateGraph(StateFlow const &stateFlow,
                        bool isSwapArgumentsAllowedInCrossover) {
    using InputNodeRef = std::reference_wrapper<InputNode>;
    using MutateNodeRef = std::reference_wrapper<MutateNode>;
    using CrossoverNodeRef = std::reference_wrapper<CrossoverNode>;
    auto constexpr InputNodeIndex = std::in_place_index<0>;
    auto constexpr MutateNodeIndex = std::in_place_index<1>;
    auto constexpr CrossoverNodeIndex = std::in_place_index<2>;
    using NodeRef = std::variant<InputNodeRef, MutateNodeRef, CrossoverNodeRef>;

    auto tbbFlow = TBBFlow{};

    tbbFlow.graphPtr =
        std::unique_ptr<tbb::flow::graph>(new tbb::flow::graph{});

    tbbFlow.inputIndices.reserve(stateFlow.GetInitialStates().size());
    tbbFlow.inputNodes.reserve(stateFlow.GetInitialStates().size());
    tbbFlow.evaluateNodes.reserve(stateFlow.GetNEvaluates() -
                                  stateFlow.GetNInitialEvaluates());
    tbbFlow.mutateNodes.reserve(stateFlow.GetNMutates());
    tbbFlow.crossoverJoinNodes.reserve(stateFlow.GetNCrossovers());
    tbbFlow.crossoverNodes.reserve(stateFlow.GetNCrossovers());

    auto nodes = std::unordered_map<State, NodeRef>{};

#ifndef NDEBUG
    auto resolvedEvaluates = std::unordered_set<State>{};
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
      auto &&node = AddInput(tbbFlow, stateFlow.GetIndex(state), state);
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
            auto &&node = AddMutate(tbbFlow, parentNodeRef.get(),
                                    IsCopyRequired(op), state);
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
                  auto &&node = AddCrossover(
                      tbbFlow, parentNodeRef0.get(), parentNodeRef1.get(),
                      IsCopyRequired(op0), IsCopyRequired(op1),
                      isSwapArgumentsAllowedInCrossover, state);
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
      assert(!resolvedEvaluates.contains(state));
      resolvedEvaluates.insert(state);
      evaluateCount++;
#endif // !NDEBUG
      if (stateFlow.IsInitialState(state))
        return;
      std::visit(
          [&](auto nodeRef) {
            AddEvaluate(tbbFlow, nodeRef.get(), /*isCopy*/ false, state);
          },
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

    return tbbFlow;
  }

  static std::unordered_set<size_t>
  EvaluateNonEvaluateInitialIndices(StateFlow const &stateFlow) {
    auto availableIndices = std::unordered_set<size_t>{};
    for (auto i = size_t{0}; i < stateFlow.GetNEvaluates(); ++i)
      availableIndices.insert(i);
    for (auto state : stateFlow.GetInitialStates())
      if (stateFlow.IsEvaluate(state))
        availableIndices.erase(stateFlow.GetIndex(state));
    return availableIndices;
  }

  size_t GetPopulationSize() const noexcept {
    return tbbFlow.stateFlow.GetNEvaluates();
  }

  void RunTaskFlow(Population const &population) {
    assert(population.size() == GetPopulationSize());
#ifndef NDEBUG
    workBuffer.clear();
    isComputedSet.clear();
    isEvaluatedSet.clear();
#endif // !NDEBUG
    evaluateBuffer.clear();

    for (auto i = size_t{0}; i < tbbFlow.inputNodes.size(); ++i)
      tbbFlow.inputNodes.at(i).try_put(
          &population.at(tbbFlow.inputIndices.at(i)));
    (*tbbFlow.graphPtr).wait_for_all();
  }

  void MoveResultsFromBuffer(Population &population, Grades &grades) noexcept {
    assert(evaluateBuffer.size() == tbbFlow.nonEvaluateInitialIndices.size());
    assert(population.size() == GetPopulationSize());
    assert(grades.size() == GetPopulationSize());

    auto ebIt = evaluateBuffer.begin();
    auto idxIt = tbbFlow.nonEvaluateInitialIndices.begin();
    for (auto i = size_t{0}, e = evaluateBuffer.size(); i != e; ++i) {
      auto index = *idxIt;
      auto [dnaPtr, grade] = *ebIt;
      population.at(index) = std::move(*dnaPtr);
      grades.at(index) = grade;
      ++ebIt;
      ++idxIt;
    }
  }

  // Debug functions
  void Register(DNAPtr dnaPtr) {
#ifndef NDEBUG
    CheckRace(dnaPtr);
    if (workBuffer.count(dnaPtr) > 0)
      ++workBuffer.at(dnaPtr);
    else
      workBuffer.emplace(dnaPtr, 1);
    assert(workBuffer.at(dnaPtr) == 1);
#endif // !NDEBUG
  }
  void Unregister(DNAPtr dnaPtr) {
#ifndef NDEBUG
    --workBuffer.at(dnaPtr);
    assert(workBuffer.at(dnaPtr) == 0);
#endif // !NDEBUG
  }
  void CheckRace(DNAPtr dnaPtr) {
#ifndef NDEBUG
    if (evaluateBuffer.count(dnaPtr) > 0) {
      std::cerr << "Found DNA which is going to be modified, "
                   "but it is already reserved for final evaluate"
                << std::endl;
      std::cerr << "Evaluate buffer size: " << evaluateBuffer.size()
                << std::endl;
      std::cerr << "Element address: " << dnaPtr << std::endl;
      DumpStateFlow();
      assert(false);
    }
    if (workBuffer.count(dnaPtr) > 0 && workBuffer.at(dnaPtr) > 0) {
      std::cerr << "Found DNA which is going to be modified by one operation, "
                   "but it is currently in progress in another operation"
                << std::endl;
      std::cerr << "Work buffer size: " << workBuffer.size() << std::endl;
      std::cerr << "Element address: " << dnaPtr << std::endl;
      DumpStateFlow();
      assert(false);
    }
#endif // !NDEBUG
  }
  void RegisterCompute(State state) {
#ifndef NDEBUG
    assert(isEvaluatedSet.count(state) == 0 &&
           "Somehow evaluate happened before compute...");
    if (isComputedSet.count(state) != 0) {
      std::cout << "Found an attempt to compute same node twice!";
      DumpStateFlow();
      assert(false);
    }
    isComputedSet.insert(state);
#endif // !NDEBUG
  }
  void RegisterEvaluate(State state) {
#ifndef NDEBUG
    assert(isComputedSet.count(state) != 0 &&
           "Somehow evaluate is happening before compute...");
    if (isEvaluatedSet.count(state) != 0) {
      std::cout << "Found an attempt to evaluate same node twice!";
      DumpStateFlow();
      assert(false);
    }
    isEvaluatedSet.insert(state);
#endif // !NDEBUG
  }
  void DumpStateFlow() {
#ifndef NDEBUG
    auto dump = std::ofstream("dump.dot");
    tbbFlow.stateFlow.Print(dump, isComputedSet, isEvaluatedSet);
    dump.close();
#endif // !NDEBUG
  }
};

} // namespace Evolution

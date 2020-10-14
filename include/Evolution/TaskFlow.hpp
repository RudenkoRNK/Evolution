#pragma once
#define NOMINMAX
#include "Evolution/GeneratorTraits.hpp"
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlowDebugger.hpp"
#include "Utility/TypeTraits.hpp"
#include <algorithm>
#include <boost/container/small_vector.hpp>
#include <cassert>
#include <execution>
#include <functional>
#include <map>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/flow_graph.h>
#include <tbb/parallel_for_each.h>
#include <utility>
#include <variant>
#include <vector>

namespace Evolution {
template <class EvaluateFG, class MutateFG, class CrossoverFG>
class TaskFlow final {

private:
  using TypeTraits = Utility::TypeTraits;
  using GeneratorTraits = Utility::GeneratorTraits;
  template <class Callable>
  using ArgumentTraits = Utility::ArgumentTraits<Callable>;

  using EvaluateFunction = typename GeneratorTraits::Function<EvaluateFG>;
  using MutateFunction = typename GeneratorTraits::Function<MutateFG>;
  using CrossoverFunction = typename GeneratorTraits::Function<CrossoverFG>;

  template <class T>
  bool constexpr static isVariant =
      TypeTraits::isInstanceOf<std::remove_reference_t<T>, std::variant>;

public:
  using DNA = std::remove_cvref_t<
      typename ArgumentTraits<EvaluateFunction>::template Type<1>>;

private:
  // DNA should be a modifiable self-sufficient type
  static_assert(!std::is_const_v<DNA>);
  static_assert(!std::is_reference_v<DNA>);
  // static_assert(noexcept(DNA(std::declval<DNA>())));

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

  // Mutate and crossover must not return constant values
  static_assert(!ArgumentTraits<MutateFunction>::template isConst<0>);
  static_assert(!ArgumentTraits<CrossoverFunction>::template isConst<0>);

  auto constexpr static isMutateInPlace =
      (!ArgumentTraits<MutateFunction>::template isConst<1>);
  auto constexpr static isMutateMovable =
      ArgumentTraits<MutateFunction>::template isRValueReference<1> ||
      ArgumentTraits<MutateFunction>::template isValue<1>;
  auto constexpr static isCrossoverInPlaceFirst =
      !ArgumentTraits<CrossoverFunction>::template isConst<1>;
  auto constexpr static isCrossoverInPlaceSecond =
      !ArgumentTraits<CrossoverFunction>::template isConst<2>;
  auto constexpr static isCrossoverMovableFirst =
      ArgumentTraits<CrossoverFunction>::template isRValueReference<1> ||
      ArgumentTraits<CrossoverFunction>::template isValue<1>;
  auto constexpr static isCrossoverMovableSecond =
      ArgumentTraits<CrossoverFunction>::template isRValueReference<2> ||
      ArgumentTraits<CrossoverFunction>::template isValue<2>;

  using State = StateFlow::State;
  using StateSet = StateFlow::StateSet;
  using StateVector = StateFlow::StateVector;
  using Operation = StateFlow::Operation;
  using OperationSet = StateFlow::OperationSet;
  using IndexSet = StateFlow::IndexSet;

  using DNAPtr = std::shared_ptr<DNA>;

  template <class Input, class Output = tbb::flow::continue_msg,
            class Policy = tbb::flow::queueing,
            class Allocator = tbb::flow::interface11::null_type>
  using FunctionNode =
      tbb::flow::function_node<Input, Output, Policy, Allocator>;

  auto constexpr static DefaultPolicyIndex = std::in_place_index<0>;
  auto constexpr static LightweightPolicyIndex = std::in_place_index<1>;
  using InputNode = FunctionNode<DNA *, DNAPtr>;
  using EvaluateNode =
      std::variant<FunctionNode<DNAPtr, int>,
                   FunctionNode<DNAPtr, int, tbb::flow::lightweight>>;
  using MutateNode =
      std::variant<FunctionNode<DNAPtr, DNAPtr>,
                   FunctionNode<DNAPtr, DNAPtr, tbb::flow::lightweight>>;
  using CrossoverNode = std::variant<
      FunctionNode<std::tuple<DNAPtr, DNAPtr>, DNAPtr>,
      FunctionNode<std::tuple<DNAPtr, DNAPtr>, DNAPtr, tbb::flow::lightweight>>;
  using CrossoverJoinNode = tbb::flow::join_node<std::tuple<DNAPtr, DNAPtr>>;
  using EvaluateTFG =
      typename GeneratorTraits::ThreadGeneratorOrFunction<EvaluateFG>;
  using MutateTFG =
      typename GeneratorTraits::ThreadGeneratorOrFunction<MutateFG>;
  using CrossoverTFG =
      typename GeneratorTraits::ThreadGeneratorOrFunction<CrossoverFG>;

  using NodeType = typename TaskFlowDebugger<DNA>::NodeType;

  struct TBBFlow {
    // tbb::flow::graph does not have copy or move assignment
    std::unique_ptr<tbb::flow::graph> graphPtr;
    std::vector<size_t> inputIndices;
    std::vector<InputNode> inputNodes;
    std::vector<EvaluateNode> evaluateNodes;
    std::vector<MutateNode> mutateNodes;
    std::vector<CrossoverJoinNode> crossoverJoinNodes;
    std::vector<CrossoverNode> crossoverNodes;
    std::unordered_set<size_t> nonEvaluateInitialIndices;
    bool isEvaluateLightweight;
    bool isMutateLightweight;
    bool isCrossoverLightweight;
  };

public:
  using Population = std::vector<DNA>;
  using Grades = std::vector<double>;

  TaskFlow(EvaluateFG const &Evaluate, MutateFG const &Mutate,
           CrossoverFG const &Crossover, StateFlow const &stateFlow,
           bool isEvaluateLightweight = false, bool isMutateLightweight = false,
           bool isCrossoverLightweight = false)
      : tbbFlow(GenerateTBBFlow(stateFlow, isEvaluateLightweight,
                                isMutateLightweight, isCrossoverLightweight)),
        stateFlow(stateFlow), debugger(stateFlow),
        evaluateTFG(GeneratorTraits::GetThreadGeneratorOrFunction(Evaluate)),
        mutateTFG(GeneratorTraits::GetThreadGeneratorOrFunction(Mutate)),
        crossoverTFG(GeneratorTraits::GetThreadGeneratorOrFunction(Crossover)) {
  }

  void Run(Population &population, Grades &grades) {
    RunTaskFlow(population);
    MoveResultsFromBuffer(population, grades);
  }

  Grades EvaluatePopulation(Population const &population) {
    auto indices = Utility::GetIndices(population.size());
    auto grades = Grades(population.size());
    tbb::parallel_for_each(indices.begin(), indices.end(), [&](size_t index) {
      auto &&Evaluate = GetEvaluateFunction();
      grades.at(index) = Evaluate(population.at(index));
    });
    return grades;
  }

  void SetStateFlow(StateFlow &&stateFlow) {
    debugger.SetStateFlow(stateFlow);
    auto tbbFlow_ = GenerateTBBFlow(stateFlow, tbbFlow.isEvaluateLightweight,
                                    tbbFlow.isMutateLightweight,
                                    tbbFlow.isCrossoverLightweight);
    tbbFlow.inputNodes.clear();
    tbbFlow.evaluateNodes.clear();
    tbbFlow.mutateNodes.clear();
    tbbFlow.crossoverJoinNodes.clear();
    tbbFlow.crossoverNodes.clear();
    tbbFlow = std::move(tbbFlow_);
    this->stateFlow = std::move(stateFlow);
  }

  StateFlow const &GetStateFlow() const noexcept { return stateFlow; }

  bool static IsEvaluateLightweight(EvaluateFG const &Evaluate,
                                    DNA const &dna) {
    return IsFGLightweight(Evaluate, dna);
  }
  bool static IsMutateLightweight(MutateFG const &Mutate, DNA const &dna) {
    if constexpr (!isMutateInPlace)
      return IsFGLightweight(Mutate, dna);
    auto dna_ = DNA(dna);
    if constexpr (isMutateMovable)
      return IsFGLightweight(Mutate, std::move(dna_));
    return IsFGLightweight(Mutate, dna_);
  }
  bool static IsCrossoverLightweight(CrossoverFG const &Crossover,
                                     DNA const &dna0, DNA const &dna1) {
    if constexpr (!isCrossoverInPlaceFirst)
      return IsCrossoverLightweight_(Crossover, dna0, dna1);
    auto dna0_ = DNA(dna0);
    if constexpr (!isCrossoverMovableFirst)
      return IsCrossoverLightweight_(Crossover, dna0_, dna1);
    return IsCrossoverLightweight_(Crossover, std::move(dna0_), dna1);
  }

private:
  TBBFlow tbbFlow;
  StateFlow stateFlow;
  TaskFlowDebugger<DNA> debugger;
  tbb::concurrent_unordered_map<DNAPtr, double, std::hash<DNAPtr>>
      evaluateBuffer;
  EvaluateTFG evaluateTFG;
  MutateTFG mutateTFG;
  CrossoverTFG crossoverTFG;

  EvaluateFunction &GetEvaluateFunction() {
    return GeneratorTraits::GetFunction<EvaluateFG>(evaluateTFG);
  }
  MutateFunction &GetMutateFunction() {
    return GeneratorTraits::GetFunction<MutateFG>(mutateTFG);
  }
  CrossoverFunction &GetCrossoverFunction() {
    return GeneratorTraits::GetFunction<CrossoverFG>(crossoverTFG);
  }

  DNAPtr CopyHelper(DNA const &src) const { return DNAPtr(new DNA(src)); }
  DNAPtr MoveHelper(DNA &&src) const { return DNAPtr(new DNA(std::move(src))); }
  DNAPtr InputHelper(DNA *iSrc, bool isCopy, State state) {
    debugger.Register(iSrc, state, isCopy, NodeType::Input);
    auto ret = DNAPtr{};
    if (isCopy)
      ret = CopyHelper(*iSrc);
    else
      ret = MoveHelper(std::move(*iSrc));
    debugger.Unregister(iSrc, state);
    debugger.RegisterOutput(ret.get(), state);
    return ret;
  }
  void EvaluateHelper(DNAPtr iSrc, bool isCopy, State state) {
    auto iSrcOrig = iSrc;
    debugger.Register(iSrcOrig.get(), state, isCopy, NodeType::Evaluate);

    // Actually this functionality should not be used
    assert(!isCopy);

    if (isCopy)
      iSrc = CopyHelper(*iSrc);
    auto &&Evaluate = GetEvaluateFunction();
    auto grade = Evaluate(*iSrc);
    evaluateBuffer.emplace(iSrc, grade);

    debugger.Unregister(iSrcOrig.get(), state);
  }
  DNAPtr MutateHelper(DNAPtr iSrc, bool isCopy, State state) {
    auto iSrcOrig = iSrc;
    debugger.Register(iSrcOrig.get(), state, isCopy, NodeType::Mutate);

    auto isCopyHelp = isMutateInPlace && isCopy;
    auto isPtrMove = !isCopy || isCopyHelp;
    if (isCopyHelp)
      iSrc = CopyHelper(*iSrc);

    auto &&Mutate = GetMutateFunction();
    auto MutatePtr = [&](DNAPtr iSrc) {
      if constexpr (isMutateMovable) {
        assert(!isCopy || iSrcOrig != iSrc);
        return Mutate(std::move(*iSrc));
      } else {
        assert(!isCopy || !isMutateInPlace);
        return Mutate(*iSrc);
      }
    };

    auto ret = DNAPtr{};
    auto &&dst = MutatePtr(iSrc);
    if (!isPtrMove)
      ret = MoveHelper(std::move(dst));
    else {
      *iSrc = std::move(dst);
      ret = iSrc;
    }

    debugger.Unregister(iSrcOrig.get(), state);
    debugger.RegisterOutput(ret.get(), state);
    return ret;
  }
  DNAPtr CrossoverHelper(std::tuple<DNAPtr, DNAPtr> iSrcs, bool isCopy0,
                         bool isCopy1, bool isSwapArgumentsAllowed,
                         State state) {
    auto [iSrc0, iSrc1] = iSrcs;
    auto [iSrcOrig0, iSrcOrig1] = iSrcs;
    debugger.Register(iSrcOrig0.get(), state, isCopy0, NodeType::Crossover);
    debugger.Register(iSrcOrig1.get(), state, isCopy1, NodeType::Crossover);

    auto isSwap = ((isCrossoverInPlaceFirst && !isCrossoverInPlaceSecond &&
                    isCopy0 && !isCopy1) ||
                   (isCrossoverInPlaceSecond && !isCrossoverInPlaceFirst &&
                    isCopy1 && !isCopy0)) &&
                  isSwapArgumentsAllowed;
    if (isSwap) {
      std::swap(iSrc0, iSrc1);
      std::swap(iSrcOrig0, iSrcOrig1);
      std::swap(isCopy0, isCopy1);
    }

    auto isCopyHelp0 = isCrossoverInPlaceFirst && isCopy0;
    auto isCopyHelp1 = isCrossoverInPlaceSecond && isCopy1;
    auto isPtrMove0 = !isCopy0 || isCopyHelp0;
    auto isPtrMove1 = !isCopy1 || isCopyHelp1;

    if (isCopyHelp0)
      iSrc0 = CopyHelper(*iSrc0);
    if (isCopyHelp1)
      iSrc1 = CopyHelper(*iSrc1);

    auto &&Crossover = GetCrossoverFunction();
    auto CrossoverPtr = [&](DNAPtr iSrc0, DNAPtr iSrc1) {
      if constexpr (isCrossoverMovableFirst) {
        assert(!isCopy0 || iSrcOrig0 != iSrc0);
        if constexpr (isCrossoverMovableSecond) {
          assert(!isCopy1 || iSrcOrig1 != iSrc1);
          return Crossover(std::move(*iSrc0), std::move(*iSrc1));
        } else {
          assert(!isCopy1 || !isCrossoverInPlaceSecond);
          return Crossover(std::move(*iSrc0), *iSrc1);
        }
      } else {
        assert(!isCopy0 || !isCrossoverInPlaceFirst);
        if constexpr (isCrossoverMovableSecond) {
          assert(!isCopy1 || iSrcOrig1 != iSrc1);
          return Crossover(*iSrc0, std::move(*iSrc1));
        } else {
          assert(!isCopy1 || !isCrossoverInPlaceSecond);
          return Crossover(*iSrc0, *iSrc1);
        }
      }
    };

    auto ret = DNAPtr{};
    auto &&dst = CrossoverPtr(iSrc0, iSrc1);
    if (!isPtrMove0 && !isPtrMove1)
      ret = MoveHelper(std::move(dst));
    else if (isPtrMove0 && !isPtrMove1) {
      *iSrc0 = std::move(dst);
      ret = iSrc0;
    } else if (isPtrMove1 && !isPtrMove0) {
      *iSrc1 = std::move(dst);
      ret = iSrc1;
    } else {
      if constexpr (isCrossoverInPlaceFirst && !isCrossoverInPlaceSecond) {
        *iSrc0 = std::move(dst);
        ret = iSrc0;
      } else if constexpr (isCrossoverInPlaceSecond &&
                           !isCrossoverInPlaceFirst) {
        *iSrc1 = std::move(dst);
        ret = iSrc1;
      } else {
        *iSrc0 = std::move(dst);
        ret = iSrc0;
      }
    }

    debugger.Unregister(iSrcOrig0.get(), state);
    debugger.Unregister(iSrcOrig1.get(), state);
    debugger.RegisterOutput(ret.get(), state);
    return ret;
  }

  template <class Node, class... Args>
  Node MakeNode(bool isLightweight, Args &&... args) {
    if (isLightweight)
      return Node(LightweightPolicyIndex, std::forward<Args>(args)...);
    else
      return Node(DefaultPolicyIndex, std::forward<Args>(args)...);
  }

  InputNode &AddInput(TBBFlow &tbbFlow, size_t index, bool isCopy,
                      State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.inputNodes.capacity() > tbbFlow.inputNodes.size());
    tbbFlow.inputIndices.push_back(index);
    tbbFlow.inputNodes.push_back(
        InputNode(*tbbFlow.graphPtr, tbb::flow::concurrency::serial,
                  [&, isCopy, state](DNA *dna) {
                    return InputHelper(dna, isCopy, state);
                  }));
    assert(tbbFlow.inputIndices.size() == tbbFlow.inputNodes.size());
    return tbbFlow.inputNodes.back();
  }
  template <class Node>
  EvaluateNode &AddEvaluate(TBBFlow &tbbFlow, Node &predecessor, bool isCopy,
                            State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.evaluateNodes.capacity() > tbbFlow.evaluateNodes.size());
    tbbFlow.evaluateNodes.push_back(MakeNode<EvaluateNode>(
        tbbFlow.isEvaluateLightweight, *tbbFlow.graphPtr,
        tbb::flow::concurrency::serial, [&, isCopy, state](DNAPtr iSrc) {
          EvaluateHelper(iSrc, isCopy, state);
          return 0;
        }));
    if constexpr (isVariant<Node>)
      std::visit(
          [&](auto &predNode, auto &succNode) {
            tbb::flow::make_edge(predNode, succNode);
          },
          predecessor, tbbFlow.evaluateNodes.back());
    else
      std::visit(
          [&](auto &succNode) { tbb::flow::make_edge(predecessor, succNode); },
          tbbFlow.evaluateNodes.back());
    return tbbFlow.evaluateNodes.back();
  }
  template <class Node>
  MutateNode &AddMutate(TBBFlow &tbbFlow, Node &predecessor, bool isCopy,
                        State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.mutateNodes.capacity() > tbbFlow.mutateNodes.size());
    tbbFlow.mutateNodes.push_back(MakeNode<MutateNode>(
        tbbFlow.isMutateLightweight, *tbbFlow.graphPtr,
        tbb::flow::concurrency::serial, [&, isCopy, state](DNAPtr iSrc) {
          return MutateHelper(iSrc, isCopy, state);
        }));
    if constexpr (isVariant<Node>)
      std::visit(
          [&](auto &predNode, auto &succNode) {
            tbb::flow::make_edge(predNode, succNode);
          },
          predecessor, tbbFlow.mutateNodes.back());
    else
      std::visit(
          [&](auto &succNode) { tbb::flow::make_edge(predecessor, succNode); },
          tbbFlow.mutateNodes.back());
    return tbbFlow.mutateNodes.back();
  }
  template <class Node0, class Node1>
  CrossoverNode &AddCrossover(TBBFlow &tbbFlow, Node0 &predecessor0,
                              Node1 &predecessor1, bool isCopy0, bool isCopy1,
                              bool isSwapArgumentsAllowed, State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.crossoverNodes.capacity() > tbbFlow.crossoverNodes.size());
    tbbFlow.crossoverNodes.push_back(MakeNode<CrossoverNode>(
        tbbFlow.isCrossoverLightweight, *tbbFlow.graphPtr,
        tbb::flow::concurrency::serial,
        [&, isCopy0, isCopy1, isSwapArgumentsAllowed,
         state](std::tuple<DNAPtr, DNAPtr> iSrcs) {
          return CrossoverHelper(iSrcs, isCopy0, isCopy1,
                                 isSwapArgumentsAllowed, state);
        }));
    tbbFlow.crossoverJoinNodes.push_back(CrossoverJoinNode(*tbbFlow.graphPtr));
    assert(tbbFlow.crossoverJoinNodes.size() == tbbFlow.crossoverNodes.size());
    if constexpr (isVariant<Node0>)
      std::visit(
          [&](auto &predNode) {
            tbb::flow::make_edge(
                predNode, input_port<0>(tbbFlow.crossoverJoinNodes.back()));
          },
          predecessor0);
    else
      tbb::flow::make_edge(predecessor0,
                           input_port<0>(tbbFlow.crossoverJoinNodes.back()));
    if constexpr (isVariant<Node1>)
      std::visit(
          [&](auto &predNode) {
            tbb::flow::make_edge(
                predNode, input_port<1>(tbbFlow.crossoverJoinNodes.back()));
          },
          predecessor1);
    else
      tbb::flow::make_edge(predecessor1,
                           input_port<1>(tbbFlow.crossoverJoinNodes.back()));
    std::visit(
        [&](auto &succNode) {
          tbb::flow::make_edge(tbbFlow.crossoverJoinNodes.back(), succNode);
        },
        tbbFlow.crossoverNodes.back());
    return tbbFlow.crossoverNodes.back();
  }

  TBBFlow GenerateTBBFlow(StateFlow const &stateFlow,
                          bool isEvaluateLightweight, bool isMutateLightweight,
                          bool isCrossoverLightweight) {
    assert(!stateFlow.IsNotReady());
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

    tbbFlow.isEvaluateLightweight = isEvaluateLightweight;
    tbbFlow.isMutateLightweight = isMutateLightweight;
    tbbFlow.isCrossoverLightweight = isCrossoverLightweight;

    tbbFlow.inputIndices.reserve(stateFlow.GetInitialStates().size());
    tbbFlow.inputNodes.reserve(stateFlow.GetInitialStates().size());
    tbbFlow.evaluateNodes.reserve(stateFlow.GetNEvaluates() -
                                  stateFlow.GetNInitialEvaluates());
    tbbFlow.mutateNodes.reserve(stateFlow.GetNMutates());
    tbbFlow.crossoverJoinNodes.reserve(stateFlow.GetNCrossovers());
    tbbFlow.crossoverNodes.reserve(stateFlow.GetNCrossovers());

    auto nodes = std::unordered_map<State, NodeRef>{};
    auto loneInitials = std::unordered_set<State>{};

#ifndef NDEBUG
    auto resolvedEvaluates = std::unordered_set<State>{};
    auto maxIterations = stateFlow.GetNStates() + 1;
    auto whileGuard = size_t{0};
    auto evaluateCount = size_t{0};
#endif // !NDEBUG

    auto IsResolved = [&](State state) {
      return nodes.contains(state) || loneInitials.contains(state);
    };
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
      if (stateFlow.IsEvaluate(parent) && !stateFlow.IsInitialState(parent))
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
      if (stateFlow.IsLeaf(state)) {
        assert(stateFlow.IsEvaluate(state));
        loneInitials.insert(state);
        return;
      }
      auto isCopy = stateFlow.IsEvaluate(state);
      auto &&node = AddInput(tbbFlow, stateFlow.GetIndex(state), isCopy, state);
      nodes.emplace(state, NodeRef(InputNodeIndex, std::ref(node)));
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
          [&](auto parentNodeRef0, auto parentNodeRef1) {
            auto &&node = AddCrossover(
                tbbFlow, parentNodeRef0.get(), parentNodeRef1.get(),
                IsCopyRequired(op0), IsCopyRequired(op1),
                stateFlow.IsSwapArgumentsAllowedInCrossover(), state);
            nodes.emplace(state, NodeRef(CrossoverNodeIndex, std::ref(node)));
          },
          parentNodeVar0, parentNodeVar1);
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
    assert(nodes.size() + loneInitials.size() == stateFlow.GetNStates());
    assert(evaluateCount == stateFlow.GetNEvaluates());

    tbbFlow.nonEvaluateInitialIndices =
        EvaluateNonEvaluateInitialIndices(stateFlow);

    return tbbFlow;
  }

  std::unordered_set<size_t> static EvaluateNonEvaluateInitialIndices(
      StateFlow const &stateFlow) {
    auto availableIndices = std::unordered_set<size_t>{};
    for (auto i = size_t{0}; i < stateFlow.GetNEvaluates(); ++i)
      availableIndices.insert(i);
    for (auto state : stateFlow.GetInitialStates())
      if (stateFlow.IsEvaluate(state))
        availableIndices.erase(stateFlow.GetIndex(state));
    return availableIndices;
  }

  size_t GetPopulationSize() const noexcept {
    return stateFlow.GetNEvaluates();
  }

  void RunTaskFlow(Population &population) {
    assert(population.size() == GetPopulationSize());
    debugger.Clear();
    evaluateBuffer.clear();

    for (auto i = size_t{0}; i < tbbFlow.inputNodes.size(); ++i)
      tbbFlow.inputNodes.at(i).try_put(
          &population.at(tbbFlow.inputIndices.at(i)));
    tbbFlow.graphPtr->wait_for_all();
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

  template <class DNAFirst>
  bool static IsCrossoverLightweight_(CrossoverFG const &Crossover,
                                      DNAFirst &&dna0, DNA const &dna1) {
    if constexpr (!isCrossoverInPlaceSecond)
      return IsFGLightweight(Crossover, std::forward<DNAFirst>(dna0), dna1);
    auto dna1_ = DNA(dna1);
    if constexpr (isCrossoverMovableSecond)
      return IsFGLightweight(Crossover, std::forward<DNAFirst>(dna0),
                             std::move(dna1_));
    return IsFGLightweight(Crossover, std::forward<DNAFirst>(dna0), dna1_);
  }

  template <class FG, class... Args>
  bool static IsFGLightweight(FG const &Func, Args &&... args) {
    auto constexpr static maxLightweightClocks = size_t{1000000};
    auto &&Func_ =
        GeneratorTraits::GetFunctionForSingleThread<decltype(Func)>(Func);
    auto freq = 2; // clocks per nanosecond
    auto time = Utility::Benchmark(std::forward<decltype(Func_)>(Func_),
                                   std::forward<Args>(args)...);
    auto nanosecs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(time).count();
    auto clocks = freq * nanosecs;
    return clocks < maxLightweightClocks;
  }

};

} // namespace Evolution

#pragma once
#define NOMINMAX
#include "Evolution/Concepts.hpp"
#include "Evolution/GeneratorTraits.hpp"
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlowDebugger.hpp"
#include "Utility/TypeTraits.hpp"
#include <algorithm>
#include <cassert>
#include <execution>
#include <functional>
#include <map>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/flow_graph.h>
#include <utility>
#include <variant>
#include <vector>

namespace Evolution {
template <EvaluateFunctionOrGeneratorConcept EvaluateFG,
          MutateFunctionOrGeneratorConcept MutateFG,
          CrossoverFunctionOrGeneratorConcept CrossoverFG>
class TaskFlow final {

private:
  using TypeTraits = Utility::TypeTraits;
  template <typename Callable>
  using CallableTraits = Utility::CallableTraits<Callable>;

  using EvaluateFunction = typename GeneratorTraits::Function<EvaluateFG>;
  using MutateFunction = typename GeneratorTraits::Function<MutateFG>;
  using CrossoverFunction = typename GeneratorTraits::Function<CrossoverFG>;

  using EvaluateArg =
      typename Utility::CallableTraits<EvaluateFunction>::template ArgType<0>;
  using MutateArg =
      typename Utility::CallableTraits<MutateFunction>::template ArgType<0>;
  using MutateReturn =
      typename Utility::CallableTraits<MutateFunction>::ReturnType;
  using CrossoverArg0 =
      typename Utility::CallableTraits<CrossoverFunction>::template ArgType<0>;
  using CrossoverArg1 =
      typename Utility::CallableTraits<CrossoverFunction>::template ArgType<1>;
  using CrossoverReturn =
      typename Utility::CallableTraits<CrossoverFunction>::ReturnType;

  template <typename T>
  bool constexpr static isVariant =
      TypeTraits::isInstanceOf<std::variant, std::remove_reference_t<T>>;

public:
  using DNA = std::remove_cvref_t<
      typename CallableTraits<EvaluateFunction>::template ArgType<0>>;
  using Population = std::vector<DNA>;

  using Grade = std::remove_cvref_t<
      typename CallableTraits<EvaluateFunction>::ReturnType>;
  using Grades = std::vector<Grade>;

private:

  // Check that arguments and return values of functions are of type DNA
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<EvaluateArg>>);
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<MutateReturn>>);
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<MutateArg>>);
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<CrossoverReturn>>);
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<CrossoverArg0>>);
  static_assert(std::is_same_v<DNA, std::remove_cvref_t<CrossoverArg1>>);

  auto constexpr static isMutateInPlace =
      (!CallableTraits<MutateFunction>::template isConst<1>);
  auto constexpr static isMutateMovable =
      CallableTraits<MutateFunction>::template isRValueReference<1> ||
      CallableTraits<MutateFunction>::template isValue<1>;
  auto constexpr static isCrossoverInPlaceFirst =
      !CallableTraits<CrossoverFunction>::template isConst<1>;
  auto constexpr static isCrossoverInPlaceSecond =
      !CallableTraits<CrossoverFunction>::template isConst<2>;
  auto constexpr static isCrossoverMovableFirst =
      CallableTraits<CrossoverFunction>::template isRValueReference<1> ||
      CallableTraits<CrossoverFunction>::template isValue<1>;
  auto constexpr static isCrossoverMovableSecond =
      CallableTraits<CrossoverFunction>::template isRValueReference<2> ||
      CallableTraits<CrossoverFunction>::template isValue<2>;

  using State = StateFlow::State;
  using StateSet = StateFlow::StateSet;
  using StateVector = StateFlow::StateVector;
  using Operation = StateFlow::Operation;

  using DNAPtr = std::shared_ptr<DNA>;

  template <typename Input, typename Output = tbb::flow::continue_msg,
            typename Policy = tbb::flow::queueing,
            typename Allocator = tbb::flow::interface11::null_type>
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
  using EvaluateFTG =
      typename GeneratorTraits::FunctionOrTBBGenerator<EvaluateFG>;
  using MutateFTG = typename GeneratorTraits::FunctionOrTBBGenerator<MutateFG>;
  using CrossoverFTG =
      typename GeneratorTraits::FunctionOrTBBGenerator<CrossoverFG>;

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
    std::vector<size_t> initialNonEvaluateIndices;
    bool isEvaluateLightweight;
    bool isMutateLightweight;
    bool isCrossoverLightweight;
  };
  TBBFlow tbbFlow;
  StateFlow stateFlow;
  TaskFlowDebugger<DNA> debugger;
  tbb::concurrent_vector<std::pair<DNAPtr, Grade>> evaluateBuffer;
  EvaluateFTG evaluateFTG;
  MutateFTG mutateFTG;
  CrossoverFTG crossoverFTG;

public:
  TaskFlow(EvaluateFG const &Evaluate, MutateFG const &Mutate,
           CrossoverFG const &Crossover, StateFlow const &stateFlow,
           bool isEvaluateLightweight = false, bool isMutateLightweight = false,
           bool isCrossoverLightweight = false)
      : tbbFlow(GenerateTBBFlow(stateFlow, isEvaluateLightweight,
                                isMutateLightweight, isCrossoverLightweight)),
        stateFlow(stateFlow), debugger(this->stateFlow),
        evaluateFTG(GeneratorTraits::WrapFunctionOrGenerator(Evaluate)),
        mutateFTG(GeneratorTraits::WrapFunctionOrGenerator(Mutate)),
        crossoverFTG(GeneratorTraits::WrapFunctionOrGenerator(Crossover)) {}

  void Run(Population &population, Grades &grades) {
    RunTaskFlow(population);
    MoveResultsFromBuffer(population, grades);
  }

  Grades EvaluatePopulation(Population const &population) {
    auto grades = Grades{};
    if constexpr (std::is_default_constructible_v<Grade>) {
      grades.resize(population.size());
      std::transform(std::execution::par_unseq, population.begin(),
                     population.end(), grades.begin(), [&](auto const &dna) {
                       auto &&Evaluate = GetEvaluateFunction();
                       return Evaluate(dna);
                     });
    } else {
      auto gradesPar = tbb::concurrent_vector<std::pair<size_t, Grade>>{};
      grades.reserve(population.size());
      gradesPar.reserve(population.size());
      auto indices = Utility::GetIndices(population.size());
      std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                    [&](auto index) {
                      auto &&Evaluate = GetEvaluateFunction();
                      gradesPar.emplace_back(index,
                                             Evaluate(population[index]));
                    });
      std::sort(gradesPar.begin(), gradesPar.end(),
                [](auto const &lhs, auto const &rhs) {
                  return lhs.first < rhs.first;
                });
      for (auto const &[index, grade] : gradesPar) {
        assert(index == grades.size());
        grades.push_back(grade);
      }
    }
    return grades;
  }

  // Strong exception guarantee
  void SetStateFlow(StateFlow &&stateFlow) {
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
    else
      return IsFGLightweight(Mutate, dna_);
  }
  bool static IsCrossoverLightweight(CrossoverFG const &Crossover,
                                     DNA const &dna0, DNA const &dna1) {
    if constexpr (!isCrossoverInPlaceFirst)
      return IsCrossoverLightweight_(Crossover, dna0, dna1);
    auto dna0_ = DNA(dna0);
    if constexpr (!isCrossoverMovableFirst)
      return IsCrossoverLightweight_(Crossover, dna0_, dna1);
    else
      return IsCrossoverLightweight_(Crossover, std::move(dna0_), dna1);
  }

private:
  EvaluateFunction &GetEvaluateFunction() {
    return GeneratorTraits::GetFunction<EvaluateFG>(evaluateFTG);
  }
  MutateFunction &GetMutateFunction() {
    return GeneratorTraits::GetFunction<MutateFG>(mutateFTG);
  }
  CrossoverFunction &GetCrossoverFunction() {
    return GeneratorTraits::GetFunction<CrossoverFG>(crossoverFTG);
  }

  DNAPtr CopyHelper(DNA const &src) const { return DNAPtr(new DNA(src)); }
  DNAPtr MoveHelper(DNA &&src) const { return DNAPtr(new DNA(std::move(src))); }
  DNAPtr SaveHelper(DNA const &src) const { return CopyHelper(src); }
  DNAPtr SaveHelper(DNA &&src) const { return MoveHelper(std::move(src)); }
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
  void EvaluateHelper(DNAPtr iSrc, State state) {
    debugger.Register(iSrc.get(), state, /*isReadOnly=*/true,
                      NodeType::Evaluate);

    auto &&Evaluate = GetEvaluateFunction();
    auto &&grade = Evaluate(*iSrc);
    evaluateBuffer.emplace_back(iSrc, std::move(grade));
  }
  DNAPtr MutateHelper(DNAPtr iSrc, bool isCopy, State state) {
    auto iSrcOrig = iSrc;
    debugger.Register(iSrcOrig.get(), state, isCopy, NodeType::Mutate);

    auto isCopyHelp = isMutateInPlace && isCopy;
    auto isPtrMove = !isCopy || isCopyHelp;
    if (isCopyHelp)
      iSrc = CopyHelper(*iSrc);

    auto &&Mutate = GetMutateFunction();
    auto MutatePtr = [&](DNAPtr iSrc) -> decltype(auto) {
      if constexpr (isMutateMovable) {
        assert(!isCopy || iSrcOrig != iSrc);
        return Mutate(std::move(*iSrc));
      } else {
        assert(!isCopy || iSrcOrig != iSrc || !isMutateInPlace);
        return Mutate(*iSrc);
      }
    };

    auto ret = DNAPtr{};
    auto &&dst = MutatePtr(iSrc);
    if (!isPtrMove)
      ret = SaveHelper(std::move(dst));
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
    auto CrossoverPtr = [&](DNAPtr iSrc0, DNAPtr iSrc1) -> decltype(auto) {
      if constexpr (isCrossoverMovableFirst) {
        assert(!isCopy0 || iSrcOrig0 != iSrc0);
        if constexpr (isCrossoverMovableSecond) {
          assert(!isCopy1 || iSrcOrig1 != iSrc1);
          return Crossover(std::move(*iSrc0), std::move(*iSrc1));
        } else {
          assert(!isCopy1 || iSrcOrig1 != iSrc1 || !isCrossoverInPlaceSecond);
          return Crossover(std::move(*iSrc0), *iSrc1);
        }
      } else {
        assert(!isCopy0 || iSrcOrig0 != iSrc0 || !isCrossoverInPlaceFirst);
        if constexpr (isCrossoverMovableSecond) {
          assert(!isCopy1 || iSrcOrig1 != iSrc1);
          return Crossover(*iSrc0, std::move(*iSrc1));
        } else {
          assert(!isCopy1 || iSrcOrig1 != iSrc1 || !isCrossoverInPlaceSecond);
          return Crossover(*iSrc0, *iSrc1);
        }
      }
    };

    auto ret = DNAPtr{};
    auto &&dst = CrossoverPtr(iSrc0, iSrc1);
    if (!isPtrMove0 && !isPtrMove1)
      ret = SaveHelper(std::move(dst));
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

  template <typename Node, typename... Args>
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
  template <typename Node>
  EvaluateNode &AddEvaluate(TBBFlow &tbbFlow, Node &predecessor, State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.evaluateNodes.capacity() > tbbFlow.evaluateNodes.size());
    tbbFlow.evaluateNodes.push_back(MakeNode<EvaluateNode>(
        tbbFlow.isEvaluateLightweight, *tbbFlow.graphPtr,
        tbb::flow::concurrency::serial, [&, state](DNAPtr iSrc) {
          EvaluateHelper(iSrc, state);
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
  template <typename Node>
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
  template <typename Node0, typename Node1>
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
    auto isNotReady = stateFlow.IsNotReady();
    if (isNotReady)
      throw std::invalid_argument("Provided stateflow is not ready. " +
                                  isNotReady.value());
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
    auto visitedStates = std::unordered_set<State>{};
    auto resolvedStates = std::unordered_set<State>{};
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
          [&](auto nodeRef) { AddEvaluate(tbbFlow, nodeRef.get(), state); },
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
#ifndef NDEBUG
      visitedStates.insert(state);
#endif // !NDEBUG
      if (IsResolved(state))
        return;
      if (!IsResolvable(state))
        return;
#ifndef NDEBUG
      resolvedStates.insert(state);
#endif // !NDEBUG
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
    assert(visitedStates.size() == stateFlow.GetNStates());
    assert(resolvedStates.size() == stateFlow.GetNStates());
    assert(evaluateCount == stateFlow.GetNEvaluates());

    tbbFlow.initialNonEvaluateIndices =
        EvaluateInitialNonEvaluateIndices(stateFlow);

    return tbbFlow;
  }

  std::vector<size_t> static EvaluateInitialNonEvaluateIndices(
      StateFlow const &stateFlow) {
    auto availableIndices = std::vector<size_t>{};
    auto unailableIndices = std::unordered_set<size_t>{};
    for (auto state : stateFlow.GetInitialStates())
      if (stateFlow.IsEvaluate(state))
        unailableIndices.insert(stateFlow.GetIndex(state));
    for (auto i = size_t{0}; i != stateFlow.GetNEvaluates(); ++i)
      if (!unailableIndices.contains(i))
        availableIndices.push_back(i);
    assert(availableIndices.size() ==
           stateFlow.GetNEvaluates() - stateFlow.GetNInitialEvaluates());
    return availableIndices;
  }

  size_t GetPopulationSize() const noexcept {
    return stateFlow.GetNEvaluates();
  }

  void RunTaskFlow(Population &population) {
    assert(population.size() == GetPopulationSize());
    assert(tbbFlow.inputNodes.size() == tbbFlow.inputIndices.size());
    evaluateBuffer.clear();

    for (auto i = size_t{0}; i != tbbFlow.inputNodes.size(); ++i)
      tbbFlow.inputNodes[i].try_put(&population.at(tbbFlow.inputIndices[i]));
    tbbFlow.graphPtr->wait_for_all();
    debugger.Finish();
  }

  void MoveResultsFromBuffer(Population &population, Grades &grades) noexcept {
    assert(evaluateBuffer.size() == tbbFlow.initialNonEvaluateIndices.size());
    assert(population.size() == GetPopulationSize());
    assert(grades.size() == GetPopulationSize());

    for (auto i = size_t{0}, e = evaluateBuffer.size(); i != e; ++i) {
      auto index = tbbFlow.initialNonEvaluateIndices[i];
      assert(index < population.size());
      auto &&[dnaPtr, grade] = evaluateBuffer[i];
      population.at(index) = std::move(*dnaPtr);
      grades.at(index) = std::move(grade);
    }
  }

  template <typename DNAFirst>
  bool static IsCrossoverLightweight_(CrossoverFG const &Crossover,
                                      DNAFirst &&dna0, DNA const &dna1) {
    if constexpr (!isCrossoverInPlaceSecond)
      return IsFGLightweight(Crossover, std::forward<DNAFirst>(dna0), dna1);
    auto dna1_ = DNA(dna1);
    if constexpr (isCrossoverMovableSecond)
      return IsFGLightweight(Crossover, std::forward<DNAFirst>(dna0),
                             std::move(dna1_));
    else
      return IsFGLightweight(Crossover, std::forward<DNAFirst>(dna0), dna1_);
  }

  template <typename FG, typename... Args>
  bool static IsFGLightweight(FG const &Func, Args &&... args) {
    auto constexpr static maxLightweightClocks = size_t{1000000};
    auto FG_ = std::function(Func);
    auto &&Func_ = GeneratorTraits::GetFunction<decltype(Func)>(FG_);
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

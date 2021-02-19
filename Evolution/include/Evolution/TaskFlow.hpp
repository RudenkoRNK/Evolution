#pragma once
#include "Evolution/Concepts.hpp"
#include "Evolution/GeneratorTraits.hpp"
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlowDebugger.hpp"
#include "Utility/Misc.hpp"
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
struct EnvironmentOptions {
  Utility::AutoOption isEvaluateLightweight;
  Utility::AutoOption isMutateLightweight;
  Utility::AutoOption isCrossoverLightweight;
  bool allowMoveFromPopulation = true;
};

template <EvaluateFunctionOrGeneratorConcept EvaluateFG,
          MutateFunctionOrGeneratorConcept MutateFG,
          CrossoverFunctionOrGeneratorConcept CrossoverFG>
class TaskFlow final {

private:
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
      Utility::isInstanceOf<std::variant, std::remove_cvref_t<T>>;

public:
  using DNA = std::remove_cvref_t<
      typename CallableTraits<EvaluateFunction>::template ArgType<0>>;
  using Population = std::vector<DNA>;

  using Grade = std::remove_cvref_t<
      typename CallableTraits<EvaluateFunction>::ReturnType>;
  using Grades = std::vector<Grade>;

private:
  // Check that arguments and return values of functions are of type DNA
  static_assert(std::is_same_v<DNA, DNADecay<EvaluateArg>>);
  static_assert(std::is_same_v<DNA, DNADecay<MutateReturn>>);
  static_assert(std::is_same_v<DNA, DNADecay<MutateArg>>);
  static_assert(std::is_same_v<DNA, DNADecay<CrossoverReturn>>);
  static_assert(std::is_same_v<DNA, DNADecay<CrossoverArg0>>);
  static_assert(std::is_same_v<DNA, DNADecay<CrossoverArg1>>);

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
  EnvironmentOptions options;
  TBBFlow tbbFlow;
  StateFlow stateFlow;
  TaskFlowDebugger<DNA> debugger;
  EvaluateFTG evaluateFTG;
  MutateFTG mutateFTG;
  CrossoverFTG crossoverFTG;
  tbb::concurrent_vector<std::pair<DNAPtr, Grade>> evaluateBuffer;

public:
  TaskFlow(EvaluateFG const &Evaluate, MutateFG const &Mutate,
           CrossoverFG const &Crossover, StateFlow const &stateFlow,
           EnvironmentOptions const &options = EnvironmentOptions{})
      : options(options), tbbFlow(GenerateTBBFlow(stateFlow, options)),
        stateFlow(stateFlow), debugger(this->stateFlow),
        evaluateFTG(GeneratorTraits::WrapFunctionOrGenerator(Evaluate)),
        mutateFTG(GeneratorTraits::WrapFunctionOrGenerator(Mutate)),
        crossoverFTG(GeneratorTraits::WrapFunctionOrGenerator(Crossover)) {}

  // TBBFlow will have references to previous *this otherwise
  TaskFlow(TaskFlow &&other) = delete;
  TaskFlow &operator=(TaskFlow &&other) = delete;

  TaskFlow(TaskFlow const &other)
      : options(other.options),
        tbbFlow(GenerateTBBFlow(other.stateFlow, other.options)),
        stateFlow(other.stateFlow), debugger(this->stateFlow),
        evaluateFTG(other.evaluateFTG), mutateFTG(other.mutateFTG),
        crossoverFTG(other.crossoverFTG), evaluateBuffer(other.evaluateBuffer) {
  }
  TaskFlow &operator=(TaskFlow const &other) & {
    if (other == *this)
      return;
    auto options_ = other.options;
    auto debugger_ = other.debugger;
    auto evaluateFTG_ = other.evaluateFTG;
    auto mutateFTG_ = other.mutateFTG;
    auto crossoverFTG_ = other.crossoverFTG;
    auto evaluateBuffer_ = other.evaluateBuffer;
    SetStateFlow(other.stateFlow);
    options = std::move(options_);
    debugger = std::move(debugger_);
    evaluateFTG = std::move(evaluateFTG_);
    mutateFTG = std::move(mutateFTG_);
    crossoverFTG = std::move(crossoverFTG_);
    evaluateBuffer = std::move(evaluateBuffer_);
  }

  void Run(Population &population, Grades &grades) {
    auto finish = Utility::RAII([&]() noexcept {
      MoveResultsFromBuffer(population, grades);
      evaluateBuffer.clear(); // Hope this will not throw...
    });
    auto finishDebugger = Utility::RAII([&]() { debugger.Finish(); },
                                        [&]() noexcept { debugger.Reset(); });
    RunTaskFlow(population);
  }

  Grades EvaluatePopulation(Population const &population) {
    auto grades = Grades{};
    auto eHandler = Utility::ExceptionSaver{};
    if constexpr (std::is_nothrow_default_constructible_v<Grade>) {
      grades.resize(population.size());
      std::transform(std::execution::par_unseq, population.begin(),
                     population.end(), grades.begin(),
                     eHandler.Wrap([&](DNA const &dna) {
                       auto &&Evaluate = GetEvaluateFunction();
                       return Evaluate(dna);
                     }));
      eHandler.Rethrow();
    } else {
      auto gradesPar = tbb::concurrent_vector<std::pair<size_t, Grade>>{};
      grades.reserve(population.size());
      gradesPar.reserve(population.size());
      auto indices = Utility::GetIndices(population.size());
      std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                    eHandler.Wrap([&](size_t index) {
                      auto &&Evaluate = GetEvaluateFunction();
                      gradesPar.emplace_back(index,
                                             Evaluate(population[index]));
                    }));
      eHandler.Rethrow();
      std::sort(gradesPar.begin(), gradesPar.end(),
                [](auto const &lhs, auto const &rhs) {
                  return lhs.first < rhs.first;
                });
      for (auto &&[index, grade] : gradesPar) {
        assert(index == grades.size());
        grades.push_back(std::move(grade));
      }
    }
    return grades;
  }

  // Strong exception guarantee
  void SetStateFlow(StateFlow &&stateFlow) {
    auto tbbFlow_ = GenerateTBBFlow(stateFlow, options);
    tbbFlow.inputNodes.clear();
    tbbFlow.evaluateNodes.clear();
    tbbFlow.mutateNodes.clear();
    tbbFlow.crossoverJoinNodes.clear();
    tbbFlow.crossoverNodes.clear();
    tbbFlow = std::move(tbbFlow_);
    this->stateFlow = std::move(stateFlow);
  }

  StateFlow const &GetStateFlow() const &noexcept { return stateFlow; }

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

  DNAPtr CopyHelper(DNA const &src) const { return std::make_shared<DNA>(src); }
  DNAPtr SaveHelper(DNA const &src, DNAPtr saveTo = DNAPtr{}) const {
    if (saveTo) {
      *saveTo = src;
      return saveTo;
    }
    return CopyHelper(src);
  }
  DNAPtr SaveHelper(DNA &&src, DNAPtr saveTo = DNAPtr{}) const {
    if (saveTo) {
      *saveTo = std::move(src);
      return saveTo;
    }
    return std::make_shared<DNA>(std::move(src));
  }
  DNAPtr SaveHelper(std::unique_ptr<DNA> src, DNAPtr saveTo = DNAPtr{}) const {
    return std::shared_ptr<DNA>(std::move(src));
  }

  DNAPtr InputHelper(DNA *iSrc, bool isReadOnly, State state) {
    debugger.Register(iSrc, state, isReadOnly, NodeType::Input);
    auto ret = DNAPtr{};
    if (isReadOnly)
      ret = CopyHelper(*iSrc);
    else
      ret = SaveHelper(std::move(*iSrc));
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
  DNAPtr MutateHelper(DNAPtr iSrc, bool isReadOnly, State state) {
    auto iSrcOrig = iSrc;
    debugger.Register(iSrcOrig.get(), state, isReadOnly, NodeType::Mutate);

    auto isCopyHelp = isMutateInPlace && isReadOnly;
    auto isPtrMove = !isReadOnly || isCopyHelp;
    if (isCopyHelp)
      iSrc = CopyHelper(*iSrc);

    auto &&Mutate = GetMutateFunction();
    auto MutatePtr = [&](DNAPtr iSrc) -> decltype(auto) {
      if constexpr (isMutateMovable) {
        assert(!isReadOnly || iSrcOrig != iSrc);
        return Mutate(std::move(*iSrc));
      } else {
        assert(!isReadOnly || iSrcOrig != iSrc || !isMutateInPlace);
        return Mutate(*iSrc);
      }
    };

    auto ret = DNAPtr{};
    auto &&dst = MutatePtr(iSrc);
    if (!isPtrMove)
      ret = SaveHelper(std::move(dst));
    else
      ret = SaveHelper(std::move(dst), iSrc);

    debugger.Unregister(iSrcOrig.get(), state);
    debugger.RegisterOutput(ret.get(), state);
    return ret;
  }
  DNAPtr CrossoverHelper(std::tuple<DNAPtr, DNAPtr> iSrcs, bool isReadOnly0,
                         bool isReadOnly1, bool isSwapArgumentsAllowed,
                         State state) {
    auto [iSrc0, iSrc1] = iSrcs;
    auto [iSrcOrig0, iSrcOrig1] = iSrcs;
    debugger.Register(iSrcOrig0.get(), state, isReadOnly0, NodeType::Crossover);
    debugger.Register(iSrcOrig1.get(), state, isReadOnly1, NodeType::Crossover);

    auto isSwap = ((isCrossoverInPlaceFirst && !isCrossoverInPlaceSecond &&
                    isReadOnly0 && !isReadOnly1) ||
                   (isCrossoverInPlaceSecond && !isCrossoverInPlaceFirst &&
                    isReadOnly1 && !isReadOnly0)) &&
                  isSwapArgumentsAllowed;
    if (isSwap) {
      std::swap(iSrc0, iSrc1);
      std::swap(iSrcOrig0, iSrcOrig1);
      std::swap(isReadOnly0, isReadOnly1);
    }

    auto isCopyHelp0 = isCrossoverInPlaceFirst && isReadOnly0;
    auto isCopyHelp1 = isCrossoverInPlaceSecond && isReadOnly1;
    auto isPtrMove0 = !isReadOnly0 || isCopyHelp0;
    auto isPtrMove1 = !isReadOnly1 || isCopyHelp1;

    if (isCopyHelp0)
      iSrc0 = CopyHelper(*iSrc0);
    if (isCopyHelp1)
      iSrc1 = CopyHelper(*iSrc1);

    auto &&Crossover = GetCrossoverFunction();
    auto CrossoverPtr = [&](DNAPtr iSrc0, DNAPtr iSrc1) -> decltype(auto) {
      if constexpr (isCrossoverMovableFirst) {
        assert(!isReadOnly0 || iSrcOrig0 != iSrc0);
        if constexpr (isCrossoverMovableSecond) {
          assert(!isReadOnly1 || iSrcOrig1 != iSrc1);
          return Crossover(std::move(*iSrc0), std::move(*iSrc1));
        } else {
          assert(!isReadOnly1 || iSrcOrig1 != iSrc1 ||
                 !isCrossoverInPlaceSecond);
          return Crossover(std::move(*iSrc0), *iSrc1);
        }
      } else {
        assert(!isReadOnly0 || iSrcOrig0 != iSrc0 || !isCrossoverInPlaceFirst);
        if constexpr (isCrossoverMovableSecond) {
          assert(!isReadOnly1 || iSrcOrig1 != iSrc1);
          return Crossover(*iSrc0, std::move(*iSrc1));
        } else {
          assert(!isReadOnly1 || iSrcOrig1 != iSrc1 ||
                 !isCrossoverInPlaceSecond);
          return Crossover(*iSrc0, *iSrc1);
        }
      }
    };

    auto ret = DNAPtr{};
    auto &&dst = CrossoverPtr(iSrc0, iSrc1);
    if (!isPtrMove0 && !isPtrMove1)
      ret = SaveHelper(std::move(dst));
    else if (isPtrMove0 && !isPtrMove1)
      ret = SaveHelper(std::move(dst), iSrc0);
    else if (isPtrMove1 && !isPtrMove0)
      ret = SaveHelper(std::move(dst), iSrc1);
    else {
      if constexpr (isCrossoverInPlaceFirst && !isCrossoverInPlaceSecond)
        ret = SaveHelper(std::move(dst), iSrc0);
      else if (isCrossoverInPlaceSecond && !isCrossoverInPlaceFirst)
        ret = SaveHelper(std::move(dst), iSrc1);
      else
        ret = SaveHelper(std::move(dst), iSrc0);
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

  InputNode &AddInput(TBBFlow &tbbFlow, size_t index, bool isReadOnly,
                      State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.inputNodes.capacity() > tbbFlow.inputNodes.size());
    tbbFlow.inputIndices.push_back(index);
    tbbFlow.inputNodes.emplace_back(
        *tbbFlow.graphPtr, tbb::flow::concurrency::serial,
        [&, isReadOnly, state](DNA *dna) {
          return InputHelper(dna, isReadOnly, state);
        });
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
  MutateNode &AddMutate(TBBFlow &tbbFlow, Node &predecessor, bool isReadOnly,
                        State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.mutateNodes.capacity() > tbbFlow.mutateNodes.size());
    tbbFlow.mutateNodes.push_back(MakeNode<MutateNode>(
        tbbFlow.isMutateLightweight, *tbbFlow.graphPtr,
        tbb::flow::concurrency::serial, [&, isReadOnly, state](DNAPtr iSrc) {
          return MutateHelper(iSrc, isReadOnly, state);
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
                              Node1 &predecessor1, bool isReadOnly0,
                              bool isReadOnly1, bool isSwapArgumentsAllowed,
                              State state) {
    // Should be preallocated to avoid refs invalidation
    assert(tbbFlow.crossoverNodes.capacity() > tbbFlow.crossoverNodes.size());
    tbbFlow.crossoverNodes.push_back(MakeNode<CrossoverNode>(
        tbbFlow.isCrossoverLightweight, *tbbFlow.graphPtr,
        tbb::flow::concurrency::serial,
        [&, isReadOnly0, isReadOnly1, isSwapArgumentsAllowed,
         state](std::tuple<DNAPtr, DNAPtr> iSrcs) {
          return CrossoverHelper(iSrcs, isReadOnly0, isReadOnly1,
                                 isSwapArgumentsAllowed, state);
        }));
    tbbFlow.crossoverJoinNodes.emplace_back(*tbbFlow.graphPtr);
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
                          EnvironmentOptions const &options) {
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

    tbbFlow.isEvaluateLightweight = options.isEvaluateLightweight.isTrue();
    tbbFlow.isMutateLightweight = options.isMutateLightweight.isTrue();
    tbbFlow.isCrossoverLightweight = options.isCrossoverLightweight.isTrue();

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
      auto isReadOnly =
          stateFlow.IsEvaluate(state) || !options.allowMoveFromPopulation;
      auto &&node =
          AddInput(tbbFlow, stateFlow.GetIndex(state), isReadOnly, state);
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
    assert(evaluateBuffer.size() == 0);

    for (auto i = size_t{0}; i != tbbFlow.inputNodes.size(); ++i)
      tbbFlow.inputNodes[i].try_put(&population.at(tbbFlow.inputIndices[i]));
    try {
      tbbFlow.graphPtr->wait_for_all();
    } catch (...) {
      SetStateFlow(StateFlow(stateFlow));
      tbbFlow.graphPtr->reset();
      throw;
    }
  }

  void MoveResultsFromBuffer(Population &population, Grades &grades) noexcept {
    assert(evaluateBuffer.size() <= tbbFlow.initialNonEvaluateIndices.size());
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
};

} // namespace Evolution

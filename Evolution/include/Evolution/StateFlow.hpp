#pragma once
#include "Utility/Misc.hpp"
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <exception>
#include <optional>
#include <utility>
#define NOMINMAX

namespace Evolution {
class StateFlow final {
private:
  auto constexpr static UndefinedIndex = size_t(-1);
  struct StateProperties final {
    bool isEvaluate = false;
    size_t index = UndefinedIndex;
  };
  struct OperationProperties final {};

  using StateGraph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                            StateProperties, OperationProperties>;

public:
  // Be careful: vertex_descriptors are invalidated when call remove_vertex
  using State = StateGraph::vertex_descriptor;
  using StateIterator = StateGraph::vertex_iterator;
  using ParentStateIterator = StateGraph::inv_adjacency_iterator;
  using ChildStateIterator = StateGraph::adjacency_iterator;
  using StateIteratorRange = std::pair<StateIterator, StateIterator>;
  using ParentStateIteratorRange =
      std::pair<ParentStateIterator, ParentStateIterator>;
  using ChildStateIteratorRange =
      std::pair<ChildStateIterator, ChildStateIterator>;
  using Operation = StateGraph::edge_descriptor;
  using OperationIterator = StateGraph::edge_iterator;
  using InOperationIterator = StateGraph::in_edge_iterator;
  using OutOperationIterator = StateGraph::out_edge_iterator;
  using OperationIteratorRange =
      std::pair<OperationIterator, OperationIterator>;
  using InOperationIteratorRange =
      std::pair<InOperationIterator, InOperationIterator>;
  using OutOperationIteratorRange =
      std::pair<OutOperationIterator, OutOperationIterator>;
  using StateSet = std::unordered_set<State>;
  using StateVector = std::vector<State>;

private:
  StateGraph G;
  StateVector initialStates;
  size_t nEvaluates = 0;
  size_t nInitialEvaluates = 0;
  size_t nMutates = 0;
  size_t nCrossovers = 0;
  bool isSwapArgumentsAllowedInCrossover = false;

public:
  // Construction
  State GetOrAddInitialState(size_t index) {
    assert(index != UndefinedIndex);
    auto &&inits = GetInitialStates();
    auto s = std::find_if(inits.begin(), inits.end(),
                          [&](auto state) { return GetIndex(state) == index; });
    if (s != inits.end())
      return *s;
    auto g = Utility::RAII([]() {}, [&]() noexcept { Clear(); });
    auto state = boost::add_vertex({.index = index}, G);
    initialStates.push_back(state);
    return state;
  }
  State AddMutate(State state) {
    auto g = Utility::RAII([]() {}, [&]() noexcept { Clear(); });
    auto ret = boost::add_vertex(StateProperties{}, G);
    boost::add_edge(state, ret, {}, G);
    ++nMutates;
    return ret;
  }
  State AddCrossover(State state0, State state1) {
    if (state0 == state1)
      throw std::invalid_argument("Cannot crossover two same states");
    auto g = Utility::RAII([]() {}, [&]() noexcept { Clear(); });
    auto ret = boost::add_vertex(StateProperties{}, G);
    boost::add_edge(state0, ret, {}, G);
    boost::add_edge(state1, ret, {}, G);
    ++nCrossovers;
    return ret;
  }
  void SetEvaluate(State state) {
    if (IsEvaluate(state))
      return;
    auto &&props = G[state];
    if (IsInitialState(state))
      ++nInitialEvaluates;
    props.isEvaluate = true;
    ++nEvaluates;
  }
  void SetSwapArgumentsAllowedInCrossover(
      bool isSwapArgumentsAllowedInCrossover = true) noexcept {
    this->isSwapArgumentsAllowedInCrossover = isSwapArgumentsAllowedInCrossover;
  }
  void Clear() noexcept {
    nEvaluates = 0;
    nMutates = 0;
    nCrossovers = 0;
    nInitialEvaluates = 0;
    isSwapArgumentsAllowedInCrossover = false;
    initialStates.clear();
    // This is not noexcept actually, but without assuming so it is impossible
    // to maintain basic exception safety at all
    G.clear();
  }

  // State/Operation Access
  StateIteratorRange GetStates() const { return boost::vertices(G); }
  OperationIteratorRange GetOperations() const { return boost::edges(G); }
  State GetAnyParent(State state) const {
    if (IsInitialState(state))
      throw std::invalid_argument("State must not be initial");
    auto range = boost::inv_adjacent_vertices(state, G);
    assert(range.second - range.first <= 2);
    return *range.first;
  }
  ChildStateIteratorRange GetChildStates(State state) const {
    return boost::adjacent_vertices(state, G);
  }
  Operation GetAnyInOperation(State state) const {
    if (IsInitialState(state))
      throw std::invalid_argument("State must not be initial");
    auto const &range = boost::in_edges(state, G);
    assert(range.second - range.first <= 2);
    assert(range.second != range.first);
    return *range.first;
  }
  OutOperationIteratorRange GetOutOperations(State state) const {
    return boost::out_edges(state, G);
  }
  State GetSource(Operation operation) const {
    return boost::source(operation, G);
  }
  State GetTarget(Operation operation) const {
    return boost::target(operation, G);
  }
  Operation GetCrossoverPair(Operation operation) const {
    if (!IsCrossover(operation))
      throw std::invalid_argument("Operation must be a crossover");
    auto const &[oB, oE] = boost::in_edges(GetTarget(operation), G);
    assert(oE - oB == 2);
    if (operation == *oB)
      return *(oB + 1);
    return *oB;
  }
  State GetOtherParent(State parent, State child) const {
    auto [op, presence] = boost::edge(parent, child, G);
    if (!presence)
      throw std::invalid_argument("No edge between parent and child");
    return GetSource(GetCrossoverPair(op));
  }
  StateVector const &GetInitialStates() const { return initialStates; }

  // Counts
  size_t GetOutDegree(State state) const { return boost::out_degree(state, G); }
  size_t GetNStates() const { return boost::num_vertices(G); }
  size_t GetNOperations() const { return boost::num_edges(G); }
  size_t GetNInitials() const { return GetInitialStates().size(); }
  size_t GetNInitialEvaluates() const { return nInitialEvaluates; }
  size_t GetNEvaluates() const { return nEvaluates; }
  size_t GetNMutates() const { return nMutates; }
  size_t GetNCrossovers() const { return nCrossovers; }

  // Properties
  bool IsMutate(Operation operation) const {
    return IsMutate(GetTarget(operation));
  }
  bool IsMutate(State state) const { return boost::in_degree(state, G) == 1; }
  bool IsCrossover(Operation operation) const {
    return IsCrossover(GetTarget(operation));
  }
  bool IsCrossover(State state) const {
    return boost::in_degree(state, G) == 2;
  }
  bool IsInitialState(State state) const {
    return boost::in_degree(state, G) == 0;
  }
  bool IsIndexSet(State state) const {
    return GetIndex(state) != UndefinedIndex;
  }
  bool IsEvaluate(State state) const { return G[state].isEvaluate; }
  bool IsLeaf(State state) const { return GetOutDegree(state) == 0; }
  bool IsSwapArgumentsAllowedInCrossover() const noexcept {
    return isSwapArgumentsAllowedInCrossover;
  }
  size_t GetIndex(State state) const {
    auto index = G[state].index;
    return index;
  }

  // Tools
  template <typename States, typename ActFunction, typename IsAddChildFunction>
  void DepthFirstSearch(States const &startStates, ActFunction &&Act,
                        IsAddChildFunction &&IsAddChild) const {
    auto visited = StateSet{};
    auto currentPath = StateVector{};
    for (auto start : startStates) {
      if (visited.contains(start))
        continue;
      currentPath.push_back(start);
      while (currentPath.size() > 0) {
        Act(std::as_const(currentPath));
        visited.insert(currentPath.back());

        auto isNextFound = false;
        while (!isNextFound && currentPath.size() > 0) {
          auto const &[nextsB, nextsE] = GetChildStates(currentPath.back());
          auto const &next = std::find_if(nextsB, nextsE, [&](auto state) {
            return !visited.contains(state) &&
                   IsAddChild(std::as_const(currentPath), std::as_const(state));
          });
          if (next != nextsE) {
            isNextFound = true;
            currentPath.push_back(*next);
          } else {
            currentPath.pop_back();
          }
        }
      }
    }
  }
  template <typename States, typename ActFunction>
  void DepthFirstSearch(States const &startStates, ActFunction &&Act) const {
    DepthFirstSearch(startStates, std::forward<ActFunction>(Act),
                     [](StateVector const &path, State child) { return true; });
  }

  template <typename States, typename ActFunction, typename IsAddChildFunction>
  void BreadthFirstSearch(States const &startStates, ActFunction &&Act,
                          IsAddChildFunction &&IsAddChild) const {
    auto visited = StateSet{};
    auto currentGen = StateSet(startStates.begin(), startStates.end());
    while (currentGen.size() > 0) {
      auto nextGen = StateSet{};
      for (auto state : currentGen) {
        Act(state);
        visited.insert(state);

        auto const &[nextsB, nextsE] = GetChildStates(state);
        for (auto nextI = nextsB; nextI != nextsE; ++nextI) {
          if (visited.contains(*nextI) || !IsAddChild(*nextI))
            continue;
          nextGen.insert(*nextI);
        }
      }
      currentGen = std::move(nextGen);
    }
  }
  template <typename States, typename ActFunction>
  void BreadthFirstSearch(States const &startStates, ActFunction &&Act) const {
    BreadthFirstSearch(startStates, std::forward<ActFunction>(Act),
                       [](State child) { return true; });
  }

  // Debug & Verification
  std::optional<State> FindUnevaluatedLeaf() const {
    auto leaf = std::optional<State>{};
    auto IsAllLeavesEval = true;
    BreadthFirstSearch(
        GetInitialStates(),
        [&](State state) {
          if (IsLeaf(state) && !IsEvaluate(state)) {
            IsAllLeavesEval = false;
            leaf = state;
          }
        },
        [&](State state) { return IsAllLeavesEval; });
    return leaf;
  }
  std::optional<std::string> IsNotReady() const {
    // The function defines whether this stateFlow can be used in TaskFlow
    auto nEvaluates = GetNEvaluates();
    auto nInitials = GetNInitials();
    if (nInitials == 0)
      return {};
    if (nInitials > nEvaluates)
      return "Number of evaluates must be greater or equal than number of "
             "initial states. Number of evaluates: " +
             std::to_string(nEvaluates) +
             ". Number of initial states: " + std::to_string(nInitials) + ".";
    auto maxState = *std::max_element(
        initialStates.begin(), initialStates.end(),
        [&](auto s1, auto s2) { return GetIndex(s1) < GetIndex(s2); });
    if (GetIndex(maxState) >= nEvaluates)
      return "Number of evaluates must be greater than max index. Number of "
             "evaluates: " +
             std::to_string(nEvaluates) +
             ". Maximum index among initial states: " +
             std::to_string(GetIndex(maxState)) +
             ". Faulty state: " + std::to_string(maxState) + ".";
    if (auto state = FindUnevaluatedLeaf())
      return "All leaf states must be evaluated. Unevaluated state: " +
             std::to_string(state.value()) + ".";
    return {};
  }
  void Print(std::ostream &out) const {
    auto VertexWriter = [&](std::ostream &out, State state) {
      auto index =
          IsIndexSet(state) ? " (" + std::to_string(GetIndex(state)) + ")" : "";
      auto shape = IsEvaluate(state) ? "diamond" : "circle";
      out << "[label=\"" << state << index << "\", ";
      out << "shape=" << shape << "]";
    };
    boost::write_graphviz(out, G, VertexWriter);
  }
};

} // namespace Evolution

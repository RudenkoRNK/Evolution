#pragma once
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/transform_value_property_map.hpp>
#include <utility>
#define NOMINMAX

namespace Evolution {
class StateFlow final {
public:
  enum class OperationType {
    Mutate,
    CrossoverArgFirst,
    CrossoverArgSecond,
  };

private:
  auto constexpr static UndefinedIndex = size_t(-1);
  struct OperationProperties final {
    OperationType operation;
  };
  struct StateProperties final {
    bool isEvaluate = false;
    size_t index = UndefinedIndex;
  };

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
  using IndexSet = std::unordered_set<size_t>;
  using StateVector = std::vector<State>;
  using OperationSet = std::unordered_set<Operation>;

private:
  StateGraph G;
  StateSet initialStates;
  StateSet evaluateStates;
  size_t nInitialEvaluates = 0;
  size_t nEvaluates = 0;
  size_t nMutates = 0;
  size_t nCrossovers = 0;
  State maxIndexState;

  template <bool IsOppositeDirection>
  auto GetChildStatesDir(State state) const {
    if constexpr (IsOppositeDirection) {
      return boost::inv_adjacent_vertices(state, G);
    } else {
      return GetChildStates(state);
    }
  }

public:
  // Construction
  State GetOrAddInitialState(size_t index) {
    assert(index != UndefinedIndex);
    auto &&inits = GetInitialStates();
    auto s = std::find_if(inits.begin(), inits.end(),
                          [&](auto state) { return GetIndex(state) == index; });
    if (s != inits.end())
      return *s;
    auto state = boost::add_vertex({.index = index}, G);
    initialStates.insert(state);
    if (initialStates.size() == 0)
      maxIndexState = state;
    maxIndexState = GetIndex(maxIndexState) > index ? maxIndexState : state;
    return state;
  }
  State AddMutate(State state) {
    auto ret = boost::add_vertex(StateProperties{}, G);
    boost::add_edge(state, ret, {OperationType::Mutate}, G);
    ++nMutates;
    return ret;
  }
  State AddCrossover(State state0, State state1) {
    assert(state0 != state1);
    auto ret = boost::add_vertex(StateProperties{}, G);
    boost::add_edge(state0, ret,
                    {
                        OperationType::CrossoverArgFirst,
                    },
                    G);
    boost::add_edge(state1, ret,
                    {
                        OperationType::CrossoverArgSecond,
                    },
                    G);
    ++nCrossovers;
    return ret;
  }
  void SetEvaluate(State state) {
    G[state].isEvaluate = true;
    evaluateStates.insert(state);
    ++nEvaluates;
    if (IsInitialState(state))
      ++nInitialEvaluates;
  }

  // State/Operation Access
  StateIteratorRange GetStates() const { return boost::vertices(G); }
  OperationIteratorRange GetOperations() const { return boost::edges(G); }
  State GetAnyParent(State state) const {
    assert(!IsInitialState(state));
    auto range = boost::inv_adjacent_vertices(state, G);
    assert(range.second - range.first <= 2);
    return *range.first;
  }
  ChildStateIteratorRange GetChildStates(State state) const {
    return boost::adjacent_vertices(state, G);
  }
  Operation GetAnyInOperation(State state) const {
    assert(!IsInitialState(state));
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
    assert(IsCrossover(operation));
    auto const &[oB, oE] = boost::in_edges(GetTarget(operation), G);
    assert(oE - oB == 2);
    if (operation == *oB)
      return *(oB + 1);
    return *oB;
  }
  State GetCrossoverPair(State parent, State child) const {
    auto e = boost::edge(parent, child, G);
    assert(e.second);
    return GetSource(e.first);
  }
  StateSet const &GetInitialStates() const noexcept { return initialStates; }
  StateSet const &GetEvaluateStates() const noexcept { return evaluateStates; }

  // Counts
  size_t GetOutDegree(State state) const { return boost::out_degree(state, G); }
  size_t GetNStates() const { return boost::num_vertices(G); }
  size_t GetNOperations() const { return boost::num_edges(G); }
  size_t GetNInitialEvaluates() const noexcept { return nInitialEvaluates; }
  size_t GetNEvaluates() const noexcept { return nEvaluates; }
  size_t GetNMutates() const noexcept { return nMutates; }
  size_t GetNCrossovers() const noexcept { return nCrossovers; }

  // Properties
  bool IsMutate(Operation operation) const {
    return GetOperationType(operation) == OperationType::Mutate;
  }
  bool IsCrossover(Operation operation) const {
    auto type = GetOperationType(operation);
    return type == OperationType::CrossoverArgFirst ||
           type == OperationType::CrossoverArgSecond;
  }
  bool IsInitialState(State state) const {
    return GetInitialStates().contains(state);
  }
  bool IsIndexSet(State state) const {
    return GetIndex(state) != UndefinedIndex;
  }
  bool IsEvaluate(State state) const {
    if (G[state].isEvaluate) {
      assert(evaluateStates.contains(state));
    }
    return G[state].isEvaluate;
  }
  bool IsLeaf(State state) const { return GetOutDegree(state) == 0; }
  size_t GetIndex(State state) const {
    auto index = G[state].index;
    assert(index != UndefinedIndex);
    return index;
  }
  State GetMaxIndexState() const noexcept {
    assert(GetNStates() > 0);
    return maxIndexState;
  }
  OperationType GetOperationType(Operation operation) const {
    return G[operation].operation;
  }

  // Verification
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
  bool Verify() const {
    auto nEvaluates = GetNEvaluates();
    return GetNStates() > 0 && GetInitialStates().size() <= nEvaluates &&
           GetIndex(GetMaxIndexState()) < nEvaluates && !FindUnevaluatedLeaf();
  }

  // Tools
  template <bool IsOppositeDirection = false, class ActFunction,
            class IsAddChildFunction>
  void DepthFirstSearch(StateSet const &startStates, ActFunction &&Act,
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
          auto const &[nextsB, nextsE] =
              GetChildStatesDir<IsOppositeDirection>(currentPath.back());
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
  template <bool IsOppositeDirection = false, class ActFunction>
  void DepthFirstSearch(StateSet const &startStates, ActFunction &&Act) const {
    DepthFirstSearch<IsOppositeDirection>(
        startStates, std::forward<ActFunction>(Act),
        [](StateVector const &path, State child) { return true; });
  }

  template <bool IsOppositeDirection = false, class ActFunction,
            class IsAddChildFunction>
  void BreadthFirstSearch(StateSet const &startStates, ActFunction &&Act,
                          IsAddChildFunction &&IsAddChild) const {
    auto visited = StateSet{};
    auto currentGen = startStates;
    auto nextGen = StateSet{};

    while (currentGen.size() > 0) {
      for (auto state : currentGen) {
        Act(state);
        visited.insert(state);

        auto const &[nextsB, nextsE] =
            GetChildStatesDir<IsOppositeDirection>(state);
        for (auto nextI = nextsB; nextI != nextsE; ++nextI) {
          if (visited.contains(*nextI) || !IsAddChild(*nextI))
            continue;
          nextGen.insert(*nextI);
        }
      }
      currentGen = std::move(nextGen);
    }
  }
  template <bool IsOppositeDirection = false, class ActFunction>
  void BreadthFirstSearch(StateSet const &startStates,
                          ActFunction &&Act) const {
    BreadthFirstSearch<IsOppositeDirection>(startStates,
                                            std::forward<ActFunction>(Act),
                                            [](State child) { return true; });
  }
};

} // namespace Evolution

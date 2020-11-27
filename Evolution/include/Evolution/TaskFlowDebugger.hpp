#pragma once
#include "Evolution/StateFlow.hpp"
#include <boost/container/small_vector.hpp>
#include <cassert>
#include <mutex>
#include <tbb/concurrent_unordered_map.h>

namespace Evolution {
template <typename DNA> class TaskFlowDebugger final {
private:
  using State = StateFlow::State;
  using Addresses = boost::container::small_vector<DNA *, 3>;
  using Poll =
      tbb::concurrent_unordered_map<DNA *, std::atomic_int, std::hash<DNA *>>;

#ifndef NDEBUG
  StateFlow &stateFlow;
  Poll writePoll;
  Poll readPoll;
  tbb::concurrent_unordered_map<State, Addresses> inputAddress;
  tbb::concurrent_unordered_map<State, DNA *> outputAddress;
  std::mutex dumpMutex;
#endif // !NDEBUG

public:
  enum class NodeType {
    Input,
    Evaluate,
    Mutate,
    Crossover,
  };

  TaskFlowDebugger(StateFlow &stateFlow) noexcept
#ifndef NDEBUG
      : stateFlow(stateFlow)
#endif // !NDEBUG
  {
  }

  void Register(DNA *dnaPtr, State state, bool isReadOnly, NodeType nodeType) {
#ifndef NDEBUG
    RegisterState(dnaPtr, state, nodeType);
    RegisterAddress(dnaPtr, state, isReadOnly);
#endif // !NDEBUG
  }

  void Unregister(DNA *dnaPtr, State state) {
#ifndef NDEBUG
    UnregisterAddress(dnaPtr, state);
#endif // !NDEBUG
  }

  void RegisterOutput(DNA *dnaPtr, State state) {
#ifndef NDEBUG
    CheckOutputOrder(dnaPtr, state);
    outputAddress.emplace(state, dnaPtr);
#endif // !NDEBUG
  }

  void Finish() {
#ifndef NDEBUG
    auto &&[sb, se] = stateFlow.GetStates();
    for (auto si = sb; si != se; ++si)
      CheckComputed(nullptr, *si);
    for (auto si = sb; si != se; ++si)
      if (stateFlow.IsEvaluate(*si))
        CheckEvaluated(nullptr, *si);
    CheckWritePollEmpty();
    Clear();
#endif // !NDEBUG
  }

  void Reset() {
#ifndef NDEBUG
    Clear();
#endif // !NDEBUG
  }

private:
#ifndef NDEBUG
  // Methods checking for tbb::flow and StateFlow synchronization
  void RegisterState(DNA *dnaPtr, State state, NodeType nodeType) {
    CheckType(dnaPtr, state, nodeType);
    CheckParentsComputed(dnaPtr, state);
    if (nodeType != NodeType::Evaluate)
      CheckChildrenNotComputed(dnaPtr, state);
    CheckComputeOrder(dnaPtr, state, nodeType);
    inputAddress[state].push_back(dnaPtr);
  }
  void CheckType(DNA *dnaPtr, State state, NodeType nodeType) {
    auto msg = std::string{
        "tbb::flow and StateFlow are out of sync. Found an attempt to compute "
        "state with incompatible method "};
    switch (nodeType) {
    case NodeType::Input:
      CheckError(dnaPtr, state, stateFlow.IsInitialState(state),
                 msg + "(input)");
      break;
    case NodeType::Evaluate:
      CheckError(dnaPtr, state, stateFlow.IsEvaluate(state),
                 msg + "(evaluate)");
      break;
    case NodeType::Mutate:
      CheckError(dnaPtr, state, stateFlow.IsMutate(state), msg + "(mutate)");
      break;
    case NodeType::Crossover:
      CheckError(dnaPtr, state, stateFlow.IsCrossover(state),
                 msg + "(crossover)");
      break;
    default:
      assert(false && "Invalid node type");
      break;
    }
  }
  void CheckParentsComputed(DNA *dnaPtr, State state) {
    if (stateFlow.IsInitialState(state))
      return;
    auto p1 = stateFlow.GetAnyParent(state);
    CheckComputed(dnaPtr, p1);
    if (stateFlow.IsCrossover(state))
      CheckComputed(dnaPtr, stateFlow.GetOtherParent(p1, state));
  }
  void CheckChildrenNotComputed(DNA *dnaPtr, State state) {
    auto &&[cB, cE] = stateFlow.GetChildStates(state);
    for (auto i = cB; i != cE; ++i)
      CheckNotComputed(dnaPtr, *i);
  }
  void CheckComputeOrder(DNA *dnaPtr, State state, NodeType nodeType) {
    auto msg = std::string{
        "tbb::flow and StateFlow are out of sync. Found an "
        "attempt to compute states in the wrong order. Faulty operation: "};
    auto inCnt = GetInputCount(state);
    auto opCnt = stateFlow.IsCrossover(state) ? 2 : 1;

    switch (nodeType) {
    case NodeType::Input:
      CheckError(dnaPtr, state, inCnt == 0, msg + "input");
      break;
    case NodeType::Evaluate:
      CheckError(dnaPtr, state, inCnt == opCnt, msg + "evaluate");
      break;
    case NodeType::Mutate:
      CheckError(dnaPtr, state, inCnt == 0, msg + "mutate");
      break;
    case NodeType::Crossover:
      CheckError(dnaPtr, state, inCnt == 0 || inCnt == 1, msg + "crossover");
      break;
    default:
      assert(false && "Invalid node type");
      break;
    }
  }
  void CheckOutputOrder(DNA *dnaPtr, State state) {
    CheckError(dnaPtr, state, outputAddress.count(state) == 0,
               "Found an attempt to register second output for the same state");
  }
  void CheckComputed(DNA *dnaPtr, State state) {
    auto realInCnt = GetInputCount(state);
    auto realOutCnt = GetOutputCount(state);
    auto isInitialEvaluate =
        stateFlow.IsInitialState(state) && stateFlow.IsEvaluate(state);
    auto isLoneInitial =
        isInitialEvaluate && stateFlow.GetOutDegree(state) == 0;
    auto isTBBEvaluate = stateFlow.IsEvaluate(state) && !isInitialEvaluate;
    if (isLoneInitial) {
      CheckError(
          dnaPtr, state, realInCnt == 0 && realOutCnt == 0,
          "tbb::flow and StateFlow out of sync. Found lone-initial-evaluate "
          "state which was computed, while it should not be");
      return;
    }
    auto inCnt = stateFlow.IsCrossover(state) ? 2 : 1;
    auto outCnt = 1;
    bool isInCountCorrect = realInCnt == inCnt;
    isInCountCorrect |= isTBBEvaluate && realInCnt == inCnt + 1;
    auto isOutCountCorrect = realOutCnt == outCnt;
    CheckError(dnaPtr, state, isInCountCorrect && isOutCountCorrect,
               "tbb::flow and StateFlow out of sync. Found state which should "
               "be computed, but it is not");
  }
  void CheckEvaluated(DNA *dnaPtr, State state) {
    assert(stateFlow.IsEvaluate(state));
    auto isInitialEvaluate =
        stateFlow.IsInitialState(state) && stateFlow.IsEvaluate(state);
    auto isTBBEvaluate = stateFlow.IsEvaluate(state) && !isInitialEvaluate;
    if (!isTBBEvaluate)
      return;
    auto inCnt = stateFlow.IsCrossover(state) ? 3 : 2;
    CheckError(dnaPtr, state, GetInputCount(state) == inCnt,
               "tbb::flow and StateFlow out of sync. Found state which should "
               "be evaluated, but it is not");
    auto evPtr = inputAddress.at(state).at(inCnt - 1);
    CheckError(evPtr, state, readPoll.count(evPtr) > 0,
               "tbb::flow and StateFlow out of sync. Found address which "
               "should be reserved for read but it is not");
  }
  void CheckNotComputed(DNA *dnaPtr, State state) {
    auto realInCnt = GetInputCount(state);
    auto realOutCnt = GetOutputCount(state);
    CheckError(dnaPtr, state, realInCnt == 0 && realOutCnt == 0,
               "tbb::flow and StateFlow out of sync. Found state which should "
               "not be computed, but it is");
  }
  size_t GetInputCount(State state) { return inputAddress[state].size(); }
  size_t GetOutputCount(State state) { return outputAddress.count(state); }

  // Methods checking for race conditions
  void RegisterAddress(DNA *dnaPtr, State state, bool isReadOnly) {
    if (isReadOnly)
      RegisterRead(dnaPtr, state);
    else
      RegisterWrite(dnaPtr, state);
  }
  void RegisterRead(DNA *dnaPtr, State state) {
    CheckRegistry(dnaPtr, state);
    CheckRace(dnaPtr, state, /*isReadOnly=*/true);
    if (readPoll.count(dnaPtr) == 0)
      readPoll.emplace(dnaPtr, 0);
    ++readPoll.at(dnaPtr);
  }
  void RegisterWrite(DNA *dnaPtr, State state) {
    CheckRegistry(dnaPtr, state);
    CheckRace(dnaPtr, state, /*isReadOnly=*/false);
    if (writePoll.count(dnaPtr) == 0)
      // emplace will not have effect if key is already in a map
      writePoll.emplace(dnaPtr, 0);
    ++writePoll.at(dnaPtr);
  }
  void UnregisterAddress(DNA *dnaPtr, State state) {
    CheckRegistry(dnaPtr, state);
    CheckRegistered(dnaPtr, state);
    if (writePoll.count(dnaPtr) > 0 && writePoll.at(dnaPtr) > 0)
      --writePoll.at(dnaPtr);
    else
      --readPoll.at(dnaPtr);
  }
  void CheckRegistry(DNA *dnaPtr, State state) {
    auto writeCnt = GetWriteCount(dnaPtr);
    auto readCnt = GetReadCount(dnaPtr);
    CheckError(dnaPtr, state, writeCnt >= 0 && readCnt >= 0,
               "Found dna which has more unregisters than registrations");
    CheckError(dnaPtr, state, writeCnt < 2,
               "Found dna which entered two modifiyng operations");
    CheckError(dnaPtr, state, !writeCnt || !readCnt,
               "Found race condition with simultaneous read and write");
  }
  void CheckRegistered(DNA *dnaPtr, State state) {
    auto writeCnt = GetWriteCount(dnaPtr);
    auto readCnt = GetReadCount(dnaPtr);
    CheckError(dnaPtr, state, writeCnt || readCnt,
               "Found dna which should be registered but it is not");
  }
  void CheckRace(DNA *dnaPtr, State state, bool isReadOnly) {
    auto isInWritePoll = GetWriteCount(dnaPtr) > 0;
    auto isInReadPoll = GetReadCount(dnaPtr) > 0;
    CheckError(dnaPtr, state, !isInWritePoll && (isReadOnly || !isInReadPoll),
               "Found DNA which is going to be modified by one operation, "
               "but it is currently in progress in another operation");
  }
  void CheckWritePollEmpty() {
    for (auto &&[dna, counter] : writePoll) {
      CheckError(
          dna, State{}, counter == 0,
          "Found DNA which is should not be registered for write, but it is");
    }
  }
  int GetWriteCount(DNA *dnaPtr) {
    return writePoll.count(dnaPtr) > 0 ? static_cast<int>(writePoll.at(dnaPtr))
                                       : 0;
  }
  int GetReadCount(DNA *dnaPtr) {
    return readPoll.count(dnaPtr) > 0 ? static_cast<int>(readPoll.at(dnaPtr))
                                      : 0;
  }

  // Common methods
  void CheckError(DNA *dnaPtr, State state, bool condition,
                  std::string const &msg) {
    if (condition)
      return;
    auto stateType = std::string{};
    if (stateFlow.IsInitialState(state))
      stateType += "initial";
    else if (stateFlow.IsMutate(state))
      stateType += "mutate";
    else if (stateFlow.IsCrossover(state))
      stateType += "crossover";
    if (stateFlow.IsEvaluate(state))
      stateType += ", evaluate";
    else
      stateType += ", non-evaluate";

    std::cerr << msg << std::endl;
    std::cerr << "Element state:   " << state << " (" << stateType << ")"
              << std::endl;
    std::cerr << "Element address: " << dnaPtr << std::endl;
    DumpStateFlow();
    assert(false);
  }
  void DumpStateFlow() {
    // TODO: implement this with collected debug info
    auto lock = std::lock_guard(dumpMutex);
    auto dump = std::ofstream("dump.dot");
    stateFlow.Print(dump);
    dump.close();
  }

  void Clear() {
    writePoll.clear();
    readPoll.clear();
    inputAddress.clear();
    outputAddress.clear();
  }
#endif // !NDEBUG
};

} // namespace Evolution

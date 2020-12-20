#pragma once
#include "Evolution/TaskFlow.hpp"

namespace Evolution {
template <EvaluateFunctionOrGeneratorConcept EvaluateFG,
          MutateFunctionOrGeneratorConcept MutateFG,
          CrossoverFunctionOrGeneratorConcept CrossoverFG>
class TaskFlowContainer final {
private:
  std::unique_ptr<TaskFlow<EvaluateFG, MutateFG, CrossoverFG>> taskFlowPtr;
#ifndef NDEBUG
  bool movedFrom = true;
#endif // !NDEBUG

public:
  using TaskFlowInst = TaskFlow<EvaluateFG, MutateFG, CrossoverFG>;

  TaskFlowContainer(EvaluateFG const &Evaluate, MutateFG const &Mutate,
                    CrossoverFG const &Crossover, StateFlow const &stateFlow,
                    EnvironmentOptions const &options = EnvironmentOptions{})
      : taskFlowPtr(std::make_unique<TaskFlowInst>(Evaluate, Mutate, Crossover,
                                                   stateFlow, options)) {
#ifndef NDEBUG
    movedFrom = false;
#endif // !NDEBUG
  }
  TaskFlowContainer(TaskFlowContainer &&other) noexcept { swap(other); }
  TaskFlowContainer(TaskFlowContainer const &other) { Copy(other); }
  TaskFlowContainer &operator=(TaskFlowContainer &&other) noexcept {
    swap(other);
    return *this;
  }
  TaskFlowContainer &operator=(TaskFlowContainer const &other) {
    Copy(other);
    return *this;
  }

  void swap(TaskFlowContainer &other) noexcept {
    std::swap(taskFlowPtr, other.taskFlowPtr);
#ifndef NDEBUG
    std::swap(movedFrom, other.movedFrom);
#endif // !NDEBUG
  }

  TaskFlowInst const &Get() const {
    assert(!movedFrom);
    return *taskFlowPtr;
  }
  TaskFlowInst &Get() {
    assert(!movedFrom);
    return *taskFlowPtr;
  }
  operator TaskFlowInst &() {
    assert(!movedFrom);
    return *taskFlowPtr;
  }
  operator TaskFlowInst const &() const {
    assert(!movedFrom);
    return *taskFlowPtr;
  }

private:
  void Copy(TaskFlowContainer const &other) {
    if (!other.taskFlowPtr) {
      taskFlowPtr = nullptr;
#ifndef NDEBUG
      movedFrom = other.movedFrom;
      assert(movedFrom);
#endif // !NDEBUG
      return;
    }
    taskFlowPtr = std::make_unique<TaskFlowInst>(*other.taskFlowPtr);
#ifndef NDEBUG
    movedFrom = other.movedFrom;
    assert(!movedFrom);
#endif // !NDEBUG
  }
};
} // namespace Evolution

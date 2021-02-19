#pragma once
#include "Evolution/TaskFlow.hpp"

namespace Evolution {
template <EvaluateFunctionOrGeneratorConcept EvaluateFG,
          MutateFunctionOrGeneratorConcept MutateFG,
          CrossoverFunctionOrGeneratorConcept CrossoverFG>
class TaskFlowContainer final {
  // TODO: TaskFlow is a part of TaskFlowContainer. This should be inverted
private:
  using TaskFlowPtr =
      std::unique_ptr<TaskFlow<EvaluateFG, MutateFG, CrossoverFG>>;
  TaskFlowPtr taskFlowPtr;

public:
  using TaskFlowInst = TaskFlow<EvaluateFG, MutateFG, CrossoverFG>;
  TaskFlowContainer() noexcept = default;
  TaskFlowContainer(EvaluateFG const &Evaluate, MutateFG const &Mutate,
                    CrossoverFG const &Crossover, StateFlow const &stateFlow,
                    EnvironmentOptions const &options = EnvironmentOptions{})
      : taskFlowPtr(std::make_unique<TaskFlowInst>(Evaluate, Mutate, Crossover,
                                                   stateFlow, options)) {}
  TaskFlowContainer(TaskFlowContainer &&other) noexcept { swap(other); }
  TaskFlowContainer(TaskFlowContainer const &other)
      : taskFlowPtr(Copy(other)) {}
  TaskFlowContainer &operator=(TaskFlowContainer &&other) & noexcept {
    swap(other);
    return *this;
  }
  TaskFlowContainer &operator=(TaskFlowContainer const &other) & {
    taskFlowPtr = Copy(other);
    return *this;
  }

  void swap(TaskFlowContainer &other) noexcept {
    std::swap(taskFlowPtr, other.taskFlowPtr);
  }

  TaskFlowInst const &Get() const & {
    assert(taskFlowPtr);
    return *taskFlowPtr;
  }
  TaskFlowInst &Get() & {
    assert(taskFlowPtr);
    return *taskFlowPtr;
  }
  operator TaskFlowInst &() & {
    assert(taskFlowPtr);
    return *taskFlowPtr;
  }
  operator TaskFlowInst const &() const & {
    assert(taskFlowPtr);
    return *taskFlowPtr;
  }

  ~TaskFlowContainer() noexcept = default;

private:
  static TaskFlowPtr Copy(TaskFlowContainer const &other) {
    auto ptr = TaskFlowPtr{};
    if (other.taskFlowPtr)
      ptr = std::make_unique<TaskFlowInst>(*other.taskFlowPtr);
    return ptr;
  }
};
} // namespace Evolution

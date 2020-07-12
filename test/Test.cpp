#define BOOST_TEST_MODULE Test

#include "Evolution/Environment.hpp"
#include "Evolution/StateFlow.hpp"
#include "Evolution/TaskFlow.hpp"
#include "Evolution/Utility.hpp"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/flow_graph.h"
#include <boost/test/included/unit_test.hpp>

using namespace Evolution;

BOOST_AUTO_TEST_CASE(first_test) {
  auto sf = StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  BOOST_TEST(!sf.Verify());
  sf.SetEvaluate(s0);
  BOOST_TEST(sf.Verify());
}

BOOST_AUTO_TEST_CASE(second_test) {
  auto sf = StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  auto s1 = sf.GetOrAddInitialState(1);
  auto s2 = sf.AddCrossover(s0, s1);
  sf.SetEvaluate(s0);
  sf.SetEvaluate(s2);

  auto Evaluate = [](int x) { return x * 1.0; };
  auto Mutate = [](int x) { return x + 1; };
  auto Crossover = [](int x, int y) { return x + y; };
  auto population = std::vector<int>{1, 1};
  auto grades = std::vector<double>{0, 0};

  auto env =
      Environment(std::move(Evaluate), std::move(Mutate), std::move(Crossover),
                  std::move(population), std::move(grades), sf);
  env.Run();
  env.Run();
  env.Run();
  population = env.GetPopulation();
  grades = env.GetGrades();
  BOOST_TEST(population.at(0) == 5);
  BOOST_TEST(population.at(1) == 3);
  BOOST_TEST(grades.at(0) == 5);
  BOOST_TEST(grades.at(1) == 3);
}

BOOST_AUTO_TEST_CASE(perm_test) {
  auto perm = std::vector<size_t>{5, 2, 3, 0, 1, 4};
  auto v = std::vector<size_t>(perm.size());
  std::iota(v.begin(), v.end(), 0);
  Permute(v, perm);
  BOOST_TEST(v == perm);
}

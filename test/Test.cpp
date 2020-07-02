#define BOOST_TEST_MODULE Test

#include "Evolution/Evolution.hpp"
#include "Evolution/StateFlow.hpp"
#include "Evolution/TBBFlow.hpp"
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

  auto flow = TBBFlow(
      std::move(Evaluate), std::move(Mutate), std::move(Crossover),
      []() { return 1; }, std::move(sf));
  flow.AllowSwapArgumentsInCrossover();
  flow.Run();
  flow.Run();
  flow.Run();
  auto population = flow.GetPopulation();
  auto grades = flow.GetGrades();
  BOOST_TEST(population.at(0) == 1);
  BOOST_TEST(population.at(1) == 4);
  BOOST_TEST(grades.at(0) == 1);
  BOOST_TEST(grades.at(1) == 4);
}

BOOST_AUTO_TEST_CASE(perm_test) {
  auto perm = std::vector<size_t>{5, 2, 3, 0, 1, 4};
  auto v = std::vector<size_t>(perm.size());
  std::iota(v.begin(), v.end(), 0);
  Permute(v, perm, std::identity{});
  BOOST_TEST(v == perm);
}

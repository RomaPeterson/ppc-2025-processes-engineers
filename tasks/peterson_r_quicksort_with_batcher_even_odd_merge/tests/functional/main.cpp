#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "peterson_r_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"
#include "peterson_r_quicksort_with_batcher_even_odd_merge/mpi/include/ops_mpi.hpp"
#include "peterson_r_quicksort_with_batcher_even_odd_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

class PetersonRQuicksortWithBatcherEvenOddMergeFuncTests
    : public ppc::util::BaseRunFuncTests<InputArray, OutputArray, TestScenario> {
 public:
  static std::string PrintTestParam(const TestScenario &test_case) {
    return test_case.first;
  }

 protected:
  void SetUp() override {
    const TestScenario params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_dataset_ = params.second;
  }

  bool CheckTestOutputData(OutputArray &actual_results) override {
    if (test_dataset_.empty()) {
      return actual_results.empty();
    }

    OutputArray expected_results = test_dataset_;
    std::ranges::sort(expected_results);

    return expected_results == actual_results;
  }

  InputArray GetTestInputData() override {
    return test_dataset_;
  }

 private:
  InputArray test_dataset_;
};

namespace {

TEST_P(PetersonRQuicksortWithBatcherEvenOddMergeFuncTests, SortingAccuracyTest) {
  ExecuteTest(GetParam());
}

const std::array<TestScenario, 12> kTestDatasets = {
    std::make_pair("empty_array", std::vector<double>{}),
    std::make_pair("single_value", std::vector<double>{42.0}),
    std::make_pair("duplicates_and_negatives", std::vector<double>{5.0, -1.0, 3.2, 3.2, 0.0}),
    std::make_pair("already_sorted", std::vector<double>{1.0, 2.0, 3.0, 4.0}),
    std::make_pair("reverse_order", std::vector<double>{4.0, 3.0, 2.0, 1.0}),
    std::make_pair("mixed_values", std::vector<double>{9.1, -7.3, 0.0, 5.5, -7.3, 2.2}),
    std::make_pair("odd_count", std::vector<double>{10.0, 3.0, 5.0, 7.0, 2.0, 8.0, 6.0}),
    std::make_pair("even_count", std::vector<double>{8.0, -2.0, 4.0, 9.0, 0.0, -5.0}),
    std::make_pair("multiple_duplicates", std::vector<double>{1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0}),
    std::make_pair("wide_range", std::vector<double>{1e9, -1e9, 5.5, -12345.6, 9999.9, 0.0}),
    std::make_pair("precision_check", std::vector<double>{0.1, 0.1000001, 0.0999999, -0.1, -0.1000001}),
    std::make_pair("extended_random",
                   std::vector<double>{12.3, -7.7, 5.5, 0.0, 2.2, 2.2, -3.3, 9.9, -1.1, 4.4, 6.6, -8.8, 7.7}),
};

const auto kTestConfiguration =
    std::tuple_cat(ppc::util::AddFuncTask<PetersonRQuicksortWithBatcherEvenOddMergeMPI, InputArray>(
                       kTestDatasets, PPC_SETTINGS_peterson_r_quicksort_with_batcher_even_odd_merge),
                   ppc::util::AddFuncTask<PetersonRQuicksortWithBatcherEvenOddMergeSEQ, InputArray>(
                       kTestDatasets, PPC_SETTINGS_peterson_r_quicksort_with_batcher_even_odd_merge));

const auto kGtestParameters = ppc::util::ExpandToValues(kTestConfiguration);

const auto kTestSuiteName = PetersonRQuicksortWithBatcherEvenOddMergeFuncTests::PrintFuncTestName<
    PetersonRQuicksortWithBatcherEvenOddMergeFuncTests>;

INSTANTIATE_TEST_SUITE_P(SortingVerification, PetersonRQuicksortWithBatcherEvenOddMergeFuncTests, kGtestParameters,
                         kTestSuiteName);

}  // namespace

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

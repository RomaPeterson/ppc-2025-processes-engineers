#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "peterson_r_qsort_batcher_oddeven_merge/common/include/common.hpp"
#include "peterson_r_qsort_batcher_oddeven_merge/mpi/include/ops_mpi.hpp"
#include "peterson_r_qsort_batcher_oddeven_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace peterson_r_qsort_batcher_oddeven_merge {

class PetersonRQsortBatcherOddEvenMergeFuncTests
    : public ppc::util::BaseRunFuncTests<InputArray, OutputArray, TestScenario> {
 public:
  static std::string FormatTestName(const TestScenario &test_case) {
    return test_case.first;
  }

 protected:
  void SetUp() override {
    const TestScenario parameters =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_dataset_ = parameters.second;
  }

  bool ValidateResults(OutputArray &actual_results) final {
    if (test_dataset_.empty()) {
      return actual_results.empty();
    }

    OutputArray expected_results = test_dataset_;
    // Используем std::sort для проверки
    std::sort(expected_results.begin(), expected_results.end());

    return expected_results == actual_results;
  }

  InputArray GetTestData() final {
    return test_dataset_;
  }

 private:
  InputArray test_dataset_;
};

namespace {

TEST_P(PetersonRQsortBatcherOddEvenMergeFuncTests, SortingAccuracyTest) {
  RunAlgorithmTest(GetParam());
}

const std::array<TestScenario, 12> test_datasets = {
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

const auto test_configuration = std::tuple_cat(ppc::util::AddFuncTask<PetersonRQsortBatcherOddEvenMergeMPI, InputArray>(
                                                   test_datasets, PPC_SETTINGS_peterson_r_qsort_batcher_oddeven_merge),
                                               ppc::util::AddFuncTask<PetersonRQsortBatcherOddEvenMergeSEQ, InputArray>(
                                                   test_datasets, PPC_SETTINGS_peterson_r_qsort_batcher_oddeven_merge));

const auto gtest_parameters = ppc::util::ExpandToValues(test_configuration);

const auto test_suite_name =
    PetersonRQsortBatcherOddEvenMergeFuncTests::PrintFuncTestName<PetersonRQsortBatcherOddEvenMergeFuncTests>;

INSTANTIATE_TEST_SUITE_P(SortingVerification, PetersonRQsortBatcherOddEvenMergeFuncTests, gtest_parameters,
                         test_suite_name);

}  // namespace

}  // namespace peterson_r_qsort_batcher_oddeven_merge

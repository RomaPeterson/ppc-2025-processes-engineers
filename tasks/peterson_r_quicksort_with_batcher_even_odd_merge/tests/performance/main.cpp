#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "peterson_r_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"
#include "peterson_r_quicksort_with_batcher_even_odd_merge/mpi/include/ops_mpi.hpp"
#include "peterson_r_quicksort_with_batcher_even_odd_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

class PetersonRQuicksortWithBatcherEvenOddMergePerfTests : public ppc::util::BaseRunPerfTests<InputArray, OutputArray> {
 private:
  InputArray performance_data_;
  OutputArray reference_result_;

  size_t base_length_ = 1000000;
  size_t replication_factor_ = 8;
  unsigned int random_seed_ = 42;

  void SetUp() override {
    performance_data_ = GeneratePerformanceData(base_length_, replication_factor_, random_seed_);
    reference_result_ = performance_data_;
    // Используем std::sort для подготовки эталона
    std::sort(reference_result_.begin(), reference_result_.end());
  }

  // Переопределяем методы базового класса BaseRunPerfTests
  bool CheckTestOutputData(OutputArray &actual_output) override {
    return (reference_result_ == actual_output);
  }

  InputArray GetTestInputData() override {
    return performance_data_;
  }

  static std::vector<double> GeneratePerformanceData(size_t base_size, size_t replication_count, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);

    std::vector<double> base_dataset(base_size);
    for (double &element : base_dataset) {
      element = distribution(generator);
    }

    std::vector<double> full_dataset;
    full_dataset.reserve(base_size * replication_count);
    for (size_t i = 0; i < replication_count; ++i) {
      full_dataset.insert(full_dataset.end(), base_dataset.begin(), base_dataset.end());
    }
    return full_dataset;
  }
};

TEST_P(PetersonRQuicksortWithBatcherEvenOddMergePerfTests, PerformanceEvaluation) {
  // Вызов метода из базового класса
  this->ExecuteTest(GetParam());
}

const auto extended_performance_tests =
    ppc::util::MakeAllPerfTasks<InputArray, PetersonRQuicksortWithBatcherEvenOddMergeMPI,
                                PetersonRQuicksortWithBatcherEvenOddMergeSEQ>(
        PPC_SETTINGS_peterson_r_quicksort_with_batcher_even_odd_merge);

const auto performance_test_values = ppc::util::TupleToGTestValues(extended_performance_tests);

const auto performance_test_name = PetersonRQuicksortWithBatcherEvenOddMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerformanceAnalysis, PetersonRQuicksortWithBatcherEvenOddMergePerfTests,
                         performance_test_values, performance_test_name);

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

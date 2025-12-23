#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

#include "peterson_r_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"
#include "peterson_r_quicksort_with_batcher_even_odd_merge/mpi/include/ops_mpi.hpp"
#include "peterson_r_quicksort_with_batcher_even_odd_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

class PetersonRQuicksortWithBatcherEvenOddMergePerfTests : public ppc::util::BaseRunPerfTests<InputArray, OutputArray> {
 private:
  InputArray input_data_;
  OutputArray expected_output_;

  std::size_t base_length_ = 1000000;
  std::size_t scale_factor_ = 8;
  unsigned int seed_ = 42;

  void SetUp() override {
    input_data_ = GenerateData(base_length_, scale_factor_, seed_);
    expected_output_ = input_data_;
    std::qsort(expected_output_.data(), expected_output_.size(), sizeof(double), [](const void *a, const void *b) {
      const double arg1 = *static_cast<const double *>(a);
      const double arg2 = *static_cast<const double *>(b);
      if (arg1 < arg2) {
        return -1;
      }
      if (arg1 > arg2) {
        return 1;
      }
      return 0;
    });
  }

  bool CheckTestOutputData(OutputArray &output_data) final {
    return expected_output_ == output_data;
  }

  InputArray GetTestInputData() final {
    return input_data_;
  }

  static std::vector<double> GenerateData(std::size_t base_length, std::size_t scale_factor, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    std::vector<double> base(base_length);
    for (double &value : base) {
      value = dist(gen);
    }

    std::vector<double> data;
    data.reserve(base_length * scale_factor);
    for (std::size_t i = 0; i < scale_factor; ++i) {
      data.insert(data.end(), base.begin(), base.end());
    }
    return data;
  }
};

TEST_P(PetersonRQuicksortWithBatcherEvenOddMergePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InputArray, PetersonRQuicksortWithBatcherEvenOddMergeMPI,
                                                       PetersonRQuicksortWithBatcherEvenOddMergeSEQ>(
    PPC_SETTINGS_peterson_r_quicksort_with_batcher_even_odd_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PetersonRQuicksortWithBatcherEvenOddMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PetersonRQuicksortWithBatcherEvenOddMergePerfTests, kGtestValues, kPerfTestName);

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

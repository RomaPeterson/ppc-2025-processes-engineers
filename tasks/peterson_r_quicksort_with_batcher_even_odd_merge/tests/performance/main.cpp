#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "peterson_r_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"
#include "peterson_r_quicksort_with_batcher_even_odd_merge/mpi/include/ops_mpi.hpp"
#include "peterson_r_quicksort_with_batcher_even_odd_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

class PetersonRQuicksortWithBatcherEvenOddMergePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 70000000;
  InType input_data_;
  OutType res_;

  void SetUp() override {
    std::vector<int> vec(kCount_);
    for (int i = 0; i < kCount_; i++) {
      vec[i] = kCount_ - i;
    }
    input_data_ = vec;
    // std::sort(vec.begin(), vec.end());
    std::ranges::sort(vec);
    res_ = vec;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return res_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PetersonRQuicksortWithBatcherEvenOddMergePerfTests, QuicksortWithBatcherEvenOddMergePerf) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PetersonRQuicksortWithBatcherEvenOddMergeMPI,
                                                       PetersonRQuicksortWithBatcherEvenOddMergeSEQ>(
    PPC_SETTINGS_peterson_r_quicksort_with_batcher_even_odd_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PetersonRQuicksortWithBatcherEvenOddMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PetersonRQuicksortWithBatcherEvenOddMergePerf,
                         PetersonRQuicksortWithBatcherEvenOddMergePerfTests, kGtestValues, kPerfTestName);

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

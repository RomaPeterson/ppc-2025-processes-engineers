#include <gtest/gtest.h>

#include "peterson_r_min_val_matrix/common/include/common.hpp"
#include "peterson_r_min_val_matrix/mpi/include/ops_mpi.hpp"
#include "peterson_r_min_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace peterson_r_min_val_matrix {

class PetersonRunPerfTest
    : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override { input_data_ = 100; }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty() &&
           output_data.size() == static_cast<size_t>(input_data_);
  }

  InType GetTestInputData() final { return input_data_; }

 private:
  InType input_data_ = 0;
};

TEST_P(PetersonRunPerfTest, RunPerfModes) { ExecuteTest(GetParam()); }

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<
    InType, PetersonRMinValMatrixMPI, PetersonRMinValMatrixSEQ>(
    PPC_SETTINGS_peterson_r_min_val_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PetersonRunPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PetersonRunPerfTest, kGtestValues,
                         kPerfTestName);

}  // namespace peterson_r_min_val_matrix

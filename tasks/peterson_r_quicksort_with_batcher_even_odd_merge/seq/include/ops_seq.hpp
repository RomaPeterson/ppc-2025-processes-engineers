#pragma once

#include "peterson_r_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

class PetersonRQuicksortWithBatcherEvenOddMergeSEQ : public CoreAlgorithm {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PetersonRQuicksortWithBatcherEvenOddMergeSEQ(const InputArray &source);

 private:
  bool VerificationStep() override;
  bool PreparationStep() override;
  bool ExecutionStep() override;
  bool CompletionStep() override;
};

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

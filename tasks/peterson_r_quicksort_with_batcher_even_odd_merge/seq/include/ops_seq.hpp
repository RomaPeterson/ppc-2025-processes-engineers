#pragma once

#include "peterson_r_qsort_batcher_oddeven_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peterson_r_qsort_batcher_oddeven_merge {

class PetersonRQsortBatcherOddEvenMergeSEQ : public CoreAlgorithm {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PetersonRQsortBatcherOddEvenMergeSEQ(const InputArray &source);

 private:
  bool VerificationStep() override;
  bool PreparationStep() override;
  bool ExecutionStep() override;
  bool CompletionStep() override;
};

}  // namespace peterson_r_qsort_batcher_oddeven_merge

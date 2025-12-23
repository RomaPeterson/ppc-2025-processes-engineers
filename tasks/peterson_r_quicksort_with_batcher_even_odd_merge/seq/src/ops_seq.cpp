#include "peterson_r_qsort_batcher_oddeven_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace peterson_r_qsort_batcher_oddeven_merge {

PetersonRQsortBatcherOddEvenMergeSEQ::PetersonRQsortBatcherOddEvenMergeSEQ(const InputArray &source) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = source;
  GetOutput() = {};
}

bool PetersonRQsortBatcherOddEvenMergeSEQ::VerificationStep() {
  return GetOutput().empty();
}

bool PetersonRQsortBatcherOddEvenMergeSEQ::PreparationStep() {
  return true;
}

bool PetersonRQsortBatcherOddEvenMergeSEQ::ExecutionStep() {
  const auto &original = GetInput();
  if (original.empty()) {
    return true;
  }

  std::vector<double> processing_buffer = original;
  // Используем современный std::sort вместо старого qsort
  std::sort(processing_buffer.begin(), processing_buffer.end());

  GetOutput() = std::move(processing_buffer);
  return true;
}

bool PetersonRQsortBatcherOddEvenMergeSEQ::CompletionStep() {
  return true;
}

}  // namespace peterson_r_qsort_batcher_oddeven_merge

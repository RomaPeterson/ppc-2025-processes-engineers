#include "peterson_r_quicksort_with_batcher_even_odd_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

PetersonRQuicksortWithBatcherEvenOddMergeSEQ::PetersonRQuicksortWithBatcherEvenOddMergeSEQ(const InputArray &source) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = source;
  GetOutput() = {};
}

bool PetersonRQuicksortWithBatcherEvenOddMergeSEQ::VerificationStep() {
  return GetOutput().empty();
}

bool PetersonRQuicksortWithBatcherEvenOddMergeSEQ::PreparationStep() {
  return true;
}

bool PetersonRQuicksortWithBatcherEvenOddMergeSEQ::ExecutionStep() {
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

bool PetersonRQuicksortWithBatcherEvenOddMergeSEQ::CompletionStep() {
  return true;
}

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

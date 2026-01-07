#include "peterson_r_quicksort_with_batcher_even_odd_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "peterson_r_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

PetersonRQuicksortWithBatcherEvenOddMergeSEQ::PetersonRQuicksortWithBatcherEvenOddMergeSEQ(const InputArray &source) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = source;
  GetOutput() = {};
}

bool PetersonRQuicksortWithBatcherEvenOddMergeSEQ::ValidationImpl() {
  return GetOutput().empty();
}

bool PetersonRQuicksortWithBatcherEvenOddMergeSEQ::PreProcessingImpl() {
  return true;
}

bool PetersonRQuicksortWithBatcherEvenOddMergeSEQ::RunImpl() {
  const auto &original = GetInput();
  if (original.empty()) {
    return true;
  }

  std::vector<double> processing_buffer = original;
  std::ranges::sort(processing_buffer);

  GetOutput() = std::move(processing_buffer);
  return true;
}

bool PetersonRQuicksortWithBatcherEvenOddMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

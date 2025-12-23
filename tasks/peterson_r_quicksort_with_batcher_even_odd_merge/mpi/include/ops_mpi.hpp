#pragma once

#include <mpi.h>

#include <utility>
#include <vector>

#include "peterson_r_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

class PetersonRQuicksortWithBatcherEvenOddMergeMPI : public CoreAlgorithm {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PetersonRQuicksortWithBatcherEvenOddMergeMPI(const InputArray &source);

 private:
  int process_id_{};
  int process_count_{};

  // Имена этих методов фиксированы в базовом классе Task!
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Внутренние методы
  void TransmitDimensions(size_t &input_size, size_t &aligned_size);
  void DispatchData(const size_t &aligned_size, const std::vector<double> &prepared_input,
                    std::vector<int> &block_sizes, std::vector<int> &block_positions,
                    std::vector<double> &local_block) const;

  void CreateComparatorSet(std::vector<std::pair<int, int>> &comparators) const;

  static void BatcherSort(std::vector<int> &procs, std::vector<std::pair<int, int>> &comparators);
  static void BatcherMerge(std::vector<int> &procs, std::vector<std::pair<int, int>> &comparators);

  void ExecuteComparisons(const std::vector<int> &block_sizes, std::vector<double> &local_block,
                          const std::vector<std::pair<int, int>> &comparators) const;

  static void CombineSegments(const std::vector<double> &local_values, const std::vector<double> &remote_values,
                              std::vector<double> &merged_result, bool select_minimum);
};

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

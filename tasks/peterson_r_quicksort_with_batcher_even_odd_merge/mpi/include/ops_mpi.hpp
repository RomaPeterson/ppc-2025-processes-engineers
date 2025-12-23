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

  // Переопределение виртуальных методов Task
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Служебные методы обмена/распределения
  void TransmitDimensions(size_t &input_size, size_t &aligned_size);
  void DispatchData(const size_t &aligned_size, const std::vector<double> &prepared_input,
                    std::vector<int> &block_sizes, std::vector<int> &block_positions,
                    std::vector<double> &local_block) const;

  // Построение сети Бэтчера (odd-even merge) – стековая реализация
  void CreateComparatorSet(std::vector<std::pair<int, int>> &comparators) const;
  static void ConstructSortSequence(const std::vector<int> &process_list,
                                    std::vector<std::pair<int, int>> &comparators);
  static void ConstructMergeSequence(const std::vector<int> &upper_group, const std::vector<int> &lower_group,
                                     std::vector<std::pair<int, int>> &comparators);
  static std::pair<std::vector<int>, std::vector<int>> SplitByParity(const std::vector<int> &elements);

  // Выполнение сети компараторов
  void ExecuteComparisons(const std::vector<int> &block_sizes, std::vector<double> &local_block,
                          const std::vector<std::pair<int, int>> &comparators) const;

  static void CombineSegments(const std::vector<double> &local_values, const std::vector<double> &remote_values,
                              std::vector<double> &merged_result, bool select_minimum);
};

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

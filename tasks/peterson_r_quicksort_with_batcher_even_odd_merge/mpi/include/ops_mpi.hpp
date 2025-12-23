#pragma once

#include <mpi.h>

#include <utility>
#include <vector>

#include "peterson_r_qsort_batcher_oddeven_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peterson_r_qsort_batcher_oddeven_merge {

class PetersonRQsortBatcherOddEvenMergeMPI : public CoreAlgorithm {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PetersonRQsortBatcherOddEvenMergeMPI(const InputArray &source);

 private:
  int process_id_{};
  int process_count_{};

  bool VerificationStep() override;
  bool PreparationStep() override;
  bool ExecutionStep() override;
  bool CompletionStep() override;

  // Методы синхронизации и распределения данных
  void TransmitDimensions(size_t &input_size, size_t &aligned_size);
  void DispatchData(const size_t &aligned_size, const std::vector<double> &prepared_input,
                    std::vector<int> &block_sizes, std::vector<int> &block_positions,
                    std::vector<double> &local_block) const;

  // Методы построения сети Бэтчера (Рекурсивная реализация)
  void CreateComparatorSet(std::vector<std::pair<int, int>> &comparators) const;

  static void BatcherSort(std::vector<int> &procs, std::vector<std::pair<int, int>> &comparators);
  static void BatcherMerge(std::vector<int> &procs, std::vector<std::pair<int, int>> &comparators);

  // Выполнение обменов
  void ExecuteComparisons(const std::vector<int> &block_sizes, std::vector<double> &local_block,
                          const std::vector<std::pair<int, int>> &comparators) const;

  static void CombineSegments(const std::vector<double> &local_values, const std::vector<double> &remote_values,
                              std::vector<double> &merged_result, bool select_minimum);
};

}  // namespace peterson_r_qsort_batcher_oddeven_merge

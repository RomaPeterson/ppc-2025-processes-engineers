#include "peterson_r_quicksort_with_batcher_even_odd_merge/mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

PetersonRQuicksortWithBatcherEvenOddMergeMPI::PetersonRQuicksortWithBatcherEvenOddMergeMPI(const InputArray &source) {
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id_);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count_);

  SetTypeOfTask(GetStaticTypeOfTask());
  if (process_id_ == 0) {
    GetInput() = source;
  }
  GetOutput() = std::vector<double>();
}

bool PetersonRQuicksortWithBatcherEvenOddMergeMPI::VerificationStep() {
  return GetOutput().empty();
}

bool PetersonRQuicksortWithBatcherEvenOddMergeMPI::PreparationStep() {
  return true;
}

bool PetersonRQuicksortWithBatcherEvenOddMergeMPI::ExecutionStep() {
  size_t original_dimension = 0;
  size_t adjusted_dimension = 0;

  // 1. Обмен размерами
  TransmitDimensions(original_dimension, adjusted_dimension);

  if (original_dimension == 0) {
    return true;
  }

  // 2. Подготовка данных (padding)
  std::vector<double> prepared_source;
  if (process_id_ == 0) {
    prepared_source = GetInput();
    if (adjusted_dimension > original_dimension) {
      prepared_source.resize(adjusted_dimension, std::numeric_limits<double>::infinity());
    }
  }

  // 3. Рассылка данных (Scatter)
  std::vector<int> counts;
  std::vector<int> displs;
  std::vector<double> local_segment;
  DispatchData(adjusted_dimension, prepared_source, counts, displs, local_segment);

  // 4. Локальная сортировка (используем std::sort)
  std::sort(local_segment.begin(), local_segment.end());

  // 5. Построение сети Бэтчера (новая рекурсивная логика)
  std::vector<std::pair<int, int>> comparator_sequence;
  CreateComparatorSet(comparator_sequence);

  // 6. Выполнение обменов по сети
  ExecuteComparisons(counts, local_segment, comparator_sequence);

  // 7. Сбор результатов (Gather)
  std::vector<double> assembled_result;
  if (process_id_ == 0) {
    assembled_result.resize(adjusted_dimension);
  }

  MPI_Gatherv(local_segment.data(), static_cast<int>(local_segment.size()), MPI_DOUBLE, assembled_result.data(),
              counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // 8. Финализация
  if (process_id_ == 0) {
    assembled_result.resize(original_dimension);
    GetOutput() = std::move(assembled_result);
  }

  // Валидация для всех процессов
  GetOutput().resize(original_dimension);
  MPI_Bcast(GetOutput().data(), static_cast<int>(original_dimension), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::TransmitDimensions(size_t &input_size, size_t &aligned_size) {
  if (process_id_ == 0) {
    input_size = GetInput().size();
    const size_t remainder = input_size % process_count_;
    aligned_size = input_size + (remainder == 0 ? 0 : (process_count_ - remainder));
  }

  MPI_Bcast(&input_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&aligned_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
}

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::DispatchData(const size_t &aligned_size,
                                                                const std::vector<double> &prepared_input,
                                                                std::vector<int> &counts, std::vector<int> &displs,
                                                                std::vector<double> &local_block) const {
  const int base_block = static_cast<int>(aligned_size / process_count_);

  counts.resize(process_count_);
  displs.resize(process_count_);

  for (int i = 0; i < process_count_; i++) {
    counts[i] = base_block;
    displs[i] = i * base_block;
  }

  const int local_dim = counts[process_id_];
  local_block.resize(local_dim);

  MPI_Scatterv(prepared_input.data(), counts.data(), displs.data(), MPI_DOUBLE, local_block.data(), local_dim,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// --- Рекурсивное построение сети (вместо стека) ---

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::CreateComparatorSet(
    std::vector<std::pair<int, int>> &comparators) const {
  std::vector<int> procs(process_count_);
  for (int i = 0; i < process_count_; i++) {
    procs[i] = i;
  }
  BatcherSort(procs, comparators);
}

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::BatcherSort(std::vector<int> &procs,
                                                               std::vector<std::pair<int, int>> &comparators) {
  if (procs.size() <= 1) {
    return;
  }

  int mid = procs.size() / 2;
  std::vector<int> left(procs.begin(), procs.begin() + mid);
  std::vector<int> right(procs.begin() + mid, procs.end());

  BatcherSort(left, comparators);
  BatcherSort(right, comparators);
  BatcherMerge(procs, comparators);
}

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::BatcherMerge(std::vector<int> &procs,
                                                                std::vector<std::pair<int, int>> &comparators) {
  int n = procs.size();
  if (n <= 1) {
    return;
  }

  if (n == 2) {
    comparators.emplace_back(procs[0], procs[1]);
    return;
  }

  std::vector<int> odd, even;
  odd.reserve(n / 2 + 1);
  even.reserve(n / 2 + 1);

  for (int i = 0; i < n; i++) {
    if (i % 2 == 0) {
      even.push_back(procs[i]);
    } else {
      odd.push_back(procs[i]);
    }
  }

  BatcherMerge(even, comparators);
  BatcherMerge(odd, comparators);

  for (int i = 1; i < n - 1; i += 2) {
    comparators.emplace_back(procs[i], procs[i + 1]);
  }
}

// --- Выполнение обменов и слияния ---

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::ExecuteComparisons(
    const std::vector<int> &counts, std::vector<double> &local_block,
    const std::vector<std::pair<int, int>> &comparators) const {
  std::vector<double> partner_buffer;
  std::vector<double> temp_buffer;

  for (const auto &comp : comparators) {
    const int p1 = comp.first;
    const int p2 = comp.second;

    if (process_id_ != p1 && process_id_ != p2) {
      continue;
    }

    const int partner = (process_id_ == p1) ? p2 : p1;
    const int my_size = counts[process_id_];
    const int partner_size = counts[partner];

    partner_buffer.resize(partner_size);
    temp_buffer.resize(my_size);

    MPI_Status status;
    MPI_Sendrecv(local_block.data(), my_size, MPI_DOUBLE, partner, 0, partner_buffer.data(), partner_size, MPI_DOUBLE,
                 partner, 0, MPI_COMM_WORLD, &status);

    // true, если мы "меньший" процесс в паре (храним меньшие значения)
    bool keep_min = (process_id_ == std::min(p1, p2));
    CombineSegments(local_block, partner_buffer, temp_buffer, keep_min);

    local_block = std::move(temp_buffer);
  }
}

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::CombineSegments(const std::vector<double> &local_values,
                                                                   const std::vector<double> &remote_values,
                                                                   std::vector<double> &merged_result,
                                                                   bool select_minimum) {
  // Создаем буфер для полного слияния
  std::vector<double> full_merge(local_values.size() + remote_values.size());

  // Используем std::merge (оптимизированное слияние)
  std::merge(local_values.begin(), local_values.end(), remote_values.begin(), remote_values.end(), full_merge.begin());

  if (select_minimum) {
    // Копируем первую половину
    std::copy(full_merge.begin(), full_merge.begin() + local_values.size(), merged_result.begin());
  } else {
    // Копируем вторую половину (с конца, чтобы учесть размер текущего процесса)
    std::copy(full_merge.end() - local_values.size(), full_merge.end(), merged_result.begin());
  }
}

bool PetersonRQuicksortWithBatcherEvenOddMergeMPI::CompletionStep() {
  return true;
}

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

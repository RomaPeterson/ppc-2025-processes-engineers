#include "peterson_r_quicksort_with_batcher_even_odd_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

#include "peterson_r_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"

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

bool PetersonRQuicksortWithBatcherEvenOddMergeMPI::ValidationImpl() {
  return GetOutput().empty();
}

bool PetersonRQuicksortWithBatcherEvenOddMergeMPI::PreProcessingImpl() {
  return true;
}

bool PetersonRQuicksortWithBatcherEvenOddMergeMPI::RunImpl() {
  std::size_t original_dimension = 0;
  std::size_t adjusted_dimension = 0;

  // [1] Обмен размерами
  TransmitDimensions(original_dimension, adjusted_dimension);

  if (original_dimension == 0) {
    return true;
  }

  // [2] Подготовка данных (padding до кратности числу процессов)
  std::vector<double> prepared_source;
  if (process_id_ == 0) {
    prepared_source = GetInput();
    if (adjusted_dimension > original_dimension) {
      prepared_source.resize(adjusted_dimension, std::numeric_limits<double>::infinity());
    }
  }

  // [3] Scatter
  std::vector<int> block_sizes;
  std::vector<int> block_positions;
  std::vector<double> local_segment;
  DispatchData(adjusted_dimension, prepared_source, block_sizes, block_positions, local_segment);

  // [4] Локальная сортировка
  std::ranges::sort(local_segment);

  // [5] Сеть компараторов Бэтчера
  std::vector<std::pair<int, int>> comparator_sequence;
  CreateComparatorSet(comparator_sequence);

  // [6] Выполнение сети
  ExecuteComparisons(block_sizes, local_segment, comparator_sequence);

  // [7] Сбор результата
  std::vector<double> assembled_result;
  if (process_id_ == 0) {
    assembled_result.resize(adjusted_dimension);
  }

  MPI_Gatherv(local_segment.data(), static_cast<int>(local_segment.size()), MPI_DOUBLE, assembled_result.data(),
              block_sizes.data(), block_positions.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // [8] Усечение padding и рассылка результата
  if (process_id_ == 0) {
    assembled_result.resize(original_dimension);
    GetOutput() = std::move(assembled_result);
  }

  GetOutput().resize(original_dimension);
  MPI_Bcast(GetOutput().data(), static_cast<int>(original_dimension), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::TransmitDimensions(std::size_t &input_size,
                                                                      std::size_t &aligned_size) {
  if (process_id_ == 0) {
    input_size = GetInput().size();
    const std::size_t remainder = input_size % static_cast<std::size_t>(process_count_);
    aligned_size = input_size + (remainder == 0 ? 0 : (static_cast<std::size_t>(process_count_) - remainder));
  }

  MPI_Bcast(&input_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&aligned_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
}

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::DispatchData(const std::size_t &aligned_size,
                                                                const std::vector<double> &prepared_input,
                                                                std::vector<int> &block_sizes,
                                                                std::vector<int> &block_positions,
                                                                std::vector<double> &local_block) const {
  const int base_block = static_cast<int>(aligned_size / static_cast<std::size_t>(process_count_));

  block_sizes.resize(static_cast<std::size_t>(process_count_));
  block_positions.resize(static_cast<std::size_t>(process_count_));

  for (int i = 0; i < process_count_; i++) {
    block_sizes[static_cast<std::size_t>(i)] = base_block;
    block_positions[static_cast<std::size_t>(i)] = i * base_block;
  }

  const int local_dim = block_sizes[static_cast<std::size_t>(process_id_)];
  local_block.resize(static_cast<std::size_t>(local_dim));

  MPI_Scatterv(prepared_input.data(), block_sizes.data(), block_positions.data(), MPI_DOUBLE, local_block.data(),
               local_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// -------------------- Построение сети Бэтчера --------------------

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::CreateComparatorSet(
    std::vector<std::pair<int, int>> &comparators) const {
  std::vector<int> process_ids(static_cast<std::size_t>(process_count_));
  for (int i = 0; i < process_count_; i++) {
    process_ids[static_cast<std::size_t>(i)] = i;
  }

  ConstructSortSequence(process_ids, comparators);
}

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::ConstructSortSequence(
    const std::vector<int> &process_list, std::vector<std::pair<int, int>> &comparators) {
  // Стек задач: (подмножество процессов, флаг "это стадия слияния?")
  std::stack<std::pair<std::vector<int>, bool>> task_stack;
  task_stack.emplace(process_list, false);

  while (!task_stack.empty()) {
    auto [current, merge_flag] = task_stack.top();
    task_stack.pop();

    if (current.size() <= 1) {
      continue;
    }

    const std::size_t mid = current.size() / 2;
    std::vector<int> left_group(current.begin(), current.begin() + static_cast<std::ptrdiff_t>(mid));
    std::vector<int> right_group(current.begin() + static_cast<std::ptrdiff_t>(mid), current.end());

    if (merge_flag) {
      ConstructMergeSequence(left_group, right_group, comparators);
      continue;
    }

    // Сначала рекурсивно сортируем левую/правую части, затем их сливаем
    task_stack.emplace(current, true);
    task_stack.emplace(right_group, false);
    task_stack.emplace(left_group, false);
  }
}

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::ConstructMergeSequence(
    const std::vector<int> &upper_group, const std::vector<int> &lower_group,
    std::vector<std::pair<int, int>> &comparators) {
  // Стек задач: (верхняя часть, нижняя часть, флаг "фаза слияния?")
  std::stack<std::tuple<std::vector<int>, std::vector<int>, bool>> task_stack;
  task_stack.emplace(upper_group, lower_group, false);

  while (!task_stack.empty()) {
    auto [upper_part, lower_part, merge_flag] = task_stack.top();
    task_stack.pop();

    const std::size_t total_count = upper_part.size() + lower_part.size();

    if (total_count <= 1) {
      continue;
    }
    if (total_count == 2) {
      comparators.emplace_back(upper_part[0], lower_part[0]);
      continue;
    }

    if (!merge_flag) {
      auto [upper_odd, upper_even] = SplitByParity(upper_part);
      auto [lower_odd, lower_even] = SplitByParity(lower_part);

      task_stack.emplace(upper_part, lower_part, true);
      task_stack.emplace(upper_even, lower_even, false);
      task_stack.emplace(upper_odd, lower_odd, false);
      continue;
    }

    std::vector<int> combined;
    combined.reserve(total_count);
    combined.insert(combined.end(), upper_part.begin(), upper_part.end());
    combined.insert(combined.end(), lower_part.begin(), lower_part.end());

    for (std::size_t i = 1; i + 1 < combined.size(); i += 2) {
      comparators.emplace_back(combined[i], combined[i + 1]);
    }
  }
}

std::pair<std::vector<int>, std::vector<int>> PetersonRQuicksortWithBatcherEvenOddMergeMPI::SplitByParity(
    const std::vector<int> &elements) {
  std::vector<int> odd_elements;
  std::vector<int> even_elements;
  odd_elements.reserve((elements.size() / 2) + 1);
  even_elements.reserve((elements.size() / 2) + 1);

  for (std::size_t i = 0; i < elements.size(); i++) {
    if ((i % 2U) == 0U) {
      even_elements.push_back(elements[i]);
    } else {
      odd_elements.push_back(elements[i]);
    }
  }
  return std::make_pair(std::move(odd_elements), std::move(even_elements));
}

// -------------------- Выполнение сети компараторов --------------------

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::ExecuteComparisons(
    const std::vector<int> &block_sizes, std::vector<double> &local_block,
    const std::vector<std::pair<int, int>> &comparators) const {
  for (const auto &comp : comparators) {
    const int p1 = comp.first;
    const int p2 = comp.second;

    if (process_id_ != p1 && process_id_ != p2) {
      continue;
    }

    const int partner = (process_id_ == p1) ? p2 : p1;
    const auto my_size = static_cast<std::size_t>(block_sizes[static_cast<std::size_t>(process_id_)]);
    const auto partner_size = static_cast<std::size_t>(block_sizes[static_cast<std::size_t>(partner)]);

    std::vector<double> partner_buffer(partner_size);
    std::vector<double> temp_buffer(my_size);

    MPI_Status status{};
    MPI_Sendrecv(local_block.data(), static_cast<int>(my_size), MPI_DOUBLE, partner, 0, partner_buffer.data(),
                 static_cast<int>(partner_size), MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, &status);

    // ВАЖНО: направление выбирается по ПЕРВОМУ элементу пары, как в оригинальном решении
    const bool keep_min = (process_id_ == p1);
    CombineSegments(local_block, partner_buffer, temp_buffer, keep_min);

    local_block = std::move(temp_buffer);
  }
}

void PetersonRQuicksortWithBatcherEvenOddMergeMPI::CombineSegments(const std::vector<double> &local_values,
                                                                   const std::vector<double> &remote_values,
                                                                   std::vector<double> &merged_result,
                                                                   bool select_minimum) {
  const std::size_t local_size = local_values.size();
  std::vector<double> full_merge(local_size + remote_values.size());

  std::ranges::merge(local_values, remote_values, full_merge.begin());

  const auto local_offset = static_cast<std::ptrdiff_t>(local_size);

  if (select_minimum) {
    std::copy(full_merge.begin(), full_merge.begin() + local_offset, merged_result.begin());
  } else {
    std::copy(full_merge.end() - local_offset, full_merge.end(), merged_result.begin());
  }
}

bool PetersonRQuicksortWithBatcherEvenOddMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

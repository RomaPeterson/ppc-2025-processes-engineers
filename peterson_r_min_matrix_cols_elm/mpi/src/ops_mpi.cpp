#include "peterson_r_min_matrix_cols_elm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "peterson_r_min_matrix_cols_elm/common/include/common.hpp"

namespace peterson_r_min_matrix_cols_elm {

PetersonRMinMatrixColsElmMPI::PetersonRMinMatrixColsElmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool PetersonRMinMatrixColsElmMPI::ValidationImpl() {
  const std::size_t m = std::get<0>(GetInput());
  const std::size_t n = std::get<1>(GetInput());
  const std::vector<int> &val = std::get<2>(GetInput());

  const bool has_positive_shape = (m > 0) && (n > 0);
  const bool size_matches = has_positive_shape && (val.size() / n == m) && (val.size() % n == 0);
  const bool fits_mpi_count = n <= static_cast<std::size_t>(std::numeric_limits<int>::max());

  valid_ = has_positive_shape && size_matches && fits_mpi_count;
  return valid_;
}

bool PetersonRMinMatrixColsElmMPI::PreProcessingImpl() {
  return valid_;
}

bool PetersonRMinMatrixColsElmMPI::RunImpl() {
  if (!valid_) {
    return false;
  }

  const std::size_t m = std::get<0>(GetInput());
  const std::size_t n = std::get<1>(GetInput());
  const std::vector<int> &val = std::get<2>(GetInput());

  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  const std::size_t base_rows = m / static_cast<std::size_t>(mpi_size);
  const std::size_t remainder = m % static_cast<std::size_t>(mpi_size);
  const auto rank_u = static_cast<std::size_t>(rank);
  const std::size_t local_rows = base_rows + ((rank_u < remainder) ? 1U : 0U);
  const std::size_t start_row = (base_rows * rank_u) + std::min(remainder, rank_u);

  std::vector<int> local_min(n, std::numeric_limits<int>::max());
  const std::size_t start_index = start_row * n;
  // Равномерно распределяем строки по рангам и собираем минимумы столбцов локально.
  for (std::size_t row = 0; row < local_rows; row++) {
    const std::size_t offset = start_index + (row * n);
    for (std::size_t col = 0; col < n; col++) {
      const int current = val[offset + col];
      local_min[col] = std::min(local_min[col], current);
    }
  }

  std::vector<int> global_min(n, std::numeric_limits<int>::max());
  MPI_Reduce(local_min.data(), global_min.data(), static_cast<int>(n), MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  // Рассылаем результаты редукции, чтобы все ранги вернули единый ответ.
  MPI_Bcast(global_min.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = std::move(global_min);

  return true;
}

bool PetersonRMinMatrixColsElmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace peterson_r_min_matrix_cols_elm

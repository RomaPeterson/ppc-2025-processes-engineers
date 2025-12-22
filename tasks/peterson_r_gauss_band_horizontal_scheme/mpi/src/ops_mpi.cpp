#include "peterson_r_gauss_band_horizontal_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "peterson_r_gauss_band_horizontal_scheme/common/include/common.hpp"

namespace peterson_r_gauss_band_horizontal_scheme {

PetersonRGaussBandHorizontalSchemeMPI::PetersonRGaussBandHorizontalSchemeMPI(const InType &input_data) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetInput() = input_data;
  }

  GetOutput().clear();
}

bool PetersonRGaussBandHorizontalSchemeMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int validation_status = 1;
  if (rank == 0) {
    validation_status = ValidateInputData(GetInput());
  }

  MPI_Bcast(&validation_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return validation_status != 0;
}

int PetersonRGaussBandHorizontalSchemeMPI::ValidateInputData(const InType &input_data) {
  if (input_data.empty()) {
    return 0;
  }

  const size_t n = input_data.size();
  const size_t cols = input_data[0].size();
  if (cols < n + 1) {
    return 0;
  }

  for (size_t i = 1; i < n; ++i) {
    if (input_data[i].size() != cols) {
      return 0;
    }
  }

  return 1;
}

bool PetersonRGaussBandHorizontalSchemeMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool PetersonRGaussBandHorizontalSchemeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  InType augmented_matrix;
  size_t n = 0;
  size_t cols = 0;

  if (rank == 0) {
    augmented_matrix = GetInput();
    n = augmented_matrix.size();
    cols = augmented_matrix[0].size();
  }

  // Рассылаем размеры матрицы всем процессам
  int n_int = static_cast<int>(n);
  int cols_int = static_cast<int>(cols);
  MPI_Bcast(&n_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

  n = static_cast<size_t>(n_int);
  cols = static_cast<size_t>(cols_int);

  InType local_matrix;
  std::vector<int> map_global_to_local(n, -1);

  SplitRows(augmented_matrix, n, cols, rank, size, local_matrix, map_global_to_local);

  if (!ForwardEliminationMPI(local_matrix, map_global_to_local, n, cols, rank, size)) {
    return false;
  }

  GetOutput() = BackSubstitutionMPI(local_matrix, map_global_to_local, n, cols, rank, size);
  return true;
}

void PetersonRGaussBandHorizontalSchemeMPI::SplitRows(const InType &matrix, size_t n, size_t cols, int rank, int size,
                                                      InType &local_matrix, std::vector<int> &map_global_to_local) {
  // Вычисляем, сколько строк будет у текущего процесса
  int local_count = 0;
  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(i) % size == rank) {
      ++local_count;
    }
  }

  local_matrix.assign(local_count, std::vector<double>(cols));
  int local_index = 0;

  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(i) % size == rank) {
      map_global_to_local[i] = local_index;
      if (rank == 0) {
        local_matrix[local_index] = matrix[i];
      } else {
        MPI_Recv(local_matrix[local_index].data(), static_cast<int>(cols), MPI_DOUBLE, 0, static_cast<int>(i),
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      ++local_index;
    } else if (rank == 0) {
      MPI_Send(matrix[i].data(), static_cast<int>(cols), MPI_DOUBLE, static_cast<int>(i) % size, static_cast<int>(i),
               MPI_COMM_WORLD);
    }
  }
}

bool PetersonRGaussBandHorizontalSchemeMPI::ForwardEliminationMPI(InType &local_matrix,
                                                                  const std::vector<int> &map_global_to_local, size_t n,
                                                                  size_t cols, int rank, int size) {
  for (size_t k = 0; k < n; ++k) {
    const int owner = static_cast<int>(k) % size;

    std::vector<double> pivot_row(cols);
    if (rank == owner) {
      pivot_row = local_matrix[map_global_to_local[k]];
    }

    MPI_Bcast(pivot_row.data(), static_cast<int>(cols), MPI_DOUBLE, owner, MPI_COMM_WORLD);

    if (std::abs(pivot_row[k]) < 1e-10) {
      return false;
    }

    EliminateMPI(local_matrix, map_global_to_local, k, n, cols, pivot_row);
  }
  return true;
}

void PetersonRGaussBandHorizontalSchemeMPI::EliminateMPI(InType &local_matrix,
                                                         const std::vector<int> &map_global_to_local, size_t pivot,
                                                         size_t n, size_t cols, const std::vector<double> &pivot_row) {
  for (size_t i = 0; i < local_matrix.size(); ++i) {
    const size_t global_index = ResolveGlobalIndex(map_global_to_local, i, n);
    if (global_index > pivot && std::abs(local_matrix[i][pivot]) > 1e-10) {
      const double factor = local_matrix[i][pivot] / pivot_row[pivot];
      for (size_t j = pivot; j < cols; ++j) {
        local_matrix[i][j] -= factor * pivot_row[j];
      }
    }
  }
}

size_t PetersonRGaussBandHorizontalSchemeMPI::ResolveGlobalIndex(const std::vector<int> &map_global_to_local,
                                                                 size_t local_index, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const int mapped_value = map_global_to_local[i];
    // Используем std::cmp_equal для безопасного сравнения знакового и беззнакового целого
    if (mapped_value >= 0 && std::cmp_equal(static_cast<size_t>(mapped_value), local_index)) {
      return i;
    }
  }
  return 0;
}

std::vector<double> PetersonRGaussBandHorizontalSchemeMPI::BackSubstitutionMPI(
    const InType &local_matrix, const std::vector<int> &map_global_to_local, size_t n, size_t cols, int rank,
    int size) {
  std::vector<double> result(n, 0.0);

  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    double sum = 0.0;
    const int owner = i % size;

    if (rank == owner) {
      const int local_index = map_global_to_local[static_cast<size_t>(i)];
      for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
        sum += local_matrix[static_cast<size_t>(local_index)][j] * result[j];
      }
      result[static_cast<size_t>(i)] = (local_matrix[static_cast<size_t>(local_index)][cols - 1] - sum) /
                                       local_matrix[static_cast<size_t>(local_index)][static_cast<size_t>(i)];
    }

    MPI_Bcast(&result[static_cast<size_t>(i)], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
  }

  return result;
}

bool PetersonRGaussBandHorizontalSchemeMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank == 0 ? !GetOutput().empty() : true;
}

}  // namespace peterson_r_gauss_band_horizontal_scheme

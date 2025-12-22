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
  std::vector<size_t> local_to_global;  // обратное отображение

  SplitRows(augmented_matrix, n, cols, rank, size, local_matrix, map_global_to_local, local_to_global);

  if (!ForwardEliminationMPI(local_matrix, map_global_to_local, n, cols, rank, size)) {
    return false;
  }

  GetOutput() = BackSubstitutionMPI(local_matrix, map_global_to_local, n, cols, rank, size);
  return true;
}

void PetersonRGaussBandHorizontalSchemeMPI::SplitRows(const InType &matrix, size_t n, size_t cols, int rank, int size,
                                                      InType &local_matrix, std::vector<int> &map_global_to_local,
                                                      std::vector<size_t> &local_to_global) {
  // Вычисляем, сколько строк будет у текущего процесса
  int local_count = 0;
  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(i) % size == rank) {
      ++local_count;
    }
  }

  local_matrix.assign(local_count, std::vector<double>(cols));
  local_to_global.clear();
  local_to_global.reserve(local_count);

  // Процесс 0 распределяет строки
  if (rank == 0) {
    for (size_t i = 0; i < n; ++i) {
      int dest_rank = static_cast<int>(i) % size;
      if (dest_rank == 0) {
        // Строка остается у процесса 0
        int local_idx = map_global_to_local[i];
        local_matrix[local_idx] = matrix[i];
        local_to_global.push_back(i);
      } else {
        // Отправляем строку другому процессу
        MPI_Send(matrix[i].data(), static_cast<int>(cols), MPI_DOUBLE, dest_rank, static_cast<int>(i), MPI_COMM_WORLD);
      }
    }
  } else {
    // Принимаем свои строки от процесса 0
    for (size_t i = rank; i < n; i += size) {
      int local_idx = map_global_to_local[i];
      MPI_Recv(local_matrix[local_idx].data(), static_cast<int>(cols), MPI_DOUBLE, 0, static_cast<int>(i),
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_to_global.push_back(i);
    }
  }

  // Заполняем map_global_to_local
  for (size_t i = 0; i < local_to_global.size(); ++i) {
    map_global_to_local[local_to_global[i]] = static_cast<int>(i);
  }
}

bool PetersonRGaussBandHorizontalSchemeMPI::ForwardEliminationMPI(InType &local_matrix,
                                                                  const std::vector<int> &map_global_to_local, size_t n,
                                                                  size_t cols, int rank, int size) {
  for (size_t k = 0; k < n; ++k) {
    int owner = static_cast<int>(k) % size;
    std::vector<double> pivot_row(cols);

    // Владелец строки k подготавливает ее для рассылки
    if (rank == owner) {
      int local_idx = map_global_to_local[k];
      if (local_idx >= 0) {
        pivot_row = local_matrix[static_cast<size_t>(local_idx)];
      }
    }

    // Рассылаем pivot_row всем процессам
    MPI_Bcast(pivot_row.data(), static_cast<int>(cols), MPI_DOUBLE, owner, MPI_COMM_WORLD);

    // Проверяем, что ведущий элемент не слишком мал
    if (std::abs(pivot_row[k]) < 1e-10) {
      return false;
    }

    // Выполняем исключение для локальных строк
    for (size_t i = 0; i < local_matrix.size(); ++i) {
      size_t global_i = local_to_global[i];  // используем предвычисленное отображение
      if (global_i > k && std::abs(local_matrix[i][k]) > 1e-10) {
        double factor = local_matrix[i][k] / pivot_row[k];
        for (size_t j = k; j < cols; ++j) {
          local_matrix[i][j] -= factor * pivot_row[j];
        }
      }
    }
  }
  return true;
}

size_t PetersonRGaussBandHorizontalSchemeMPI::ResolveGlobalIndex(const std::vector<int> &map_global_to_local,
                                                                 size_t local_index, size_t n) {
  // Более эффективная реализация через предвычисленный вектор
  if (local_index < local_to_global.size()) {
    return local_to_global[local_index];
  }
  return n;  // возвращаем n в случае ошибки
}

std::vector<double> PetersonRGaussBandHorizontalSchemeMPI::BackSubstitutionMPI(
    const InType &local_matrix, const std::vector<int> &map_global_to_local, size_t n, size_t cols, int rank,
    int size) {
  std::vector<double> result(n, 0.0);

  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    double sum = 0.0;
    int owner = i % size;

    if (rank == owner) {
      int local_idx = map_global_to_local[static_cast<size_t>(i)];
      if (local_idx >= 0) {
        for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
          sum += local_matrix[static_cast<size_t>(local_idx)][j] * result[j];
        }
        result[static_cast<size_t>(i)] = (local_matrix[static_cast<size_t>(local_idx)][cols - 1] - sum) /
                                         local_matrix[static_cast<size_t>(local_idx)][static_cast<size_t>(i)];
      }
    }

    // Рассылаем вычисленное значение всем процессам
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

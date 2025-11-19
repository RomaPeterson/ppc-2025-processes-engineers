#include "peterson_r_min_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "peterson_r_min_val_matrix/common/include/common.hpp"

namespace peterson_r_min_val_matrix {

PetersonRMinValMatrixMPI::PetersonRMinValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool PetersonRMinValMatrixMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput().empty());
}

bool PetersonRMinValMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool PetersonRMinValMatrixMPI::RunImpl() {
  auto input = GetInput();
  if (input == 0) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = input;

  // Распределение работы между процессами
  const int chunk_size = (n + size - 1) / size;
  const int start = rank * chunk_size;
  const int end = std::min(start + chunk_size, n);

  std::vector<int> local_result;
  local_result.reserve(chunk_size);

  // Каждый процесс вычисляет свою часть столбцов
  for (int j = start; j < end; ++j) {
    int min_val = j + 1;
    for (int i = 1; i < n; ++i) {
      const int val = i * n + j + 1;
      if (val < min_val) {
        min_val = val;
      }
    }
    local_result.push_back(min_val);
  }

  // Сбор результатов на процессе 0
  if (rank == 0) {
    GetOutput().resize(n);
  }

  // Собираем количество элементов от каждого процесса
  const int local_size = static_cast<int>(local_result.size());
  std::vector<int> recv_counts(size);
  MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  // Вычисляем смещения для Gatherv
  std::vector<int> displs(size, 0);
  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
  }

  // Собираем все результаты
  MPI_Gatherv(local_result.data(), local_size, MPI_INT, GetOutput().data(),
              recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  // Рассылаем полный результат всем процессам
  MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool PetersonRMinValMatrixMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace peterson_r_min_val_matrix

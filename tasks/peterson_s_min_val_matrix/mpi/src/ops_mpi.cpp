#include "peterson_s_min_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "peterson_s_min_val_matrix/common/include/common.hpp"

namespace peterson_s_min_val_matrix {

PetersonSMinValMatrixMPI::PetersonSMinValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool PetersonSMinValMatrixMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput().empty());
}

bool PetersonSMinValMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().reserve(GetInput());
  return true;
}

bool PetersonSMinValMatrixMPI::RunImpl() {
  InType n = GetInput();
  if (n == 0) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int cols_per_proc = n / size;
  int remainder = n % size;
  int start_col = rank * cols_per_proc + std::min(rank, remainder);
  int num_local_cols = cols_per_proc + (rank < remainder ? 1 : 0);

  std::vector<InType> local_mins;
  local_mins.reserve(num_local_cols);

  for (int j = start_col; j < start_col + num_local_cols; ++j) {
    InType min_val = j + 1;
    for (int i = 1; i < n; ++i) {
      InType current_val = i * n + j + 1;
      min_val = std::min(min_val, current_val);
    }
    local_mins.push_back(min_val);
  }

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  for (int i = 0; i < size; ++i) {
    int proc_cols = cols_per_proc + (i < remainder ? 1 : 0);
    recvcounts[i] = proc_cols;
    displs[i] = i * cols_per_proc + std::min(i, remainder);
  }

  GetOutput().resize(n);

  if (rank == 0) {
    MPI_Gatherv(local_mins.data(), num_local_cols, MPI_INT, GetOutput().data(),
                recvcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(local_mins.data(), num_local_cols, MPI_INT, nullptr,
                recvcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
  }

  MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool PetersonSMinValMatrixMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace peterson_s_min_val_matrix

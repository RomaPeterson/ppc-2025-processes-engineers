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
  if (n == 0) return false;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int cols_per_proc = n / size;
  int remainder = n % size;
  int start_col = rank * cols_per_proc + (rank < remainder ? rank : remainder);
  int num_local_cols = cols_per_proc + (rank < remainder ? 1 : 0);

  std::vector<InType> local_mins(num_local_cols);

  for (int j = 0; j < num_local_cols; ++j) {
    int col_idx = start_col + j;
    InType min_val = col_idx + 1;
    for (int i = 1; i < n; ++i) {
      InType val = i * n + col_idx + 1;
      if (val < min_val) min_val = val;
    }
    local_mins[j] = min_val;
  }

  std::vector<int> recvcounts(size), displs(size);
  for (int i = 0; i < size; ++i) {
    recvcounts[i] = cols_per_proc + (i < remainder ? 1 : 0);
    displs[i] = i * cols_per_proc + (i < remainder ? i : remainder);
  }

  GetOutput().resize(n);

  MPI_Gatherv(local_mins.data(), num_local_cols, MPI_INT,
              rank == 0 ? GetOutput().data() : nullptr, recvcounts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool PetersonSMinValMatrixMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace peterson_s_min_val_matrix

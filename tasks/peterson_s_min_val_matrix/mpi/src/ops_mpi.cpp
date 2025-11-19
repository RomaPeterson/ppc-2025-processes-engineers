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
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return GetOutput().empty();
  }
  return true;
}

bool PetersonSMinValMatrixMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    GetOutput().clear();
  }
  return true;
}

bool PetersonSMinValMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = GetInput();
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    if (rank == 0) {
      GetOutput().clear();
    }
    return true;
  }

  int chunk_size = n / size;
  int remainder = n % size;

  int start_col = (rank * chunk_size) + std::min(rank, remainder);
  int end_col = start_col + chunk_size + (rank < remainder ? 1 : 0);

  std::vector<int> local_mins;
  for (int j = start_col; j < end_col; ++j) {
    int min_val = j + 1;
    for (int i = 1; i < n; ++i) {
      int current_val = (i * n) + j + 1;
      if (current_val < min_val) {
        min_val = current_val;
      }
    }
    local_mins.push_back(min_val);
  }

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; ++i) {
    int proc_cols = chunk_size + (i < remainder ? 1 : 0);
    recvcounts[i] = proc_cols;
    displs[i] = (i * chunk_size) + std::min(i, remainder);
  }

  std::vector<int> global_mins(n);
  MPI_Gatherv(local_mins.data(), static_cast<int>(local_mins.size()), MPI_INT,
              global_mins.data(), recvcounts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  MPI_Bcast(global_mins.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_mins;
  return true;
}

bool PetersonSMinValMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace peterson_s_min_val_matrix

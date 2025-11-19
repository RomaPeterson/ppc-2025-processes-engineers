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
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return GetOutput().empty();
  }
  return true;
}

bool PetersonRMinValMatrixMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().clear();
  }
  return true;
}

bool PetersonRMinValMatrixMPI::RunImpl() {
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

  int chunk = n / size;
  int rem = n % size;
  int start = rank * chunk + std::min(rank, rem);
  int end = start + chunk + (rank < rem ? 1 : 0);

  std::vector<int> local;
  for (int j = start; j < end; ++j) {
    int min_val = j + 1;
    for (int i = 1; i < n; ++i) {
      int val = i * n + j + 1;
      min_val = std::min(min_val, val);
    }
    local.push_back(min_val);
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; ++i) {
    counts[i] = chunk + (i < rem ? 1 : 0);
    displs[i] = i * chunk + std::min(i, rem);
  }

  std::vector<int> result(n);
  MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_INT,
              result.data(), counts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  MPI_Bcast(result.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = result;
  return true;
}

bool PetersonRMinValMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace peterson_r_min_val_matrix

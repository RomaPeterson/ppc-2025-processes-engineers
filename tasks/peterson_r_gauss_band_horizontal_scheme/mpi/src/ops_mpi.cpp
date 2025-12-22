roma, [22.12.2025 16:43]
#include "peterson_r_gauss_band_horizontal_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <cmath>
#include <vector>

namespace peterson_r_gauss_band_horizontal_scheme {

PetersonRGaussBandHorizontalSchemeMPI::PetersonRGaussBandHorizontalSchemeMPI(const InType& inputData) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int petersonRomanRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &petersonRomanRank);

  if (petersonRomanRank == 0) {
    GetInput() = inputData;
  }

  GetOutput().clear();
}

bool PetersonRGaussBandHorizontalSchemeMPI::ValidationImpl() {
  int petersonRomanRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &petersonRomanRank);

  int validationStatus = 1;
  if (petersonRomanRank == 0) {
    validationStatus = ValidateInputData(GetInput());
  }

  MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return validationStatus != 0;
}

int PetersonRGaussBandHorizontalSchemeMPI::ValidateInputData(const InType& inputData) {
  if (inputData.empty()) return 0;

  size_t n = inputData.size();
  size_t cols = inputData[0].size();
  if (cols < n + 1) return 0;

  for (size_t i = 1; i < n; ++i) {
    if (inputData[i].size() != cols) return 0;
  }

  return 1;
}

bool PetersonRGaussBandHorizontalSchemeMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool PetersonRGaussBandHorizontalSchemeMPI::RunImpl() {
  int petersonRomanRank = 0;
  int petersonRomanSize = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &petersonRomanRank);
  MPI_Comm_size(MPI_COMM_WORLD, &petersonRomanSize);

  InType petersonRomanMatrix;
  size_t petersonRomanN = 0;
  size_t petersonRomanCols = 0;

  if (petersonRomanRank == 0) {
    petersonRomanMatrix = GetInput();
    petersonRomanN = petersonRomanMatrix.size();
    petersonRomanCols = petersonRomanMatrix[0].size();
  }

  int nInt = static_cast<int>(petersonRomanN);
  int cInt = static_cast<int>(petersonRomanCols);
  MPI_Bcast(&nInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

  petersonRomanN = static_cast<size_t>(nInt);
  petersonRomanCols = static_cast<size_t>(cInt);

  InType localMatrix;
  std::vector<int> mapGlobalToLocal(petersonRomanN, -1);

  SplitRows(petersonRomanMatrix, petersonRomanN, petersonRomanCols,
            petersonRomanRank, petersonRomanSize,
            localMatrix, mapGlobalToLocal);

  if (!ForwardEliminationMPI(localMatrix, mapGlobalToLocal,
                             petersonRomanN, petersonRomanCols,
                             petersonRomanRank, petersonRomanSize)) {
    return false;
  }

  GetOutput() = BackSubstitutionMPI(localMatrix, mapGlobalToLocal,
                                   petersonRomanN, petersonRomanCols,
                                   petersonRomanRank, petersonRomanSize);
  return true;
}

void PetersonRGaussBandHorizontalSchemeMPI::SplitRows(
    const InType& matrix, size_t n, size_t cols,
    int rank, int size, InType& localMatrix,
    std::vector<int>& mapGlobalToLocal) {

  int localCount = 0;
  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(i) % size == rank) {
      ++localCount;
    }
  }

  localMatrix.assign(localCount, std::vector<double>(cols));
  int localIndex = 0;

  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(i) % size == rank) {
      mapGlobalToLocal[i] = localIndex;
      if (rank == 0) {
        localMatrix[localIndex] = matrix[i];
      } else {
        MPI_Recv(localMatrix[localIndex].data(), static_cast<int>(cols),
                 MPI_DOUBLE, 0, static_cast<int>(i),
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      ++localIndex;
    } else if (rank == 0) {
      MPI_Send(matrix[i].data(), static_cast<int>(cols),
               MPI_DOUBLE, static_cast<int>(i) % size,
               static_cast<int>(i), MPI_COMM_WORLD);
    }
  }
}

bool PetersonRGaussBandHorizontalSchemeMPI::ForwardEliminationMPI(
    InType& localMatrix, const std::vector<int>& mapGlobalToLocal,
    size_t n, size_t cols, int rank, int size) {

  for (size_t k = 0; k < n; ++k) {
    int owner = static_cast<int>(k) % size;

std::vector<double> pivotRow(cols);

    if (rank == owner) {
      pivotRow = localMatrix[mapGlobalToLocal[k]];
    }

    MPI_Bcast(pivotRow.data(), static_cast<int>(cols),
              MPI_DOUBLE, owner, MPI_COMM_WORLD);

    if (std::abs(pivotRow[k]) < 1e-10) return false;

    EliminateMPI(localMatrix, mapGlobalToLocal, k, n, cols, pivotRow);
  }
  return true;
}

void PetersonRGaussBandHorizontalSchemeMPI::EliminateMPI(
    InType& localMatrix, const std::vector<int>& mapGlobalToLocal,
    size_t pivot, size_t n, size_t cols,
    const std::vector<double>& pivotRow) {

  for (size_t i = 0; i < localMatrix.size(); ++i) {
    size_t globalIdx = ResolveGlobalIndex(mapGlobalToLocal, i, n);
    if (globalIdx > pivot && std::abs(localMatrix[i][pivot]) > 1e-10) {
      double factor = localMatrix[i][pivot] / pivotRow[pivot];
      for (size_t j = pivot; j < cols; ++j) {
        localMatrix[i][j] -= factor * pivotRow[j];
      }
    }
  }
}

size_t PetersonRGaussBandHorizontalSchemeMPI::ResolveGlobalIndex(
    const std::vector<int>& mapGlobalToLocal,
    size_t localIndex, size_t n) {

  for (size_t i = 0; i < n; ++i) {
    if (mapGlobalToLocal[i] == static_cast<int>(localIndex)) return i;
  }
  return 0;
}

std::vector<double> PetersonRGaussBandHorizontalSchemeMPI::BackSubstitutionMPI(
    const InType& localMatrix, const std::vector<int>& mapGlobalToLocal,
    size_t n, size_t cols, int rank, int size) {

  std::vector<double> result(n, 0.0);

  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    double sum = 0.0;
    int owner = i % size;

    if (rank == owner) {
      int localIdx = mapGlobalToLocal[i];
      for (size_t j = i + 1; j < n; ++j) {
        sum += localMatrix[localIdx][j] * result[j];
      }
      result[i] = (localMatrix[localIdx][cols - 1] - sum) /
                  localMatrix[localIdx][i];
    }

    MPI_Bcast(&result[i], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
  }

  return result;
}

bool PetersonRGaussBandHorizontalSchemeMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank == 0 ? !GetOutput().empty() : true;
}

}  // namespace peterson_r_gauss_band_horizontal_scheme

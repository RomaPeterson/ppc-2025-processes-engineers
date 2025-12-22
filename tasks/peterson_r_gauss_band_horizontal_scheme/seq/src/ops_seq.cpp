#include "peterson_r_gauss_band_horizontal_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace peterson_r_gauss_band_horizontal_scheme {

PetersonRGaussBandHorizontalSchemeSEQ::PetersonRGaussBandHorizontalSchemeSEQ(const InType& inputData) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = inputData;
  GetOutput().clear();
}

bool PetersonRGaussBandHorizontalSchemeSEQ::ValidationImpl() {
  const InType& petersonRomanMatrix = GetInput();
  if (petersonRomanMatrix.empty()) return false;

  size_t petersonRomanN = petersonRomanMatrix.size();
  size_t petersonRomanCols = petersonRomanMatrix[0].size();

  if (petersonRomanCols < petersonRomanN + 1) return false;

  for (size_t i = 1; i < petersonRomanN; ++i) {
    if (petersonRomanMatrix[i].size() != petersonRomanCols) return false;
  }
  return true;
}

bool PetersonRGaussBandHorizontalSchemeSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool PetersonRGaussBandHorizontalSchemeSEQ::RunImpl() {
  InType petersonRomanAugmented = GetInput();
  size_t petersonRomanN = petersonRomanAugmented.size();
  size_t petersonRomanCols = petersonRomanAugmented[0].size();

  if (!ForwardElimination(petersonRomanAugmented, petersonRomanN, petersonRomanCols)) {
    return false;
  }

  GetOutput() = BackSubstitution(petersonRomanAugmented, petersonRomanN, petersonRomanCols);
  return true;
}

bool PetersonRGaussBandHorizontalSchemeSEQ::ForwardElimination(
    InType& matrix, size_t n, size_t cols) {
  for (size_t k = 0; k < n; ++k) {
    size_t pivotRow = ChoosePivotRow(matrix, k, n);
    if (pivotRow != k) {
      std::swap(matrix[k], matrix[pivotRow]);
    }

    if (std::abs(matrix[k][k]) < 1e-10) return false;

    Eliminate(matrix, k, n, cols);
  }
  return true;
}

size_t PetersonRGaussBandHorizontalSchemeSEQ::ChoosePivotRow(
    const InType& matrix, size_t k, size_t n) {
  size_t bestRow = k;
  double bestValue = std::abs(matrix[k][k]);

  for (size_t i = k + 1; i < n; ++i) {
    double current = std::abs(matrix[i][k]);
    if (current > bestValue) {
      bestValue = current;
      bestRow = i;
    }
  }
  return bestRow;
}

void PetersonRGaussBandHorizontalSchemeSEQ::Eliminate(
    InType& matrix, size_t k, size_t n, size_t cols) {
  for (size_t i = k + 1; i < n; ++i) {
    if (std::abs(matrix[i][k]) > 1e-10) {
      double factor = matrix[i][k] / matrix[k][k];
      for (size_t j = k; j < cols; ++j) {
        matrix[i][j] -= factor * matrix[k][j];
      }
    }
  }
}

std::vector<double> PetersonRGaussBandHorizontalSchemeSEQ::BackSubstitution(
    const InType& matrix, size_t n, size_t cols) {
  std::vector<double> petersonRomanResult(n, 0.0);

  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    double sum = 0.0;
    for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
      sum += matrix[static_cast<size_t>(i)][j] * petersonRomanResult[j];
    }

    petersonRomanResult[static_cast<size_t>(i)] =
        (matrix[static_cast<size_t>(i)][cols - 1] - sum) /
        matrix[static_cast<size_t>(i)][static_cast<size_t>(i)];
  }
  return petersonRomanResult;
}

bool PetersonRGaussBandHorizontalSchemeSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace peterson_r_gauss_band_horizontal_scheme

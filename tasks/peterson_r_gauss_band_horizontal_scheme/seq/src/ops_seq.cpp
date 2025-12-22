#include "peterson_r_gauss_band_horizontal_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace peterson_r_gauss_band_horizontal_scheme {

PetersonRGaussBandHorizontalSchemeSEQ::PetersonRGaussBandHorizontalSchemeSEQ(const InType &input_data) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input_data;
  GetOutput().clear();
}

bool PetersonRGaussBandHorizontalSchemeSEQ::ValidationImpl() {
  const InType &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  const size_t n = matrix.size();
  const size_t cols = matrix[0].size();

  if (cols < n + 1) {
    return false;
  }

  for (size_t i = 1; i < n; ++i) {
    if (matrix[i].size() != cols) {
      return false;
    }
  }

  return true;
}

bool PetersonRGaussBandHorizontalSchemeSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool PetersonRGaussBandHorizontalSchemeSEQ::RunImpl() {
  InType augmented_matrix = GetInput();
  const size_t n = augmented_matrix.size();
  const size_t cols = augmented_matrix[0].size();

  if (!ForwardElimination(augmented_matrix, n, cols)) {
    return false;
  }

  GetOutput() = BackSubstitution(augmented_matrix, n, cols);
  return true;
}

bool PetersonRGaussBandHorizontalSchemeSEQ::ForwardElimination(InType &matrix, size_t n, size_t cols) {
  for (size_t k = 0; k < n; ++k) {
    const size_t pivot_row = ChoosePivotRow(matrix, k, n);
    if (pivot_row != k) {
      std::swap(matrix[k], matrix[pivot_row]);
    }

    if (std::abs(matrix[k][k]) < 1e-10) {
      return false;
    }

    Eliminate(matrix, k, n, cols);
  }
  return true;
}

size_t PetersonRGaussBandHorizontalSchemeSEQ::ChoosePivotRow(const InType &matrix, size_t k, size_t n) {
  size_t best_row = k;
  double best_value = std::abs(matrix[k][k]);

  for (size_t i = k + 1; i < n; ++i) {
    const double current = std::abs(matrix[i][k]);
    if (current > best_value) {
      best_value = current;
      best_row = i;
    }
  }

  return best_row;
}

void PetersonRGaussBandHorizontalSchemeSEQ::Eliminate(InType &matrix, size_t k, size_t n, size_t cols) {
  for (size_t i = k + 1; i < n; ++i) {
    if (std::abs(matrix[i][k]) > 1e-10) {
      const double factor = matrix[i][k] / matrix[k][k];
      for (size_t j = k; j < cols; ++j) {
        matrix[i][j] -= factor * matrix[k][j];
      }
    }
  }
}

std::vector<double> PetersonRGaussBandHorizontalSchemeSEQ::BackSubstitution(const InType &matrix, size_t n,
                                                                            size_t cols) {
  std::vector<double> result(n, 0.0);

  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    double sum = 0.0;
    for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
      sum += matrix[static_cast<size_t>(i)][j] * result[j];
    }

    result[static_cast<size_t>(i)] =
        (matrix[static_cast<size_t>(i)][cols - 1] - sum) / matrix[static_cast<size_t>(i)][static_cast<size_t>(i)];
  }

  return result;
}

bool PetersonRGaussBandHorizontalSchemeSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace peterson_r_gauss_band_horizontal_scheme

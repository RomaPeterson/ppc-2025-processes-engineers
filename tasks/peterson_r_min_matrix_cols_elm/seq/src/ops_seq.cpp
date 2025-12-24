#include "peterson_r_min_matrix_cols_elm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "peterson_r_min_matrix_cols_elm/common/include/common.hpp"

namespace peterson_r_min_matrix_cols_elm {
PetersonRMinMatrixColsElmSEQ::PetersonRMinMatrixColsElmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool PetersonRMinMatrixColsElmSEQ::ValidationImpl() {
  const std::size_t m = std::get<0>(GetInput());
  const std::size_t n = std::get<1>(GetInput());
  const std::vector<int> &val = std::get<2>(GetInput());

  const bool has_positive_shape = (m > 0) && (n > 0);
  const bool size_matches = has_positive_shape && (val.size() / n == m) && (val.size() % n == 0);

  valid_ = has_positive_shape && size_matches;
  return valid_;
}

bool PetersonRMinMatrixColsElmSEQ::PreProcessingImpl() {
  return valid_;
}

bool PetersonRMinMatrixColsElmSEQ::RunImpl() {
  if (!valid_) {
    return false;
  }

  const std::size_t m = std::get<0>(GetInput());
  const std::size_t n = std::get<1>(GetInput());
  const std::vector<int> &val = std::get<2>(GetInput());

  std::vector<int> min_cols(n, std::numeric_limits<int>::max());

  for (std::size_t row = 0; row < m; row++) {
    const std::size_t offset = row * n;
    for (std::size_t col = 0; col < n; col++) {
      const int current = val[offset + col];
      min_cols[col] = std::min(min_cols[col], current);
    }
  }

  GetOutput() = std::move(min_cols);
  return true;
}

bool PetersonRMinMatrixColsElmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace peterson_r_min_matrix_cols_elm

#include "peterson_s_min_val_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "peterson_s_min_val_matrix/common/include/common.hpp"

namespace peterson_s_min_val_matrix {

PetersonSMinValMatrixSEQ::PetersonSMinValMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool PetersonSMinValMatrixSEQ::ValidationImpl() {
  return GetOutput().empty();
}

bool PetersonSMinValMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool PetersonSMinValMatrixSEQ::RunImpl() {
  int n = GetInput();
  if (n == 0) {
    GetOutput().clear();
    return true;
  }

  std::vector<int> result(n);
  for (int j = 0; j < n; ++j) {
    int min_val = j + 1;
    for (int i = 1; i < n; ++i) {
      int val = i * n + j + 1;
      min_val = std::min(min_val, val);
    }
    result[j] = min_val;
  }

  GetOutput() = result;
  return true;
}

bool PetersonSMinValMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace peterson_s_min_val_matrix

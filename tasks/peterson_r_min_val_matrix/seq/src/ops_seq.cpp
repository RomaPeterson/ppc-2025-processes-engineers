#include "peterson_r_min_val_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "peterson_r_min_val_matrix/common/include/common.hpp"

namespace peterson_r_min_val_matrix {

PetersonRMinValMatrixSEQ::PetersonRMinValMatrixSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool PetersonRMinValMatrixSEQ::ValidationImpl() {
  return (GetInput() >= kMinMatrixSize) && 
         (GetInput() <= kMaxMatrixSize) && 
         (GetOutput().empty());
}

bool PetersonRMinValMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool PetersonRMinValMatrixSEQ::RunImpl() {
  const auto input = GetInput();
  if (input <= 0) {
    return false;
  }

  const int n = input;
  std::vector<int> result(n);

  for (int j = 0; j < n; ++j) {
    int min_val = j + 1;
    for (int i = 1; i < n; ++i) {
      const int val = i * n + j + 1;
      if (val < min_val) {
        min_val = val;
      }
    }
    result[j] = min_val;
  }

  GetOutput() = result;
  return true;
}

bool PetersonRMinValMatrixSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace peterson_r_min_val_matrix

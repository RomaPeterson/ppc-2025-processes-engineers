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
  return (GetInput() > 0) && (GetOutput().empty());
}

bool PetersonRMinValMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool PetersonRMinValMatrixSEQ::RunImpl() {
  auto input = GetInput();
  if (input == 0) {
    return false;
  }

  std::vector<int> result(input);
  for (int j = 0; j < input; ++j) {
    int min_val = j + 1;
    for (int i = 1; i < input; ++i) {
      int val = i * input + j + 1;
      min_val = std::min(min_val, val);
    }
    result[j] = min_val;
  }

  GetOutput() = result;
  return true;
}

bool PetersonRMinValMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace peterson_r_min_val_matrix

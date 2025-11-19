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
  return (GetInput() > 0) && (GetOutput().empty());
}

bool PetersonSMinValMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().reserve(GetInput());
  return true;
}

bool PetersonSMinValMatrixSEQ::RunImpl() {
  InType n = GetInput();
  if (n == 0) return false;

  GetOutput().resize(n);

  for (InType j = 0; j < n; ++j) {
    InType min_val = j + 1;
    for (InType i = 1; i < n; ++i) {
      InType val = i * n + j + 1;
      if (val < min_val) min_val = val;
    }
    GetOutput()[j] = min_val;
  }

  return true;
}

bool PetersonSMinValMatrixSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace peterson_s_min_val_matrix

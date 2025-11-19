#pragma once

#include "peterson_s_min_val_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peterson_s_min_val_matrix {

class PetersonSMinValMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PetersonSMinValMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace peterson_s_min_val_matrix

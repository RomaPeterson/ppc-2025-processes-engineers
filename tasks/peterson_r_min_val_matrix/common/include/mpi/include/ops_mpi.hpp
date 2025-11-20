#pragma once

#include "peterson_r_min_val_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peterson_r_min_val_matrix {

class PetersonRMinValMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PetersonRMinValMatrixMPI(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace peterson_r_min_val_matrix

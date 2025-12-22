#pragma once

#include "peterson_r_min_matrix_cols_elm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peterson_r_min_matrix_cols_elm {

class PetersonRMinMatrixColsElmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PetersonRMinMatrixColsElmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool valid_ = false;
};

}  // namespace peterson_r_min_matrix_cols_elm

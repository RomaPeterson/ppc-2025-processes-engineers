#pragma once

#include "peterson_r_min_matrix_cols_elm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peterson_r_matrix_min_by_columns {

class PetersonRMinMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PetersonRMinMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace peterson_r_matrix_min_by_columns

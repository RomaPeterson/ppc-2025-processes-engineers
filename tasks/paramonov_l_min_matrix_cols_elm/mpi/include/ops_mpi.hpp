#pragma once

#include "paramonov_l_min_matrix_cols_elm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace paramonov_l_min_matrix_cols_elm{

class ParamonovLMinMatrixColsElmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ParamonovLMinMatrixColsElmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> t_matrix_;
  bool valid_ = false;
};

}  // namespace paramonov_l_min_matrix_cols_elm

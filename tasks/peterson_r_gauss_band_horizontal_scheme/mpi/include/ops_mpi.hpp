#pragma once

#include <cstddef>
#include <vector>

#include "peterson_r_gauss_band_horizontal_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peterson_r_gauss_band_horizontal_scheme {

class PetersonRGaussBandHorizontalSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit PetersonRGaussBandHorizontalSchemeMPI(const InType& inputData);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int ValidateInputData(const InType& inputData);
  static void SplitRows(const InType& matrix, size_t n, size_t cols, int rank, int size,
                        InType& localMatrix, std::vector<int>& mapGlobalToLocal);
  static bool ForwardEliminationMPI(InType& localMatrix, const std::vector<int>& mapGlobalToLocal,
                                    size_t n, size_t cols, int rank, int size);
  static void EliminateMPI(InType& localMatrix, const std::vector<int>& mapGlobalToLocal,
                           size_t pivot, size_t n, size_t cols,
                           const std::vector<double>& pivotRow);
  static size_t ResolveGlobalIndex(const std::vector<int>& mapGlobalToLocal,
                                   size_t localIndex, size_t n);
  static std::vector<double> BackSubstitutionMPI(const InType& localMatrix,
                                                 const std::vector<int>& mapGlobalToLocal,
                                                 size_t n, size_t cols,
                                                 int rank, int size);
};

}  // namespace peterson_r_gauss_band_horizontal_scheme

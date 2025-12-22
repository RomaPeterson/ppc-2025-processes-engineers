#pragma once

#include <cstddef>
#include <vector>

#include "peterson_r_gauss_band_horizontal_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peterson_r_gauss_band_horizontal_scheme {

class PetersonRGaussBandHorizontalSchemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit PetersonRGaussBandHorizontalSchemeSEQ(const InType& inputData);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static bool ForwardElimination(InType& matrix, size_t n, size_t cols);
  static size_t ChoosePivotRow(const InType& matrix, size_t k, size_t n);
  static void Eliminate(InType& matrix, size_t k, size_t n, size_t cols);
  static std::vector<double> BackSubstitution(const InType& matrix, size_t n, size_t cols);
};

}  // namespace peterson_r_gauss_band_horizontal_scheme

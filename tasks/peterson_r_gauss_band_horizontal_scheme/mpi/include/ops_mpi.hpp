#pragma once

#include <vector>

#include "peterson_r_gauss_band_horizontal_scheme/common/include/common.hpp"

namespace peterson_r_gauss_band_horizontal_scheme {

class PetersonRGaussBandHorizontalSchemeMPI : public BaseTask {
 public:
  explicit PetersonRGaussBandHorizontalSchemeMPI(const InType &input_data);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int ValidateInputData(const InType &input_data);

  void SplitRows(const InType &matrix, size_t n, size_t cols, int rank, int size, InType &local_matrix,
                 std::vector<int> &map_global_to_local);

  bool ForwardEliminationMPI(InType &local_matrix, const std::vector<int> &map_global_to_local, size_t n, size_t cols,
                             int rank, int size);

  void EliminateMPI(InType &local_matrix, const std::vector<int> &map_global_to_local, size_t pivot, size_t n,
                    size_t cols, const std::vector<double> &pivot_row);

  size_t ResolveGlobalIndex(const std::vector<int> &map_global_to_local, size_t local_index, size_t n);

  std::vector<double> BackSubstitutionMPI(const InType &local_matrix, const std::vector<int> &map_global_to_local,
                                          size_t n, size_t cols, int rank, int size);
};

}  // namespace peterson_r_gauss_band_horizontal_scheme

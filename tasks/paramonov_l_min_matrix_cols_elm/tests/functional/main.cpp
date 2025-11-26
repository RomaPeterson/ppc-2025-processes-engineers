#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "paramonov_l_min_matrix_cols_elm/common/include/common.hpp"
#include "paramonov_l_min_matrix_cols_elm/mpi/include/ops_mpi.hpp"
#include "paramonov_l_min_matrix_cols_elm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace paramonov_l_min_matrix_cols_elm {

class ParamonovLMinMatrixColsElmTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    if (std::get<1>(params) != "") {
      GetDataFromFile(params);
    } else {
      Generate(params);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // реализована не стандратная проверка,
    // так как вектор ответа в процессе с ранком 0 имеет больший размер
    // для уменьшения времени на выделение лишней памяти
    for (std::size_t i = 0; i < correct_test_output_data_.size(); i++) {
      if (output_data[i] != correct_test_output_data_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<int> correct_test_output_data_;

  void Generate(const TestType &params) {
    std::size_t m = std::get<2>(params)[0];
    std::size_t n = std::get<2>(params)[1];
    int seed = std::get<2>(params)[2];

    std::mt19937 gen(seed);
    std::uniform_int_distribution<> idis(-10, 20);

    std::vector<int> val(m * n);
    std::vector<int> answer(n);
    for (std::size_t i = 0; i < n; i++) {
      val[i] = idis(gen);
      answer[i] = val[i];
    }

    for (std::size_t i = 1; i < m; i++) {
      for (std::size_t j = 0; j < n; j++) {
        val[i * n + j] = idis(gen);
        if (answer[j] > val[i * n + j]) {
          answer[j] = val[i * n + j];
        }
      }
    }
    input_data_ = std::make_tuple(m, n, val);
    correct_test_output_data_ = answer;
  }

  void GetDataFromFile(const TestType &params) {
    std::size_t m = 0;
    std::size_t n = 0;
    std::string local = std::get<1>(params) + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_paramonov_l_min_matrix_cols_elm, local);
    std::ifstream file(abs_path);
    if (file.is_open() == false) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }
    file >> m;
    file >> n;
    std::vector<int> val(m * n);
    for (std::size_t i = 0; i < val.size(); i++) {
      file >> val[i];
    }
    input_data_ = std::make_tuple(m, n, val);
    correct_test_output_data_ = std::get<2>(params);
  }
};

namespace {

TEST_P(ParamonovLMinMatrixColsElmTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple("Matrix_3_3_from_1_to_9", "test_matrix_3_3", std::vector<int>({1, 2, 3})),
    std::make_tuple("Generate_7_7", "", std::vector<int>({7, 7, 123})),
    // std::make_tuple("Generate_1000_1000", "", std::vector<int>({1000, 1000, 123})),
    // std::make_tuple("Generate_77_88", "", std::vector<int>({77, 88, 123})),
    std::make_tuple("Generate_7_8", "", std::vector<int>({7, 8, 123}))};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ParamonovLMinMatrixColsElmMPI, InType>(
                                               kTestParam, PPC_SETTINGS_paramonov_l_min_matrix_cols_elm),
                                           ppc::util::AddFuncTask<ParamonovLMinMatrixColsElmSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_paramonov_l_min_matrix_cols_elm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ParamonovLMinMatrixColsElmTests::PrintFuncTestName<ParamonovLMinMatrixColsElmTests>;

INSTANTIATE_TEST_SUITE_P(GenMatrixTests, ParamonovLMinMatrixColsElmTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace paramonov_l_min_matrix_cols_elm

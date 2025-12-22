#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "peterson_r_min_matrix_cols_elm/common/include/common.hpp"
#include "peterson_r_min_matrix_cols_elm/mpi/include/ops_mpi.hpp"
#include "peterson_r_min_matrix_cols_elm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace peterson_r_min_matrix_cols_elm {

class PetersonRMinMatrixColsElmTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
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

  void Prepare(const TestType &params) {
    if (!std::get<1>(params).empty()) {
      GetDataFromFile(params);
    } else {
      Generate(params);
    }
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
        const std::size_t idx = (i * n) + j;
        val[idx] = idis(gen);
        answer[j] = std::min(answer[j], val[idx]);
      }
    }
    input_data_ = std::make_tuple(m, n, val);
    correct_test_output_data_ = answer;
  }

  void GetDataFromFile(const TestType &params) {
    std::size_t m = 0;
    std::size_t n = 0;
    std::string local = std::get<1>(params) + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_peterson_r_min_matrix_cols_elm, local);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }
    file >> m;
    file >> n;
    std::vector<int> val(m * n);
    for (auto &value : val) {
      file >> value;
    }
    input_data_ = std::make_tuple(m, n, val);
    correct_test_output_data_ = std::get<2>(params);
  }
};

TEST(PetersonRMinMatrixColsElmSEQ, ValidationFailsOnInvalidShape) {
  InType bad_shape = std::make_tuple(0, 5, std::vector<int>({1, 2, 3}));
  PetersonRMinMatrixColsElmSEQ task(bad_shape);
  // Проверяем, что валидация отсекает некорректные размеры.
  EXPECT_FALSE(task.Validation());
}

namespace {

const std::array<TestType, 6> kTestParam = {
    std::make_tuple("Matrix_3_3_from_1_to_9", "test_matrix_3_3", std::vector<int>({1, 2, 3})),
    std::make_tuple("Matrix_4_5_with_negatives", "test_matrix_4_5_neg", std::vector<int>({-3, -5, -4, -2, -7})),
    std::make_tuple("Generate_7_7", "", std::vector<int>({7, 7, 123})),
    std::make_tuple("Generate_7_8", "", std::vector<int>({7, 8, 123})),
    std::make_tuple("Generate_single_row", "", std::vector<int>({1, 5, 42})),
    std::make_tuple("Generate_single_column", "", std::vector<int>({9, 1, 321}))};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<PetersonRMinMatrixColsElmMPI, InType>(
                                               kTestParam, PPC_SETTINGS_peterson_r_min_matrix_cols_elm),
                                           ppc::util::AddFuncTask<PetersonRMinMatrixColsElmSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_peterson_r_min_matrix_cols_elm));

template <typename Tuple, std::size_t... Is>
auto TupleToArray(const Tuple &tuple, std::index_sequence<Is...> /*unused*/) {
  return std::array<ppc::util::FuncTestParam<InType, OutType, TestType>, sizeof...(Is)>{std::get<Is>(tuple)...};
}

constexpr std::size_t kTasksCount = std::tuple_size_v<decltype(kTestTasksList)>;
const auto kTestTasksArray = TupleToArray(kTestTasksList, std::make_index_sequence<kTasksCount>{});

TEST_F(PetersonRMinMatrixColsElmTests, MatmulFromPic) {
  for (const auto &param : kTestTasksArray) {
    Prepare(std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(param));
    ExecuteTest(param);
  }
}

}  // namespace

}  // namespace peterson_r_min_matrix_cols_elm

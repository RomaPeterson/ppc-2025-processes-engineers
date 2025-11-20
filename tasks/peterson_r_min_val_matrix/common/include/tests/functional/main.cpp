#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <tuple>
#include <vector>
#include <stdexcept>
#include <string>

#include "peterson_r_min_val_matrix/common/include/common.hpp"
#include "peterson_r_min_val_matrix/mpi/include/ops_mpi.hpp"
#include "peterson_r_min_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace peterson_r_min_val_matrix {

class PetersonRunFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    int width = -1;
    int height = -1;
    int channels = -1;
    
    // Read image (теперь картинка есть в data/)
    {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_peterson_r_min_val_matrix, "pic.jpg");
      auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, 0);
      if (data == nullptr) {
        throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
      }
      stbi_image_free(data);
    }

    TestType params = std::get<static_cast<std::size_t>(
        ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    
    // Generate input based on image properties
    input_data_ = std::min(width, 10) + std::min(height, 10);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto expected = GenerateExpectedResult(input_data_);
    return output_data == expected;
  }

  InType GetTestInputData() final { return input_data_; }

 private:
  std::vector<int> GenerateExpectedResult(int n) {
    std::vector<int> result(n);
    for (int j = 0; j < n; ++j) {
      result[j] = j + 1;
    }
    return result;
  }

  InType input_data_ = 0;
};

namespace {

TEST_P(PetersonRunFuncTests, MinValMatrixTest) { ExecuteTest(GetParam()); }

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PetersonRMinValMatrixMPI, InType>(
        kTestParam, PPC_SETTINGS_peterson_r_min_val_matrix),
    ppc::util::AddFuncTask<PetersonRMinValMatrixSEQ, InType>(
        kTestParam, PPC_SETTINGS_peterson_r_min_val_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    PetersonRunFuncTests::PrintFuncTestName<PetersonRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(MinValMatrixTests, PetersonRunFuncTests, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace peterson_r_min_val_matrix

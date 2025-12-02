#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "peterson_r_min_matrix_cols_elm/common/include/common.hpp"
#include "peterson_r_min_matrix_cols_elm/mpi/include/ops_mpi.hpp"
#include "peterson_r_min_matrix_cols_elm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace peterson_r_min_matrix_cols_elm {

    class PetersonRMinMatrixColsElmPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
        std::vector<int> correct_test_output_data_;
        InType input_data_;

        void SetUp() override {
            Generate(10000, 10000, 123);
        }

        bool CheckTestOutputData(OutType& output_data) final {
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

        void Generate(std::size_t m, std::size_t n, int seed) {
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
                        answer[j] = std::min(answer[j], val[i * n + j]);
                    }
                }
            }
            input_data_ = std::make_tuple(m, n, val);
            correct_test_output_data_ = answer;
        }
    };

    namespace {

        TEST_P(PetersonRMinMatrixColsElmPerfTest, RunPerfModes) {
            ExecuteTest(GetParam());
        }

        const auto kAllPerfTasks =
            ppc::util::MakeAllPerfTasks<InType, PetersonRMinMatrixColsElmMPI, PetersonRMinMatrixColsElmSEQ>(
                PPC_SETTINGS_peterson_r_min_matrix_cols_elm);

        const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

        const auto kPerfTestName = PetersonRMinMatrixColsElmPerfTest::CustomPerfTestName;

        INSTANTIATE_TEST_SUITE_P(RunModeTests, PetersonRMinMatrixColsElmPerfTest, kGtestValues, kPerfTestName);

    }  // namespace

}  // namespace peterson_r_min_matrix_cols_elm
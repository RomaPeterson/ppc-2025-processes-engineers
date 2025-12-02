#include "peterson_r_min_matrix_cols_elm/seq/include/ops_seq.hpp"

#include <algorithm>  
#include <cstddef>    
#include <vector>

#include "peterson_r_min_matrix_cols_elm/common/include/common.hpp"

namespace peterson_r_min_matrix_cols_elm {

    PetersonRMinMatrixColsElmSEQ::PetersonRMinMatrixColsElmSEQ(const InType& in) {
        SetTypeOfTask(GetStaticTypeOfTask());
        GetInput() = in;
        GetOutput() = std::vector<int>();
    }

    bool PetersonRMinMatrixColsElmSEQ::ValidationImpl() {
        std::size_t m = std::get<0>(GetInput());
        std::size_t n = std::get<1>(GetInput());
        std::vector<int>& val = std::get<2>(GetInput());
        valid_ = (n > 0) && (m > 0) && (val.size() == (n * m));
        return valid_;
    }

    bool PetersonRMinMatrixColsElmSEQ::PreProcessingImpl() {
        if (valid_) {
            std::size_t m = std::get<0>(GetInput());
            std::size_t n = std::get<1>(GetInput());
            std::vector<int>& val = std::get<2>(GetInput());
            t_matrix_ = std::vector<int>(n * m);
            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < n; j++) {
                    t_matrix_[(j * m) + i] = val[(i * n) + j];
                }
            }
            return true;
        }
        return false;
    }

    bool PetersonRMinMatrixColsElmSEQ::RunImpl() {
        if (!valid_) {
            return false;
        }

        std::size_t m = std::get<0>(GetInput());
        std::size_t n = std::get<1>(GetInput());

        std::vector<int> min_cols_elm(n);
        for (std::size_t i = 0; i < n; i++) {
            min_cols_elm[i] = t_matrix_[i * m];
            for (std::size_t j = 1; j < m; j++) {
                if (min_cols_elm[i] > t_matrix_[i * m + j]) {
                    min_cols_elm[i] = std::min(min_cols_elm[i], t_matrix_[i * m + j]);
                }
            }
        }

        GetOutput() = min_cols_elm;
        return true;
    }

    bool PetersonRMinMatrixColsElmSEQ::PostProcessingImpl() {
        return true;
    }

}  // namespace peterson_r_min_matrix_cols_elm
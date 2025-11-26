#pragma once

#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace paramonov_l_min_matrix_cols_elm {

using InType = std::tuple<std::size_t, std::size_t, std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::string, std::string, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace paramonov_l_min_matrix_cols_elm

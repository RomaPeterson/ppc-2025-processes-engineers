#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace peterson_r_min_val_matrix {

using InType = int;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

constexpr int kMaxMatrixSize = 10000;
constexpr int kMinMatrixSize = 1;

}  // namespace peterson_r_min_val_matrix

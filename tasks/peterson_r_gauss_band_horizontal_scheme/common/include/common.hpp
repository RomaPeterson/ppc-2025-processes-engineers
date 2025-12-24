#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace peterson_r_gauss_band_horizontal_scheme {

using InType = std::vector<std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace peterson_r_gauss_band_horizontal_scheme

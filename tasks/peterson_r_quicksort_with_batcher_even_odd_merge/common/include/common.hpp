#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::string, std::vector<int>, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace peterson_r_quicksort_with_batcher_even_odd_merge {

using InputArray = std::vector<double>;
using OutputArray = std::vector<double>;
using TestScenario = std::pair<std::string, std::vector<double>>;
using CoreAlgorithm = ppc::task::Task<InputArray, OutputArray>;

}  // namespace peterson_r_quicksort_with_batcher_even_odd_merge

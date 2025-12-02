#include "peterson_r_min_matrix_cols_elm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>    
#include <cstddef>      
#include <iostream>     
#include <limits>
#include <string>       
#include <vector>

#include "peterson_r_min_matrix_cols_elm/common/include/common.hpp"

namespace peterson_r_min_matrix_cols_elm {

	PetersonRMinMatrixColsElmMPI::PetersonRMinMatrixColsElmMPI(const InType& in) {
		SetTypeOfTask(GetStaticTypeOfTask());
		GetInput() = in;
		GetOutput() = std::vector<int>();
	}

	bool PetersonRMinMatrixColsElmMPI::ValidationImpl() {
		std::size_t m = std::get<0>(GetInput());
		std::size_t n = std::get<1>(GetInput());
		std::vector<int>& val = std::get<2>(GetInput());
		valid_ = (n > 0) && (m > 0) && (val.size() == (n * m));
		return valid_;
	}

	bool PetersonRMinMatrixColsElmMPI::PreProcessingImpl() {
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

	bool PetersonRMinMatrixColsElmMPI::RunImpl() {
		if (!valid_) {
			return false;
		}

		std::size_t m = std::get<0>(GetInput());
		std::size_t n = std::get<1>(GetInput());

		int rank = 0;
		int mpi_size = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

		std::size_t procesess_step = t_matrix_.size() / mpi_size;
		std::size_t start = procesess_step * rank;
		std::size_t end = procesess_step * (rank + 1);

		if (rank == mpi_size - 1) {
			end = t_matrix_.size();
		}

		std::vector<int> min_cols_elm;
		int imax = std::numeric_limits<int>::max();
		if (rank == 0) {
			min_cols_elm.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(mpi_size), imax);
		}
		else {
			min_cols_elm.resize(n, imax);
		}

		if (m > 0) {
			std::size_t row = start / m;
			min_cols_elm[row] = t_matrix_[start];

			for (std::size_t i = start; i < end; i++) {
				if (i == (row + 1) * m) {
					row++;
					min_cols_elm[row] = t_matrix_[i];
				}
				if (min_cols_elm[row] > t_matrix_[i]) {
					min_cols_elm[row] = std::min(min_cols_elm[row], t_matrix_[i]);
				}
			}
		}

		int n_int = static_cast<int>(n);
		MPI_Gather(min_cols_elm.data(), n_int, MPI_INT, min_cols_elm.data(), n_int, MPI_INT, 0, MPI_COMM_WORLD);

		if (rank == 0) {
			for (std::size_t i = 0; i < n; i++) {
				for (int j = 0; j < mpi_size; j++) {
					if (min_cols_elm[i] > min_cols_elm[static_cast<std::size_t>(j) * n + i]) {
						min_cols_elm[i] = std::min(min_cols_elm[i], min_cols_elm[static_cast<std::size_t>(j) * n + i]);
					}
				}
			}
		}

		MPI_Bcast(min_cols_elm.data(), n_int, MPI_INT, 0, MPI_COMM_WORLD);


		std::vector<int> result(min_cols_elm.begin(), min_cols_elm.begin() + n_int);
		GetOutput() = result;

		return true;
	}

	bool PetersonRMinMatrixColsElmMPI::PostProcessingImpl() {
		return true;
	}

}  // namespace peterson_r_min_matrix_cols_elm
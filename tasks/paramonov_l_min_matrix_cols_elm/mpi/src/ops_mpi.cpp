#include "paramonov_l_min_matrix_cols_elm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <limits>
#include <numeric>
#include <vector>


#include "paramonov_l_min_matrix_cols_elm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace paramonov_l_min_matrix_cols_elm {

ParamonovLMinMatrixColsElmMPI::ParamonovLMinMatrixColsElmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool ParamonovLMinMatrixColsElmMPI::ValidationImpl() {
  std::size_t m = std::get<0>(GetInput());
  std::size_t n = std::get<1>(GetInput());
  std::vector<int> &val = std::get<2>(GetInput());
  valid_ = (n > 0) && (m > 0) && (val.size() == (n * m));
  return valid_;
}

bool ParamonovLMinMatrixColsElmMPI::PreProcessingImpl() {
  if(valid_) {
    std::size_t m = std::get<0>(GetInput());
    std::size_t n = std::get<1>(GetInput());
    std::vector<int> &val = std::get<2>(GetInput());
    t_matrix_ = std::vector<int>(n * m);
    for(std::size_t i = 0; i < m; i++) {
      for(std::size_t j = 0; j < n; j++) {
        t_matrix_[(j * m) + i] = val[(i * n) + j];
      }
    }
    return true;
  }
  return false;
}

bool ParamonovLMinMatrixColsElmMPI::RunImpl() {
  //проверка корректности данных
  if(!valid_) {
    return false;
  }
  //получение размера матрицы
  std::size_t m = std::get<0>(GetInput());
  std::size_t n = std::get<1>(GetInput());
  
  //debug
  std::string deb = "\n\n-----------\n";
  for(std::size_t i = 0; i < n; i++) {
    for(std::size_t j = 0; j < m; j++) {
      deb += std::to_string(t_matrix_[i*m + j]) + " ";
    }
    deb += "\n";
  }
  std::cout << deb;

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
  if(rank == 0) {
    min_cols_elm.resize(n * mpi_size, imax);
  } else {
    min_cols_elm.resize(n, imax);
  }
  std::size_t row = start / m;
  min_cols_elm[row] = t_matrix_[start];

  for(std::size_t i = start; i < end; i++) {
    if(i == (row + 1) * m) {
      row++;
      min_cols_elm[row] = t_matrix_[i];
    }
    if(min_cols_elm[row] > t_matrix_[i]) {
      min_cols_elm[row] = t_matrix_[i];
    }
  }
  
  MPI_Gather(min_cols_elm.data(), n, MPI_INT, min_cols_elm.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  if(rank == 0) {
    for(std::size_t i = 0; i < n; i++) {
      for(int j = 0; j < mpi_size; j++) {
        if(min_cols_elm[i] > min_cols_elm[j * n + i]) {
          min_cols_elm[i] = min_cols_elm[j * n + i];
        }
      }
    }
  }
  MPI_Bcast(min_cols_elm.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = min_cols_elm;

  return true;
}

bool ParamonovLMinMatrixColsElmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace paramonov_l_min_matrix_cols_elm

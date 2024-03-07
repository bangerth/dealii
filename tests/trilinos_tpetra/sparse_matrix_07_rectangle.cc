// ---------------------------------------------------------------------
//
// Copyright (C) 2011 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------



// test LinearAlgebra::TpetraWrappers::SparseMatrix<double>::reinit with a
// dealii::SparseMatrix with a separate sparsity pattern that is partly subset
// and partly superset

#include <deal.II/base/utilities.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>
#include <deal.II/lac/trilinos_tpetra_sparsity_pattern.h>

#include <iostream>

#include "../tests.h"


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  initlog();

  SparsityPattern sparsity(4, 5, 5);
  sparsity.add(1, 2);
  sparsity.add(2, 3);
  sparsity.add(3, 4);
  sparsity.add(0, 4);
  sparsity.compress();
  SparseMatrix<double> matrix(sparsity);
  {
    double value = 1;
    for (SparseMatrix<double>::iterator p = matrix.begin(); p != matrix.end();
         ++p, ++value)
      p->value() = value;
  }
  deallog << "Original:" << std::endl;
  matrix.print_formatted(deallog.get_file_stream());

  // create a separate sparsity pattern to use
  SparsityPattern xsparsity(4, 5, 5);
  xsparsity.add(1, 2);
  xsparsity.add(2, 3);
  xsparsity.add(2, 4);
  xsparsity.add(2, 1);
  xsparsity.compress();


  // now copy everything into a Trilinos matrix
  const auto local_rows = complete_index_set(4);
  const auto local_cols = complete_index_set(5);
  LinearAlgebra::TpetraWrappers::SparseMatrix<double> tmatrix;
  tmatrix.reinit(
    local_rows, local_cols, matrix, MPI_COMM_SELF, 0, true, &xsparsity);

  deallog << "Copy structure only:" << std::endl;
  tmatrix.print(deallog.get_file_stream());
}

// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2020 by the deal.II authors
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


// test ReductionControl with Trilinos solver
// This test is adapted from tests/trilinos/solver_03.cc


#include <deal.II/base/utilities.h>

#include <deal.II/lac/trilinos_tpetra_precondition.h>
#include <deal.II/lac/trilinos_tpetra_solver.h>
#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>
#include <deal.II/lac/vector_memory.h>

#include <iostream>
#include <typeinfo>

#include "../tests.h"

#include "../testmatrix.h"


int
main(int argc, char **argv)
{
  initlog();
  deallog << std::setprecision(4);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());


  {
    const unsigned int size = 32;
    unsigned int       dim  = (size - 1) * (size - 1);

    deallog << "Size " << size << " Unknowns " << dim << std::endl;

    // Make matrix
    FDMatrix               testproblem(size, size);
    DynamicSparsityPattern csp(dim, dim);
    testproblem.five_point_structure(csp);
    LinearAlgebra::TpetraWrappers::SparseMatrix<double> A;
    A.reinit(csp);
    testproblem.five_point(A);

    LinearAlgebra::TpetraWrappers::Vector<double> f;
    f.reinit(complete_index_set(dim), MPI_COMM_WORLD);
    LinearAlgebra::TpetraWrappers::Vector<double> u;
    u.reinit(complete_index_set(dim), MPI_COMM_WORLD);
    f = 1.;
    A.compress(VectorOperation::insert);
    f.compress(VectorOperation::insert);
    u.compress(VectorOperation::insert);

    LinearAlgebra::TpetraWrappers::PreconditionJacobi preconditioner;
    preconditioner.initialize(A);

    deallog.push("Rel tol");
    {
      // Expects success
      ReductionControl                        control(2000, 1.e-30, 1e-6);
      LinearAlgebra::TpetraWrappers::SolverCG solver(control);
      check_solver_within_range(solver.solve(A, u, f, preconditioner),
                                control.last_step(),
                                49,
                                51);
    }
    deallog.pop();

    u = 0.;
    deallog.push("Abs tol");
    {
      // Expects success
      ReductionControl                        control(2000, 1.e-3, 1e-9);
      LinearAlgebra::TpetraWrappers::SolverCG solver(control);
      check_solver_within_range(solver.solve(A, u, f, preconditioner),
                                control.last_step(),
                                42,
                                44);
    }
    deallog.pop();

    u = 0.;
    deallog.push("Iterations");
    {
      // Expects failure
      ReductionControl                        control(20, 1.e-30, 1e-6);
      LinearAlgebra::TpetraWrappers::SolverCG solver(control);
      check_solver_within_range(solver.solve(A, u, f, preconditioner),
                                control.last_step(),
                                0,
                                19);
    }
    deallog.pop();
  }
}

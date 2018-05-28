/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2018 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, Colorado State University
 *         Yong-Yong Cai, Beijing Computational Science Research Center
 */

// @sect3{Include files}
// The program starts with the usual include files, all of which you should
// have seen before by now:
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>


// Then the usual placing of all content of this program into a namespace and
// the importation of the deal.II namespace into the one we will work in:
namespace Step58
{
  using namespace dealii;

  // @sect3{The <code>NonlinearSchroedingerEquation</code> class}
  //
  template <int dim>
  class NonlinearSchroedingerEquation
  {
  public:
    NonlinearSchroedingerEquation();
    void run();

  private:
    void setup_system();
    void assemble_matrices();
    void do_half_phase_step();
    void do_full_spatial_step();
    void output_results() const;


    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    ConstraintMatrix constraints;

    SparsityPattern                    sparsity_pattern;
    SparseMatrix<std::complex<double>> system_matrix;
    SparseMatrix<std::complex<double>> rhs_matrix;

    Vector<std::complex<double>> solution;
    Vector<std::complex<double>> old_solution;
    Vector<std::complex<double>> system_rhs;

    double       time;
    double       time_step;
    unsigned int timestep_number;

    double kappa;
  };



  // @sect3{Equation data}

  // Before we go on filling in the details of the main class, let us define
  // the equation data corresponding to the problem, i.e. initial values, as
  // well as a right hand side class. (We will reuse the initial conditions
  // also for the boundary values, which we simply keep constant.) We do so
  // using classes derived
  // from the Function class template that has been used many times before, so
  // the following should not look surprising. The only point of interest is
  // that we here have a complex-valued problem, so we have to provide the
  // second template argument of the Function class (which would otherwise
  // default to `double`). Furthermore, the return type of the `value()`
  // functions is then of course also complex.
  //
  // What precisely these functions return has been discussed at the end of
  // the Introduction section.
  template <int dim>
  class InitialValues : public Function<dim, std::complex<double>>
  {
  public:
    InitialValues() : Function<dim, std::complex<double>>(1)
    {}

    virtual std::complex<double>
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };



  template <int dim>
  std::complex<double>
  InitialValues<dim>::value(const Point<dim> & p,
                            const unsigned int component) const
  {
    static_assert(dim == 2, "This initial condition only works in 2d.");

    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));


    std::complex<double> value = {1, 0};

    const std::vector<Point<dim>> vortex_centers = {
      {0, -0.5}, {0, +0.5}, {0.5, 0}};
    const std::vector<double> vortex_strengths = {1, 1, 1};
    AssertDimension(vortex_centers.size(), vortex_strengths.size());

    const std::complex<double> i(0, 1);

    for (unsigned int c = 0; c < vortex_centers.size(); ++c)
      {
        const auto distance = p - vortex_centers[c];

        const double r   = distance.norm();
        const double phi = std::atan2(distance[1], distance[0]);
        value *= vortex_strengths[c] * (r * std::exp(i * phi));
      }

    return value;
  }



  template <int dim>
  class Potential : public Function<dim>
  {
  public:
    Potential() = default;
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };



  template <int dim>
  double Potential<dim>::value(const Point<dim> & p,
                               const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return p * p;
  }



  // @sect3{Implementation of the <code>NonlinearSchroedingerEquation</code>
  // class}

  //
  template <int dim>
  NonlinearSchroedingerEquation<dim>::NonlinearSchroedingerEquation() :
    fe(2),
    dof_handler(triangulation),
    time(0),
    time_step(1. / 64),
    timestep_number(1),
    kappa(0)
  {}


  // @sect4{WaveEquation::setup_system}

  // The next function is the one that sets up the mesh, DoFHandler, and
  // matrices and vectors at the beginning of the program, i.e. before the
  // first time step. The first few lines are pretty much standard if you've
  // read through the tutorial programs at least up to step-6:
  template <int dim>
  void NonlinearSchroedingerEquation<dim>::setup_system()
  {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(7);

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;

    dof_handler.distribute_dofs(fe);

    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    rhs_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.close();
  }



  template <int dim>
  void NonlinearSchroedingerEquation<dim>::assemble_matrices()
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<std::complex<double>> cell_matrix_lhs(dofs_per_cell,
                                                     dofs_per_cell);
    FullMatrix<std::complex<double>> cell_matrix_rhs(dofs_per_cell,
                                                     dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double>                  potential_values(n_q_points);
    Potential<dim>                       potential;

    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        cell_matrix_lhs = std::complex<double>(0.);
        cell_matrix_rhs = std::complex<double>(0.);

        fe_values.reinit(cell);

        potential.value_list(fe_values.get_quadrature_points(),
                             potential_values);

        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                for (unsigned int l = 0; l < dofs_per_cell; ++l)
                  {
                    const std::complex<double> i(0, 1);

                    cell_matrix_lhs(k, l) +=
                      (-i * fe_values.shape_value(k, q_index) *
                         fe_values.shape_value(l, q_index) +
                       time_step / 4 * fe_values.shape_grad(k, q_index) *
                         fe_values.shape_grad(l, q_index) +
                       time_step / 2 * potential_values[q_index] *
                         fe_values.shape_value(k, q_index) *
                         fe_values.shape_value(l, q_index)) *
                      fe_values.JxW(q_index);

                    cell_matrix_lhs(k, l) +=
                      (-i * fe_values.shape_value(k, q_index) *
                         fe_values.shape_value(l, q_index) -
                       time_step / 4 * fe_values.shape_grad(k, q_index) *
                         fe_values.shape_grad(l, q_index) -
                       time_step / 2 * potential_values[q_index] *
                         fe_values.shape_value(k, q_index) *
                         fe_values.shape_value(l, q_index)) *
                      fe_values.JxW(q_index);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix_lhs, local_dof_indices, system_matrix);
        constraints.distribute_local_to_global(
          cell_matrix_rhs, local_dof_indices, rhs_matrix);
      }
  }

  // $\psi^{(3)}(t_{n+1}) &= e^{-i\kappa|\psi^{(2)}(t_{n+1})|^2 \tfrac
  //  12\Delta t} \; \psi^{(2)}(t_{n+1})$.
  template <int dim>
  void NonlinearSchroedingerEquation<dim>::do_half_phase_step()
  {
    for (auto &value : solution)
      {
        const std::complex<double> i(0, 1);
        const double               magnitude = std::abs(value);

        value = std::exp(-i * kappa * magnitude * magnitude * (time_step / 2)) *
                value;
      }
  }


  namespace
  {
  void split_complex_linear_system (const SparseMatrix<std::complex<double>> &A,
		  const Vector<std::complex<double>> &b,
		  BlockSparseMatrix<double> &split_A,
		  BlockVector<double> &split_b)
  {
	  Assert (A.m() == A.n(), ExcMessage ("The matrix in the linear system is not square."));
	  Assert (A.m() == b.size(), ExcMessage ("Matrix and right hand side sizes do not match."));

	  const unsigned int n = b.size();
	  split_A.reinit ({n,n});
	  split_A.block(0,0).reinit (A.get_sparsity_pattern());
	  split_A.block(0,1).reinit (A.get_sparsity_pattern());
	  split_A.block(1,0).reinit (A.get_sparsity_pattern());
	  split_A.block(1,1).reinit (A.get_sparsity_pattern());
	  split_b.reinit (BlockIndices{n,n});

	  {
		  SparseMatrix<std::complex<double>>::const_iterator it_A = A.begin();
		  SparseMatrix<double>::iterator it_split_A_00 = split_A.block(0,0).begin();
		  SparseMatrix<double>::iterator it_split_A_01 = split_A.block(0,1).begin();
		  SparseMatrix<double>::iterator it_split_A_10 = split_A.block(1,0).begin();
		  SparseMatrix<double>::iterator it_split_A_11 = split_A.block(1,1).begin();

		  for (; it_A!=A.end(); ++it_A)
		  {
			  std::complex<double> A_value = it_A->value();
			  it_split_A_00->value() = std::real(A_value);
			  it_split_A_11->value() = std::real(A_value);
			  it_split_A_01->value() = std::imag(A_value);
			  it_split_A_10->value() = -std::imag(A_value);
		  }
	  }

	  for (unsigned int i=0; i<b.size(); ++i)
	  {
		  split_b.block(0)[i] = std::real(b[i]);
		  split_b.block(1)[i] = std::imag(b[i]);
	  }
  }

  }

  template <int dim>
  void NonlinearSchroedingerEquation<dim>::do_full_spatial_step()
  {
	  rhs_matrix.vmult (system_rhs, solution);

	  BlockSparseMatrix<double> split_A;
	  BlockVector<double> split_b;
	  split_complex_linear_system(system_matrix, system_rhs, split_A, split_b);

	  SparseDirectUMFPACK direct_solver;
	  direct_solver.solve (split_A, split_b);
  }



  namespace DataPostprocessors
  {
    template <int dim>
    class ComplexMagnitude : public DataPostprocessorScalar<dim>
    {
    public:
      ComplexMagnitude();

      virtual void evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>> &               computed_quantities) const;
    };

    template <int dim>
    ComplexMagnitude<dim>::ComplexMagnitude() :
      DataPostprocessorScalar<dim>("Magnitude", update_values)
    {}


    template <int dim>
    void ComplexMagnitude<dim>::evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &               computed_quantities) const
    {
      Assert(computed_quantities.size() == inputs.solution_values.size(),
             ExcDimensionMismatch(computed_quantities.size(),
                                  inputs.solution_values.size()));

      for (unsigned int i = 0; i < computed_quantities.size(); i++)
        {
          Assert(computed_quantities[i].size() == 1,
                 ExcDimensionMismatch(computed_quantities[i].size(), 1));
          Assert(inputs.solution_values[i].size() == 2,
                 ExcDimensionMismatch(inputs.solution_values[i].size(), 2));

          computed_quantities[i](0) = std::abs(std::complex<double>(
            inputs.solution_values[i](0), inputs.solution_values[i](1)));
        }
    }
  } // namespace DataPostprocessors


  template <int dim>
  void NonlinearSchroedingerEquation<dim>::output_results() const
  {
    Vector<double> magnitude(dof_handler.n_dofs());
    Vector<double> phase(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      {
        magnitude(i) = std::abs(solution(i));
        phase(i)     = std::arg(solution(i));
      }

    //    DataPostprocessors::ComplexMagnitude<dim> complex_magnitude;

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "Psi");

    //    data_out.add_data_vector (solution, complex_magnitude);

    data_out.add_data_vector(magnitude, "magnitude");
    data_out.add_data_vector(phase, "phase");
    data_out.build_patches();

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }



  template <int dim>
  void NonlinearSchroedingerEquation<dim>::run()
  {
    setup_system();
    assemble_matrices();

    VectorTools::interpolate(dof_handler, InitialValues<dim>(), solution);

    for (time = 0; time <= 5; time += time_step, ++timestep_number)
      {
        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        do_half_phase_step();
        do_full_spatial_step();
        do_half_phase_step();

        output_results();

        old_solution = solution;
      }
  }
} // namespace Step58



int main()
{
  try
    {
      using namespace dealii;
      using namespace Step58;

      NonlinearSchroedingerEquation<2> nse;
      nse.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}

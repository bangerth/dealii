/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Colorado State University, 2023
 */


#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/petsc_ts.h>

#include <fstream>
#include <iostream>


namespace Step86
{
  using namespace dealii;


  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation();
    void run();

  private:
    void setup_system();

    void implicit_function(const double                      time,
                           const PETScWrappers::MPI::Vector &solution,
                           const PETScWrappers::MPI::Vector &solution_dot,
                           PETScWrappers::MPI::Vector &      dst) const;

    void
    assemble_implicit_jacobian(const double                      time,
                               const PETScWrappers::MPI::Vector &solution_dot,
                               const PETScWrappers::MPI::Vector &solution,
                               const double                      shift);

    void output_results(const double                      time,
                        const PETScWrappers::MPI::Vector &solution,
                        const unsigned int timestep_number) const;

    void solve_with_jacobian(const PETScWrappers::MPI::Vector &src,
                             PETScWrappers::MPI::Vector &      dst) const;

    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    mutable AffineConstraints<double> constraints;

    PETScWrappers::MPI::SparseMatrix jacobian_matrix;

    PETScWrappers::MPI::Vector solution;

    PETScWrappers::TimeStepperData time_stepper_data;
  };



  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>()
      , period(0.2)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    const double period;
  };



  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());

    const double time = this->get_time();
    const double point_within_period =
      (time / period - std::floor(time / period));

    if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
      {
        if ((p[0] > 0.5) && (p[1] > -0.5))
          return 1;
        else
          return 0;
      }
    else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
      {
        if ((p[0] > -0.5) && (p[1] > 0.5))
          return 1;
        else
          return 0;
      }
    else
      return 0;
  }



  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };



  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                    const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }



  template <int dim>
  HeatEquation<dim>::HeatEquation()
    : fe(1)
    , dof_handler(triangulation)
    , time_stepper_data("", "beuler", 0.0, 1.0)
  {}



  template <int dim>
  void HeatEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    BoundaryValues<dim> boundary_values_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             boundary_values_function,
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);

    // directly initialize from dsp, no need for the regular sparsity pattern:
    // mass_matrix.reinit(dof_handler.locally_owned_dofs(), dsp, MPI_COMM_SELF);
    // laplace_matrix.reinit(dof_handler.locally_owned_dofs(), dsp,
    // MPI_COMM_SELF);
    jacobian_matrix.reinit(dof_handler.locally_owned_dofs(),
                           dsp,
                           MPI_COMM_SELF);

    solution.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
  }


  template <int dim>
  void HeatEquation<dim>::implicit_function(
    const double                      time,
    const PETScWrappers::MPI::Vector &solution,
    const PETScWrappers::MPI::Vector &solution_dot,
    PETScWrappers::MPI::Vector &      dst) const
  {
    RightHandSide<dim> rhs_function;
    rhs_function.set_time(time);

    BoundaryValues<dim> boundary_values_function;
    boundary_values_function.set_time(time);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             boundary_values_function,
                                             constraints);

    PETScWrappers::MPI::Vector local_solution(solution);
    PETScWrappers::MPI::Vector local_solution_dot(solution_dot);
    constraints.distribute(local_solution);
    constraints.set_zero(local_solution_dot);


    QGauss<dim>   quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> solution_gradients(n_q_points);
    std::vector<double>         solution_dot_values(n_q_points);

    Vector<double> cell_residual(dofs_per_cell);

    dst = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        fe_values.get_function_gradients(local_solution, solution_gradients);
        fe_values.get_function_values(local_solution_dot, solution_dot_values);

        cell->get_dof_indices(local_dof_indices);

        cell_residual = 0;
        for (const unsigned int q : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            {
              cell_residual(i) +=
                (fe_values.shape_value(i, q) * solution_dot_values[q] +
                 fe_values.shape_grad(i, q) * solution_gradients[q] -
                 rhs_function.value(fe_values.quadrature_point(q)) *
                   fe_values.shape_value(i, q)) *
                fe_values.JxW(q);
            }
        constraints.distribute_local_to_global(cell_residual,
                                               local_dof_indices,
                                               dst);
      }
    dst.compress(VectorOperation::add);
    // Now we correct the entries corresponding to constrained degrees of
    // freedom. local_solution[c] contains the constrained value of the
    // solution at the constrained degree of freedom c.
    for (const auto &c : constraints.get_lines())
      if (c.inhomogeneity != 0.0)
        dst[c.index] = local_solution[c.index] - solution[c.index];
    dst.compress(VectorOperation::insert);
  }


  template <int dim>
  void HeatEquation<dim>::assemble_implicit_jacobian(
    const double,
    const PETScWrappers::MPI::Vector &,
    const PETScWrappers::MPI::Vector &,
    const double shift)
  {
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);

    QGauss<dim>   quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> solution_gradients(n_q_points);
    std::vector<double>         solution_dot_values(n_q_points);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    jacobian_matrix = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        cell->get_dof_indices(local_dof_indices);

        cell_matrix = 0;
        for (const unsigned int q : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              {
                cell_matrix(i, j) +=
                  (shift * fe_values.shape_value(i, q) *
                     fe_values.shape_value(j, q) +
                   fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q)) *
                  fe_values.JxW(q);
              }
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               jacobian_matrix);
      }
    jacobian_matrix.compress(VectorOperation::add);
    // Now we correct the entries corresponding to constrained degrees of
    // freedom. We want the Jacobian to be one on inhomoegeneous Dirichlet BCs
    for (const auto &c : constraints.get_lines())
      if (c.inhomogeneity != 0.0)
        jacobian_matrix.set(c.index, c.index, 1.0);
    jacobian_matrix.compress(VectorOperation::insert);
  }


  template <int dim>
  void
  HeatEquation<dim>::solve_with_jacobian(const PETScWrappers::MPI::Vector &src,
                                         PETScWrappers::MPI::Vector &dst) const
  {
    SolverControl           solver_control(1000, 1e-8 * src.l2_norm());
    PETScWrappers::SolverCG cg(solver_control);

    PETScWrappers::PreconditionSSOR preconditioner;
    preconditioner.initialize(
      jacobian_matrix, PETScWrappers::PreconditionSSOR::AdditionalData(1.0));

    cg.solve(jacobian_matrix, dst, src, preconditioner);

    constraints.distribute(dst);

    std::cout << "     " << solver_control.last_step() << " CG iterations."
              << std::endl;
  }



  template <int dim>
  void
  HeatEquation<dim>::output_results(const double                      time,
                                    const PETScWrappers::MPI::Vector &solution,
                                    const unsigned int timestep_number) const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");

    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }


  template <int dim>
  void HeatEquation<dim>::refine_mesh(const unsigned int min_grid_level,
                                      const unsigned int max_grid_level)
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      solution,
      estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.6,
                                                      0.4);

    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();

    SolutionTransfer<dim, PETScWrappers::MPI::Vector> solution_trans(
      dof_handler);

    PETScWrappers::MPI::Vector previous_solution;
    previous_solution = solution;

    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    triangulation.execute_coarsening_and_refinement();
    setup_system();

    solution_trans.interpolate(previous_solution, solution);
    constraints.distribute(solution);
  }



  template <int dim>
  void HeatEquation<dim>::run()
  {
    const unsigned int initial_global_refinement = 5;
    // const unsigned int n_adaptive_pre_refinement_steps = 0;

    GridGenerator::hyper_L(triangulation);
    triangulation.refine_global(initial_global_refinement);

    setup_system();

    VectorTools::interpolate(dof_handler,
                             Functions::ZeroFunction<dim>(),
                             solution);

    PETScWrappers::TimeStepper<PETScWrappers::MPI::Vector,
                               PETScWrappers::MPI::SparseMatrix>
      petsc_ts(time_stepper_data);

    petsc_ts.set_matrix(jacobian_matrix);
    petsc_ts.implicit_function =
      [&](const auto t, const auto &y, const auto &y_dot, auto &res) {
        this->implicit_function(t, y, y_dot, res);
      };

    petsc_ts.setup_jacobian =
      [&](const auto t, const auto &y, const auto &y_dot, const auto alpha) {
        this->assemble_implicit_jacobian(t, y, y_dot, alpha);
      };

    petsc_ts.solve_with_jacobian = [&](const auto &src, auto &dst) {
      this->solve_with_jacobian(src, dst);
    };

    petsc_ts.monitor =
      [&](const auto t, const auto &solution, const auto step_number) {
        std::cout << "Time step " << step_number << " at t=" << t << std::endl;
        this->output_results(t, solution, step_number);
      };


    petsc_ts.solve(solution);
  }
} // namespace Step86



int main(int argc, char **argv)
{
  try
    {
      using namespace Step86;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      HeatEquation<2> heat_equation_solver;
      heat_equation_solver.run();
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

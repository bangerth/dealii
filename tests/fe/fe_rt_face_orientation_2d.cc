// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2018 by the deal.II authors
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


// 

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/vector.h>

#include <sstream>

#include "../tests.h"



template <int dim>
void
test(const unsigned int degree)
{
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube (triangulation);
  triangulation.refine_global(1);

  FE_RaviartThomas<dim> fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::vector<Vector<double>> shape_functions(dof_handler.n_dofs(),
                                              Vector<double>(dof_handler.n_dofs()));
  for (unsigned int i=0; i<dof_handler.n_dofs(); ++i)
    shape_functions[i][i] = 1;
  
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  for (unsigned int i=0; i<dof_handler.n_dofs(); ++i)
    data_out.add_data_vector(shape_functions[i],
                             std::vector<std::string>(dim,"phi_" + std::to_string(i)),
                             DataOut<dim>::type_automatic,
                             std::vector<DataComponentInterpretation::DataComponentInterpretation>(dim, DataComponentInterpretation::component_is_part_of_vector));
  data_out.build_patches(4);
  
  data_out.write_vtk(deallog.get_file_stream());
}



int
main()
{
  std::ofstream logfile("output");
  deallog << std::setprecision(3);

  deallog.attach(logfile);

  test<2>(0);
}

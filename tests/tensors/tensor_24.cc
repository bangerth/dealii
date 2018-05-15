// ---------------------------------------------------------------------
//
// Copyright (C) 2006 - 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// check Tensor<2,dim>::component_to_unrolled_index and the
// other way round

#include "../tests.h"
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>


template <int dim>
void
check ()
{
  typedef Tensor<2,dim> S;
  for (unsigned int i=0; i<S::n_independent_components; ++i)
    {
      deallog << i << "  --  "
              << S::unrolled_to_component_indices (i)
              << std::endl;
      Assert (S::component_to_unrolled_index
              (S::unrolled_to_component_indices (i))
              ==
              i,
              ExcInternalError());
    }
}


int
main ()
{
  std::ofstream logfile("output");
  deallog << std::setprecision(3);
  deallog.attach(logfile);

  check<1> ();
  check<2> ();
  check<3> ();

  deallog << "OK" << std::endl;
}

// ---------------------------------------------------------------------
//
// Copyright (C) 2010 - 2017 by the deal.II authors
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


// check serialization for Table<2, int>

#include "serialization.h"
#include <deal.II/base/table.h>
#include <boost/serialization/vector.hpp>


void
test ()
{
  unsigned int index1 = 3, index2 = 4;
  TableIndices<2> indices1(index1, index2);
  unsigned int sum_of_indices = index1 + index2;

  Table<2, int> t1(index1, index2);
  Table<2, int> t2(index1, index2);

  index1 = 2;
  index2 = 5;
  Table<2, int> t3(index1, index2);

  unsigned int counter = 0;
  for (unsigned int i1 = 0; i1 < indices1[0]; ++i1)
    {
      for (unsigned int i2 = 0; i2 < indices1[1]; ++i2)
        {
          t1[i1][i2] = counter ++;
          t2[i1][i2] = counter + sum_of_indices;
        }
    }
  verify (t1, t2);

  verify (t1, t3);
}


int
main ()
{
  std::ofstream logfile("output");
  deallog << std::setprecision(3);
  deallog.attach(logfile);

  test ();

  deallog << "OK" << std::endl;
}

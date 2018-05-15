// ---------------------------------------------------------------------
//
// Copyright (C) 2017 by the deal.II authors
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


// Similar to tensor_product_matrix_02.cc except testing with
// different mass and laplace matrices for each tensor direction, respectively.

#include "../tests.h"
#include "../testmatrix.h"
#include <deal.II/lac/tensor_product_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>


template <int dim, int size>
void
do_test()
{
  deallog << "Testing dim=" << dim << ", degree=" << size << std::endl;
  FullMatrix<double> init_mass(size, size);
  FullMatrix<double> init_laplace(size, size);
  for (unsigned int i=0; i<size; ++i)
    {
      init_mass(i,i) = 2./3. ;
      if (i > 0)
        init_mass(i,i-1) = 1./6. ;
      if (i<size-1)
        init_mass(i,i+1) = 1./6. ;
      init_laplace(i,i) = 2. ;
      if (i > 0)
        init_laplace(i,i-1) = -1. ;
      if (i < size-1)
        init_laplace(i,i+1) = -1. ;
    }

  std::array<FullMatrix<double>, dim> mass ;
  std::array<FullMatrix<double>, dim> laplace ;
  for (unsigned int dir = 0; dir<dim; ++dir)
    {
      for (unsigned int i=0; i<size; ++i)
        {
          init_mass(i,i) *= 4./3. ;
          init_laplace(i,i) *= 5./4. ;
        }
      mass[dir] = init_mass ;
      laplace[dir] = init_laplace ;
    }

  TensorProductMatrixSymmetricSum<dim,double,size> mat;
  mat.reinit(mass, laplace);
  Vector<double> v1(mat.m()), v2(mat.m()), v3(mat.m());
  for (unsigned int i=0; i<v1.size(); ++i)
    v1(i) = (2*i+1)%23;

  const ArrayView<double> view1(v1.begin(), v1.size()) ;
  const ArrayView<double> view2(v2.begin(), v2.size()) ;
  const ArrayView<double> view3(v3.begin(), v3.size()) ;
  mat.vmult(view2, view1);
  mat.apply_inverse(view3, view2);
  v3 -= v1;
  deallog << "Verification of vmult and inverse: " << v3.linfty_norm() << std::endl;

  FullMatrix<double> full(v1.size(), v1.size());
  full = 0. ;
  for (unsigned int dir = 0; dir<dim; ++dir)
    for (unsigned int i=0, c=0; i<(dim>2?size:1); ++i)
      for (unsigned int j=0; j<(dim>1?size:1); ++j)
        for (unsigned int k=0; k<size; ++k, ++c)
          for (unsigned int ii=0, cc=0; ii<(dim>2?size:1); ++ii)
            for (unsigned int jj=0; jj<(dim>1?size:1); ++jj)
              for (unsigned int kk=0; kk<size; ++kk, ++cc)
                if (dim == 1)
                  full(c,cc) = laplace[0](k,kk);
                else if (dim==2)
                  full(c,cc) = laplace[1](j,jj)*mass[0](k,kk) + mass[1](j,jj)*laplace[0](k,kk);
                else if (dim==3)
                  full(c,cc)
                    = laplace[2](i,ii) * mass[1](j,jj)*mass[0](k,kk)
                      + mass[2](i,ii) * (laplace[1](j,jj)*mass[0](k,kk) + mass[1](j,jj)*laplace[0](k,kk));
  full.vmult(v3, v1);
  v3 -= v2;
  deallog << "Verification of vmult: " << v3.linfty_norm() << std::endl;

  full.gauss_jordan();
  full.vmult(v3, v1);
  mat.apply_inverse(view2, view1);
  v3 -= v2;
  deallog << "Verification of inverse: " << v3.linfty_norm() << std::endl;
}


int
main()
{
  initlog();

  do_test<1,1>();
  do_test<1,2>();
  do_test<1,5>();
  do_test<2,1>();
  do_test<2,2>();
  do_test<2,5>();
  do_test<2,11>();
  do_test<3,1>();
  do_test<3,2>();
  do_test<3,3>();
  do_test<3,7>();

  return 0;
}

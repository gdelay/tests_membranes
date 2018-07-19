/*
 *       /\         DISK++, a template library for DIscontinuous SKeletal
 *      /__\        methods.
 *     /_\/_\
 *    /\    /\      Matteo Cicuttin (C) 2016, 2017, 2018
 *   /__\  /__\     matteo.cicuttin@enpc.fr
 *  /_\/_\/_\/_\    École Nationale des Ponts et Chaussées - CERMICS
 *
 * This file is copyright of the following authors:
 * Matteo Cicuttin (C) 2016, 2017, 2018         matteo.cicuttin@enpc.fr
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * If you use this code or parts of it for scientific publications, you
 * are required to cite it as following:
 *
 * Implementation of Discontinuous Skeletal methods on arbitrary-dimensional,
 * polytopal meshes using generic programming.
 * M. Cicuttin, D. A. Di Pietro, A. Ern.
 * Journal of Computational and Applied Mathematics.
 * DOI: 10.1016/j.cam.2017.09.017
 */


#include <iostream>
#include <iomanip>
#include <regex>

#include <unistd.h>

#include "bases/bases.hpp"
#include "revolution/quadratures"
#include "methods/hho"

#include "core/loaders/loader.hpp"

#include "common.hpp"

template<typename Mesh>
struct test_functor
{
    /* Expect k+1 convergence */
    typename Mesh::scalar_type
    operator()(const Mesh& msh, size_t degree) const
    {
        typedef Mesh mesh_type;
        typedef typename mesh_type::cell        cell_type;
        typedef typename mesh_type::face        face_type;
        typedef typename mesh_type::scalar_type scalar_type;
        typedef typename mesh_type::point_type  point_type;

        typedef Matrix<scalar_type, Mesh::dimension, 1> ret_type;

        auto f = make_vector_testing_data(msh);

        typename disk::hho_degree_info hdi(degree);

        scalar_type error = 0.0;
        for (auto& cl : msh)
        {
            Matrix<scalar_type, Dynamic, 1> proj = disk::project_function(msh, cl, hdi, f);
            auto gr = disk::make_hho_vector_laplacian(msh, cl, hdi);

            size_t rec_size = disk::vector_basis_size(hdi.reconstruction_degree(), Mesh::dimension, Mesh::dimension);

            Matrix<scalar_type, Dynamic, 1> reconstr = Matrix<scalar_type, Dynamic, 1>::Zero(rec_size);
            reconstr.tail(rec_size-Mesh::dimension) = gr.first * proj;

            for (size_t i = 0; i < Mesh::dimension; i++)
                reconstr(i) = proj(i);

            auto cb = disk::make_vector_monomial_basis(msh, cl, hdi.reconstruction_degree());
            Matrix<scalar_type, Dynamic, Dynamic> mass = disk::make_mass_matrix(msh, cl, cb);
            Matrix<scalar_type, Dynamic, 1> rhs = disk::make_rhs(msh, cl, cb, f);
            Matrix<scalar_type, Dynamic, 1> exp_reconstr = mass.llt().solve(rhs);

            Matrix<scalar_type, Dynamic, 1> diff = reconstr - exp_reconstr;

            Matrix<scalar_type, Dynamic, Dynamic> stiffness = disk::make_stiffness_matrix(msh, cl, cb);

            error += diff.dot(stiffness*diff);
        }

        return std::sqrt(error);
    }

    size_t
    expected_rate(size_t k)
    {
        return k+1;
    }
};

int main(void)
{
    tester<test_functor> tstr;
    tstr.run();
    return 0;
}

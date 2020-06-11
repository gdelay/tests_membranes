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

#pragma once

#include <iomanip>

#include "core/loaders/loader.hpp"
#include "contrib/sol2/sol.hpp"
#include "contrib/colormanip.h"

const size_t MIN_TEST_DEGREE = 0;
const size_t MAX_TEST_DEGREE = 3;

using namespace Eigen;

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, T ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
    // unless the result is subnormal
           || std::abs(x-y) < std::numeric_limits<T>::min();
}


/*****************************************************************************************/
template<typename Mesh>
struct scalar_testing_function;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct scalar_testing_function< Mesh<T,2,Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
    }
};

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct scalar_testing_function< Mesh<T,3,Storage> >
{
    typedef Mesh<T,3,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y()) * std::sin(M_PI * pt.z());
    }
};

template<typename Mesh>
auto make_scalar_testing_data(const Mesh& msh)
{
    return scalar_testing_function<Mesh>();
}

/*****************************************************************************************/
template<typename Mesh>
struct scalar_testing_function_grad;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct scalar_testing_function_grad<Mesh<T, 2, Storage>>
{
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type      point_type;
    typedef Matrix<scalar_type, 2, 1>           vector_type;

    vector_type
    operator()(const point_type& pt) const
    {
        vector_type ret;
        ret(0) = M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y());
        ret(1) = M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y());

        return ret;
    }
};

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct scalar_testing_function_grad<Mesh<T, 3, Storage>>
{
    typedef Mesh<T, 3, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type      point_type;
    typedef Matrix<scalar_type, 3, 1>           vector_type;

    vector_type
    operator()(const point_type& pt) const
    {
        vector_type ret;
        ret(0) = M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y()) * std::sin(M_PI * pt.z());
        ret(1) = M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y()) * std::sin(M_PI * pt.z());
        ret(2) = M_PI * std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y()) * std::cos(M_PI * pt.z());

        return ret;
    }
};


template<typename Mesh>
auto
make_scalar_testing_data_grad(const Mesh& msh)
{
    return scalar_testing_function_grad<Mesh>();
}

/*****************************************************************************************/
template<typename Mesh>
struct vector_testing_function;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct vector_testing_function< Mesh<T,2,Storage> >
{
	typedef Mesh<T,2,Storage> 				mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
    typedef Matrix<scalar_type, 2, 1> 		ret_type;

    ret_type operator()(const point_type& pt) const
    {
    	ret_type ret;
        ret(0) = std::sin(2. * M_PI * pt.x());
        ret(1) = std::sin(2. * M_PI * pt.y());
        return ret;
    }
};

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct vector_testing_function< Mesh<T,3,Storage> >
{
	typedef Mesh<T,3,Storage> 				mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
    typedef Matrix<scalar_type, 3, 1> 		ret_type;

    ret_type operator()(const point_type& pt) const
    {
    	ret_type ret;
        ret(0) = std::sin(2. * M_PI * pt.x());
        ret(1) = std::sin(2. * M_PI * pt.y());
        ret(2) = std::sin(2. * M_PI * pt.z());
        return ret;
    }
};

template<typename Mesh>
auto make_vector_testing_data(const Mesh& msh)
{
	return vector_testing_function<Mesh>();
}

/*****************************************************************************************/
template<typename Mesh>
struct vector_testing_function_div;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct vector_testing_function_div< Mesh<T,2,Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
    typedef Matrix<scalar_type, 2, 1>       ret_type;

    scalar_type operator()(const point_type& pt) const
    {
        return 2. * M_PI * std::cos(2. * M_PI * pt.x()) +
               2. * M_PI * std::cos(2. * M_PI * pt.y());
    }
};

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct vector_testing_function_div< Mesh<T,3,Storage> >
{
    typedef Mesh<T,3,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
    typedef Matrix<scalar_type, 3, 1>       ret_type;

    scalar_type operator()(const point_type& pt) const
    {
        return 2. * M_PI * std::cos(2. * M_PI * pt.x()) +
               2. * M_PI * std::cos(2. * M_PI * pt.y()) +
               2. * M_PI * std::cos(2. * M_PI * pt.z());
    }
};

template<typename Mesh>
auto make_vector_testing_data_div(const Mesh& msh)
{
    return vector_testing_function_div<Mesh>();
}

/*****************************************************************************************/

template<typename T>
std::vector< disk::generic_mesh<T, 2> >
get_triangle_generic_meshes(void)
{
	std::vector<std::string> meshfiles;
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_1.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_2.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_3.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_4.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_5.typ1");

    typedef disk::generic_mesh<T, 2>  mesh_type;

    std::vector< mesh_type > ret;
    for (size_t i = 0; i < meshfiles.size(); i++)
    {
        mesh_type msh;
        disk::fvca5_mesh_loader<T, 2> loader;

        if (!loader.read_mesh(meshfiles.at(i)))
        {
            std::cout << "Problem loading mesh." << std::endl;
            continue;
        }
        loader.populate_mesh(msh);

        ret.push_back(msh);
    }

    return ret;
}

template<typename T>
std::vector< disk::generic_mesh<T, 2> >
get_polygonal_generic_meshes(void)
{
	std::vector<std::string> meshfiles;
    meshfiles.push_back("../../../diskpp/meshes/2D_hex/fvca5/hexagonal_1.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_hex/fvca5/hexagonal_2.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_hex/fvca5/hexagonal_3.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_hex/fvca5/hexagonal_4.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_hex/fvca5/hexagonal_5.typ1");

    typedef disk::generic_mesh<T, 2>  mesh_type;

    std::vector< mesh_type > ret;
    for (size_t i = 0; i < meshfiles.size(); i++)
    {
        mesh_type msh;
        disk::fvca5_mesh_loader<T, 2> loader;

        if (!loader.read_mesh(meshfiles.at(i)))
        {
            std::cout << "Problem loading mesh." << std::endl;
            continue;
        }
        loader.populate_mesh(msh);

        ret.push_back(msh);
    }

    return ret;
}

template<typename T>
std::vector< disk::simplicial_mesh<T, 2> >
get_triangle_netgen_meshes(void)
{
	std::vector<std::string> meshfiles;
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/tri01.mesh2d");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/tri02.mesh2d");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/tri03.mesh2d");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/tri04.mesh2d");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/tri05.mesh2d");


    typedef disk::simplicial_mesh<T, 2>  mesh_type;

    std::vector< mesh_type > ret;
    for (size_t i = 0; i < meshfiles.size(); i++)
    {
        mesh_type msh;
        disk::netgen_mesh_loader<T, 2> loader;

        if (!loader.read_mesh(meshfiles.at(i)))
        {
            std::cout << "Problem loading mesh." << std::endl;
            continue;
        }
        loader.populate_mesh(msh);

        ret.push_back(msh);
    }

    return ret;
}

template<typename T>
std::vector< disk::simplicial_mesh<T, 3> >
get_tetrahedra_netgen_meshes(void)
{
    std::vector<std::string> meshfiles;
    meshfiles.push_back("../../../diskpp/meshes/3D_tetras/netgen/cube1.mesh");
    meshfiles.push_back("../../../diskpp/meshes/3D_tetras/netgen/cube2.mesh");
    meshfiles.push_back("../../../diskpp/meshes/3D_tetras/netgen/cube3.mesh");
    meshfiles.push_back("../../../diskpp/meshes/3D_tetras/netgen/cube4.mesh");
    meshfiles.push_back("../../../diskpp/meshes/3D_tetras/netgen/cube5.mesh");


    typedef disk::simplicial_mesh<T, 3>  mesh_type;

    std::vector< mesh_type > ret;
    for (size_t i = 0; i < meshfiles.size(); i++)
    {
        mesh_type msh;
        disk::netgen_mesh_loader<T, 3> loader;

        if (!loader.read_mesh(meshfiles.at(i)))
        {
            std::cout << "Problem loading mesh." << std::endl;
            continue;
        }
        loader.populate_mesh(msh);

        ret.push_back(msh);
    }

    return ret;
}

template<typename T>
std::vector< disk::cartesian_mesh<T, 3> >
get_cartesian_3d_diskpp_meshes(void)
{
    std::vector<std::string> meshfiles;
    meshfiles.push_back("../../../diskpp/meshes/3D_hexa/diskpp/testmesh-2-2-2.hex");
    meshfiles.push_back("../../../diskpp/meshes/3D_hexa/diskpp/testmesh-4-4-4.hex");
    meshfiles.push_back("../../../diskpp/meshes/3D_hexa/diskpp/testmesh-8-8-8.hex");
    meshfiles.push_back("../../../diskpp/meshes/3D_hexa/diskpp/testmesh-16-16-16.hex");
    meshfiles.push_back("../../../diskpp/meshes/3D_hexa/diskpp/testmesh-32-32-32.hex");

    typedef disk::cartesian_mesh<T, 3>  mesh_type;

    std::vector< mesh_type > ret;
    for (size_t i = 0; i < meshfiles.size(); i++)
    {
        mesh_type msh;
        disk::cartesian_mesh_loader<T, 3> loader;

        if (!loader.read_mesh(meshfiles.at(i)))
        {
            std::cout << "Problem loading mesh." << std::endl;
            continue;
        }
        loader.populate_mesh(msh);

        ret.push_back(msh);
    }

    return ret;
}

template<typename T>
std::vector< disk::generic_mesh<T, 3> >
get_generic_fvca6_meshes(void)
{
    std::vector<std::string> meshfiles;
    meshfiles.push_back("../../../diskpp/meshes/3D_general/fvca6/dbls_10.msh");
    meshfiles.push_back("../../../diskpp/meshes/3D_general/fvca6/dbls_20.msh");
    meshfiles.push_back("../../../diskpp/meshes/3D_general/fvca6/dbls_30.msh");
    meshfiles.push_back("../../../diskpp/meshes/3D_general/fvca6/dbls_40.msh");

    typedef disk::generic_mesh<T, 3>  mesh_type;

    std::vector< mesh_type > ret;
    for (size_t i = 0; i < meshfiles.size(); i++)
    {
        mesh_type msh;
        disk::fvca6_mesh_loader<T, 3> loader;

        if (!loader.read_mesh(meshfiles.at(i)))
        {
            std::cout << "Problem loading mesh." << std::endl;
            continue;
        }
        loader.populate_mesh(msh);

        ret.push_back(msh);
    }

    return ret;
}

template<typename T>
std::vector<disk::generic_mesh<T, 3>>
get_tetrahedra_fvca6_meshes(void)
{
    std::vector<std::string> meshfiles;
    meshfiles.push_back("../../../diskpp/meshes/3D_tetras/fvca6/tet.0.msh");
    meshfiles.push_back("../../../diskpp/meshes/3D_tetras/fvca6/tet.1.msh");
    meshfiles.push_back("../../../diskpp/meshes/3D_tetras/fvca6/tet.2.msh");
    meshfiles.push_back("../../../diskpp/meshes/3D_tetras/fvca6/tet.3.msh");

    typedef disk::generic_mesh<T, 3> mesh_type;

    std::vector<mesh_type> ret;
    for (size_t i = 0; i < meshfiles.size(); i++)
    {
        mesh_type                     msh;
        disk::fvca6_mesh_loader<T, 3> loader;

        if (!loader.read_mesh(meshfiles.at(i)))
        {
            std::cout << "Problem loading mesh." << std::endl;
            continue;
        }
        loader.populate_mesh(msh);

        ret.push_back(msh);
    }

    return ret;
}

template<typename T>
std::vector< disk::generic_mesh<T, 2> >
get_quad_generic_meshes(void)
{
	std::vector<std::string> meshfiles;
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_1.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_2.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_3.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_4.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_5.typ1");

    typedef disk::generic_mesh<T, 2>  mesh_type;

    std::vector< mesh_type > ret;
    for (size_t i = 0; i < meshfiles.size(); i++)
    {
        mesh_type msh;
        disk::fvca5_mesh_loader<T, 2> loader;

        if (!loader.read_mesh(meshfiles.at(i)))
        {
            std::cout << "Problem loading mesh." << std::endl;
            continue;
        }
        loader.populate_mesh(msh);

        /*
        // ADD A RANDOM TRANSFORM HERE
        auto tr = [](const typename mesh_type::point_type& pt) -> auto {

            auto px = -1 * ( 1-pt.x() ) + 1 * pt.x();
            auto py = -1 * ( 1-pt.y() ) + 1 * pt.y();
            return typename mesh_type::point_type({px, py});
        };

        msh.transform(tr);
        */

        ret.push_back(msh);
    }

    return ret;
}

template<typename T>
std::vector< disk::cartesian_mesh<T, 2> >
get_cartesian_2d_diskpp_meshes(void)
{
    std::vector<std::string> meshfiles;
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/diskpp/testmesh-2-2.quad");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/diskpp/testmesh-4-4.quad");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/diskpp/testmesh-8-8.quad");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/diskpp/testmesh-16-16.quad");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/diskpp/testmesh-32-32.quad");
    // meshfiles.push_back("../../../diskpp/meshes/2D_quads/diskpp/testmesh-64-64.quad");

    typedef disk::cartesian_mesh<T, 2>  mesh_type;

    std::vector< mesh_type > ret;
    for (size_t i = 0; i < meshfiles.size(); i++)
    {
        mesh_type msh;
        disk::cartesian_mesh_loader<T, 2> loader;

        if (!loader.read_mesh(meshfiles.at(i)))
        {
            std::cout << "Problem loading mesh." << std::endl;
            continue;
        }
        loader.populate_mesh(msh);

        ret.push_back(msh);
    }

    return ret;
}

template<typename Mesh, typename Function>
void
do_testing(std::vector<Mesh>& meshes, const Function& run_test,
           const std::function<size_t(size_t)>& expected_rate,
           size_t min_test_degree = MIN_TEST_DEGREE,
           size_t max_test_degree = MAX_TEST_DEGREE)
{
	using T = typename Mesh::coordinate_type;

	for (size_t k = min_test_degree; k <= max_test_degree; k++)
    {
        std::cout << "  Testing degree " << k << " (expected rate is ";
        std::cout << expected_rate(k) << ")" << std::endl;

        std::vector<T> mesh_hs;
        std::vector<T> l2_errors;

        for(auto& msh : meshes)
        {
            auto error = run_test(msh, k);
            mesh_hs.push_back( disk::average_diameter(msh) );
            l2_errors.push_back(error);
        }

        for (size_t i = 0; i < mesh_hs.size(); i++)
        {
            if (i == 0)
            {
                std::cout << "    ";
                std::cout << std::scientific << std::setprecision(5) << mesh_hs.at(i) << "    ";
                std::cout << std::scientific << std::setprecision(5) << l2_errors.at(i);
                std::cout << "     -- " << std::endl;
            }
            else
            {
                auto rate = std::log( l2_errors.at(i)/l2_errors.at(i-1) ) /
                            std::log( mesh_hs.at(i)/mesh_hs.at(i-1) );
                std::cout << "    ";
                std::cout << std::scientific << std::setprecision(5) << mesh_hs.at(i) << "    ";
                std::cout << std::scientific << std::setprecision(5) << l2_errors.at(i) << "    ";
                std::cout << std::defaultfloat << std::setprecision(3) << rate << "    ";

                if ( rate < expected_rate(k)-0.5 )
                    std::cout << "[" << red << "FAIL" << nocolor << "]";
                else if ( rate > expected_rate(k)+0.5 )
                    std::cout << "[" << yellow << "FAIL" << nocolor << "]";
                else
                    std::cout << "[" << green << " OK " << nocolor << "]";

                std::cout << std::endl;
            }
        }
    }
}

template< template<typename> class TestFunctor >
class tester
{
    template<typename Mesh>
    TestFunctor<Mesh>
    get_test_functor(const std::vector<Mesh>& meshes)
    {
        return TestFunctor<Mesh>();
    }

    void
    test_triangles_generic(size_t min_degree = MIN_TEST_DEGREE, size_t max_degree = MAX_TEST_DEGREE)
    {
        std::cout << yellow << "Mesh under test: triangles on generic mesh";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_triangle_generic_meshes<T>();
        auto tf = get_test_functor(meshes);
        auto er = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er, min_degree, max_degree);
    }

    void
    test_polygonal_generic(size_t min_degree = MIN_TEST_DEGREE, size_t max_degree = MAX_TEST_DEGREE)
    {
        std::cout << yellow << "Mesh under test: polygons on generic mesh";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_polygonal_generic_meshes<T>();
        auto tf = get_test_functor(meshes);
        auto er = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er, min_degree, max_degree);
    }

    void
    test_triangles_netgen(size_t min_degree = MIN_TEST_DEGREE, size_t max_degree = MAX_TEST_DEGREE)
    {
        std::cout << yellow << "Mesh under test: triangles on netgen mesh";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_triangle_netgen_meshes<T>();
        auto tf = get_test_functor(meshes);
        auto er = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er, min_degree, max_degree);
    }

    void
    test_quads(size_t min_degree = MIN_TEST_DEGREE, size_t max_degree = MAX_TEST_DEGREE)
    {
        std::cout << yellow << "Mesh under test: quads on generic mesh";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_quad_generic_meshes<T>();
        auto tf = get_test_functor(meshes);
        auto er = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er, min_degree, max_degree);
    }

    void
    test_cartesian_2d_diskpp(size_t min_degree = MIN_TEST_DEGREE, size_t max_degree = MAX_TEST_DEGREE)
    {
        std::cout << yellow << "Mesh under test: 2D cartesian mesh (DiSk++)";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_cartesian_2d_diskpp_meshes<T>();
        auto tf = get_test_functor(meshes);
        auto er = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er, min_degree, max_degree);
    }

    void
    test_tetrahedra_netgen(size_t min_degree = MIN_TEST_DEGREE, size_t max_degree = MAX_TEST_DEGREE)
    {
        std::cout << yellow << "Mesh under test: tetrahedra on netgen mesh";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_tetrahedra_netgen_meshes<T>();
        auto tf = get_test_functor(meshes);
        auto er = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er, min_degree, max_degree);
    }

    void
    test_cartesian_3d_diskpp(size_t min_degree = MIN_TEST_DEGREE, size_t max_degree = MAX_TEST_DEGREE)
    {
        std::cout << yellow << "Mesh under test: 3D cartesian mesh (DiSk++)";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_cartesian_3d_diskpp_meshes<T>();
        auto tf = get_test_functor(meshes);
        auto er = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er, min_degree, max_degree);
    }

    void
    test_generic_fvca6(size_t min_degree = MIN_TEST_DEGREE, size_t max_degree = MAX_TEST_DEGREE)
    {
        std::cout << yellow << "Mesh under test: polyhedra on generic mesh";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_generic_fvca6_meshes<T>();
        auto tf = get_test_functor(meshes);
        auto er = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er, min_degree, max_degree);
    }

public:
  int
  run(size_t min_degree = MIN_TEST_DEGREE, size_t max_degree = MAX_TEST_DEGREE)
  {
      sol::state lua;

      bool crash_on_nan           = false;
      bool do_triangles_generic   = true;
      bool do_polygonal_generic   = true;
      bool do_triangles_netgen    = true;
      bool do_quads               = true;
      bool do_cartesian_2d_diskpp = true;
      bool do_tetrahedra_netgen   = true;
      bool do_cartesian_3d_diskpp = true;
      bool do_generic_fvca6       = true;

      auto r = lua.do_file("test_config.lua");
      if (r.valid())
      {
          crash_on_nan           = lua["crash_on_nan"].get_or(false);
          do_triangles_generic   = lua["do_triangles_generic"].get_or(false);
          do_polygonal_generic   = lua["do_polygonal_generic"].get_or(false);
          do_triangles_netgen    = lua["do_triangles_netgen"].get_or(false);
          do_quads               = lua["do_quads"].get_or(false);
          do_cartesian_2d_diskpp = lua["do_cartesian_2d_diskpp"].get_or(false);
          do_tetrahedra_netgen   = lua["do_tetrahedra_netgen"].get_or(false);
          do_cartesian_3d_diskpp = lua["do_cartesian_3d_diskpp"].get_or(false);
          do_generic_fvca6       = lua["do_generic_fvca6"].get_or(false);
      }

      if (crash_on_nan)
          _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);

      if (do_triangles_generic)
          test_triangles_generic(min_degree, max_degree);

      if (do_triangles_netgen)
          test_triangles_netgen(min_degree, max_degree);

      if (do_polygonal_generic)
          test_polygonal_generic(min_degree, max_degree);

      if (do_quads)
          test_quads(min_degree, max_degree);

      if (do_cartesian_2d_diskpp)
          test_cartesian_2d_diskpp(min_degree, max_degree);

      if (do_tetrahedra_netgen)
          test_tetrahedra_netgen(min_degree, max_degree);

      if (do_cartesian_3d_diskpp)
          test_cartesian_3d_diskpp(min_degree, max_degree);

      if (do_generic_fvca6)
          test_generic_fvca6(min_degree, max_degree);

      return 0;
    }
};

template<template<typename> class TestFunctor>
class tester_simplicial
{
    template<typename Mesh>
    TestFunctor<Mesh>
    get_test_functor(const std::vector<Mesh>& meshes)
    {
        return TestFunctor<Mesh>();
    }

    void
    test_triangles_generic(void)
    {
        std::cout << yellow << "Mesh under test: triangles on generic mesh";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_triangle_generic_meshes<T>();
        auto tf     = get_test_functor(meshes);
        auto er     = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er);
    }

    void
    test_triangles_netgen(void)
    {
        std::cout << yellow << "Mesh under test: triangles on netgen mesh";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_triangle_netgen_meshes<T>();
        auto tf     = get_test_functor(meshes);
        auto er     = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er);
    }

    void
    test_tetrahedra_netgen(void)
    {
        std::cout << yellow << "Mesh under test: tetrahedra on netgen mesh";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_tetrahedra_netgen_meshes<T>();
        auto tf     = get_test_functor(meshes);
        auto er     = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er);
    }

    void
    test_tetrahedra_fvca6(void)
    {
        std::cout << yellow << "Mesh under test: tetrahedra on generic mesh";
        std::cout << nocolor << std::endl;
        using T = double;

        auto meshes = get_tetrahedra_fvca6_meshes<T>();
        auto tf     = get_test_functor(meshes);
        auto er     = [&](size_t k) { return tf.expected_rate(k); };
        do_testing(meshes, tf, er);
    }

  public:
    int
    run(void)
    {
        sol::state lua;

        bool crash_on_nan         = false;
        bool do_triangles_generic = true;
        bool do_triangles_netgen  = true;
        bool do_tetrahedra_netgen = true;
        bool do_tetrahedra_fvca6  = true;

        auto r = lua.do_file("test_config.lua");
        if (r.valid())
        {
            crash_on_nan         = lua["crash_on_nan"].get_or(false);
            do_triangles_generic = lua["do_triangles_generic"].get_or(false);
            do_triangles_netgen  = lua["do_triangles_netgen"].get_or(false);
            do_tetrahedra_netgen = lua["do_tetrahedra_netgen"].get_or(false);
            do_tetrahedra_fvca6  = lua["do_tetrahedra_fvca6"].get_or(false);
        }

        if (crash_on_nan)
            _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);

        if (do_triangles_generic)
            test_triangles_generic();

        if (do_triangles_netgen)
            test_triangles_netgen();

        if (do_tetrahedra_netgen)
            test_tetrahedra_netgen();

        if (do_tetrahedra_fvca6)
            test_tetrahedra_fvca6();

        return 0;
    }
};